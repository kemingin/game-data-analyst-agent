# -*- coding: utf-8 -*-
"""
Function Calling 工具层（Phase 3）
================================================================================
【工具层在架构里的位置】

      LLM（只做决策：「我该调哪个工具、传什么参数」）
              ↓ tool_calls
      本模块（只做执行：「验证参数 → 走 Phase 2 的链路 → 把结果整理成文本」）
              ↓ tool 消息
      LLM（基于真实数据作答）

  这一层是「LLM 与现实世界之间唯一的接口」，也是整个项目安全设计的落点：
  **LLM 永远拿不到数据库连接，也永远写不出 SQL。** 它只能点菜（选 metric_id），
  做菜（生成 SQL、校验、执行）全部由程序完成。

【为什么只给 3 个工具？】

  工具数量与模型选错率成正比。这 3 个工具刚好构成一条不可跳过的路径：

      list_metrics        不确定用户问的是哪个指标 → 先看目录
      get_metric_detail   确认口径、参数名、取值范围
      query_metric        拿真实数字 ← **唯一**能产出数字的工具

  这么设计的直接收益：只要「所有数字都必须来自 query_metric」这一条成立，
  模型就没有编造数据的空间。相比在提示词里写十遍「不要编造」，
  用工具集来约束是机制层面的保证，可靠得多。

【为什么工具的返回值要同时给「文本」和「结构化数据」？】

  · content —— Markdown/JSON 文本，喂给 LLM 看（它只能读文本）
  · data    —— 结构化字典，留给 Phase 4 前端画图表用

  两者由同一次查询产生，不会出现「图表和 Agent 说的数字对不上」这种尴尬。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src import config as cfg
from src.exceptions import GameAgentError
from src.metrics.registry import Metric, MetricRegistry, get_registry
from src.sqlgen.executor import QueryExecutor
from src.sqlgen.generator import GeneratedSQL, SQLGenerator

if TYPE_CHECKING:
    # 只为类型标注导入，运行期不加载 —— 避免「Agent 层 import 组装根」的反向依赖
    from src.context import DatasetContext

# ===========================================================================
# 一、工具定义（OpenAI Function Calling 的 JSON Schema）
# ===========================================================================
# 【description 为什么要写这么细？】
#   模型是**靠 description 来决定要不要调用这个工具**的，它不看你写的 Python 代码。
#   描述里必须回答三个问题：这个工具能干什么、什么时候该用它、参数怎么填。
#   实践中的经验是：description 写得越具体，模型乱调/漏调的概率越低。
#
# 【parameters 里为什么都用 "object"？】
#   Function Calling 的参数规范是 JSON Schema 的一个子集，
#   顶层必须是 object（因为参数最终要序列化成一个 JSON 对象）。
#   即使 list_metrics 不需要任何参数，也要写 properties: {} + required: []，
#   不能省略 —— 有些模型实现会因为缺少 schema 而拒绝这个工具。
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_metrics",
            "description": (
                "列出系统当前支持的全部指标，包含 metric_id、中文名、分类、单位、"
                "别名和参数名。当你无法确定用户的问题对应哪个 metric_id 时，"
                "先调用这个工具查看目录。不需要任何参数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_detail",
            "description": (
                "查询某个指标的完整口径细节：业务定义、参数列表（含类型、默认值、"
                "取值范围）、输出字段含义、以及容易算错的注意事项。"
                "在调用 query_metric 之前用它确认参数应该怎么写。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_id": {
                        "type": "string",
                        "description": "指标 ID，必须来自 list_metrics 的结果，例如 retention_rate",
                    }
                },
                "required": ["metric_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_metric",
            "description": (
                "按指标 ID 和参数查询真实数据，返回结果表格。"
                "这是唯一能获取数字的工具 —— 回答中出现的任何数字都必须来自它的返回结果，"
                "严禁凭记忆或推理编造。参数不确定时传空对象 {}，系统会自动套用默认值。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_id": {
                        "type": "string",
                        "description": "指标 ID，例如 dau、retention_rate",
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "指标参数键值对，例如 {\"day_n\": 7} 表示 7 日留存，"
                            "{\"start_date\": \"2026-09-01\", \"end_date\": \"2026-09-07\"} 表示指定区间。"
                            "不确定就传 {}。"
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["metric_id"],
            },
        },
    },
]


# ===========================================================================
# 二、工具执行结果
# ===========================================================================

@dataclass
class ToolResult:
    """一次工具调用的结果。

    ok 这个字段很关键：Agent 主循环会把 content 原样塞回给模型作为「观察结果」，
    模型看到「工具执行失败：参数 day_n 不在允许范围内」之后，通常能自己改对参数重试。
    这就是 Agent 的自我纠错能力 —— 关键不在于不出错，而在于错误能被结构化成
    模型看得懂的反馈，而不是一个让程序崩掉的异常。
    """

    name: str
    ok: bool
    content: str                                       # 给 LLM 看的文本
    data: dict[str, Any] = field(default_factory=dict)  # 给前端用的结构化数据
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化（Phase 4 前端会用）。"""
        return {
            "name": self.name,
            "ok": self.ok,
            "content": self.content,
            "data": self.data,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


# ===========================================================================
# 三、工具执行器
# ===========================================================================

class ToolExecutor:
    """负责「调用哪个函数、传什么参、怎么把结果讲给模型听」。

    依赖注入说明（和 Phase 2 的 SQLGenerator 一致）：
      registry / generator / executor 都可以从外部传入，
      这样单元测试就能用固定的假窗口、或直接复用真实数据库，互不干扰。
    """

    def __init__(
        self,
        registry: MetricRegistry | None = None,
        generator: SQLGenerator | None = None,
        executor: QueryExecutor | None = None,
        max_rows_to_llm: int | None = None,
        context: "DatasetContext | None" = None,
    ) -> None:
        """依赖优先级：显式传入的 registry/generator/executor > context > 全局默认。

        【为什么要加 context，而不是只靠现有三个参数？】
          因为现有三个参数的默认值互相**不联动**：
              ToolExecutor(registry=上传库的方案)
          会造出「A 方案的 generator + 全局库的 executor」这样一个组合 ——
          查询能跑通，但数字是错的，而且过程区显示的 SQL 看起来完全正常。
          context 提供的正是「一整套自洽的装配」，把这种错配在 API 层面消除。

        【为什么 context 先兜底、显式参数后覆盖？】
          这样「只传 context」和「传 context + 覆盖某一个组件」都能工作，
          不需要两套代码路径。测试里传一个假的 executor 仍然有效。
        """
        self.context = context
        if context is not None:
            self.registry: MetricRegistry = registry or context.registry
            self.generator: SQLGenerator = generator or context.build_generator()
            self.executor: QueryExecutor = executor or context.build_executor()
        else:
            # 未传 context → 与改造前逐行相同的旧行为（默认数据集路径）
            self.registry = registry or get_registry()
            self.generator = generator or SQLGenerator(self.registry)
            self.executor = executor or QueryExecutor()
        self.max_rows_to_llm: int = (
            cfg.LLM_TOOL_MAX_ROWS if max_rows_to_llm is None else max_rows_to_llm
        )

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """给 LLM 的工具清单（每次返回一份新的 list，防止外部误改模块常量）。"""
        return list(TOOL_DEFINITIONS)

    # ------------------------------------------------------------------
    # 分发入口
    # ------------------------------------------------------------------
    def execute(self, name: str, arguments: dict[str, Any] | str | None = None) -> ToolResult:
        """执行一个工具。

        这是本模块唯一的对外入口，主循环只需要调用它。
        刻意**不抛异常**：所有失败都转成 ok=False 的 ToolResult，
        让 Agent 有「看到错误 → 重新尝试」的机会，而不是让整轮对话崩掉。
        （和 Phase 2 的 QueryExecutor.execute 是同一个设计原则。）
        """
        start = time.perf_counter()

        # 步骤 1：解析参数。模型给的是 JSON 字符串，偶尔会写坏。
        args, parse_error = self._parse_arguments(arguments)
        if parse_error:
            message = (
                f"工具 `{name}` 的参数解析失败：{parse_error}\n"
                f"请重新调用，并确保参数是一个合法的 JSON 对象。"
            )
            return ToolResult(
                name=name,
                ok=False,
                content=message,
                error=parse_error,
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )

        # 步骤 2：找处理函数。约定「工具名 = 方法名去掉 _tool_ 前缀」，
        # 这样新增工具只要写一个方法，不用再维护一张映射表。
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            available = ", ".join(d["function"]["name"] for d in TOOL_DEFINITIONS)
            return ToolResult(
                name=name,
                ok=False,
                content=f"不存在名为 `{name}` 的工具。可用工具：{available}",
                error=f"未知工具：{name}",
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )

        # 步骤 3：执行。业务异常（指标不存在、参数非法、SQL 被拦截……）
        # 都是「预期内」的失败，转成文本反馈给模型即可。
        try:
            result = handler(args)
            result.elapsed_ms = (time.perf_counter() - start) * 1000
            return result
        except GameAgentError as exc:
            return ToolResult(
                name=name,
                ok=False,
                content=f"工具 `{name}` 执行失败：{exc.message}",
                error=exc.message,
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - 未预期错误也要兜住，绝不让 Agent 崩掉
            return ToolResult(
                name=name,
                ok=False,
                content=(
                    f"工具 `{name}` 出现未预期的错误：{exc}。"
                    f"请换一种方式提问，或先调用 list_metrics 确认可用指标。"
                ),
                error=str(exc),
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )

    @staticmethod
    def _parse_arguments(arguments: dict[str, Any] | str | None) -> tuple[dict[str, Any], str | None]:
        """把模型给的参数统一成 dict。

        为什么还要处理字符串？虽然 LLMClient 已经解析过一次了，
        但 ToolExecutor 是一个独立可复用的组件（测试、脚本、Phase 4 都可能直接调），
        不能假设调用方一定传 dict。在边界处做一次归一化是稳妥的做法。
        """
        if arguments is None:
            return {}, None
        if isinstance(arguments, dict):
            return arguments, None
        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                return {}, None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                return {}, str(exc)
            if not isinstance(parsed, dict):
                return {}, f"参数必须是一个 JSON 对象，实际是 {type(parsed).__name__}"
            return parsed, None
        return {}, f"参数类型不支持：{type(arguments).__name__}"

    # ------------------------------------------------------------------
    # 工具 1：指标目录
    # ------------------------------------------------------------------
    def _tool_list_metrics(self, args: dict[str, Any]) -> ToolResult:
        """列出全部指标。不接收参数（多余的参数直接忽略，没必要为此报错）。"""
        metrics = self.registry.all()
        lines = [f"系统当前支持 {len(metrics)} 个指标："]
        lines.extend(metric.summary_line() for metric in metrics)
        lines.append("")
        lines.append("如需某个指标的详细口径与参数说明，请调用 get_metric_detail。")

        return ToolResult(
            name="list_metrics",
            ok=True,
            content="\n".join(lines),
            data={
                "count": len(metrics),
                "metrics": [
                    {
                        "metric_id": m.metric_id,
                        "metric_name": m.metric_name,
                        "category": m.category,
                        "unit": m.unit,
                        "aliases": list(m.aliases),
                        "params": list(m.param_names),
                    }
                    for m in metrics
                ],
            },
        )

    # ------------------------------------------------------------------
    # 工具 2：单指标详情
    # ------------------------------------------------------------------
    def _tool_get_metric_detail(self, args: dict[str, Any]) -> ToolResult:
        """返回一个指标的口径细节。

        注意这里**没有用 try 捕获 MetricNotFoundError** ——
        因为 execute() 已经统一兜底了，Error 会被转成「工具执行失败：……」
        并把「可用指标列表」带出去，模型据此就能自己纠正 ID 重试。
        少写一层重复的 try/except，代码更好维护。
        """
        metric_id = str(args.get("metric_id", "")).strip()
        if not metric_id:
            return ToolResult(
                name="get_metric_detail",
                ok=False,
                content="缺少必填参数 metric_id。请先调用 list_metrics 查看可用指标。",
                error="缺少参数 metric_id",
            )

        metric = self.registry.get(metric_id)      # 不存在 → MetricNotFoundError
        return ToolResult(
            name="get_metric_detail",
            ok=True,
            content=self._render_metric_detail(metric),
            data=self._metric_to_dict(metric),
        )

    @staticmethod
    def _metric_to_dict(metric: Metric) -> dict[str, Any]:
        """把指标定义转成结构化字典（前端「指标口径卡片」会用到）。"""
        return {
            "metric_id": metric.metric_id,
            "metric_name": metric.metric_name,
            "category": metric.category,
            "unit": metric.unit,
            "good_direction": metric.good_direction,
            "business_definition": metric.business_definition,
            "source_tables": list(metric.source_tables),
            "dimensions": list(metric.dimensions),
            "params": [
                {
                    "name": p.name,
                    "type": p.type,
                    "label": p.label,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": list(p.enum),
                }
                for p in metric.params
            ],
            "output_columns": [dict(c) for c in metric.output_columns],
            "notes": metric.notes,
        }

    @staticmethod
    def _render_metric_detail(metric: Metric) -> str:
        """把指标定义渲染成给模型读的文本。

        为什么用这种「冒号 + 短句」的朴素格式，而不是 Markdown 表格？
          因为模型要的是**信息密度**而不是排版美观。
          每个字段独占一行、标签固定，模型定位信息最快，token 也最省。
        """
        lines = [
            f"指标 ID：{metric.metric_id}",
            f"指标名称：{metric.metric_name}",
            f"分类：{metric.category or '-'}｜单位：{metric.unit or '-'}｜"
            f"方向：{'越高越好' if metric.good_direction == 'up' else '越低越好'}",
            f"业务定义：{metric.business_definition}",
            f"数据来源表：{', '.join(metric.source_tables)}",
            f"支持维度：{', '.join(metric.dimensions) or '-'}",
        ]

        if metric.params:
            lines.append("参数列表：")
            for p in metric.params:
                detail = [
                    f"名称={p.name}",
                    f"类型={p.type}",
                    f"含义={p.label or p.description}",
                    f"必填={'是' if p.required else '否'}",
                    f"默认值={p.default if p.has_default else '无'}",
                ]
                if p.enum:
                    detail.append("取值范围=" + "/".join(str(v) for v in p.enum))
                lines.append("  - " + "，".join(detail))
        else:
            lines.append("参数列表：无（该指标无需参数）")

        if metric.output_columns:
            lines.append("输出字段：")
            for col in metric.output_columns:
                lines.append(f"  - {col.get('name')}：{col.get('description', '')}")

        if metric.notes:
            lines.append(f"注意事项：{metric.notes}")

        lines.append(
            "调用 query_metric 时，params 只能是上面「名称」列出的键；"
            "不填的键会自动使用默认值。"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 工具 3：查询指标数据（唯一能拿到数字的工具）
    # ------------------------------------------------------------------
    def _tool_query_metric(self, args: dict[str, Any]) -> ToolResult:
        """按 metric_id + params 生成 SQL、校验、执行，返回结果。

        这一段的代码非常短 —— 因为真正的复杂度都在 Phase 2 里做好了：
        生成（模板填参）、校验（四层安全）、执行（只读+超时+行数上限）。
        工具层只是把它们串起来，正说明分层设计起到了作用：
        每一层都足够简单，组合起来就能干一件不小的事。
        """
        metric_id = str(args.get("metric_id", "")).strip()
        if not metric_id:
            return ToolResult(
                name="query_metric",
                ok=False,
                content="缺少必填参数 metric_id。请先调用 list_metrics 查看可用指标。",
                error="缺少参数 metric_id",
            )

        params = args.get("params") or {}
        if not isinstance(params, dict):
            return ToolResult(
                name="query_metric",
                ok=False,
                content=(
                    f"参数 params 必须是 JSON 对象（键值对），"
                    f"实际收到 {type(params).__name__}。例如 {{\"day_n\": 7}}。"
                ),
                error="params 类型错误",
            )

        # 第 1 步：生成（含参数校验、默认值填充、日期表达式解析）
        generated = self.generator.generate(metric_id, params)

        # 第 2 步 + 第 3 步：校验 + 执行（QueryExecutor.execute 默认先过校验器）
        result = self.executor.execute(
            generated.sql,
            generated.params,
            allowed_tables=generated.source_tables,
        )

        if not result.success:
            # 执行失败也要把「用了什么参数」说清楚，模型才知道该往哪个方向调整
            return ToolResult(
                name="query_metric",
                ok=False,
                content=(
                    f"指标 `{metric_id}` 查询失败：{result.error}\n"
                    f"本次使用的参数：{generated.params}\n"
                    f"建议：缩小时间范围、减少对比维度，或先用默认参数重试。"
                ),
                error=result.error,
                data={
                    "metric_id": metric_id,
                    "params": generated.params,
                    "rendered_sql": generated.rendered_sql,
                },
            )

        return ToolResult(
            name="query_metric",
            ok=True,
            content=self._render_query_observation(generated, result),
            data={
                "metric_id": generated.metric_id,
                "metric_name": generated.metric_name,
                "unit": generated.unit,
                "params": generated.params,
                "columns": result.columns,
                "rows": result.to_dicts(),        # 结构化数据，给前端画图
                "row_count": result.row_count,
                "truncated": result.truncated,
                "rendered_sql": generated.rendered_sql,   # 供前端「查看 SQL」用
                "elapsed_ms": round(result.elapsed_ms, 2),
            },
        )

    def _render_query_observation(self, generated: GeneratedSQL, result: Any) -> str:
        """把查询结果渲染成给模型看的「观察结果」。

        设计取舍：为什么把表格截断到 max_rows_to_llm 行？
          因为每一行都会变成 token（要花钱、也占上下文），
          而 SQL_MAX_ROWS=500 是给「人看的大表格」的上限。
          对 Agent 来说，看清趋势只需要前几十行；真需要全量数据时，
          正确做法是让用户在前端看表格，而不是把 500 行塞进模型上下文。
        """
        header = (
            f"指标：{generated.metric_name}（{generated.metric_id}）"
            f"｜单位：{generated.unit or '-'}"
            f"｜参数：{generated.params}"
            f"｜返回 {result.row_count} 行，耗时 {result.elapsed_ms:.1f}ms"
        )
        table = result.to_markdown(max_preview=self.max_rows_to_llm)
        footer = (
            "以上是数据库返回的真实结果。请严格基于这些数字作答，"
            "不要推测未返回的数据，也不要对数字做任何未经计算的改动。"
        )
        return "\n".join([header, "", table, "", footer])