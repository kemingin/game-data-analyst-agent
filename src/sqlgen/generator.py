# -*- coding: utf-8 -*-
"""
SQL 生成模块（Phase 2）
================================================================================
【核心思想：模板填充，而不是让模型写 SQL】

  生成一条可执行 SQL 的全过程：

      metric_id + params
            │
            ├─ 1. 取指标定义（含预审过的 SQL 模板）
            ├─ 2. 合并参数默认值（用户没说时间，就用「最近 7 天」）
            ├─ 3. 类型校验 + 取值范围校验 + 日期区间合理性校验
            ├─ 4. 把 @data_end-6 这类「相对日期表达式」解析成真实日期
            ├─ 5. 检查模板占位符与参数是否一一对应（防模板写错）
            └─ 6. 产出两份 SQL：
                     sql          —— 含 :param 占位符，用于「参数绑定」执行（防注入）
                     rendered_sql —— 值已内联，仅用于前端透明展示给人看

【为什么必须产出两份 SQL？（这是本模块最重要的设计决策）】

  · 执行用参数绑定（sql + params）：
      cursor.execute(sql, params)
    SQL 的「结构」和「数据」彻底分离，哪怕参数里塞了 `'; DROP TABLE x --`，
    数据库也只会把它当成一个普通的字符串值，不可能变成可执行语句。
    这是防 SQL 注入的根本手段，比任何「关键字黑名单」都可靠。

  · 展示用渲染版（rendered_sql）：
    产品需求是「用户能看到 Agent 到底执行了什么 SQL」。
    如果直接把绑定后的 SQL 展示，用户看到的是 `:start_date`，看不懂。
    所以另外渲染一份把值填进去的版本，只给人看、绝不用于执行。
    渲染时会对单引号做转义（`'` → `''`），保证展示版也是语法正确的。

【为什么日期支持 @data_end-6 这种表达式？】

  数据是「截止到某天」的，如果默认值写死 '2026-09-11'，
  数据窗口一滚动（比如明天变成 09-18），默认值就过期了，得改 JSON。
  写成相对表达式后，「最近 7 天」永远成立，这是数据产品里
  「相对时间口径」的标准做法。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from src import config as cfg
from src.exceptions import MetricParamError
from src.metrics.registry import Metric, MetricRegistry, get_registry

# ---------------------------------------------------------------------------
# 正则：识别模板里的 :param 占位符
# ---------------------------------------------------------------------------
# 说明：SQL 里也可能出现 PostgreSQL 的 :: 类型转换写法，但 SQLite 用不到，
#       这里简单地禁止占位符前紧跟冒号即可（用 (?<!:) 反向否定断言）。
PLACEHOLDER_RE = re.compile(r"(?<!:):([A-Za-z_]\w*)")

# ---------------------------------------------------------------------------
# 正则：识别相对日期表达式
# ---------------------------------------------------------------------------
# 支持两种偏移写法，可以叠加：
#   @data_end            锚点本身
#   @data_end-6          字面数字偏移
#   @data_end-:day_n     引用另一个「已解析的整数参数」做偏移
#   @data_end-6-:day_n   叠加（数据截止日往前 6 天，再往前 day_n 天）
#
# 【为什么要支持 :参数 偏移？】
#   留存类指标的观察窗口和留存天数是绑死的：算 D7 就必须把批次窗口整体前移 7 天，
#   否则「最近 7 天注册的用户」和「第 7 天可观测」这两个条件交集为空，结果永远是 NULL。
#   把这件事写进默认值表达式，比让 LLM 去算日期可靠得多 ——
#   口径的确定性应该由语义层保证，而不是靠模型临场推理。
DATE_EXPR_RE = re.compile(
    r"^@(data_end|data_start)((?:[+-](?:\d+|:[A-Za-z_]\w*))+)?$"
)

# 拆出表达式里的每一个偏移项，例如 "-6-:day_n" → [('-', '6'), ('-', ':day_n')]
_OFFSET_TERM_RE = re.compile(r"([+-])(\d+|:[A-Za-z_]\w*)")

# 需要校验「先后顺序」的参数对：(起始, 结束)
ORDERED_PAIRS: tuple[tuple[str, str], ...] = (
    ("start_date", "end_date"),
    ("cohort_start", "cohort_end"),
)


@dataclass
class GeneratedSQL:
    """一条生成好的、待执行的 SQL 及其全部上下文。

    Phase 3 的 Function Calling 工具和 Phase 4 的前端都会用到它。
    """

    metric_id: str
    metric_name: str
    sql: str                       # 含 :param 占位符，交给 sqlite3 做参数绑定
    params: dict[str, Any]         # 绑定值
    rendered_sql: str              # 展示版（值已内联），仅用于人类阅读
    source_tables: tuple[str, ...] # 数据来源表，交给校验器做白名单校验
    unit: str = ""
    output_columns: tuple[dict[str, str], ...] = field(default_factory=tuple)

    def explain(self) -> str:
        """人类可读的执行说明，用于日志与前端「执行过程透明展示」。"""
        return (
            f"指标：{self.metric_name}（{self.metric_id}）\n"
            f"来源表：{', '.join(self.source_tables)}\n"
            f"参数：{self.params}\n"
            f"SQL：\n{self.rendered_sql}"
        )


class SQLGenerator:
    """按 metric_id + params 生成受控 SQL。"""

    def __init__(
        self,
        registry: MetricRegistry | None = None,
        data_start: date | None = None,
        data_end: date | None = None,
    ) -> None:
        # 这三个参数做成可注入的，是为了让单元测试可以固定一个「假的数据窗口」，
        # 不受真实 config 变化影响（测试要稳定、可复现）。
        self.registry = registry or get_registry()
        self.data_start: date = data_start or cfg.DATA_START
        self.data_end: date = data_end or cfg.DATA_END

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def generate(self, metric_id: str, params: dict[str, Any] | None = None) -> GeneratedSQL:
        """根据指标 ID 与参数生成 SQL。

        异常：
            MetricNotFoundError —— metric_id 不存在
            MetricParamError    —— 参数缺失/类型错误/取值越界/日期区间倒挂
        """
        metric = self.registry.get(metric_id)          # 不存在会抛 MetricNotFoundError
        provided = dict(params or {})

        self._check_unknown_params(metric, provided)
        resolved = self._resolve_params(metric, provided)
        self._check_ordered_pairs(metric, resolved)
        self._check_template_placeholders(metric)

        return GeneratedSQL(
            metric_id=metric.metric_id,
            metric_name=metric.metric_name,
            sql=metric.sql_template,
            params=resolved,
            rendered_sql=self.render(metric.sql_template, resolved),
            source_tables=metric.source_tables,
            unit=metric.unit,
            output_columns=metric.output_columns,
        )

    # ------------------------------------------------------------------
    # 分步校验（拆成小函数，每一步的报错信息都能说清楚「到底哪不对」）
    # ------------------------------------------------------------------
    @staticmethod
    def _check_unknown_params(metric: Metric, provided: dict[str, Any]) -> None:
        """拒绝未声明的参数。

        为什么要拒绝，而不是忽略？
          如果静默忽略，LLM 抽错参数名（比如把 day_n 写成 days）时会「看起来成功」，
          但实际用的是默认值，结果悄悄错了 —— 这种 bug 最难查。
          直接报错，让 Agent 有机会重试或提示用户，才是正确做法。
        """
        unknown = set(provided) - set(metric.param_names)
        if unknown:
            raise MetricParamError(
                f"指标 {metric.metric_id} 不支持参数：{', '.join(sorted(unknown))}；"
                f"可用参数：{', '.join(metric.param_names) or '无'}"
            )

    def _resolve_params(self, metric: Metric, provided: dict[str, Any]) -> dict[str, Any]:
        """合并默认值 → 类型校验 → 取值解析，产出最终绑定值。

        注意：这里**按 JSON 里参数的声明顺序**逐个解析。
        日期参数的默认值可以写成 @data_end-:day_n 去引用前面的整数参数，
        所以「被引用的参数必须声明在被引用者之前」是这个语义层的一条隐含约定，
        注册表自检测试会保证它。
        """
        resolved: dict[str, Any] = {}

        for param in metric.params:
            raw = provided.get(param.name, param.default)

            if raw is None:
                raise MetricParamError(
                    f"指标 {metric.metric_id} 缺少必填参数 {param.name}"
                    f"（{param.label or param.description}）"
                )

            value = self._coerce(param, raw, resolved)

            if param.enum and value not in param.enum:
                raise MetricParamError(
                    f"参数 {param.name} 的取值 {value!r} 不在允许范围内，"
                    f"只能是：{', '.join(str(v) for v in param.enum)}"
                )

            resolved[param.name] = value

        return resolved

    def _coerce(self, param: Any, value: Any, resolved: dict[str, Any]) -> Any:
        """按参数声明把外部传入的值转换成规范类型。

        LLM 抽出来的参数经常是字符串（"7" 而不是 7），所以必须做一次显式转换，
        而不是指望调用方传对类型。

        resolved 是「已经解析完的参数」，日期默认值里可以用 :其他参数 引用它，
        所以这里必须把上下文传下去。
        """
        if param.type == "date":
            return self._coerce_date(param.name, value, resolved)

        if param.type == "int":
            # bool 是 int 的子类，True 会被当成 1，容易掩盖错误，所以单独拦掉
            if isinstance(value, bool):
                raise MetricParamError(f"参数 {param.name} 需要整数，收到布尔值 {value!r}")
            if isinstance(value, int):
                return value
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                return int(value.strip())
            raise MetricParamError(f"参数 {param.name} 需要整数，收到 {value!r}")

        # 其余（string）统一转成字符串
        return str(value)

    def _coerce_date(self, name: str, value: Any, resolved: dict[str, Any]) -> str:
        """把日期参数规范成 'YYYY-MM-DD' 字符串。

        支持：
          1. datetime.date / datetime.datetime 对象
          2. '2026-09-10'、'2026/09/10'、'2026-09-10 12:00:00'
          3. 相对表达式 '@data_end-6' / '@data_start' / '@data_end-:day_n'
        """
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()

        if not isinstance(value, str):
            raise MetricParamError(f"参数 {name} 需要日期，收到 {value!r}")

        text = value.strip()

        # ---- 相对表达式 ----
        if text.startswith("@"):
            return self._resolve_date_expr(name, text, resolved).isoformat()

        # ---- 绝对日期：允许 '/' 分隔与带时间部分，统一规范化 ----
        candidate = text.replace("/", "-")
        if len(candidate) > 10:
            candidate = candidate[:10]
        try:
            parsed = date.fromisoformat(candidate)
        except ValueError as exc:
            raise MetricParamError(
                f"参数 {name} 的日期格式无法识别：{value!r}，"
                f"请使用 YYYY-MM-DD，或 @data_end-6 这类相对表达式"
            ) from exc

        # 数据是有边界的：问一个超出数据窗口的日期，不是「查询失败」，
        # 而是「知识缺失」。在这里显式报错，Agent 才能诚实地说
        # 「我的数据只覆盖 2026-06-20 ~ 2026-09-17」。这是 Phase 5 评测里
        # 「知识缺失问题」这一类场景的正确处理方式。
        if not (self.data_start <= parsed <= self.data_end):
            raise MetricParamError(
                f"参数 {name} = {parsed.isoformat()} 超出了可用数据范围 "
                f"{self.data_start.isoformat()} ~ {self.data_end.isoformat()}"
            )
        return parsed.isoformat()

    def _resolve_date_expr(
        self, name: str, expr: str, resolved: dict[str, Any]
    ) -> date:
        """把 '@data_end-6' / '@data_end-6-:day_n' 解析成真实日期。

        多个偏移项按书写顺序依次叠加；偏移量可以写成字面数字，
        也可以用 `:参数名` 引用同一条 SQL 里「声明在前、已经解析好」的整数参数。
        """
        match = DATE_EXPR_RE.match(expr)
        if not match:
            raise MetricParamError(
                f"参数 {name} 的日期表达式无法解析：{expr!r}；"
                f"支持 @data_start / @data_end，可带偏移如 @data_end-6 "
                f"或引用整数参数如 @data_end-:day_n"
            )

        anchor, terms = match.group(1), match.group(2) or ""
        base = self.data_end if anchor == "data_end" else self.data_start

        for sign, token in _OFFSET_TERM_RE.findall(terms):
            days = self._offset_days(name, token, resolved)
            base += timedelta(days=days if sign == "+" else -days)

        return base

    @staticmethod
    def _offset_days(name: str, token: str, resolved: dict[str, Any]) -> int:
        """解析单个偏移项：字面数字直接返回，`:参数名` 则去已解析结果里取值。"""
        if not token.startswith(":"):
            return int(token)

        ref = token[1:]
        if ref not in resolved:
            raise MetricParamError(
                f"参数 {name} 的日期表达式引用了参数 :{ref}，"
                f"但它尚未解析（被引用的参数必须声明在它前面）"
            )

        value = resolved[ref]
        if not isinstance(value, int):
            raise MetricParamError(
                f"参数 {name} 的日期表达式只能引用整数参数，"
                f"但 :{ref} 的值是 {value!r}"
            )
        return value

    @staticmethod
    def _check_ordered_pairs(metric: Metric, resolved: dict[str, Any]) -> None:
        """校验「起始日期 <= 结束日期」。

        日期倒挂是 LLM 抽取时间区间时最常见的错误之一，
        如果不拦，SQL 会正常执行但返回空结果，用户会以为「真的没数据」。
        """
        for start_key, end_key in ORDERED_PAIRS:
            if start_key in resolved and end_key in resolved:
                if resolved[start_key] > resolved[end_key]:
                    raise MetricParamError(
                        f"时间区间不合法：{start_key}({resolved[start_key]}) "
                        f"晚于 {end_key}({resolved[end_key]})"
                    )

    @staticmethod
    def _check_template_placeholders(metric: Metric) -> None:
        """校验模板占位符 ↔ 声明参数一一对应。

        这是一个「配置自检」：如果模板里写了 :day_n 但 params 里没声明，
        执行时会报「找不到绑定参数」；反过来声明了却没用上，说明 JSON 写错了。
        在生成阶段就把它拦住，比等到执行时报错更早、更好定位。

        这条不变量由单元测试保证，运行期再检查一次是「防御性编程」。
        """
        in_template = set(PLACEHOLDER_RE.findall(metric.sql_template))
        declared = set(metric.param_names)

        missing = in_template - declared
        if missing:
            raise MetricParamError(
                f"指标 {metric.metric_id} 的 SQL 模板使用了未声明的占位符："
                f"{', '.join(sorted(missing))}"
            )

        unused = declared - in_template
        if unused:
            raise MetricParamError(
                f"指标 {metric.metric_id} 声明了未在 SQL 模板中使用的参数："
                f"{', '.join(sorted(unused))}"
            )

    # ------------------------------------------------------------------
    # 展示版渲染
    # ------------------------------------------------------------------
    @staticmethod
    def render(sql_template: str, params: dict[str, Any]) -> str:
        """把占位符替换成字面量，生成「给人看」的 SQL。

        安全提示：这个函数的结果**只用于展示**，绝不能拿去执行。
        字符串值会用单引号包裹并把内部的单引号翻倍（SQL 标准的转义方式），
        所以即使值里含有单引号，展示出来的 SQL 也是语法正确的。
        """

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in params:
                return match.group(0)      # 未知占位符原样保留（会被校验器发现）
            value = params[name]
            if isinstance(value, str):
                return "'" + value.replace("'", "''") + "'"
            return str(value)

        return PLACEHOLDER_RE.sub(replace, sql_template)