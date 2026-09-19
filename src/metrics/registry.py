# -*- coding: utf-8 -*-
"""
指标语义层 · 注册表加载器（Phase 2）
================================================================================
【这一层到底解决什么问题？】

  如果让大模型直接写 SQL，会同时踩三个坑：
    1. 幻觉表名/字段名 —— 它不知道我们的库里叫 user_daily_snapshot 还是 daily_active；
    2. 口径漂移 —— 同一个「留存率」，今天用注册日为分母、明天用活跃日为分母；
    3. 安全风险 —— 它可能写出 DELETE，或者被用户用提示词注入带偏。

  所以我们把「指标」变成一种**受控的资产**：
    用户问题  →  LLM 只做一件事：映射到 metric_id + 抽取参数
              →  程序按注册表里的预审 SQL 模板填参生成 SQL
              →  校验 → 执行

  这样做的好处（设计要点）：
    · LLM 的职责被收窄成「分类 + 抽取」，准确率远高于「生成 SQL」；
    · 口径由业务方在 JSON 里定义，改口径不改代码，属于「配置驱动」；
    · SQL 模板是白名单资产，天然免疫大部分注入风险。

【为什么用 JSON 而不是 Python dict？】
  · JSON 是纯数据，可以被产品/运营/数据分析师直接维护和评审，不需要他们会 Python；
  · 未来要把它做成数据库表或后台管理页面时，迁移成本最低；
  · 逻辑（本文件）与数据（JSON）分离，是配置驱动设计的标准做法。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.exceptions import MetricNotFoundError

# 注册表文件的默认位置：与本文件同目录
DEFAULT_REGISTRY_PATH: Path = Path(__file__).resolve().with_name("metrics_registry.json")

# 指标名里括号内的内容，例如「用户粘性（DAU/MAU）」里的「DAU/MAU」
_PARENTHETICAL = re.compile(r"[（(]([^）)]*)[）)]")
# 别名归一化时要忽略的字符：括号、逗号、顿号、句点、正斜杠、空白
_ALIAS_NOISE = re.compile(r"[（）()，,。/、\s]")
# 括号内多个缩写之间的分隔符，例如「MAU，滚动 30 天」用逗号分隔
_ABBR_SPLIT = re.compile(r"[/、,，]")


# ===========================================================================
# 一、数据结构：把 JSON 的每个字段变成有类型的对象
# ===========================================================================

@dataclass(frozen=True)
class MetricParam:
    """指标的一个参数（例如留存率的「第 N 天」）。

    frozen=True 表示「不可变」：加载后就不能再改，避免运行期被意外修改，
    这类配置对象用不可变数据结构是更安全的做法。
    """

    name: str                 # 参数名，必须与 SQL 模板里的 :name 占位符一致
    type: str                 # 参数类型：date / int / string
    description: str = ""
    label: str = ""           # 中文名，给前端和 LLM 看
    required: bool = False
    default: Any = None       # 默认值；date 类型支持 "@data_end-6" 这种相对表达式
    enum: tuple[Any, ...] = ()  # 允许的取值集合（空元组表示不限制）

    @property
    def has_default(self) -> bool:
        """是否声明了默认值。

        注意不能写成 `if self.default`：默认值可能是 0（比如 day_n=0 不合法，
        但 window_days 之类完全可能是 0），用 is not None 判断才正确。
        """
        return self.default is not None


@dataclass(frozen=True)
class Metric:
    """一个指标定义。字段与 JSON 一一对应。"""

    metric_id: str                          # 指标 ID，如 retention_rate
    metric_name: str                        # 中文名称
    business_definition: str                # 业务定义（口径说明，最重要）
    sql_template: str                       # SQL 模板（含 :param 占位符）
    source_tables: tuple[str, ...]          # 数据来源表（会被安全校验器强制）
    owner: str                              # 负责人
    updated_at: str                         # 更新时间
    aliases: tuple[str, ...] = ()           # 同义词/别名，用于关键词粗匹配
    category: str = ""                      # 分类：活跃 / 留存 / 商业化 / 渠道 / 版本
    unit: str = ""                          # 单位：人 / % / 元
    good_direction: str = "up"              # 指标越高越好(up)还是越低越好(down)
    dimensions: tuple[str, ...] = ()        # 支持的对比维度
    params: tuple[MetricParam, ...] = ()    # 参数列表
    output_columns: tuple[dict[str, str], ...] = ()  # 输出字段说明
    notes: str = ""                         # 实现备注 / 易错点提示

    def get_param(self, name: str) -> MetricParam | None:
        """按名字取参数定义，找不到返回 None。"""
        for p in self.params:
            if p.name == name:
                return p
        return None

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def summary_line(self) -> str:
        """完整版一行式摘要，供 list_metrics 工具使用（带分类）。"""
        return (
            f"- {self.metric_id} | {self.metric_name} | 分类: {self.category} | "
            f"单位: {self.unit} | 别名: {self.alias_text()} | "
            f"参数: {', '.join(self.param_names) if self.params else '无'}"
        )

    def catalog_line(self) -> str:
        """**紧凑版**目录行，专供系统提示词使用（Phase 6 新增）。

        和 summary_line() 的唯一区别是去掉了「分类」：
          · 分类（活跃 / 留存 / 商业化 / 渠道 / 版本）只服务于前端「指标字典」
            的分组展示，对「用户问题 → metric_id」这个映射任务没有任何区分度 ——
            模型不会因为看到「分类: 留存」就更容易把「次留」映射到 retention_rate。
          · 而 summary_line() 被 list_metrics 工具使用。工具是「按需调用」的，
            调用频率低，多带一个字段换来人读起来更清楚，是划算的。

        一句话：**提示词里的信息要省着放，工具里的信息要放全。**
        两者用不同的渲染方法，而不是共用一个「差不多就行」的格式。
        """
        return (
            f"- {self.metric_id} | {self.metric_name} | "
            f"单位: {self.unit or '-'} | 别名: {self.alias_text()} | "
            f"参数: {', '.join(self.param_names) if self.params else '无'}"
        )

    def alias_text(self, limit: int = 6) -> str:
        """去重后的别名串（给上面两个渲染方法共用）。"""
        alias = self._dedup_aliases()[:limit]
        return "/".join(alias) if alias else "-"

    def _dedup_aliases(self) -> list[str]:
        """去重列表：剔除与指标名 / 指标名括号缩写 / metric_id 重复的别名。

        【Phase 6 · 提示词优化的核心动作】

          为什么盯着别名？因为指标目录是系统提示词里唯一的「固定开销」：
          它每一轮对话都要完整进入模型上下文，却只服务于「映射 metric_id」
          这一件事。Phase 5 实测：一个知识边界类问题（0 次工具调用）
          仍要消耗 2,316 tokens，其中近一半就是这个目录。所以目录的每一行
          都值得按 token 审一遍。

          去重规则（保守，绝不丢有用信息）：
            1. 与指标名主干重复的别名 —— 例：dau 名称是「日活跃用户数（DAU）」，
               别名里的「日活跃用户数」与主干一字不差，删掉它不损失任何召回；
            2. 与指标名括号内缩写重复的别名 —— 例：arppu 名称自带「(ARPPU)」，
               别名里再写一个「ARPPU」纯属浪费；用户说「ARPPU」照样能被名称匹配；
            3. 与 metric_id 重复的别名 —— 同理，metric_id 本身就在行首。

          实测 12 个指标的目录从 1,390 字符降到约 1,260 字符，保留的仍全是
          「次留 / 客单价 / 拉新量」这类有区分度的口语词。

        【为什么不干脆把别名全删掉，能省得更多？】
          算过账：别名全删能再省约 500 token / 次请求，但模型会失去口语词的
          映射线索，大概率先去调一次 list_metrics 确认 —— 而一次工具调用会让
          **整个上下文再发一遍**（2,500+ token）。省 500 换 2,500 是亏本买卖。
          结论：最省 token 的做法不是删信息，而是把信息放在最便宜的位置。
        """
        name = self.metric_name
        # ① 提取名称里括号中的缩写（去掉括号后的主干才是「名称本体」）
        core = _PARENTHETICAL.sub("", name)
        abbr: set[str] = set()
        for group in _PARENTHETICAL.findall(name):
            for part in _ABBR_SPLIT.split(group):
                if part.strip():
                    abbr.add(self._normalize_alias(part))
        abbr.add(self._normalize_alias(core))
        abbr.add(self._normalize_alias(self.metric_id))

        seen: set[str] = set()
        kept: list[str] = []
        for a in self.aliases:
            n = self._normalize_alias(a)
            if not n or n in abbr or n in seen:
                continue
            seen.add(n)
            kept.append(a)
        return kept

    @staticmethod
    def _normalize_alias(text: str) -> str:
        """别名归一化：去掉空格、括号、和常见分隔符号，用于等价比较。

        例：「DAU/MAU」与「DAU/ MAU」、指标名括号里的「ARPPU」与别名「ARPPU」，
        归一化后都相等，才能正确判重。（注意：正斜杠 '/'(0x2F) 也要去掉，
        否则「DAU/MAU」会被误判成和「DAU」「MAU」都不同。）
        """
        return _ALIAS_NOISE.sub("", text)


# ===========================================================================
# 二、注册表本体
# ===========================================================================

class MetricRegistry:
    """指标注册表：负责加载、检索、给 LLM 生成指标目录。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path: Path = Path(path) if path else DEFAULT_REGISTRY_PATH
        self._meta: dict[str, Any] = {}
        # 用 dict 保存，保证「同一个 metric_id 只能有一个定义」，且保持 JSON 里的顺序
        self._metrics: dict[str, Metric] = {}
        self._load()

    # ---------------- 加载 ----------------
    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"指标注册表文件不存在：{self.path}")

        # encoding="utf-8" 必须写：Windows 默认编码是 GBK，中文指标名会读乱码
        raw = json.loads(self.path.read_text(encoding="utf-8"))

        self._meta = raw.get("meta", {})
        for item in raw.get("metrics", []):
            metric = self._build_metric(item)
            if metric.metric_id in self._metrics:
                # 重复的 metric_id 是严重的数据治理事故：会让「同一个指标有两种口径」
                raise ValueError(
                    f"指标 ID 重复：{metric.metric_id}，请检查 {self.path.name}"
                )
            self._metrics[metric.metric_id] = metric

        if not self._metrics:
            raise ValueError(f"指标注册表为空：{self.path}")

    @staticmethod
    def _build_metric(item: dict[str, Any]) -> Metric:
        """把一条 JSON 记录翻译成 Metric 对象。"""
        # 必填字段缺失时，尽早抛错（Fail Fast），而不是等到运行时才报 KeyError
        required_fields = (
            "metric_id", "metric_name", "business_definition",
            "sql_template", "source_tables", "owner", "updated_at",
        )
        missing = [f for f in required_fields if not item.get(f)]
        if missing:
            raise ValueError(
                f"指标 {item.get('metric_id', '<未知>')} 缺少必填字段：{', '.join(missing)}"
            )

        params = tuple(
            MetricParam(
                name=p["name"],
                type=p.get("type", "string"),
                description=p.get("description", ""),
                label=p.get("label", ""),
                required=bool(p.get("required", False)),
                default=p.get("default"),
                enum=tuple(p.get("enum", ())),
            )
            for p in item.get("params", [])
        )

        return Metric(
            metric_id=item["metric_id"],
            metric_name=item["metric_name"],
            business_definition=item["business_definition"],
            sql_template=item["sql_template"],
            source_tables=tuple(item["source_tables"]),
            owner=item["owner"],
            updated_at=item["updated_at"],
            aliases=tuple(item.get("aliases", ())),
            category=item.get("category", ""),
            unit=item.get("unit", ""),
            good_direction=item.get("good_direction", "up"),
            dimensions=tuple(item.get("dimensions", ())),
            params=params,
            output_columns=tuple(item.get("output_columns", ())),
            notes=item.get("notes", ""),
        )

    # ---------------- 查询 ----------------
    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    def __len__(self) -> int:
        return len(self._metrics)

    def __contains__(self, metric_id: object) -> bool:
        return metric_id in self._metrics

    def get(self, metric_id: str) -> Metric:
        """按 ID 取指标；不存在时抛 MetricNotFoundError（而不是返回 None）。

        为什么抛异常而不是返回 None？
          因为「指标找不到」在 Agent 流程里是一个必须被显式处理的业务事件
          （要告诉用户「我还不认识这个指标」并推荐相近指标），
          返回 None 很容易被上层忘记判断，最后变成 AttributeError。
        """
        metric = self._metrics.get(metric_id)
        if metric is None:
            raise MetricNotFoundError(
                f"指标注册表中不存在 metric_id = {metric_id}；"
                f"可用指标：{', '.join(self.ids())}"
            )
        return metric

    def all(self) -> list[Metric]:
        """返回全部指标（保持 JSON 中的定义顺序）。"""
        return list(self._metrics.values())

    def ids(self) -> list[str]:
        return list(self._metrics.keys())

    def find_by_alias(self, text: str) -> list[Metric]:
        """关键词粗匹配：问题里出现某个别名时，返回对应指标。

        注意：这只是「粗排」，用于给 LLM 提供候选、或者在没有 LLM 时降级使用。
        真正的语义匹配由 Phase 3 的 match_metric 工具（Function Calling）负责，
        因为中文里「留不住人」和「留存率」字面完全不同，关键词匹配无能为力。

        匹配按「别名长度」从长到短排序：别名越长越具体，
        「新手引导完成率」显然应该优先于泛化的「留存率」。
        """
        lowered = text.lower()
        hits: list[tuple[int, Metric]] = []
        for metric in self._metrics.values():
            for alias in metric.aliases:
                if alias.lower() in lowered:
                    hits.append((len(alias), metric))
                    break
        hits.sort(key=lambda x: x[0], reverse=True)   # 长别名优先
        return [m for _, m in hits]

    def to_catalog_text(self) -> str:
        """生成给大模型看的「指标目录」。

        Phase 3 会把它塞进 System Prompt，让模型知道有哪些指标可用。
        为什么不直接把整个 JSON 丢给模型？因为那样 token 消耗大、噪声多，
        而且模型容易「自由发挥」去改 SQL；只给 ID / 名称 / 别名 / 参数，
        把模型的职责钉死在「映射 + 抽参」这一件事上。

        【Phase 6 · 为什么表头里的「数据窗口」被删掉了？】
          系统提示词里已经有一整段【你的数据边界】在讲数据窗口（含 T+1 更新、
          「最近 7 天」具体指哪一段），目录表头再重复一遍纯属冗余。
          提示词优化的第一条纪律就是：**同一件事只说一遍，而且要放在最该说的地方。**
        """
        lines = [f"【可用指标目录】（共 {len(self)} 个）"]
        for metric in self._metrics.values():
            # 用紧凑版（不带分类）——这段文本每次请求都要进上下文，按 token 审过
            lines.append(metric.catalog_line())
        return "\n".join(lines)


# ===========================================================================
# 三、进程内单例
# ===========================================================================

@lru_cache(maxsize=4)
def get_registry(path: str | None = None) -> MetricRegistry:
    """获取注册表（带缓存）。

    为什么用 lru_cache？
      注册表是只读的配置文件，每次请求都重新读盘 + 解析 JSON 是纯浪费。
      用 lru_cache 做进程内缓存，第一次读之后就常驻内存。
      参数 path 参与缓存 key，所以测试里传入不同路径时不会被旧缓存污染。
    """
    return MetricRegistry(path)