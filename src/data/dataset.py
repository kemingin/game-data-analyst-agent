# -*- coding: utf-8 -*-
"""
数据集与指标方案的元信息管理
=============================
本模块是「多数据集」架构的地基，定义三个核心概念：

    MetricScheme   指标方案 —— 一套指标定义（= 一份 metrics_registry.json）
    Dataset        数据集   —— 一份数据（自己的 SQLite 库 + 时间窗口 + 表清单）
    DatasetStore   索引管理器 —— 持久化「有哪些数据集 / 有哪些方案」

【为什么要把「指标方案」和「数据集」拆开？】
    因为「数据」和「口径」的生命周期完全不同：
      · 数据每周换一批，口径可能一年不变；
      · 反过来，口径会随指标委员会评审而变，而历史数据集不该跟着变。
    绑在一起的话，每次换数据都要连带复制一份口径，两份各自演进之后
    就会出现「同名不同义」——同一个指标 ID 在不同数据集里算法不一样，
    这是数据治理里最危险的状态。

    拆开之后，「复用」退化成一句赋值：新建 Dataset 时 scheme_id 填同一个值即可。
    **零复制、零同步成本。** 这正是「我这套数据集跟某某数据集的指标一样」
    这个需求在数据模型上的直接落地。

【本模块的依赖边界】
    刻意**不 import MetricRegistry** —— 数据层不该依赖指标语义层。
    这里只持有注册表的**路径**，真正加载由 src/context.py（组装根）负责。
    唯一的例外是 check_scheme_compatibility()，它需要读指标定义，
    用的是方法内的延迟 import（理由见该方法注释）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import config as cfg

if TYPE_CHECKING:  # pragma: no cover - 仅供类型标注，运行期不 import
    from src.data.upload import UploadResult


# ---------------------------------------------------------------------------
# 一、ID 规范
# ---------------------------------------------------------------------------
# dataset_id 会直接变成目录名 data/datasets/<id>/。
# 【为什么用白名单正则而不是「过滤危险字符」的黑名单？】
#   因为黑名单永远列不全：引号、分号、注释符、换行、Unicode 同形字符……
#   而 dataset_id 来自用户输入的显示名，是本项目里少见的不可信输入。
#   一旦放行 "../"，一次上传就能把库写到项目目录之外（路径穿越）。
#   白名单只放行「确定安全」的字符，漏掉某个危险字符的风险为零。
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,39}$")

# 索引文件的格式版本。
# 【为什么现在就加？】索引是持久化状态文件，将来改字段格式时，
#   需要靠它判断「要不要迁移」；事后补这个字段就得先猜历史格式。
_INDEX_VERSION = 1


# ---------------------------------------------------------------------------
# 二、数据集里的一张表
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetTable:
    """数据集里一张表的元信息快照（用于展示与白名单推导，不参与查询）。"""

    name: str                                  # 落库表名，即 SQL 里写的名字
    row_count: int = 0
    source_file: str = ""                      # 来源文件名；内置数据集为空串
    columns: tuple[dict[str, Any], ...] = ()    # [{"name","sql_name","sql_type","is_date"}]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "source_file": self.source_file,
            "columns": [dict(c) for c in self.columns],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetTable":
        return cls(
            name=raw["name"],
            row_count=int(raw.get("row_count", 0)),
            source_file=raw.get("source_file", ""),
            columns=tuple(dict(c) for c in raw.get("columns", ())),
        )


# ---------------------------------------------------------------------------
# 三、数据集
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    """一个数据集的元信息。

    【为什么 tables 存「快照」而不是每次去查 sqlite_master？】
      因为「SQL 白名单」和「前端表结构展示」都依赖它，而它们出现在每次请求的
      路径上。每次查一遍 sqlite_master 要多开一次连接，还会把「库文件当前是否
      可读」这个运行时状态耦合进配置读取。存快照 + 提供 refresh_tables() 手动同步，
      是「读多写少」场景下的正确取舍。
    """

    dataset_id: str          # 同时是目录名，必须通过 _ID_RE 校验
    display_name: str
    scheme_id: str           # ← 解耦的关键：只持有方案的 ID 引用，不持有指标定义
    db_path: Path
    raw_dir: Path            # 原始 CSV 存放目录，便于重建
    source: str              # "builtin" | "upload"
    created_at: str          # ISO 字符串
    data_start: date | None = None   # None = 未能从数据里识别出日期（不是「没有数据」）
    data_end: date | None = None
    tables: tuple[DatasetTable, ...] = ()
    notes: str = ""

    # ---------------- 便捷属性 ----------------
    @property
    def table_names(self) -> frozenset[str]:
        """本数据集实际存在的表名集合 —— 直接用作 SQL 校验白名单。

        【为什么白名单能直接用它？】
          因为 SQL 只允许查「本数据集真实存在的表」，这正是白名单的语义。
          改造前白名单是全局硬编码的 9 张表（validator.ALLOWED_TABLES），
          多数据集下这个概念不再成立。
        """
        return frozenset(t.name for t in self.tables)

    def cache_key(self) -> str:
        """缓存键 = dataset_id + scheme_id + created_at。

        【为什么必须把 scheme_id 也算进去？】
          因为「同一个数据集改指另一套指标方案」是合法操作（解耦的直接收益）。
          只按 dataset_id 做缓存键的话，改方案后 Agent 会继续用旧方案 ——
          这类 bug 的表现是「配置改了但不生效」，排查成本极高。
          created_at 同理：重新上传同一个数据集会更新它，缓存必须失效。
        """
        return f"{self.dataset_id}::{self.scheme_id}::{self.created_at}"

    # ---------------- 序列化 ----------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "scheme_id": self.scheme_id,
            # 路径统一存 BASE_DIR 相对形式（理由见 DatasetStore._to_stored_path）
            "db_path": DatasetStore._to_stored_path(self.db_path),
            "raw_dir": DatasetStore._to_stored_path(self.raw_dir),
            "source": self.source,
            "created_at": self.created_at,
            "data_start": self.data_start.isoformat() if self.data_start else None,
            "data_end": self.data_end.isoformat() if self.data_end else None,
            "tables": [t.to_dict() for t in self.tables],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Dataset":
        return cls(
            dataset_id=raw["dataset_id"],
            display_name=raw.get("display_name", raw["dataset_id"]),
            scheme_id=raw["scheme_id"],
            db_path=DatasetStore._from_stored_path(raw["db_path"]),
            raw_dir=DatasetStore._from_stored_path(raw.get("raw_dir", ".")),
            source=raw.get("source", "upload"),
            created_at=raw.get("created_at", ""),
            data_start=_parse_date(raw.get("data_start")),
            data_end=_parse_date(raw.get("data_end")),
            tables=tuple(DatasetTable.from_dict(t) for t in raw.get("tables", ())),
            notes=raw.get("notes", ""),
        )


def _parse_date(raw: str | None) -> date | None:
    """把 ISO 日期字符串转成 date；None / 空串 / 非法值都返回 None。

    【为什么非法值不抛错？】索引是「派生状态」而非源资产，
    一条坏日期不该让整个数据集列表加载失败。
    """
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 四、指标方案
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricScheme:
    """一套可复用的指标方案（= 一份 metrics_registry.json 资产）。

    【为什么它是独立资产，而不是 Dataset 的一个字段？】
      见模块头部的说明 —— 核心是「数据」和「口径」生命周期不同。
      抽成独立实体后，多个数据集可以引用同一个 scheme_id，零复制。
    """

    scheme_id: str
    display_name: str
    registry_path: Path
    created_at: str
    derived_from: str | None = None
    # ↑ 【预留字段】下一轮做「从数据集派生方案」时用它记录血缘（A 派生自 B）。
    #   本轮恒为 None，但先落库 —— 索引是有状态文件，事后加字段要写迁移。

    def exists(self) -> bool:
        """注册表文件是否真实存在。"""
        return self.registry_path.exists()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme_id": self.scheme_id,
            "display_name": self.display_name,
            "registry_path": DatasetStore._to_stored_path(self.registry_path),
            "created_at": self.created_at,
            "derived_from": self.derived_from,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MetricScheme":
        return cls(
            scheme_id=raw["scheme_id"],
            display_name=raw.get("display_name", raw["scheme_id"]),
            registry_path=DatasetStore._from_stored_path(raw["registry_path"]),
            created_at=raw.get("created_at", ""),
            derived_from=raw.get("derived_from"),
        )


# ---------------------------------------------------------------------------
# 五、索引管理器
# ---------------------------------------------------------------------------

class DatasetStore:
    """数据集与指标方案的索引管理器。

    【为什么用 JSON 索引文件，而不是「扫目录自动发现」？】
      扫目录只能得到「有哪些 dataset_id」，得不到 display_name、scheme_id、
      created_at 这些必须持久化的信息。而索引文件是「派生状态」而非「源资产」——
      所以它不需要进 Git：首次构造时若文件不存在，ensure_builtin() 会自动
      重建内置条目，任何克隆都能自愈。
    """

    def __init__(
        self,
        root: str | Path | None = None,
        index_path: str | Path | None = None,
        auto_builtin: bool = True,
    ) -> None:
        """root 默认 data/datasets；index_path 默认 <root>/index.json。

        【为什么两个参数都可注入？】让单元测试用 tmp_path 跑，
        不污染真实目录 —— 与 SQLGenerator / ToolExecutor 的依赖注入风格一致。
        """
        self.root: Path = Path(root) if root else cfg.DATASETS_DIR
        self.index_path: Path = Path(index_path) if index_path else (self.root / "index.json")
        self._datasets: dict[str, Dataset] = {}
        self._schemes: dict[str, MetricScheme] = {}
        self._load()
        if auto_builtin:
            self.ensure_builtin()

    # ================= 索引读写 =================

    def _load(self) -> None:
        if not self.index_path.exists():
            return  # 首次运行：留空，由 ensure_builtin() 补内置条目
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 【为什么吞掉异常而不是抛？】索引是派生状态，损坏时「重建」比
            # 「拒绝启动」更合理 —— 真实数据（库文件、CSV）都还在磁盘上，
            # 索引丢了只是要重新登记，不该让整个应用起不来。
            return

        for item in raw.get("datasets", ()):
            try:
                ds = Dataset.from_dict(item)
            except (KeyError, TypeError, ValueError):
                continue  # 单条坏记录不该拖垮整个列表
            self._datasets[ds.dataset_id] = ds
        for item in raw.get("schemes", ()):
            try:
                sc = MetricScheme.from_dict(item)
            except (KeyError, TypeError, ValueError):
                continue
            self._schemes[sc.scheme_id] = sc

    def _save(self) -> None:
        """原子写：先写 .tmp，再 os.replace 覆盖。

        【为什么不能直接 open(..., "w")？】
          因为进程在写入中途被杀（Ctrl-C、OOM、断电）会留下半截 JSON，
          下次启动直接 JSONDecodeError —— **所有数据集一起丢失**。
          os.replace 在 Windows / POSIX 上都是原子操作：要么全新，要么全旧，
          不存在「写到一半」的中间态。
        """
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_VERSION,
            "datasets": [d.to_dict() for d in self._datasets.values()],
            "schemes": [s.to_dict() for s in self._schemes.values()],
        }
        tmp = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.index_path)

    # ---------------- 路径相对化 ----------------

    @staticmethod
    def _to_stored_path(path: Path) -> str:
        """绝对路径 → BASE_DIR 相对路径（不在 BASE_DIR 内则保留绝对形式）。

        【为什么必须相对化？】
          本项目路径里带空格和中文（`Game Data Analyst Agent`）。
          把绝对路径写进索引后，一旦换机器 / 换目录克隆，内置数据集就会指向
          一个不存在的库，而报错信息是「数据库不存在」—— 看起来像数据丢了，
          实际只是路径漂了。存相对路径则天然跟随项目根。
        """
        try:
            return path.resolve().relative_to(cfg.BASE_DIR).as_posix()
        except (ValueError, OSError):
            return str(path)

    @staticmethod
    def _from_stored_path(raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        return (cfg.BASE_DIR / p)

    # ================= ID 规范化 =================

    @staticmethod
    def normalize_id(text: str, fallback_seed: str | None = None) -> str:
        """把显示名归一成合法的 dataset_id。

        规则：转小写 → 非 [a-z0-9_] 折叠成下划线 → 合并连续下划线 →
              去首尾下划线 → 截断 40 字符。
        全中文 / 空名会折叠成空串，此时用短哈希兜底，保证合法且稳定。
        """
        folded = re.sub(r"[^a-z0-9_]+", "_", text.strip().lower())
        folded = re.sub(r"_+", "_", folded).strip("_")[:40].strip("_")
        if not folded or not _ID_RE.match(folded):
            seed = fallback_seed if fallback_seed is not None else text
            folded = "ds_" + hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]
        return folded

    def _unique_id(self, base: str) -> str:
        """确保 dataset_id 不与现有重复：重复时追加 _2 / _3 …"""
        if base not in self._datasets:
            return base
        n = 2
        while f"{base}_{n}" in self._datasets:
            n += 1
        return f"{base}_{n}"

    # ================= 数据集 =================

    def list_datasets(self) -> list[Dataset]:
        """按 created_at 升序返回；内置数据集永远排第一（作为默认选中项）。"""
        items = list(self._datasets.values())
        items.sort(key=lambda d: (d.source != "builtin", d.created_at, d.dataset_id))
        return items

    def get(self, dataset_id: str) -> Dataset:
        """取数据集；不存在抛 KeyError。

        【为什么抛异常而不是返回 None？】
          与 MetricRegistry.get 的 fail-fast 风格一致：数据集找不到是必须被
          显式处理的业务事件，返回 None 很容易被上层忘记判断，
          最后变成 AttributeError 或者更糟 —— 悄悄用了默认数据集。
        """
        if dataset_id not in self._datasets:
            raise KeyError(
                f"数据集不存在：{dataset_id}；"
                f"可用数据集：{', '.join(d.dataset_id for d in self.list_datasets()) or '（无）'}"
            )
        return self._datasets[dataset_id]

    def has(self, dataset_id: str) -> bool:
        return dataset_id in self._datasets

    def create_dataset(
        self,
        display_name: str,
        scheme_id: str,
        source: str,
        db_path: Path,
        raw_dir: Path,
        dataset_id: str | None = None,
        data_start: date | None = None,
        data_end: date | None = None,
        tables: tuple[DatasetTable, ...] = (),
        notes: str = "",
    ) -> Dataset:
        """登记一个数据集（不负责建库 / 导数，那是 upload.py 的事）。

        【为什么「登记」和「写数据」要分两步？】
          因为它们是两个会分别失败的动作。导数失败时不应该在索引里留下一个
          指向空库的「僵尸数据集」—— 那会让用户在列表里看到一个能选中、
          但一提问就报错的条目。所以顺序永远是「先写库成功 → 再登记」。
        """
        if scheme_id not in self._schemes:
            raise KeyError(f"指标方案不存在：{scheme_id}")

        ds_id = dataset_id or self.normalize_id(display_name)
        # 【为什么这里要再校验一次？】normalize_id 保证输出合法，
        # 但调用方可以显式传 dataset_id（比如内置数据集用固定 ID）。
        # 显式传入的值同样会变成目录名，所以必须过同一道白名单。
        if not _ID_RE.match(ds_id):
            raise ValueError(
                f"dataset_id 不合法：{ds_id!r}；"
                f"只允许小写字母、数字、下划线，且首字符不能是下划线，长度 1~40"
            )
        ds_id = self._unique_id(ds_id)

        dataset = Dataset(
            dataset_id=ds_id,
            display_name=display_name,
            scheme_id=scheme_id,
            db_path=Path(db_path),
            raw_dir=Path(raw_dir),
            source=source,
            created_at=datetime.now().isoformat(timespec="seconds"),
            data_start=data_start,
            data_end=data_end,
            tables=tables,
            notes=notes,
        )
        self._datasets[ds_id] = dataset
        self._save()
        return dataset

    def add_from_upload(
        self,
        files: list[tuple[str, bytes]],
        display_name: str,
        scheme_id: str,
    ) -> tuple[Dataset, list["UploadResult"]]:
        """上传建表的完整编排：建目录 → 写原始 CSV → 建库导数 → 登记索引。

        返回 (数据集, 每个文件的结果)。

        【失败语义：任何一个文件失败都不登记数据集】
          因为「部分成功的多表数据集」对下游是灾难 —— 指标模板会静默查不到表，
          报错信息是「引用了未授权的表」，而用户以为自己上传成功了。
          宁可整体失败让用户重传，也不要留下一个残缺的数据集。
        """
        from src.data.upload import import_csv_files  # 延迟 import，避免模块级循环

        ds_id = self._unique_id(self.normalize_id(display_name))
        if not _ID_RE.match(ds_id):
            raise ValueError(f"数据集名称无法生成合法 ID：{display_name!r}")

        ds_dir = self.root / ds_id
        raw_dir = ds_dir / "raw"
        db_path = ds_dir / "data.db"

        results, summary = import_csv_files(files, db_path=db_path, raw_dir=raw_dir)

        if not summary.get("ok"):
            # 不登记，但把结果返回给前端逐条展示失败原因
            failed = Dataset(
                dataset_id=ds_id,
                display_name=display_name,
                scheme_id=scheme_id,
                db_path=db_path,
                raw_dir=raw_dir,
                source="upload",
                created_at=datetime.now().isoformat(timespec="seconds"),
                notes="导入失败，未登记",
            )
            return failed, results

        tables = tuple(
            DatasetTable(
                name=t["name"],
                row_count=int(t.get("row_count", 0)),
                source_file=t.get("source_file", ""),
                columns=tuple(t.get("columns", ())),
            )
            for t in summary.get("tables", ())
        )

        dataset = self.create_dataset(
            display_name=display_name,
            scheme_id=scheme_id,
            source="upload",
            db_path=db_path,
            raw_dir=raw_dir,
            dataset_id=ds_id,
            data_start=_parse_date(summary.get("data_start")),
            data_end=_parse_date(summary.get("data_end")),
            tables=tables,
            notes=f"上传导入，共 {len(tables)} 张表",
        )
        return dataset, results

    def refresh_tables(self, dataset_id: str) -> Dataset:
        """重新扫 sqlite_master，同步 tables 快照与行数（重新上传后调用）。"""
        import sqlite3

        dataset = self.get(dataset_id)
        tables: list[DatasetTable] = []
        if dataset.db_path.exists():
            conn = sqlite3.connect(f"file:{dataset.db_path}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
                for (name,) in rows:
                    count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                    cols = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
                    tables.append(
                        DatasetTable(
                            name=name,
                            row_count=int(count),
                            columns=tuple(
                                {"sql_name": c[1], "sql_type": c[2] or "TEXT", "is_date": False}
                                for c in cols
                            ),
                        )
                    )
            finally:
                conn.close()

        updated = Dataset(
            dataset_id=dataset.dataset_id,
            display_name=dataset.display_name,
            scheme_id=dataset.scheme_id,
            db_path=dataset.db_path,
            raw_dir=dataset.raw_dir,
            source=dataset.source,
            created_at=datetime.now().isoformat(timespec="seconds"),
            data_start=dataset.data_start,
            data_end=dataset.data_end,
            tables=tuple(tables),
            notes=dataset.notes,
        )
        self._datasets[dataset_id] = updated
        self._save()
        return updated

    # ================= 指标方案 =================

    def list_schemes(self) -> list[MetricScheme]:
        items = list(self._schemes.values())
        items.sort(key=lambda s: (s.scheme_id != cfg.BUILTIN_SCHEME_ID, s.created_at))
        return items

    def get_scheme(self, scheme_id: str) -> MetricScheme:
        if scheme_id not in self._schemes:
            raise KeyError(
                f"指标方案不存在：{scheme_id}；"
                f"可用方案：{', '.join(s.scheme_id for s in self.list_schemes()) or '（无）'}"
            )
        return self._schemes[scheme_id]

    def register_scheme(
        self,
        display_name: str,
        registry_path: Path,
        scheme_id: str | None = None,
        derived_from: str | None = None,
    ) -> MetricScheme:
        """登记一套指标方案。

        【为什么本轮用不上也要提供？】
          因为「复用」的实现路径就是它：下一轮做方案派生时，
          「新方案落盘 + 调用这个方法登记」是唯一新增的代码，索引格式不用动。
          这也是把 derived_from 字段现在就写进索引的原因。
        """
        sc_id = scheme_id or self.normalize_id(display_name)
        if sc_id in self._schemes:
            return self._schemes[sc_id]

        scheme = MetricScheme(
            scheme_id=sc_id,
            display_name=display_name,
            registry_path=Path(registry_path),
            created_at=datetime.now().isoformat(timespec="seconds"),
            derived_from=derived_from,
        )
        self._schemes[sc_id] = scheme
        self._save()
        return scheme

    def set_scheme(self, dataset_id: str, scheme_id: str) -> Dataset:
        """把数据集指向另一套指标方案（解耦的直接收益：改一行引用即可）。

        【为什么必须先校验方案存在？】
          因为 scheme_id 只是一个字符串引用，写错不会有任何报错 ——
          直到某次提问时 build_context() 才抛 KeyError「指标方案不存在」。
          那时用户已经在看一个「看起来正常」的数据集了，报错位置离原因很远。
          在这里校验，等于把「引用完整性」收在写入点上。

        【为什么用 dataclasses.replace 而不是重新构造 Dataset？】
          replace 只改指定字段、其余原样复制，因此**不可能漏字段**。
          手写构造（像 refresh_tables 那样逐个字段抄）在 Dataset 加字段时
          会静默丢掉新字段 —— 这类 bug 不会报错，只会让某个功能莫名失效。
        """
        dataset = self.get(dataset_id)
        if scheme_id not in self._schemes:
            raise KeyError(
                f"指标方案不存在：{scheme_id}；"
                f"可用方案：{', '.join(s.scheme_id for s in self.list_schemes()) or '（无）'}"
            )

        updated = replace(dataset, scheme_id=scheme_id)
        self._datasets[dataset_id] = updated
        self._save()
        return updated

    # ================= 内置资产自愈 =================

    def ensure_builtin(self) -> Dataset:
        """确保内置数据集与内置方案存在（幂等）。

        【为什么内置数据集不搬文件，只是「登记」？】
          见 config.py 第七节的说明：现有库与注册表已被大量脚本、评测、
          文档引用，搬迁会让它们同时失效，而收益只是「目录整齐」。
          所以内置数据集是一个指针记录 —— 零迁移、零风险。
        """
        if cfg.BUILTIN_SCHEME_ID not in self._schemes:
            self.register_scheme(
                display_name=cfg.BUILTIN_SCHEME_NAME,
                registry_path=cfg.BASE_DIR / "src" / "metrics" / "metrics_registry.json",
                scheme_id=cfg.BUILTIN_SCHEME_ID,
            )

        if cfg.BUILTIN_DATASET_ID not in self._datasets:
            dataset = self.create_dataset(
                display_name=cfg.BUILTIN_DATASET_NAME,
                scheme_id=cfg.BUILTIN_SCHEME_ID,
                source="builtin",
                db_path=cfg.DB_PATH,
                raw_dir=cfg.GENERATED_DIR,
                dataset_id=cfg.BUILTIN_DATASET_ID,
                data_start=cfg.DATA_START,
                data_end=cfg.DATA_END,
                notes="项目内置数据集，指向 Phase 1 建好的库（未搬迁，零迁移）",
            )
            # 内置数据集的表清单直接从库里读，保证白名单与真实结构一致。
            # 【为什么在这里同步而不是写死？】写死的话，将来 schema.sql 加表
            # 就要同步改这里，漏改的后果是「新表查不了」。
            return self.refresh_tables(dataset.dataset_id)

        return self._datasets[cfg.BUILTIN_DATASET_ID]

    # ================= 方案兼容性检查 =================

    def check_scheme_compatibility(self, dataset_id: str) -> dict[str, Any]:
        """检查数据集与所选指标方案的匹配度。

        返回 {"ok", "missing_tables", "usable_metrics", "total_metrics", "message"}

        【为什么这个检查是必需的？】
          上传的数据集默认只能沿用已有方案，而内置方案的 12 个指标全部依赖
          内置的 9 张表 —— 新库里一张都没有，每一次提问都会以
          「SQL 引用了未授权的表」失败。
          与其让用户以为系统坏了，不如在选择数据集时就把这件事说清楚，
          并指向「指标方案搭建」面板（那里可以按表结构生成草稿、人工确认后启用）。
          这是「诚实失败」的完整形态：说清现状 + 给出下一步，而不是「假装能用」。

        【为什么这里用方法内延迟 import MetricRegistry？】
          本模块的依赖边界是「数据层不依赖指标语义层」，所以模块级不能 import。
          但这个方法本质上就是在问「这份数据能不能满足这套口径」，
          天然需要读指标定义。用方法内 import 把依赖限制在这一个方法的调用期，
          既保持了模块级的干净，又不用把逻辑上移到上层。
        """
        dataset = self.get(dataset_id)
        scheme = self.get_scheme(dataset.scheme_id)

        if not scheme.exists():
            return {
                "ok": False,
                "missing_tables": [],
                "usable_metrics": [],
                "total_metrics": 0,
                "message": f"指标方案文件不存在：{scheme.registry_path}",
            }

        from src.metrics.registry import MetricRegistry

        try:
            registry = MetricRegistry(scheme.registry_path)
        except (ValueError, FileNotFoundError) as exc:
            return {
                "ok": False,
                "missing_tables": [],
                "usable_metrics": [],
                "total_metrics": 0,
                "message": f"指标方案加载失败：{exc}",
            }

        present = dataset.table_names
        required: set[str] = set()
        for metric in registry.all():
            required.update(metric.source_tables)

        missing = sorted(required - present)
        usable = [
            m.metric_id
            for m in registry.all()
            if set(m.source_tables) <= present
        ]

        ok = not missing
        if ok:
            message = f"方案与数据集匹配，{len(usable)} 个指标可用。"
        else:
            message = (
                f"方案依赖的 {len(missing)} 张表在本数据集中不存在："
                f"{', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}。"
                f"可在「🛠 指标方案搭建」面板为本数据集生成专属指标草稿，"
                f"或在重新上传时改选一个与表结构匹配的已有方案；"
                f"在此之前，提问会以「引用了未授权的表」失败。"
            )

        return {
            "ok": ok,
            "missing_tables": missing,
            "usable_metrics": usable,
            "total_metrics": len(registry),
            "message": message,
        }
