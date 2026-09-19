# -*- coding: utf-8 -*-
"""
数据集运行时上下文单元测试（Phase 7）
================================================================================
被测对象：src/context.py 的 DatasetContext / build_context。

【这个文件在防什么？】
  改造前，「查哪个库 / 用哪套指标 / 日期窗口 / 表白名单」这四个值分散在四个
  全局位置。多数据集下它们必须一起变，而**接错不会报错，只会算错**：

      用 A 方案的模板 + 连 B 数据集的库
      → 若两个库恰好都有同名表，查询会成功
      → 返回一个语义完全错误的数字，过程区的 SQL 看起来毫无异常

  所以本文件的核心不是「能不能跑」，而是「**有没有串台**」：
  第三节的两条用例专门验证「同名表在两个数据集里各查各的」。

【为什么全部用 tmp_path 造真库？】
  因为隔离性只能靠「真的连上去查一次」证明。用 mock 断言
  「db_path 传对了」只能证明参数传了，不能证明查询打在了正确的库上 ——
  而后者才是我们真正要保证的。
"""

from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest

from src import config as cfg
from src.context import DatasetContext, build_context
from src.data.dataset import DatasetStore, DatasetTable
from src.exceptions import SQLSecurityError

REAL_REGISTRY: Path = cfg.BASE_DIR / "src" / "metrics" / "metrics_registry.json"


def _make_db(path: Path, table: str, values: list[tuple]) -> None:
    """造一个只有一张两列表的迷你库（表名可定制，用来验证同名表不串台）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(f'CREATE TABLE "{table}" (k TEXT, v INTEGER)')
        conn.executemany(f'INSERT INTO "{table}" VALUES (?, ?)', values)
        conn.commit()
    finally:
        conn.close()


def _store(tmp_path: Path, *, scheme_id: str = "s1") -> DatasetStore:
    store = DatasetStore(root=tmp_path / "datasets", auto_builtin=False)
    store.register_scheme(
        display_name="测试指标方案",
        registry_path=REAL_REGISTRY,
        scheme_id=scheme_id,
    )
    return store


def _register(
    store: DatasetStore,
    dataset_id: str,
    *,
    table: str,
    values: list[tuple] | None = None,
    scheme_id: str = "s1",
    data_start: date | None = None,
    data_end: date | None = None,
) -> None:
    """造库 + 登记数据集，一步到位。"""
    db_path = store.root / dataset_id / "data.db"
    _make_db(db_path, table, values if values is not None else [("a", 1)])
    store.create_dataset(
        display_name=f"数据集 {dataset_id}",
        scheme_id=scheme_id,
        source="upload",
        db_path=db_path,
        raw_dir=store.root / dataset_id / "raw",
        dataset_id=dataset_id,
        data_start=data_start,
        data_end=data_end,
        tables=(DatasetTable(name=table, row_count=len(values or [("a", 1)])),),
    )


# ---------------------------------------------------------------------------
# 一、组装：四个值必须一起来自同一个数据集
# ---------------------------------------------------------------------------

def test_context_takes_everything_from_the_dataset(tmp_path):
    """库路径 / 白名单 / 日期窗口都来自 dataset，不来自全局常量。"""
    store = _store(tmp_path)
    _register(
        store,
        "ds_a",
        table="t_a",
        data_start=date(2025, 3, 1),
        data_end=date(2025, 3, 4),
    )

    ctx = build_context("ds_a", store=store)

    assert ctx.db_path == store.root / "ds_a" / "data.db"
    assert ctx.allowed_tables == frozenset({"t_a"})
    assert ctx.data_start == date(2025, 3, 1)
    assert ctx.data_end == date(2025, 3, 4)
    assert ctx.dataset_id == "ds_a"
    assert ctx.scheme_id == "s1"


def test_date_window_priority_explicit_over_dataset_over_cfg(tmp_path):
    """日期窗口的优先级链：显式传入 > dataset > config。

    【为什么这条必须有？】内置数据集在索引里记的就是 cfg 的值，
    所以「走哪一档」结果一样 —— 这正是「改造前后行为逐字节一致」的依据。
    而上传数据集记的是 CSV 里的真实日期，用 cfg 的窗口就会让模型
    自信地给出一个超出数据范围的结论。
    """
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_a", data_start=date(2025, 3, 1), data_end=date(2025, 3, 4))
    _register(store, "ds_nodate", table="t_b")  # 没识别出日期

    # ① 显式传入最高优先
    explicit = build_context("ds_a", store=store, data_start=date(2000, 1, 1), data_end=date(2000, 1, 2))
    assert (explicit.data_start, explicit.data_end) == (date(2000, 1, 1), date(2000, 1, 2))

    # ② 其次是 dataset 里记录的
    from_dataset = build_context("ds_a", store=store)
    assert from_dataset.data_start == date(2025, 3, 1)

    # ③ 数据集没记日期时回落到 config
    from_cfg = build_context("ds_nodate", store=store)
    assert from_cfg.data_start == cfg.DATA_START
    assert from_cfg.data_end == cfg.DATA_END


def test_unknown_dataset_raises_keyerror(tmp_path):
    """数据集不存在时 fail-fast，不静默回落到内置数据集。"""
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        build_context("not_exist", store=store)


def test_cache_key_delegates_to_dataset(tmp_path):
    """上下文的缓存键就是 dataset 的缓存键（Streamlit 靠它区分 Agent 实例）。"""
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_a")
    ctx = build_context("ds_a", store=store)
    assert ctx.cache_key() == store.get("ds_a").cache_key()


def test_describe_mentions_dataset_scheme_and_window(tmp_path):
    """describe() 是前端副标题与日志的来源，三个要素都要在。"""
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_a", data_start=date(2025, 3, 1), data_end=date(2025, 3, 4))
    text = build_context("ds_a", store=store).describe()

    assert "ds_a" in text
    assert "测试指标方案" in text        # 方案的可读名，而非 ID
    assert "2025-03-01" in text and "2025-03-04" in text


# ---------------------------------------------------------------------------
# 二、白名单：校验器按数据集收紧
# ---------------------------------------------------------------------------

def test_validator_rejects_tables_outside_this_dataset(tmp_path):
    """本数据集没有的表必须被拒 —— 这是「数据集隔离」真正生效的地方。

    改造前的语义是「不能超出全库 9 张表白名单」；多数据集下「全库白名单」
    这个概念不存在了，基准变成「本数据集实际有的表」。
    """
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_a")
    ctx = build_context("ds_a", store=store)

    validator = ctx.build_validator(max_rows=50)
    with pytest.raises(SQLSecurityError):
        validator.validate("SELECT * FROM user_daily_snapshot")

    # 自己那张表则放行，并补上 LIMIT
    safe = validator.validate("SELECT * FROM t_a")
    assert safe.rstrip().endswith("LIMIT 50")


def test_metric_declaring_outside_table_is_rejected(tmp_path):
    """指标声明的来源表不在本数据集白名单里 → 配置层面的防线也要报错。"""
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_a")
    ctx = build_context("ds_a", store=store)

    with pytest.raises(SQLSecurityError) as exc:
        ctx.build_validator().validate(
            "SELECT * FROM t_a", allowed_tables=("t_a", "dim_user")
        )
    assert "本数据集的白名单" in str(exc.value)


def test_executor_shares_the_same_whitelist_as_context(tmp_path):
    """executor 内嵌的校验器必须与 ctx 一致。

    【为什么要一致】ctx.build_executor() 把 validator 一起造好，是为了让
    「行数上限」在两处相同（validator 补 LIMIT、executor 做 fetchmany 上限）。
    两者不一致会出现「SQL 里写着 LIMIT 500 但只 fetch 了 100 行」这种静默丢数据。
    """
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_a")
    ctx = build_context("ds_a", store=store)

    executor = ctx.build_executor(max_rows=77)
    assert executor.validator.allowed_tables == ctx.allowed_tables
    assert executor.max_rows == 77
    assert executor.validator.max_rows == 77


# ---------------------------------------------------------------------------
# 三、隔离性的正面证明：同名表各查各的
# ---------------------------------------------------------------------------

def test_same_table_name_in_two_datasets_returns_own_data(tmp_path):
    """两个数据集都有 t_metrics，各自查到的必须是自己的数。

    【这条是整个多数据集改造的核心断言】
      如果上下文没有把「库路径 + 白名单」绑在一起（比如缓存键漏了数据集身份），
      这里就会返回同一份数据 —— 而且不报错。用「同名表 + 不同值」来验，
      是因为它能把「配错」变成可观测的差异；若两库表名不同，
      配错会以「未授权的表」失败，反而掩盖了问题。
    """
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_metrics", values=[("dau", 111)])
    _register(store, "ds_b", table="t_metrics", values=[("dau", 222)])

    ctx_a = build_context("ds_a", store=store)
    ctx_b = build_context("ds_b", store=store)

    # 先证明两个库确实指向不同文件
    assert ctx_a.db_path != ctx_b.db_path

    result_a = ctx_a.build_executor().execute("SELECT v FROM t_metrics")
    result_b = ctx_b.build_executor().execute("SELECT v FROM t_metrics")

    assert result_a.success and result_b.success
    assert result_a.rows == [(111,)]
    assert result_b.rows == [(222,)]


def test_dataset_without_the_table_fails_instead_of_borrowing(tmp_path):
    """只有 A 有 t_metrics 时，B 查它必须失败 —— 不能「借用」A 的数据。

    这是上一条的反面：隔离不仅要求「查到自己的」，还要求
    「查不到别人的」。只测正面会漏掉「回落到全局默认库」这类实现错误。
    """
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_metrics", values=[("dau", 111)])
    _register(store, "ds_b", table="t_other", values=[("x", 1)])

    ctx_b = build_context("ds_b", store=store)
    result = ctx_b.build_executor().execute("SELECT v FROM t_metrics")

    assert result.success is False
    assert "未授权的表" in (result.error or "")


# ---------------------------------------------------------------------------
# 四、SQL 生成与提示词也按数据集走
# ---------------------------------------------------------------------------

def test_generator_uses_dataset_date_window(tmp_path):
    """SQL 生成器里的日期参数必须来自本数据集的窗口。

    【不这样会怎样】模板里的默认窗口参数会取全局 config 的日期，
    于是「最近 7 天」算出来的是内置数据集的最近 7 天，而查询打在上传库上
    —— 结果为空，用户以为数据没导进去。
    """
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_a", data_start=date(2025, 3, 1), data_end=date(2025, 3, 4))
    ctx = build_context("ds_a", store=store)

    generator = ctx.build_generator()
    assert generator.data_start == date(2025, 3, 1)
    assert generator.data_end == date(2025, 3, 4)
    assert generator.registry is ctx.registry


def test_system_prompt_states_dataset_window(tmp_path):
    """提示词里必须写本数据集的窗口与「今天」，否则模型会按错误的
    「最新可用日期」回答。"""
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_a", data_start=date(2025, 3, 1), data_end=date(2025, 3, 4))
    prompt = build_context("ds_a", store=store).build_system_prompt()

    assert "2025-03-01" in prompt
    assert "2025-03-04" in prompt
    assert str(cfg.DATA_END) not in prompt  # 不能混进全局窗口


def test_builtin_context_matches_legacy_globals(tmp_path):
    """内置数据集走同一条代码路径，且四个值都与改造前的全局值一致。

    【为什么这条是「向后兼容」的锁】改造后的默认行为必须与改造前逐字节一致。
    内置数据集在索引里登记的 db_path / 日期窗口就是 cfg 的值，
    所以这条断言一旦变红，说明「默认路径」被动过了。
    """
    store = DatasetStore(root=tmp_path / "datasets")  # auto_builtin=True
    ctx = build_context(cfg.BUILTIN_DATASET_ID, store=store)

    assert ctx.db_path == cfg.DB_PATH
    assert (ctx.data_start, ctx.data_end) == (cfg.DATA_START, cfg.DATA_END)
    assert ctx.allowed_tables == store.get(cfg.BUILTIN_DATASET_ID).table_names
    assert ctx.scheme.registry_path == cfg.BASE_DIR / "src" / "metrics" / "metrics_registry.json"


def test_context_is_frozen(tmp_path):
    """上下文是不可变的：下游某一层改了 allowed_tables 会静默影响其他调用者。"""
    store = _store(tmp_path)
    _register(store, "ds_a", table="t_a")
    ctx: DatasetContext = build_context("ds_a", store=store)

    with pytest.raises(FrozenInstanceError):
        ctx.allowed_tables = frozenset()  # type: ignore[misc]
