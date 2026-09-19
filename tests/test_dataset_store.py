# -*- coding: utf-8 -*-
"""
数据集索引管理器单元测试（Phase 7）
================================================================================
被测对象：src/data/dataset.py 的 Dataset / MetricScheme / DatasetStore。

【为什么这个文件里大量用 tmp_path，而不是真实 data/datasets/？】
  因为这些测试会**写文件**（建索引、登记数据集）。
  真实目录是运行期状态，测试往里写会污染用户的界面（列表里冒出测试数据集），
  而 DatasetStore 的 root / index_path 都是可注入的 —— 这正是当初把它设计成
  可注入的原因。用 tmp_path 后，每个用例都在自己的沙箱里跑，互不干扰、
  可并行、失败也不留痕。

【auto_builtin 为什么默认关掉？】
  它会在构造时去读真实的 106MB 库做 COUNT(*) 与 PRAGMA。绝大多数用例
  只关心「索引怎么读写」，不需要这个副作用。只有专门验证「自愈」的用例
  才显式打开它。

【测试分四组】
  一、ID 规范化 —— 这是路径穿越的唯一防线，必须逐个边界打
  二、索引读写 —— 原子性、损坏自愈、相对路径
  三、数据集 / 方案的增改查与缓存键
  四、方案兼容性检查（前端「诚实失败」提示的数据来源）
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from src import config as cfg
from src.data.dataset import DatasetStore, DatasetTable, MetricScheme, _ID_RE
from src.sqlgen.validator import ALLOWED_TABLES

# 真实的指标注册表文件：兼容性检查需要一个「真的能加载」的方案
REAL_REGISTRY: Path = cfg.BASE_DIR / "src" / "metrics" / "metrics_registry.json"


def _store(
    tmp_path: Path,
    *,
    auto_builtin: bool = False,
    scheme_id: str = "s1",
) -> DatasetStore:
    """造一个跑在沙箱里的 store，并登记一套指向真实注册表的方案。"""
    store = DatasetStore(root=tmp_path / "datasets", auto_builtin=auto_builtin)
    store.register_scheme(
        display_name="测试指标方案",
        registry_path=REAL_REGISTRY,
        scheme_id=scheme_id,
    )
    return store


def _create(
    store: DatasetStore,
    *,
    name: str = "测试数据集",
    scheme_id: str = "s1",
    dataset_id: str | None = None,
    tables=(),
    **kwargs,
):
    """create_dataset 的薄封装：只暴露用例真正关心的参数。"""
    return store.create_dataset(
        display_name=name,
        scheme_id=scheme_id,
        source="upload",
        db_path=store.root / "x" / "data.db",
        raw_dir=store.root / "x" / "raw",
        dataset_id=dataset_id,
        tables=tuple(tables),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 一、ID 规范化：路径穿越防线
# ---------------------------------------------------------------------------
# dataset_id 会直接变成目录名（data/datasets/<id>/）。它来自用户输入的
# 数据集名称，是全项目唯一的不可信输入 —— 所以这组用例是安全用例，不是
# 功能用例：一旦有一条能穿出去，一次上传就能把库写到项目目录之外。

@pytest.mark.parametrize(
    "raw",
    [
        "我的数据集",            # 全中文：白名单过滤后为空，走哈希兜底
        "",                     # 空串
        "   ",                  # 全空白
        "../../etc/passwd",     # 经典路径穿越
        "..\\..\\windows",      # Windows 反斜杠变体
        "a/b/c",                # 子路径
        "007",                  # 数字开头
        "_underscore",          # 下划线开头
        "A B C",                # 大写 + 空格
        "?!@#$%^&*()",          # 全特殊字符
        "x" * 200,              # 超长
        "数据 集 2024 版",       # 中文 + 数字混合
    ],
)
def test_normalize_id_always_returns_legal_id(raw):
    """无论输入多离谱，normalize_id 的输出都必须过白名单正则。

    这是「不变式测试」：不针对某个具体输入断言结果，而是断言
    「输出永远满足安全约束」。哪怕以后改了折叠规则，这条也不会失效。
    """
    assert _ID_RE.match(DatasetStore.normalize_id(raw))


def test_normalize_id_folds_english_name():
    """英文名走可读路径：小写 + 非字母数字折叠成下划线。"""
    assert DatasetStore.normalize_id("Sales Data 2024") == "sales_data_2024"
    assert DatasetStore.normalize_id("  my__set  ") == "my_set"


def test_normalize_id_is_stable_for_cjk():
    """中文名折叠成空 → 用哈希兜底。

    【为什么强调「稳定」？】因为同一个名称重复上传必须得到同一个 ID
    （再靠 _unique_id 加后缀），否则「重新导入同一份数据」会散成一堆
    互不相识的数据集，索引里全是孤儿。
    """
    first = DatasetStore.normalize_id("我的数据集")
    assert first.startswith("ds_")
    assert first == DatasetStore.normalize_id("我的数据集")
    assert first != DatasetStore.normalize_id("另一份数据集")


@pytest.mark.parametrize(
    "bad",
    ["../evil", "a/b", "A_b", "_x", "x" * 41, "a b", "数据", ".."],
)
def test_create_dataset_rejects_explicit_illegal_id(tmp_path, bad):
    """显式传入的 dataset_id 必须过同一道白名单。

    【为什么不能只在 normalize_id 里校验？】因为调用方可以绕过它 ——
    内置数据集就用固定 ID 显式传入。校验必须落在 create_dataset 这个
    「真正写盘之前」的关口上，而不是「生成 ID 的函数」里。

    【注意空串不在本例里】`dataset_id or normalize_id(...)` 把空串当作
    「没指定」—— 这是有意的：空串不是「想建一个叫空名字的数据集」，
    而是「我没给 ID」。所以它不会抛错，而是走自动生成。
    """
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        _create(store, dataset_id=bad)


def test_duplicate_id_gets_numeric_suffix(tmp_path):
    """重名时追加 _2 / _3，而不是覆盖 —— 覆盖会让上一份数据的索引凭空消失。"""
    store = _store(tmp_path)
    a = _create(store, name="重复名", dataset_id="dup")
    b = _create(store, name="重复名", dataset_id="dup")
    c = _create(store, name="重复名", dataset_id="dup")
    assert [a.dataset_id, b.dataset_id, c.dataset_id] == ["dup", "dup_2", "dup_3"]


# ---------------------------------------------------------------------------
# 二、索引读写
# ---------------------------------------------------------------------------

def test_index_persists_across_store_instances(tmp_path):
    """索引是持久化的：换一个 store 实例仍能读回同样的数据集。"""
    store = _store(tmp_path)
    created = _create(store, name="持久化测试", dataset_id="persist", data_start=date(2025, 1, 1))

    reopened = DatasetStore(root=tmp_path / "datasets", auto_builtin=False)
    again = reopened.get("persist")
    assert again.display_name == "持久化测试"
    assert again.data_start == date(2025, 1, 1)
    assert again.cache_key() == created.cache_key()


def test_save_is_atomic_and_leaves_no_tmp_file(tmp_path):
    """原子写：先写 .tmp 再 os.replace，成功后不留残渣。

    【为什么这条值得单独测？】因为「写坏了」的代价是全部数据集一起消失
    （索引是单一 JSON）。这条用例同时锁住两件事：索引确实生成了，
    且没有留下半截的 .tmp —— 后者说明 os.replace 真的被用上了。
    """
    store = _store(tmp_path)
    _create(store, dataset_id="atomic")
    assert store.index_path.exists()
    assert list(store.root.glob("*.tmp")) == []


def test_corrupted_index_is_rebuilt_instead_of_failing(tmp_path):
    """索引损坏时不应让应用起不来 —— 它是派生状态，重建即可。

    真实资产（库文件、CSV）都还在磁盘上，索引丢了只是要重新登记。
    """
    root = tmp_path / "datasets"
    root.mkdir(parents=True)
    (root / "index.json").write_text("{ 这不是合法 JSON", encoding="utf-8")

    store = DatasetStore(root=root)  # auto_builtin=True
    assert store.has(cfg.BUILTIN_DATASET_ID)


def test_broken_single_record_does_not_drop_the_others(tmp_path):
    """单条坏记录只跳过自己，不拖垮整个列表。"""
    root = tmp_path / "datasets"
    root.mkdir(parents=True)
    (root / "index.json").write_text(
        '{"version": 1, "datasets": ['
        '{"display_name": "缺 dataset_id 的坏记录"},'
        '{"dataset_id": "good", "display_name": "好记录", "scheme_id": "s1",'
        ' "db_path": "x.db", "raw_dir": "raw", "source": "upload"}'
        "], \"schemes\": []}",
        encoding="utf-8",
    )
    store = DatasetStore(root=root, auto_builtin=False)
    assert store.has("good")
    assert not store.has("缺 dataset_id 的坏记录")


def test_builtin_ensure_is_idempotent(tmp_path):
    """内置数据集 / 方案只登记一次，重复构造不会产生 _2 副本。"""
    store = DatasetStore(root=tmp_path / "datasets")
    assert store.has(cfg.BUILTIN_DATASET_ID)
    assert store.get_scheme(cfg.BUILTIN_SCHEME_ID).exists()

    again = DatasetStore(root=tmp_path / "datasets")
    assert [d.dataset_id for d in again.list_datasets()] == [cfg.BUILTIN_DATASET_ID]


def test_builtin_tables_come_from_real_db(tmp_path):
    """内置数据集的表清单是「从库里读出来的」，不是写死的。

    写死的话，schema.sql 加表就要同步改代码，漏改的后果是「新表查不了」
    —— 而报错是「引用了未授权的表」，看起来像指标配置错了。
    """
    store = DatasetStore(root=tmp_path / "datasets")
    assert store.get(cfg.BUILTIN_DATASET_ID).table_names == ALLOWED_TABLES


def test_stored_path_is_relative_inside_base_dir():
    """BASE_DIR 内的路径存成相对形式，换机器克隆后仍然指得对。"""
    inside = cfg.BASE_DIR / "data" / "datasets" / "demo" / "data.db"
    assert DatasetStore._to_stored_path(inside) == "data/datasets/demo/data.db"


def test_stored_path_keeps_absolute_outside_base_dir(tmp_path):
    """BASE_DIR 外的路径无法相对化，只能原样保留（不硬凑一个错路径）。"""
    stored = DatasetStore._to_stored_path(tmp_path / "outside.db")
    assert Path(stored).is_absolute()


# ---------------------------------------------------------------------------
# 三、数据集与方案
# ---------------------------------------------------------------------------

def test_list_datasets_puts_builtin_first(tmp_path):
    """内置数据集永远排第一 —— 它是默认选中项，排序即默认值。"""
    store = _store(tmp_path, auto_builtin=True)
    _create(store, name="上传的", dataset_id="zzz")
    assert store.list_datasets()[0].dataset_id == cfg.BUILTIN_DATASET_ID


def test_get_unknown_dataset_raises_keyerror(tmp_path):
    """找不到就抛 KeyError（fail-fast），不返回 None。

    返回 None 很容易被上层漏判，最后变成「悄悄用了默认数据集」——
    这比直接报错危险得多。
    """
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.get("not_exist")


def test_create_dataset_with_unknown_scheme_raises(tmp_path):
    """指向不存在的方案时必须报错，否则会在运行期变成「指标字典是空的」。"""
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        _create(store, scheme_id="no_such_scheme")


def test_register_scheme_is_idempotent(tmp_path):
    """重复登记同一 ID 返回已有方案，不覆盖 —— 覆盖会静默改掉别处引用。"""
    store = _store(tmp_path)
    first = store.get_scheme("s1")
    again = store.register_scheme(
        display_name="改了名字",
        registry_path=tmp_path / "other.json",
        scheme_id="s1",
    )
    assert again.display_name == first.display_name
    assert again.registry_path == REAL_REGISTRY


def test_cache_key_changes_with_scheme_and_created_at(tmp_path):
    """缓存键必须随「方案」与「创建时间」变化。

    【这是多数据集改造里最容易漏的一处】Streamlit 的 cache_resource 用
    「函数名 + 参数」做键。若缓存键只按 dataset_id 算，那么
    「同一个数据集改指另一套指标方案」之后，Agent 会继续用旧方案 ——
    表现是「配置改了但不生效」，排查成本极高。
    """
    store = _store(tmp_path, scheme_id="s1")
    base = _create(store, dataset_id="cache", scheme_id="s1")
    assert base.scheme_id in base.cache_key()
    assert base.dataset_id in base.cache_key()

    # 只换方案：缓存键必须跟着变
    other_scheme = replace(base, scheme_id="s2")
    assert other_scheme.cache_key() != base.cache_key()

    # 只换创建时间（等价于重新上传）：缓存键同样必须变
    reuploaded = replace(base, created_at="2099-01-01T00:00:00")
    assert reuploaded.cache_key() != base.cache_key()
    assert reuploaded.dataset_id == base.dataset_id  # 身份不变，只有版本变


# ---------------------------------------------------------------------------
# 四、方案兼容性检查
# ---------------------------------------------------------------------------

def test_compatibility_ok_when_all_tables_present(tmp_path):
    """表齐 → ok=True，且可用指标数等于总指标数。"""
    store = _store(tmp_path)
    ds = _create(
        store,
        dataset_id="full",
        tables=[DatasetTable(name=n) for n in sorted(ALLOWED_TABLES)],
    )
    assert ds.table_names == ALLOWED_TABLES

    info = store.check_scheme_compatibility("full")
    assert info["ok"] is True
    assert info["missing_tables"] == []
    assert info["usable_metrics"] and len(info["usable_metrics"]) == info["total_metrics"]


def test_compatibility_reports_missing_tables(tmp_path):
    """表不齐 → ok=False 且列出缺哪些表。

    这是前端「诚实失败」提示的数据来源：本轮不支持为新数据集自动生成方案，
    与其让用户每次提问都撞「引用了未授权的表」，不如在选择数据集时就说清楚。
    """
    store = _store(tmp_path)
    _create(store, dataset_id="partial")
    info = store.check_scheme_compatibility("partial")

    assert info["ok"] is False
    assert info["missing_tables"]
    assert "不存在" in info["message"]
    assert info["usable_metrics"] == []


def test_compatibility_reports_missing_registry_file(tmp_path):
    """方案文件本身丢了也要给一条能看懂的提示，而不是抛异常。"""
    store = _store(tmp_path, scheme_id="s_broken")
    store._schemes["s_broken"] = MetricScheme(
        scheme_id="s_broken",
        display_name="坏方案",
        registry_path=tmp_path / "not_here.json",
        created_at="",
    )
    _create(store, dataset_id="orphan", scheme_id="s_broken")

    info = store.check_scheme_compatibility("orphan")
    assert info["ok"] is False
    assert "不存在" in info["message"]
