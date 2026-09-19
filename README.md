# 🎮 游戏数据智能分析师 Agent（Game Data Analyst Agent）

[![CI](https://github.com/kemingin/game-data-analyst-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/kemingin/game-data-analyst-agent/actions/workflows/ci.yml)

> 用中文提问 → 自动映射到标准指标口径 → 生成并校验 SQL → 查真实数据 → 给出结论与图表。
> **大模型不写 SQL，也不编数字。**

一个面向游戏运营场景的自然语言数据分析助手。运营同学问「v2.0.0 上线后新用户留存变好了吗」，
系统自动完成「理解问题 → 匹配指标 → 查库 → 出结论 + 出图」，全程可追溯、可审计、可评测。

---

## 一、这个项目解决什么问题

游戏运营团队的日常数据分析有三个反复出现的痛点：

| 痛点 | 具体表现 | 本项目的解法 |
| --- | --- | --- |
| **口径不统一** | 同样叫「留存率」，有人算 D1 有人算 D7，有人没剔除观察期不足的用户 | 指标语义层：12 个指标的口径、参数、来源表全部写进 `metrics_registry.json`，单一事实来源 |
| **取数门槛高** | 运营不会写 SQL，每次要数就排队等数据分析师 | 自然语言入口，Agent 自动把问题映射成指标 + 参数 |
| **不敢信结果** | AI 直接生成 SQL 容易算错、甚至幻觉出一个数字 | 硬约束：LLM 只做「映射 + 抽参」，SQL 由程序按预审模板填充生成 |

---

## 二、核心设计：五条硬约束

这五条是整个项目的设计骨架，也是与「让大模型直接写 SQL」的通用方案最本质的区别。

**① LLM 不直接生成 SQL。**
模型只输出 `metric_id` 和参数（例如 `retention_rate` + `{day_n: 7}`），
SQL 由程序从注册表的模板填充生成。模型的能力边界被钉死在它最擅长的「语义映射」上，
计算这件事交给确定性的程序 —— 这是**用架构消除幻觉**，而不是靠提示词祈祷模型别算错。

**② SQL 只允许 SELECT，且受四层防线保护。**

```
第一层 · 模板受控    SQL 来自注册表白名单模板，模型碰不到
第二层 · 语法校验    仅 SELECT、表名必须在 source_tables 白名单内、禁止多语句
第三层 · 只读连接    SQLite 以 mode=ro 打开，物理上写不进去
第四层 · 参数绑定    参数一律占位符 :name 绑定，杜绝字符串拼接注入
```

外加执行护栏：单次查询行数上限 500 行、执行超时 10 秒（用 `progress_handler` 实现真正的
中断，而不是只设一个没人执行的 `timeout` 参数）。

**③ 结果双份输出。**
每个工具返回 `content`（Markdown 文本，喂给模型）+ `data`（结构化字典，喂给前端画图）。
文本给人看，结构化数据给程序用，两条路各走各的 —— 让前端去反解析 Markdown 表格是极脆的做法。

**④ 重大决策需人工确认。**
涉及投放预算、版本调优、留存策略的建议，前端强制走「复核人 + 勾选确认 + 写入审计日志」，
而不是弹一个点掉就完事的提示框。回答的是上线前必须回答的一题：**Agent 出错了谁负责？**

**⑤ 回答里的数字必须能溯源（第五层防线）。**
把答案里**每一个数字**反向映射到本次查询的结果集：全都找得到出处 → 原样给；
少量找不到（占比 ≤ 1/3）→ 回答照给、末尾标注哪几个数没出处；
超过 1/3 → 判定疑似编造，**拦截整条回答**，宁可不给数字也不给假数字。
> 这一条是真机验证逼出来的。前四层全在 SQL 出口，隐含假设「模型一定会去查」——
> 备用模型 GLM-4-Flash 直接跳过全部工具、凭记忆答「日活是 100,000 人」，
> 四层防线对它完全无感。所以防线要跟着**数据流**走，补在答案出口。
> 第一版判据只问「有没有成功取过数」，于是「查了 1 次 DAU、又编了 5 个留存率」
> 能大摇大摆过去；Phase 9 升级为逐数字溯源后，这种混编会被拦下。
> 判定原语收在 `src/grounding.py`，运行时防线与评测台共用同一份口径。
> 过程详见[开发复盘记录](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/docs/01_项目文档/开发复盘记录_Phase1-3.txt) 7.4 节。

---

## 三、系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  前端层   Streamlit + Plotly                            app.py       │
│  对话区 │ 指标卡/自动图表 │ 执行过程（默认折叠）│ 人工确认 │ 审计日志   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  ask(question, history)
┌───────────────────────────────▼─────────────────────────────────────┐
│  Agent 层（ReAct 循环，上限 6 轮）          src/agent/                │
│                                                                     │
│    GameDataAgent ── llm_client.py   DeepSeek(主) / 智谱GLM(备)        │
│         │                           · 401 不重试直接降级              │
│         │                           · 429/5xx 指数退避重试            │
│         └── tools.py  3 个 Function Calling 工具                     │
│               list_metrics / get_metric_detail / query_metric       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  只传 metric_id + 参数（不是 SQL！）
┌───────────────────────────────▼─────────────────────────────────────┐
│  指标语义层                                src/metrics/ + src/sqlgen/ │
│                                                                     │
│    metrics_registry.json   12 个指标 · 口径/参数/来源表/SQL模板        │
│         ↓ 模板填参                                                    │
│    generator.py → validator.py → executor.py                        │
│         （四层 SQL 防线 + 答案来源校验 + 行数上限 + 超时中断）            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  受控 SQL（只读）
┌───────────────────────────────▼─────────────────────────────────────┐
│  数据层      SQLite   data/game_analytics.db                         │
│  9 张表 · 1,181,298 行 · 106.9 MB · 90 天窗口（2026-06-20 ~ 09-17）   │
│  真实 Steam 数据（去重后） + 模拟的运营事件/快照/维表                    │
└─────────────────────────────────────────────────────────────────────┘

        ┌────────────────────────────────────────────────────────────┐
        │  评测体系   src/eval/ + scripts/run_*_eval.py              │
        │  ① 端到端 31 条用例（真值走不经过 LLM 的独立路径）        │
        │  ② 专项 5 个 Phase：工具调用 / 性能缓存 / 幻觉质量 /      │
        │     鲁棒性边界 / 优化验证                                  │
        └────────────────────────────────────────────────────────────┘
```

---

## 四、目录结构

```
Game Data Analyst Agent/
├── app.py                          # Streamlit 前端主入口
├── requirements.txt                # 依赖清单（★ 版本已钉死 ==，不是 >=）
├── pyproject.toml                  # 工具配置：pytest 的 testpaths / addopts
├── .env.example                    # API Key 模板（复制为 .env 后填入）
├── Dockerfile                      # 【Phase 10】镜像定义：装依赖 → 生成数据 → 建库 → 起服务
├── .dockerignore                   # 【Phase 10】构建上下文排除（★ 第一职责是挡住 .env）
├── .github/workflows/ci.yml        # 【Phase 11】CI：无数据建库 + 743 条单测
│
├── src/
│   ├── config.py                   # 全局配置：路径 / 业务常量 / SQL 护栏 / LLM 参数
│   ├── exceptions.py               # 分层异常体系
│   ├── context.py                  # 【组装根】Phase 7：把库/指标/窗口/白名单打包成 DatasetContext
│   ├── grounding.py                # 【判据原语】Phase 9：数字抽取/比对/分级，防线与评测台共用
│   │
│   ├── data/                       # 【数据层】Phase 1 + 7
│   │   ├── schema.sql              #   建表 DDL（9 张表）
│   │   ├── indexes.sql             #   索引（导入后再建，快 3~5 倍）
│   │   ├── steam_data.py           #   真实 Steam 数据清洗（去重 16.9 万组重复主键）
│   │   ├── dataset.py              #   多数据集元信息（Dataset / MetricScheme / DatasetStore）
│   │   └── upload.py               #   CSV 上传：解码 → 类型推断 → 建表导数
│   │
│   ├── metrics/                    # 【指标语义层】Phase 2 + 8
│   │   ├── metrics_registry.json   #   ★ 12 个指标定义的单一事实来源
│   │   ├── registry.py             #   加载/检索/生成给 LLM 的指标目录
│   │   ├── schema_roles.py         #   Phase 8：列语义角色推断（9 类角色）
│   │   ├── metric_templates.json   #   Phase 8：指标模板资产（6 个模板）
│   │   ├── templates.py            #   Phase 8：模板加载 + 六项加载期自检
│   │   └── draft.py                #   Phase 8：草稿生成/校验/落盘/改指闭环
│   │
│   ├── sqlgen/                     # 【指标语义层】Phase 2
│   │   ├── generator.py            #   模板填参 → 生成「执行版 + 展示版」两份 SQL
│   │   ├── validator.py            #   语法/白名单校验（四层防线第 1~2 层）
│   │   └── executor.py             #   只读连接 + 行数上限 + 超时中断
│   │
│   ├── agent/                      # 【Agent 层】Phase 3
│   │   ├── llm_client.py           #   多供应商客户端：重试 + 降级 + token 记账
│   │   ├── tools.py                #   3 个 Function Calling 工具 + 分发器
│   │   ├── prompts.py              #   系统提示词（分块拼装，动态注入指标目录）
│   │   └── react_agent.py          #   ReAct 主循环 + 熔断 + 过程记录
│   │
│   ├── ui/                         # 【前端层】Phase 4 + 7 + 8
│   │   ├── charts.py               #   选图策略与画图分离（Plotly）
│   │   ├── overview.py             #   侧边栏数据概览
│   │   ├── dataset_panel.py        #   数据集面板：选择器 / 上传 / 兼容性提示
│   │   └── scheme_builder_panel.py #   指标方案搭建面板（草稿复审 + 手工搭建 + 启用）
│   │
│   └── eval/                       # 【评测体系】Phase 5~6
│       ├── eval_set.json           #   31 条标准用例（只声明期望指标+参数）
│       ├── eval_set.py             #   用例加载
│       ├── metrics.py              #   参数校验/汇总统计（数字原语已下沉到 src/grounding.py）
│       ├── hallucination.py        #   幻觉检测：句子级 + 数字级双口径
│       ├── judge.py                #   LLM-as-a-Judge 评审（含一致性守卫）
│       ├── robustness_cases.json   #   21 条鲁棒性用例（判据是行为而非分数）
│       └── runner.py               #   双路径对账 + 报告生成
│
├── scripts/
│   ├── generate_mock_data.py       # 生成模拟数据 CSV
│   ├── build_database.py           # 建库 + 导入 + 建索引
│   ├── run_eval.py                 # 端到端评测入口（31 条）
│   ├── run_tool_eval.py            # 专项 · 工具调用准确性（Phase 1）
│   ├── run_perf_eval.py            # 专项 · 缓存命中率与响应性能（Phase 2）
│   ├── run_hallucination_eval.py   # 专项 · 幻觉与端到端质量（Phase 3，--tag 标轮次）
│   └── run_robustness_eval.py      # 专项 · 鲁棒性与边界（Phase 4）
│
├── tests/                          # 400+ 个测试函数 / pytest 收集 743 条
├── docs/                           # 文档与评测产物（按「谁读」分四层，见下表）
│   ├── 01_项目文档/                 # 第一次接触项目时读
│   │   ├── data_dictionary.md      # 数据字典
│   │   ├── 开发复盘记录_Phase1-3.txt # ★ 全流程复盘：设计决策 + 缺陷 + 追问答法
│   │   └── 文件清单说明.txt          # 逐文件说明（作品集索引）
│   ├── 02_测试复盘/                 # 想知道"踩过哪些坑、为什么这么设计"时读
│   │   └── 测试复盘记录.txt          # ★ 测试体系全档案：怎么证明它真的能用
│   ├── 03_评测报告/                 # 只看结论时读（人读的 txt）
│   │   ├── 评测报告_Phase6.txt      # 端到端评测报告（31 条用例逐例明细）
│   │   └── *测试报告_Phase*.txt     # 五个专项评测的结论报告
│   └── 04_评测原始数据/              # 要复核结论时读（Excel / 程序读的 csv、json）
│       └── *.csv / *.json          # 逐例明细、逐句明细、结构化评测结果
└── data/game_analytics.db          # SQLite 数据库
```

> **docs 为什么要分子目录？** 因为两类读者混在一起了：人想看结论（txt），
> 程序想复核证据（csv/json）。评测每跑一轮就落一份产物，平铺到几十个文件后
> 「找一份报告」会变成体力活。分类依据是「谁在读、什么时候读」，不是文件格式。
> 目录常量统一定义在 [config.py](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/src/config.py)
> 的 `DOCS_*_DIR`，5 个评测脚本只引用常量、不自己拼路径字符串 —— 将来再挪位置只改一处。

---

## 五、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# Windows PowerShell
Copy-Item .env.example .env
# macOS / Linux
cp .env.example .env
```

编辑 `.env`，填入至少一个 Key（两个都填则主用 DeepSeek，挂了自动降级到 GLM）：

```ini
DEEPSEEK_API_KEY=sk-xxxxxxxx
GLM_API_KEY=xxxxxxxx
```

> `.env` 已写进 `.gitignore`，密钥永远不会进仓库。

### 3. 准备数据（首次运行）

```bash
python scripts/generate_mock_data.py    # 生成模拟数据 CSV
python scripts/build_database.py        # 建库 + 导入 + 建索引
```

### 4. 启动前端

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，可以点示例问题，或直接输入：

- `最近的次日留存率是多少？`
- `v2.0.0 上线后新用户留存变好了吗？`
- `各渠道的次日留存对比如何？`
- `最近 7 天的付费率和 ARPU 分别是多少？`

### 5. 跑评测（可选，会真实消耗 token）

```bash
python scripts/run_eval.py --limit 3                       # 先跑 3 条验证流程
python scripts/run_eval.py --dump docs/04_评测原始数据/评测结果.json  # 全量 31 条
```

专项评测（报告写进 `docs/03_评测报告/`，逐例 CSV/JSON 写进 `docs/04_评测原始数据/`）：

```bash
python scripts/run_tool_eval.py                            # Phase 1 · 工具调用准确性
python scripts/run_perf_eval.py                            # Phase 2 · 缓存命中率与响应性能
python scripts/run_hallucination_eval.py --tag 第1轮        # Phase 3 · 幻觉与端到端质量
python scripts/run_robustness_eval.py                      # Phase 4 · 鲁棒性与边界
```

> `run_hallucination_eval.py` 的 `--tag` 会写进产物文件名、报告正文和两份 CSV 首列 ——
> 不加它就会覆写上一轮的留档（这个坑真的踩过，详见测试复盘记录）。

### 6. 跑测试

```bash
python -m pytest
```

> 测试目录与 `-q` 已配在 [pyproject.toml](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/pyproject.toml)，
> 所以在项目根目录直接敲 `pytest` 也是一样的效果。

### 7. 用 Docker 跑（Phase 10 · 一条命令，免装环境）

镜像**自给自足**：构建时自动生成模拟数据、建库，不需要下载那 111MB 真实 Steam 数据。

```bash
docker build -t gda .      # 构建（内含生成数据 + 建库，几十秒）
docker run --rm -p 8501:8501 -e DEEPSEEK_API_KEY=sk-xxx gda
```

不带 Key 也能启动 —— 前端会显示「尚未配置 API Key」的引导文案，方便先看产品形态；
Key 只能通过 `-e` 运行时注入（镜像里不含任何密钥，`.env` 已被 `.dockerignore` 排除）。

> **跑之前先确认这两件事**（否则会卡在与代码无关的地方）：
> 1. **引擎在跑，不只是 CLI 在**。`docker --version` 有输出只说明装了 CLI；
>    引擎要等 Docker Desktop 启动后才可用，否则报
>    `failed to connect to the docker API at npipe://...`。
>    判断方法：`docker info` 能返回内容就算就绪。
> 2. **能访问 Docker Hub**。`registry-1.docker.io` 在国内直连会超时，而
>    **Docker Desktop 默认不走 Windows 系统代理** —— 需要在
>    Settings → Resources → Proxies 里单独填（如 `http://127.0.0.1:7890`）。
>    改完必须**重启 Docker Desktop** 才生效。

> **诚实边界**：镜像里只有模拟数据（合成画像，「精简版」），真实 Steam 数据相关的
> `dim_game` / `user_game` 两张表为空（它们只服务「游戏库 / 玩家画像」类分析，
> 12 个运营指标一个都不依赖）。精简版与完整版的**指标口径与算法完全相同，但数值不同**，
> 既往评测报告的数字都是在完整版上测的，别拿精简版的数值去对报告。
> 需要完整数据时，把两个 CSV 放进 `data/raw/` 后在本机直接跑脚本即可。

> **验证边界**：镜像已在 Windows + WSL2 后端真机验证通过（2026-09-19）：
> `docker build` 成功，镜像 990MB；容器 `healthy`；宿主访问 `/_stcore/health`
> 返回 `ok`、首页 200 / 7,260 字节；容器内时区为 CST、与宿主一致；
> **容器内跑 `python -m pytest` → 743 passed**（干净 Linux + 精简版数据）。
> 已知的环境前提：本机需能访问 Docker Hub（国内需给 Docker Desktop 配代理）。
> 详见《测试复盘记录》12.3。

### 8. 持续集成（CI · 每次推送自动验）

[.github/workflows/ci.yml](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/.github/workflows/ci.yml)
在干净的 Ubuntu 上依次跑三步：

1. **无数据建库链路** —— 仓库里没有 `data/`（已被 `.gitignore` 排除），
   所以这一步是在**完全空数据**的环境里跑 `generate_mock_data.py` + `build_database.py`。
   这正是「别人 clone 下来」的真实处境。
2. **断言建库产物** —— 9 张表齐全、`dim_user` / `game_event_log` /
   `user_daily_snapshot` 非空。（`dim_game` / `user_game` 在精简版里**本该为空**，故不断言。）
3. **743 条单元测试**。

> **为什么顺序是「先建库、再测试」？** 有一批测试带 `skipif`：数据库不存在时
> 它们会**跳过**而不是失败。若顺序反了，在「建库失败」的场景下 CI 反而一片绿 ——
> 跳过是不报错的。这个顺序堵住了「假绿」。
>
> **为什么 CI 里不跑 `docker build`？** 因为上面三步已覆盖核心风险（别人 clone
> 下来能不能跑起来），且一两分钟跑完；`docker build` 要拉几百 MB 依赖，
> 每次提交都多花几分钟，而镜像正确性已用真机验证过 —— 收益不抵成本。

**依赖已钉死版本。** `requirements.txt` 用 `==` 而非 `>=`：
写 `>=` 时 pip 装的是「当天最新的兼容版」，今天和下周可能装出两套版本 ——
别人复现不出报告里的读数，CI 也可能因为上游发新版而突然变红，而红的原因和你的改动无关。
现在这些版本号来自 2026-09-19 在 `python:3.11-slim` 里的全新安装解析结果，
并已在**该环境下跑通全部 743 条单测** —— 是一组被验证过的组合，不是猜的。
升级方式：显式改版本号，然后重跑测试与 CI。

---

## 六、指标清单（12 个）

| 分类 | 指标 | 单位 | 关键口径提醒 |
| --- | --- | --- | --- |
| 活跃 | `dau` 日活跃用户数 | 人 | 唯一口径 `user_daily_snapshot.is_active = 1` |
| 活跃 | `mau` 月活跃用户数 | 人 | **滚动 30 天**，不是全窗口去重 |
| 活跃 | `stickiness` 用户粘性 | % | DAU/MAU，行业参考区间 10%~25% |
| 留存 | `retention_rate` 留存率 | % | 分母必须剔除「观察期不足 N 天」的用户 |
| 留存 | `tutorial_funnel` 新手引导漏斗 | % | 逐级转化率定位瓶颈步骤 |
| 增长 | `new_user_count` 新增用户数 | 人 | 以 `dim_user.register_date` 为准 |
| 商业化 | `payment_rate` 付费转化率 | % | 分母是活跃用户，不是全部注册用户 |
| 商业化 | `arpu` 每活跃用户平均收入 | 元 | 分母是全体活跃用户 |
| 商业化 | `arppu` 付费用户人均付费 | 元 | 分母只是付费用户，别和 ARPU 混 |
| 商业化 | `ltv` 用户生命周期价值 | 元 | 简化 LTV_N 口径，与 CAC 配对看 ROI |
| 渠道 | `channel_retention` 渠道留存对比 | % | 固定时间区间与留存天数，只让渠道变化 |
| 版本 | `version_retention` 版本前后留存对比 | % | 三档批次，必须同时看 `cohort_size` |

完整定义（业务口径、参数、来源表、SQL 模板、易错点）见
[metrics_registry.json](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/src/metrics/metrics_registry.json)。

---

## 七、评测体系

**核心原则：真值走完全不经过 LLM 的独立路径。**
如果拿 Agent 自己查出来的数据当标准答案，评测就退化成了「自己给自己打分」。

**评测集只声明期望指标 + 期望参数，不写期望数值。**
真值由评测器直接调 SQL 层现算，避免数据重新生成后用例集体失效。

**数字分四档，而不是「对上/对不上」：**

| 档位 | 判定 | 处理 |
| --- | --- | --- |
| 严格 | 与真值误差 ≤ 0.5% | 直接通过 |
| 近似 | 误差 ≤ 3%（口语化的「大概 530」） | 通过 |
| 可推导 | 可由真值经四则运算得到（如平均值、差值） | 通过 |
| 存疑 | 无法追溯来源 | **只有这一档需要人工看** |

Phase 6 基线（31 条用例 v1.1，真实调用大模型）：

| 指标 | 结果 |
| --- | --- |
| 指标映射准确率 | 100%（27/27） |
| 参数抽取准确率 | 100%（15/15） |
| 答案数字可追溯率 | 96.73%（296/306） |
| 超范围拒答率 | 100%（4/4） |
| 平均成本 | 7,125.9 tokens / 3,372 ms / 2.74 轮 |
| 缓存命中率 | 90.69%（190,332 / 输入 209,878） |

> 用例集从 Phase 5 的 18 条扩到 31 条（补长尾问法、多轮追问、边界），
> 所以「平均成本」不能和 Phase 5 的 7,160.8 直接比 —— 分母变了。
> 同一组 18 条上，prompt 精简带来的净收益是 7,160.8 → 6,796.4（-5.1%）。

**评测分三层，成本与目的都不同 —— 不是重复测：**

| 层 | 入口 | 规模 | 成本 | 回答什么问题 |
| --- | --- | --- | --- | --- |
| 单元测试 | `python -m pytest` | 400+ 个函数 / 743 条 | 零成本 | 改代码有没有改坏 |
| 端到端评测 | `scripts/run_eval.py` | 31 条用例 | 花 token | 答得准不准、贵不贵 |
| 专项评测 | `scripts/run_*_eval.py` | 5 个 Phase | 花 token | 单点深挖（见下表） |

五个专项评测的终版读数（全部合格）：

| Phase | 测什么 | 关键读数 |
| --- | --- | --- |
| 1 · 工具调用 | 工具选对没、参数抽对没 | 选择/抽取/完整性均 100%，F1 = 1.0 |
| 2 · 性能与缓存 | 快不快、贵不贵 | 缓存命中率 90.87%（稳态口径，排除冷启动） |
| 3 · 幻觉与质量 | 编没编数字、答得全不全 | 幻觉率 1.77%（5/283 句），任务完成率 100% |
| 4 · 鲁棒性与边界 | 不该答的时候乱不乱答 | 21 条行为断言 100% 通过，5 类场景全过 |
| 5 · 优化验证 | 提示词改进真的有效吗 | 无依据声明率降至 0.71%，日历/周末归因、行业基准两类问题归零 |

> 跨轮次的「率值大小」本身不可直接比（分母不同 + 答案每次重新生成），
> Phase 3 就是靠这条纠偏才没误判的。所以 Phase 5 的结论支撑是**类别归零 +
> 逐句行为证据 + 多轮基线对照**，而不是「2.83% 降到 0.71%」这一个数。

> **别把 Phase 编号混了** —— 项目里有三套编号：
> ① **开发阶段**（`src/data` Phase 1 → `metrics`/`sqlgen` Phase 2 → `agent` Phase 3 → `ui` Phase 4）；
> ② **测试专项** Phase 1~5（按测试维度分，即上表）；
> ③ **端到端评测轮次**（Phase 5 的 18 条 → Phase 6 的 31 条，按跑测轮次分）。
>
> 还有一个反直觉的结论：五轮专项里**评测台自身出错 5 次，被测对象只有 1 类真实缺陷**。
> 评测台比被测对象更容易出错，因为它没有人在盯着。
> 详见[测试复盘记录](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/docs/02_测试复盘/测试复盘记录.txt)。

---

## 八、技术亮点（速览）

1. **用架构消除幻觉** —— LLM 只做语义映射，SQL 由受控模板生成，从根上杜绝「模型算错数」。
2. **五层安全防线** —— 模板受控 / 语法校验 / 只读连接 / 参数绑定 / 答案来源校验，
   前四层管 SQL 出口，第五层管答案出口（逐数字溯源，拦「没查库就编数字」
   与「查了但混着编」），每层都有单测覆盖。
3. **真正的超时中断** —— 用 SQLite `progress_handler` 实现，而不是设一个不生效的 `timeout`。
4. **多供应商降级** —— 按错误类型区分重试策略（401 不重试直接切，429/5xx 指数退避），
   基于 OpenAI 兼容协议做抽象，新增供应商只改配置、不改代码。
   降级后前端与导出件都会给出提示（备用模型慢 4~5 倍、指令遵循更弱），
   提示的判据取自配置的主备顺序，不写死供应商名。
5. **手搓 ReAct 循环** —— 不依赖 LangChain，6 轮熔断防无限调用，过程完整记录可回溯。
6. **可评测性内建** —— 双路径对账、四档数字判定、真值不经过 LLM，把「好像变好了」变成数字。
7. **评测台自己也被审** —— 五轮专项里评测台自身出错 5 次（真值不同源、去重维度错、
   Judge 判据过严、reason 与 grounded 自相矛盾），而被测对象只有 1 类真实缺陷。
   每修一次就补一条单测防回归。**评测系统的可信度也是要挣的** ——
   否则会拿着错的尺子去"优化"一个本来没问题的模型。
8. **人工兜底环节的产品化** —— 重大决策类结论强制走复核 + 审计日志，而不是一句免责声明。

---

## 九、演示脚本（5 分钟）

按这个顺序演示，每一问都对应一个可讲的设计点，不要一次问完。

| # | 提问 | 演示时说的话（要点） |
| --- | --- | --- |
| 1 | `最近的次日留存率是多少？` | 展开「执行过程」：**模型只传了 `metric_id` 和 `day_n`，SQL 是系统生成的** —— 这是第 1 条硬约束的直观证据 |
| 2 | `v2.0.0 上线后新用户留存变好了吗？` | 答案里有三档批次和各自的 `cohort_size`。讲「相关不等于因果」，模型被要求在结论里提示节假日、渠道结构等混淆因素 |
| 3 | `各渠道的次日留存对比如何？` | 图表自动出图（不用手选），再展开原始表格。讲「对比类结论必须同时报样本量」 |
| 4 | `去年 12 月的日活是多少？` | 明确拒答并说明数据边界。**拒答能力比回答能力更值得讲** —— 一个不承认边界的分析助手是不敢用的 |
| 5 | `帮我把买量预算翻倍，投到抖音渠道` | 触发人工确认节点：填复核人 + 勾选 → 写入左侧审计日志。讲「决策责任留在人这一侧」 |

**演示前检查**：`.env` 已配 Key、`data/game_analytics.db` 存在、`streamlit run app.py` 能起。

---

## 十、文档

| 文档 | 说明 |
| --- | --- |
| [开发复盘记录](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/docs/01_项目文档/开发复盘记录_Phase1-3.txt) | 全流程复盘：每个设计决策的取舍、踩过的坑、常见追问的答法 |
| [测试复盘记录](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/docs/02_测试复盘/测试复盘记录.txt) | 测试体系全档案：三层的分工、幻觉两个口径、评测台自身五次出错、常见追问 |
| [文件清单说明](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/docs/01_项目文档/文件清单说明.txt) | 逐文件说明：117 个文件各是干什么的、在架构哪个位置、设计要点是什么 |
| [数据字典](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/docs/01_项目文档/data_dictionary.md) | 9 张表的字段说明与业务含义 |
| [评测报告](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/docs/03_评测报告/评测报告_Phase6.txt) | 端到端 31 条用例的逐例明细与总览 |
| [专项报告](file:///e:/TraeCode/Work/JAVAWork/Game%20Data%20Analyst%20Agent/docs/03_评测报告/工具调用测试报告_Phase1.txt) | 5 个专项的原始留档（工具 / 性能缓存 / 幻觉质量 / 鲁棒性边界） |

> `docs/` 下的四层子目录各有分工：**01_项目文档** 讲「项目是什么、怎么做出来的」，
> **02_测试复盘** 讲「踩过哪些坑、为什么这么设计」，**03_评测报告** 是给人读的结论（txt），
> **04_评测原始数据** 是给程序/Excel 复核用的证据（csv/json）。
> 完整目录树见第四节。

---

## 十一、技术栈

Python 3.11 · SQLite · Streamlit · Plotly · OpenAI SDK（DeepSeek / 智谱 GLM）· pytest · Docker · GitHub Actions

---

## 十二、开发与提交规范

> 既是本仓库的协作约定，也是日常操作速查卡。

### 1. 日常四步

| 步骤 | 命令 | 作用 |
| --- | --- | --- |
| ① 看 | `git status` | 哪些文件改了、哪些已进暂存区 |
| ② 搬 | `git add .` | 把改动放进暂存区（**不产生提交**） |
| ③ 存 | `git commit -F "信息文件路径"` | 生成一条**本地**提交（GitHub 还看不到） |
| ④ 推 | `git push` | 上传到远程仓库（GitHub 才看得到） |

### 2. 状态字母速记

`git status --short` 的**第一列 = 已暂存**，**第二列 = 只改了没 add**：

| 字母 | 含义 |
| --- | --- |
| `A` | 新增文件（第一次入库） |
| `M` | 修改已有文件 |
| `D` | 删除文件 |

`git add .` 会同时处理**新增 / 修改 / 删除**三种改动 —— 删除也是一次「改动」，
不 add 不 commit 的话，仓库里那份文件仍然存在（因此也还能用 `git restore` 救回来）。

### 3. 提交信息规范

- 标题一行说清**为什么改**，空一行后可补详细说明；一次提交只做一件事，便于回退与 Review
- **中文提交信息必须走文件**：`git commit -F "信息文件路径"`
  直接用 `git commit -m "中文"` 在 PowerShell 下会乱码（字符串按控制台代码页传递）。
  写信息文件时**另存为 UTF-8 编码**
- 提交信息用中文，代码注释也用中文 —— 与全仓库口径一致

### 4. 提交前自检

- [ ] `git status` 里没有 `.env`、没有 `data/` 下的大文件（密钥与 107MB 数据库都不入库）
- [ ] `python -m pytest` 全绿（当前 743 条）
- [ ] 新增 / 删除文件后，`docs/01_项目文档/文件清单说明.txt` 的条目与统计已同步更新
- [ ] README 中提到的每个路径都真实存在（含 `docs/` 四层子目录）

### 5. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `warning: CRLF will be replaced by LF` | **正常警告，不是错误**。`.gitattributes` 要求仓库内统一 LF，Windows 工作区的 CRLF 会在下次 Git 重写文件时自动转换 |
| `git push` 卡住或报 `Connection refused` | 代理未生效。git 不读 Windows 系统代理，需为仓库单独配置；恢复直连执行 `git config --local --unset http.proxy`（`https.proxy` 同理） |
| 改错了想还原 | `git restore <文件名>`；若已 `add`，先 `git restore --staged <文件名>` 再 restore 文件 |
| 提交信息显示乱码 | 改用 `-F` 读 UTF-8 文件，不要用 `-m` 直传中文 |
| 文件明明写进 `.gitignore` 还是被提交 | `.gitignore` **只对未跟踪文件生效**；已被跟踪的文件需先 `git rm --cached <文件>` 移出跟踪 |
