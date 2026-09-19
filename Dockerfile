# ============================================================================
# 游戏数据智能分析师 Agent —— 镜像定义
# ============================================================================
#
# 【这个 Dockerfile 要解决什么问题？】
#
#   在此之前，「拿到这个项目的人」要跑起来必须做四件事：
#     ① 装 Python 3.11+ 并建虚拟环境  ② pip install -r requirements.txt
#     ③ 下载 111MB 真实 Steam 数据放进 data/raw/  ④ 手动跑两个脚本生成数据、建库
#   任何一步出错都跑不起来 —— 对一个要拿给别人看的产品，这是最致命的一环。
#
#   本镜像把 ①~④ 全部固化进构建过程：build 完就能 run，不需要使用者碰数据。
#
# 【为什么镜像里不含那 111MB 真实 Steam 数据？】
#
#   那是外部原始数据（版权归属 Steam），本就被 .gitignore 排除在仓库之外，
#   同时 .dockerignore 也排除了本机的 data/ —— 目的是让**镜像构建结果只取决于
#   仓库里的代码**，而不是「构建者本机恰好有什么」。任何人 clone 后 build
#   得到的都是同一个镜像，这才叫可复现。
#   代价是镜像里只有模拟数据，真实数据相关的两张表（dim_game / user_game）为空；
#   这两张表只服务「游戏库 / 玩家画像」类分析，12 个运营指标一个都不依赖它们
#   （详见 scripts/build_database.py 里跳过分支的注释）。
#   需要完整数据时，把两个 CSV 放进 data/raw/ 并在本机直接跑脚本即可。
#
# 【为什么在 build 阶段就建库，而不是容器启动时再建？】
#   建库要跑一遍数据生成（约十几秒到一分钟）。放在 build 阶段，`docker run`
#   是秒起的；放在启动时，每次 run 都要等，且并发启动会互相踩数据库文件。
#   代价是数据被固化在镜像里 —— 想改数据窗口（config.py 的 DATA_END）需要重建镜像。
#   对「演示 + 评审」这个用途来说，启动速度比数据可调更重要。
#
# 【怎么用？】
#   构建：docker build -t gda .
#   运行：docker run --rm -p 8501:8501 -e DEEPSEEK_API_KEY=sk-xxx gda
#   国内构建卡住时：pip 拉包换镜像站 + Docker Desktop 配代理，见 README
#   「国内代理构建」小节（两条链路要分别解决，只解决一条仍会卡）。
#   不带 Key 也能启动 —— 前端会显示「尚未配置 API Key」的引导文案（见 react_agent.py
#   的 _NOT_CONFIGURED_HINT），界面照样能打开，方便先看产品形态。
#
# ★ 安全：镜像里**不含任何密钥**。.env 被 .dockerignore 排除，Key 只能通过
#   运行时环境变量注入（config.py 用 load_dotenv(override=False)，真实环境变量优先）。
# ============================================================================

# 与本地开发环境对齐的 Python 版本（本地实测 3.11.5）。
# 用 slim 变体而非完整镜像：pandas / numpy / plotly 都有预编译 wheel，
# 不需要编译工具链，省下的几百 MB 对演示镜像没有意义。
FROM python:3.11-slim

# PYTHONUNBUFFERED：让日志实时输出，否则 docker logs 要等缓冲满了才看得到 ——
#   排查「容器起来了但界面打不开」这类问题时，实时日志是唯一的线索。
# PYTHONDONTWRITEBYTECODE：容器内不需要 .pyc，少写一层垃圾。
# TZ=Asia/Shanghai：让 app.py 里 datetime.now() / date.today()（导出时间、审计
#   日志、created_at 缓存键）按北京时间计算，而不是 UTC —— 差 8 小时会在
#   演示时把「导出报告的时间戳」显示成后一天。
#   ★ 必须有下面那个 apt-get install tzdata：python:3.11-slim 基于 Debian slim，
#     glibc 默认没有 /usr/share/zoneinfo，只设 TZ 会被忽略、静默退回 UTC。
#     tzdata 是纯数据包（约 5MB），装完 TZ 才真正生效。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 装 tzdata（TZ 要生效的前提，见上方 ENV 注释）。
# 单独一层且排在依赖层之前：tzdata 的安装与 requirements.txt 无关，
# 放前面可以让「改依赖重装」时不必连带重跑 apt。
# DEBIAN_FRONTEND=noninteractive：tzdata 的安装脚本会问时区，构建环境没有
#   终端，不显式声明非交互时可能卡在等输入上 —— 一行代价换掉一类构建卡死。
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# 第一层：只装依赖
# 【为什么先 COPY requirements.txt 再 COPY 代码？】
#   Docker 按层缓存：只要 requirements.txt 没变，改代码后重新 build 会直接命中
#   这一层，跳过几分钟的 pip install。如果把 COPY . . 放在前面，任何一次改代码
#   都会让缓存失效、重装全部依赖 —— 这是写 Dockerfile 最常见的一个性能坑。
# ---------------------------------------------------------------------------

# pip 安装源，可用 --build-arg 覆盖（国内加速用，见 README「国内代理构建」小节）。
# 默认仍是官方 PyPI，不改变默认的构建行为；PIP_INDEX_URL 只是给了
# 「pip 拉包走镜像站」的退路。**换源只换拉包走哪，装的是同一组钉死 == 的版本，
# 不影响「重装出同一个环境」这一承诺。**
ARG PIP_INDEX_URL=https://pypi.org/simple

COPY requirements.txt ./
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt

# ---------------------------------------------------------------------------
# 第二层：拷代码并建库
# data/ 与 .env 已在 .dockerignore 中排除，所以这里拷进来的是纯净的源码。
# ---------------------------------------------------------------------------
COPY . .

# 建库 = 生成模拟数据（固定随机种子，结果可复现）→ 建表导数建索引 → 验收。
# 【为什么要 && 串起来？】任一步失败就让 build 失败。如果分开写，
#   第一步失败时镜像仍会构建成功，得到一个「能启动但查不到数据」的镜像 ——
#   问题会被推迟到运行时才暴露，更难定位。构建期失败远好过运行期失败。
RUN python scripts/generate_mock_data.py && \
    python scripts/build_database.py

# Streamlit 默认监听 8501
EXPOSE 8501

# 健康检查：容器里没有 curl，用 Python 标准库发一个 HTTP 请求即可。
# 【为什么需要它？】Streamlit 进程活着不代表界面可用（比如建库失败导致启动即崩）。
#   有了 healthcheck，`docker ps` 能直接看出容器是 healthy 还是 unhealthy。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

# --server.address=0.0.0.0：容器内必须监听所有网卡，否则 -p 映射进来的请求到不了。
#   这是「本地能跑、容器里打不开」最常见的原因。
# --server.headless=true：跳过首次运行要求填邮箱的交互，容器里没法交互。
# --browser.gatherUsageStats=false：关掉遥测，避免容器无谓外联。
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
