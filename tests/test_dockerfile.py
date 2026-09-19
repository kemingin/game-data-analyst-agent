# -*- coding: utf-8 -*-
"""
Dockerfile 与 README 的一致性单元测试
================================================================================
【为什么专门测这一块？】

  Dockerfile 在本项目里有一个特殊性：**它是唯一一个「改动后没有任何自动化
  验证会碰到」的交付物**。
    · 本机不一定装 Docker（开发机就没有 docker CLI）；
    · CI 里也刻意不跑 `docker build`（见 .github/workflows/ci.yml 文件头注释：
      拉几百 MB 依赖、每次多花几分钟，收益不抵成本）。

  所以它的正确性只能靠两类东西守：
    ① 构建期的 `&&` 串联 —— 守的是「构建过程本身」；
       （`RUN a.py && b.py`，任一步失败就让 build 失败，见 Dockerfile 建库那一层）
    ② 本文件 —— 守的是「**README 教别人敲的命令，Dockerfile 真的支持**」。

  ② 为什么值得单独守？因为 README 是外部读者唯一的入口，而它描述的是
  Dockerfile 的能力。一旦 Dockerfile 删掉某个 ARG、改了镜像名或脚本路径，
  README 会**静默变成假话**：命令照抄下来会失败，但仓库里没有任何红灯 ——
  连 CI 都是绿的（CI 不构建镜像）。这类「文档与实现漂移」正是本文件要拦的。

【本文件的口径】
  只做静态文本断言：不调用 docker、不联网、不读 data/ —— 因此任何机器上都能跑，
  也就能进 CI。

【已知限制 / 将来改法】
  · 只覆盖「文本层面的一致性」，不覆盖「命令真的能跑通」——
    后者需要 Docker 引擎，属于 Phase 10 真机验证的范畴（读数见《测试复盘记录》12.3）。
  · 正则按当前 README / Dockerfile 的写法写。若将来 README 改用
    `docker buildx build`、或把 tag 写进变量，需要同步放宽下面的正则。
  · 新增 `--build-arg` 用法时，若该 ARG 是 Docker 引擎预定义的（代理类），
    需加进 DOCKER_PREDEFINED_BUILD_ARGS；否则要在 Dockerfile 里补 ARG 声明。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"
README = ROOT / "README.md"

# Docker 引擎**预定义**的构建参数：不写 `ARG` 声明也能生效，所以 README 里
# 出现它们不算「文档跑在实现前面」。
# 依据：Docker 官方文档 —— 一组预定义的 ARG 变量（代理类），可在 Dockerfile
#      中直接使用而无需 ARG 指令，且不会出现在 `docker history` 里。
DOCKER_PREDEFINED_BUILD_ARGS = {
    "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "NO_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "ftp_proxy", "no_proxy", "all_proxy",
}


# ---------------------------------------------------------------------------
# 一、判据（纯函数：只吃文本，便于做反面证明）
# ---------------------------------------------------------------------------

def readme_build_args(readme_text: str) -> set[str]:
    """从 README 里抽出所有 `--build-arg KEY=...` 的 KEY。"""
    return set(re.findall(r"--build-arg\s+([A-Za-z_][A-Za-z0-9_]*)=", readme_text))


def dockerfile_args(dockerfile_text: str) -> dict[str, str | None]:
    """从 Dockerfile 里抽出所有 `ARG NAME[=默认值]`，返回 {名称: 默认值}。"""
    found = re.findall(
        r"^[ \t]*ARG[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*(?:=[ \t]*(\S+))?[ \t]*$",
        dockerfile_text,
        re.MULTILINE,
    )
    return {name: default or None for name, default in found}


def missing_build_args(readme_text: str, dockerfile_text: str) -> set[str]:
    """README 教了、但 Dockerfile 并不支持的 build-arg。

    「不支持」= 既不是 Docker 预定义的代理变量，也没有对应的 ARG 声明。
    这种情况下读者照抄命令，`--build-arg` 会被 Docker **静默忽略** ——
    不报错，只是不生效，然后卡在拉包超时上，且无从判断原因。
    """
    declared = set(dockerfile_args(dockerfile_text))
    return readme_build_args(readme_text) - declared - DOCKER_PREDEFINED_BUILD_ARGS


# ---------------------------------------------------------------------------
# 二、README 里的 --build-arg 必须真的被 Dockerfile 支持
# ---------------------------------------------------------------------------

def test_readme_mentions_at_least_one_build_arg():
    """前置断言：README 里确实有 --build-arg 用法。

    没有这一条，下面那条测试会在「README 把用法删光了」时**空集合对空集合**
    通过 —— 那是假绿。判据必须先把「有没有东西可比」钉住。
    """
    args = readme_build_args(README.read_text(encoding="utf-8"))
    assert args, "README 里一个 --build-arg 都没有了？要么补回来，要么删掉本条测试"


def test_every_readme_build_arg_is_supported_by_dockerfile():
    """★ 本文件最重要的一条：README 教的每个 build-arg，Dockerfile 都要认。

    场景还原：README 让国内用户敲
        docker build --build-arg PIP_INDEX_URL=<镜像站> -t gda .
    如果 Dockerfile 里没有 `ARG PIP_INDEX_URL`，Docker 会**静默忽略**这个参数，
    pip 照样去连 pypi.org 然后超时 —— 用户看到的是网络问题，
    而真正的原因在文档与实现的漂移上。这条测试就是拦这个。
    """
    readme_text = README.read_text(encoding="utf-8")
    dockerfile_text = DOCKERFILE.read_text(encoding="utf-8")

    missing = missing_build_args(readme_text, dockerfile_text)
    assert not missing, (
        f"README 教了但 Dockerfile 不支持的 build-arg：{sorted(missing)}。"
        "要么在 Dockerfile 里补 `ARG <名字>=<默认值>`，"
        "要么把 README 里对应的用法删掉。"
    )


def test_build_arg_check_catches_a_violation():
    """反面证明：给判据喂一个 Dockerfile 里不存在的 ARG，必须报出来。

    不做这条，上面那条测试可能是「永远为真」的 ——
    正则写错（比如漏了 `--build-arg` 的空格）、或判据恒返回空集，
    表现和「一切正常」完全一样。
    """
    fake_readme = "docker build --build-arg NOT_IN_DOCKERFILE=1 -t gda ."
    fake_dockerfile = "FROM python:3.11-slim\nARG PIP_INDEX_URL=https://pypi.org/simple\n"

    assert missing_build_args(fake_readme, fake_dockerfile) == {"NOT_IN_DOCKERFILE"}
    # 反面证明要证明的是「判据能报警」，而不是「判据永远报警」——
    # 所以同一份假 Dockerfile 配上合法的 ARG，必须不报。
    assert missing_build_args("--build-arg PIP_INDEX_URL=x", fake_dockerfile) == set()
    # 预定义的代理变量不需要 ARG 声明，也不该被报出来
    assert missing_build_args("--build-arg HTTPS_PROXY=x", fake_dockerfile) == set()


# ---------------------------------------------------------------------------
# 三、ARG 必须有默认值（「不传就与之前行为一致」这句承诺的前提）
# ---------------------------------------------------------------------------

def test_every_arg_has_a_default_value():
    """Dockerfile 里每个 ARG 都要带默认值。

    为什么值得测：本项目对 PIP_INDEX_URL 的承诺是「**不传这个参数时，构建行为
    与之前完全一致**」。如果写成光秃秃的 `ARG PIP_INDEX_URL`，不传时它是空字符串，
    那一行会变成 `pip install -i "" -r requirements.txt` —— 承诺当场失效，
    而本机没 Docker 的话根本发现不了。
    """
    declared = dockerfile_args(DOCKERFILE.read_text(encoding="utf-8"))

    assert declared, "Dockerfile 里一个 ARG 都没解析到，正则可能失效了"
    no_default = sorted(name for name, default in declared.items() if default is None)
    assert not no_default, f"这些 ARG 没有默认值：{no_default}（不传时会变成空字符串）"

    # 顺带钉住那个被 README 引用的具体值：默认必须仍是官方源，
    # 否则「默认行为不变」这句承诺就悄悄换了内容。
    assert declared.get("PIP_INDEX_URL") == "https://pypi.org/simple"


# ---------------------------------------------------------------------------
# 四、镜像名一致（README 复制粘贴就能跑的前提）
# ---------------------------------------------------------------------------

def test_image_tag_is_consistent():
    """README 里 `docker build -t X` 与 `docker run ... X` 必须是同一个 X。

    这不是吹毛求疵：README 的 Docker 小节是**复制粘贴**用的，
    build 出来的 tag 与 run 用的 tag 对不上，读者会得到
    「Unable to find image 'xxx' locally」—— 而两条命令分开看都是对的。
    """
    readme_text = README.read_text(encoding="utf-8")

    build_tags = set(re.findall(r"docker build\b[^\n]*?-t\s+(\S+)", readme_text))
    # run 行的镜像名是行尾最后一个 token（前面都是 --flag 与 flag 的值）
    run_tags = set(
        re.findall(r"docker run\b[^\n]*?[ \t]([A-Za-z0-9][A-Za-z0-9._/-]*)[ \t]*$",
                   readme_text, re.MULTILINE)
    )

    assert build_tags, "README 里没解析到 docker build -t 的镜像名，正则可能失效了"
    assert run_tags, "README 里没解析到 docker run 的镜像名，正则可能失效了"
    assert build_tags == run_tags, (
        f"README 里 build 的 tag {sorted(build_tags)} 与 run 用的 {sorted(run_tags)} 不一致"
    )

    # Dockerfile 的【怎么用？】注释里写的是同一套命令，tag 也得跟上 ——
    # 否则改 tag 时改了 README、忘了注释，下次有人照着 Dockerfile 敲就错。
    dockerfile_text = DOCKERFILE.read_text(encoding="utf-8")
    for tag in build_tags:
        assert f"-t {tag} " in dockerfile_text, (
            f"Dockerfile 的注释里没有 `-t {tag}` —— README 改了 tag，注释没跟上"
        )


# ---------------------------------------------------------------------------
# 五、build 期跑的脚本路径必须真实存在
# ---------------------------------------------------------------------------

def test_scripts_run_at_build_time_exist():
    """Dockerfile 在 build 期用 `RUN python <路径>` 跑的脚本，必须真实存在。

    路径写错时只有真正 build 才报错，而本机与 CI 都不会 build ——
    等到别人 clone 下来才发现，是最晚也最贵的暴露时机。
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    referenced = re.findall(r"python\s+([A-Za-z0-9_./-]+\.py)", text)

    assert referenced, "Dockerfile 里没解析到任何 python 脚本调用，正则可能失效了"
    missing = [path for path in referenced if not (ROOT / path).exists()]
    assert not missing, f"Dockerfile 引用了不存在的脚本：{missing}"
