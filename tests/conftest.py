# -*- coding: utf-8 -*-
"""pytest 全局配置。

【为什么需要这个文件？】
  pytest 默认只把「测试文件所在目录（tests/）」加入模块搜索路径，
  而我们的测试要 `import src.xxx`，src 在项目根目录下。
  所以在收集测试之前，先把项目根目录塞进 sys.path。

  这也是 Python 项目里最常见的「测试如何 import 源码」问题的标准解法之一，
  另一种做法是装成包（pip install -e .）后在 pyproject.toml 里配置，
  对课程/求职项目来说，这里几行代码更直观。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))