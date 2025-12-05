# Repository Guidelines

## 环境信息与沟通约定
当前仓库在 Linux (Ubuntu 22.04.2 LTS, 内核 6.8.0-85-generic, x86_64 架构) 上维护；若迁移至其他平台，请注意字体和 Plotly 渲染差异。跨团队沟通、提交说明与代码注释均应优先使用中文，必要时补充精确英文翻译。

## 项目结构与模块组织
核心分析代码位于 `src/lightweight_analysis.py`，如需扩展请在 `src/` 下新增模块并由主类统一调度。输入数据放置在 `data/` 与 `benchmark_data/`，盯市与回测结果保存在 `mtm_analysis_results/`。脚本会把可视化与缓存写到 `reports/visualization_analysis/`，发布到 GitHub Pages 时将所需文件同步到 `docs/`。临时虚拟环境目录 `quant_env/` 与各类缓存请保持在 `.gitignore` 管控范围内。

## 构建、测试与开发命令
- `python -m venv .venv && source .venv/bin/activate`：如不复用 `quant_env/`，先创建隔离环境。
- `python -m pip install -r requirements.txt`：安装 pandas、plotly、statsmodels 等依赖。
- `python src/lightweight_analysis.py`：运行完整分析流程并刷新 `reports/visualization_analysis/index.html`。
- `python -m compileall src`：快速语法检查，确保提交前无明显错误。

## 编码风格与命名约定
采用 PEP 8 默认格式，统一使用 4 空格缩进；函数与变量使用 `snake_case`，类名采用 `PascalCase`，路径操作优先 `pathlib.Path`。新增公共接口请补充类型注解。终端输出请沿用 `_print_dup`，确保字符替换逻辑一致。文档、注释与沟通优先使用中文，如需英文请保证双语说明准确一致。

## 测试指南
当前未配置自动化测试，提交前需以真实或抽样的 parquet 数据运行脚本，确认生成页面可在浏览器中正常展示，关键指标（如收益曲线、Alpha、风险敞口）与历史版本一致。若新增辅助函数，建议在 `tests/` 下编写 `pytest` 用例，并把示例数据存放于 `data/fixtures/`（保持忽略），便于本地复现。

## 提交与 Pull Request 规范
承袭现有历史，提交信息保持简洁、聚焦且倾向中文动宾短语（示例：`添加IR与夏普比率曲线`）。单次提交聚焦一个逻辑变更，避免夹带无关文件。PR 描述需包含：变更摘要、数据依赖或新增缓存说明、关键图表或指标的前后对比截图以及相关需求或 issue 链接，明确是否需要热/冷部署步骤。请确认自动生成的报告文件已按需提交或列入忽略清单。

## 安全与配置提示
数据集中包含交易明细，请遵守内部数据脱敏规范，避免将真实账户信息写入日志。外部接口调用需在离线环境完成，严禁在脚本中硬编码凭证。若需新增配置项，请通过环境变量或本地 `.env` 方案，并在 README 中补充维护说明。

## 外置资源
### 接口文档
对外部行情或基础数据接口进行修改前或遇到异常时，必须先查阅官方文档确认正确调用方式。主要参考：
- Tushare：https://tushare.pro/document/2

### 文档抓取经验
- 先用 `curl -s https://tushare.pro/document/2` 抓取入口页 HTML，再配合 `grep -n "目标关键词"` 快速定位侧栏里的 `doc_id` 链接位置，例如 `grep -n "ETF专题"`。
- 根据定位到的 `doc_id`，继续 `curl -s https://tushare.pro/document/2?doc_id=<编号>` 获取子页面全文，随后用 `sed -n '起始,终止p'` 或 `grep -n "接口示例"` 精确截取参数说明与示例片段。
- 若需批量整理链接，可对入口页执行 `grep -o "doc_id=[0-9]*" | sort -u`，再逐个按需抓取并筛选；全流程仅依赖 shell 自带文本工具即可复用。
