# Analysis_orders

对交易数据集进行诊断分析，评估高频/量化交易模型的预测准确性与执行到位情况。本仓库仅保留核心脚本 `lightweight_analysis.py`，用于快速生成可部署的可视化报告。

## 功能概览

- 读取 `data/orders.parquet` 交易明细与盯市净值数据，构建多维度执行分析
- 自动计算等权、金额加权、PnL(花出的钱)等收益指标，支持基准指数对比
- 生成 Plotly 交互式图表并汇总成 `index.html` 仪表板，可直接用于 GitHub Pages

## 环境准备

```bash
pip install -r requirements.txt
```

请确保仓库根目录存在下列数据／缓存文件：

- `data/orders.parquet`
- `data/paired_trades_fifo.parquet`（若需要配对交易分析）
- `mtm_analysis_results/daily_nav_revised.csv`（盯市净值）
- 可选缓存：`reports/first_day_capital_snapshot.json`、`data/daily_close_cache.parquet` 等

## 使用方法

```bash
python src/lightweight_analysis.py
```

脚本默认把输出写入 `reports/visualization_analysis/`。若需要部署到 GitHub Pages，可在运行后将该目录复制到 `docs/`，或将脚本中的 `self.reports_dir` 修改为仓库内的 `docs/`。

GitHub Pages 配置建议：在仓库 Settings → Pages 中选择 `main` 分支，文件夹选 `/docs`。

## 许可证

沿用原始项目的授权条款。如需变更，请根据实际需求更新。

## 更新日志

- 相对 9ee7b319d4563e3ceb71b77fbd6bafa5a9e1f2ce：新增择时能力分析页面（基于 baostock 5min 行情计算 Entry/Exit Rank 与 Edge，支持缓存与多权重视角，输出 HTML/TXT 报告）。
- 完善文档体系：补充各页面指导文档与可视化页面设计准则，记录 MathJax 渲染与卡片模板经验，便于后续维护。
- 首页导航与关联页强化：优化 index 页跳转与页内导航，新增相关页面入口，提升“交易执行分析”板块的主页面体验。
- 收益展示改进：策略 vs 基准指数累积收益对比、日绝对收益分布等页面布局与文案更新，保持轻量化输出。
- 数据与性能修正：PnL 计算分母包含空头，脚本支持 `--recompute` 参数与缓存复用；当前环境受限默认单核运行，如需并行可按权限配置 `--workers`。
