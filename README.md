# Analysis_orders

对交易数据集进行诊断分析，评估高频/量化交易模型的预测准确性与执行到位情况。本仓库仅保留核心脚本 `lightweight_analysis.py`，用于快速生成可部署的轻量级可视化报告。

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

100.115.26.69