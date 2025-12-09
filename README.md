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

### 最新更新 (v1.2.0)

**交互体验与报告生成引擎升级 (`src/lightweight_analysis.py`)**

- **页面导航体系重构**：
    - **新增快捷导航面板**：在报告顶部集成页面导航（Navigation），支持按“分析板块”（策略收益、模型性能、交易执行、滑点成本等）和“核心图表”进行快速锚点跳转。
    - **平滑滚动**：内置 JavaScript 实现页面内锚点的平滑滚动体验，优化长报表阅读效率。

- **可视化组件与前端增强**：
    - **动态徽章系统**：引入 JavaScript 动态标记逻辑，自动识别并为图表卡片添加 `NEW`（如择时能力分析）和 `UPGRADE`（如日收益率、Sharpe 净值）视觉标签，高亮更新重点。
    - **公式渲染修复**：修正“口径说明”板块中 PnL 及资金利用率的 LaTeX 公式渲染方式，采用 `<span class="math-inline">` 容器，解决部分浏览器环境下的显示异常。
    - **标准属性完善**：为所有嵌入的子图表 iframe 增加 `title` 属性，提升页面无障碍性（Accessibility）。

- **系统稳定性与流程控制**：
    - **模块级容错**：为 `factor_exposure_analysis`（因子特征暴露）计算模块增加独立异常捕获（try-except），防止因单项非核心分析失败导致整体报告生成中断。

### 近期变更历史

- **新增择时能力分析**：基于 Baostock 5分钟线数据计算 Entry/Exit Rank 与 Edge，支持缓存与多权重视角，输出独立 HTML/TXT 诊断报告。
- **收益分析深化**：
    - **策略 vs 基准**：优化累积收益对比页面的布局与交互。
    - **资金效率**：PnL 计算逻辑修正，分母纳入空头头寸，更准确反映实际资金占用。
    - **日收益分布**：更新日绝对收益分布图表的设计与统计文案。
- **文档与工程化**：
    - 建立完整的可视化页面设计准则与 LaTeX 渲染修复记录。
    - 脚本支持 `--recompute` 参数与缓存复用，优化单核/多核运行逻辑。
