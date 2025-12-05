# IC 质量监控页面说明

## 1. 覆盖页面与目的
- `ic_distribution_light.html`
- `ic_timeseries_light.html`
- `ic_stability_regime_light.html`
目的：监控因子/模型的 IC 分布、时序演化与在不同市场状态下的稳定性，评估信号有效性与稳健性。

## 2. 数据与计算口径
- 输入：IC 序列（可为日/周），来源于预测与真实收益的截面相关；市场状态由分位或阈值划分。
- 口径：
  - 分布：直方图/箱线图呈现 IC 分布、均值、标准差、偏度等。
  - 时序：IC 随时间的波动及滚动均值/标准差。
  - 稳定性：按市场状态（如多空收益分层、波动 regime）分组的均值与显著性。
- 具体字段与计算细节见 `src/lightweight_analysis.py` 相应函数。

## 3. 输出与位置
- 页面输出：`reports/visualization_analysis/` 与 `docs/` 对应文件名。
- 仪表板：因子质量/信号监控板块。

## 4. 修改历史与注意事项
- 2025-03-XX：创建说明文档。若更换 IC 定义（Pearson/Spearman）、滚动窗口或市场分层方式，请在此记录。 
