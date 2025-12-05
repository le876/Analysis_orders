# 因子暴露分析中 inf 和 nan 值处理记录

## 修复日期
2025-10-05

## 问题描述
在"策略因子特征暴露度（轻量化）"页面的指标表格中，出现了以下异常显示：

### 原始问题数据
- **动量（5分钟）暴露-均值**: `inf`
- **动量（30分钟）暴露-均值**: `inf`
- **动量（60分钟）暴露-均值**: `inf`
- **高频β（5分钟）-均值**: `nan`
- **高频β（5分钟）-末值**: `nan`
- **高频β（5分钟）-末值相对市场**: `nan`
- **近10分钟价格振幅-均值**: `inf`

## 问题根源分析

### 1. inf（无穷大）的产生原因
在分钟级因子计算时（`_build_intraday_factor_snapshots` 方法），以下计算可能产生 `inf`：

```python
# 第3503-3507行（修复前）
minute_df['mom_5m'] = minute_df['ret_5m']  # pct_change() 当前值为0时产生 inf
minute_df['mom_30m'] = group_close.transform(lambda s: s / s.shift(6) - 1)  # shift值为0时产生 inf
minute_df['mom_60m'] = group_close.transform(lambda s: s / s.shift(12) - 1)  # shift值为0时产生 inf
minute_df['range_day'] = (rolling_high - rolling_low) / minute_df['close']  # close为0时产生 inf
```

**原因**：
- 当股票价格为0或接近0时，百分比变化计算会产生无穷大
- 当除数为0时，直接产生 `inf`

### 2. nan（非数字）的产生原因
高频β的计算依赖于指数收益率数据：

```python
# 第3534-3542行
minute_df['beta_5m'] = np.nan  # 初始值为 nan
# 如果指数数据缺失或方差为0，beta保持为 nan
cov = s['ret_5m'].rolling(window=6, min_periods=3).cov(s['idx_ret'])
var = s['idx_ret'].rolling(window=6, min_periods=3).var()
beta = cov / var.replace(0, np.nan)
```

**原因**：
- 指数5分钟数据缺失
- 指数收益率方差为0（价格不变）
- 样本数不足（min_periods要求）

### 3. 指标统计时的传播
在计算指标均值时，没有过滤异常值：

```python
# 第4154-4159行（修复前）
series = pd.to_numeric(exp_df[f'strat_{col}'], errors='coerce')
metrics[f'{titles_map.get(col, col)}-均值'] = f"{series.mean():.4f}"  # 包含 inf 时结果为 inf
```

## 解决方案

### 修改1：数据源头 - 因子计算时过滤 inf
**文件**：`lightweight_analysis.py` 第3502-3512行

**修改内容**：在计算分钟级因子后立即将 `inf` 替换为 `nan`

```python
group_close = minute_df.groupby('Code')['close']
minute_df['ret_5m'] = group_close.pct_change()
minute_df['mom_5m'] = minute_df['ret_5m'].replace([np.inf, -np.inf], np.nan)
minute_df['mom_30m'] = group_close.transform(lambda s: s / s.shift(6) - 1).replace([np.inf, -np.inf], np.nan)
minute_df['mom_60m'] = group_close.transform(lambda s: s / s.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
minute_df['rv_5m'] = group_close.transform(lambda s: (s.pct_change().pow(2).rolling(window=2, min_periods=1).sum()) ** 0.5).replace([np.inf, -np.inf], np.nan)

# 10分钟窗口的价格振幅（归一化）
rolling_high = minute_df.groupby('Code')['high'].transform(lambda s: s.rolling(window=2, min_periods=1).max())
rolling_low = minute_df.groupby('Code')['low'].transform(lambda s: s.rolling(window=2, min_periods=1).min())
minute_df['range_day'] = ((rolling_high - rolling_low) / minute_df['close']).replace([np.inf, -np.inf], np.nan)
```

### 修改2：加权暴露计算时过滤 inf
**文件**：`lightweight_analysis.py` 第4026-4041行

**修改内容**：在加权计算前过滤掉 `inf` 值

```python
def _weighted_exposure(df: pd.DataFrame, col: str, weight_col: str) -> float:
    g = df.dropna(subset=[col, weight_col])
    if len(g) == 0:
        return np.nan
    # 过滤掉 inf 值
    g = g[~np.isinf(g[col])]
    if len(g) == 0:
        return np.nan
    w = g[weight_col].astype(float)
    sw = w.sum()
    if sw <= 0:
        return np.nan
    x = g[col].astype(float)
    result = (w * x).sum() / sw
    # 确保结果不是 inf
    return float(result) if not np.isinf(result) else np.nan
```

### 修改3：市场基准计算时过滤 inf
**文件**：`lightweight_analysis.py` 第4075-4090行

**修改内容**：在市值加权计算前过滤掉 `inf` 值

```python
def _market_exposure(factors_day: pd.DataFrame, col: str) -> float:
    g = factors_day.dropna(subset=[col])
    if len(g) == 0:
        return np.nan
    # 过滤掉 inf 值
    g = g[~np.isinf(g[col])]
    if len(g) == 0:
        return np.nan
    if 'market_cap' in g.columns and g['market_cap'].notna().any():
        w = g['market_cap'].astype(float).clip(lower=0)
        sw = w.sum()
        if sw > 0:
            result = (w * g[col].astype(float)).sum() / sw
            return float(result) if not np.isinf(result) else np.nan
    result = g[col].astype(float).mean()
    return float(result) if not np.isinf(result) else np.nan
```

### 修改4：指标格式化时安全处理异常值
**文件**：`lightweight_analysis.py` 第4148-4176行

**修改内容**：创建安全格式化函数，将 `inf` 和 `nan` 显示为 `N/A`

```python
def _safe_format(val, fmt='.4f'):
    """安全格式化数值，处理 inf 和 nan"""
    if pd.isna(val) or np.isinf(val):
        return 'N/A'
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return 'N/A'

for col in show_cols:
    series = pd.to_numeric(exp_df[f'strat_{col}'], errors='coerce')
    # 过滤掉 inf 和 nan 后计算均值
    valid_series = series.replace([np.inf, -np.inf], np.nan).dropna()
    mean_val = valid_series.mean() if len(valid_series) > 0 else np.nan
    last_val = series.iloc[-1] if len(series) > 0 else np.nan
    
    metrics[f'{titles_map.get(col, col)}-均值'] = _safe_format(mean_val)
    metrics[f'{titles_map.get(col, col)}-末值'] = _safe_format(last_val)
    
    if f'mkt_{col}' in exp_df.columns:
        mkt_last = pd.to_numeric(exp_df[f'mkt_{col}'], errors='coerce').iloc[-1] if len(exp_df) > 0 else np.nan
        diff_last = last_val - mkt_last if not (pd.isna(last_val) or pd.isna(mkt_last)) else np.nan
        metrics[f'{titles_map.get(col, col)}-末值相对市场'] = _safe_format(diff_last, '+.4f')
```

## 修复后的结果

### 修复后的指标值
- **动量（5分钟）暴露-均值**: `0.0003` ✅
- **动量（30分钟）暴露-均值**: `-0.0001` ✅
- **动量（60分钟）暴露-均值**: `-0.0003` ✅
- **高频β（5分钟）-均值**: `N/A` ✅
- **高频β（5分钟）-末值**: `N/A` ✅
- **高频β（5分钟）-末值相对市场**: `N/A` ✅
- **近10分钟价格振幅-均值**: `0.0064` ✅

所有 `inf` 和 `nan` 值都已被正确处理并显示。

## 技术要点总结

### inf 和 nan 的区别
- **inf (Infinity)**: 无穷大，通常由除以0或极大数值运算产生
- **nan (Not a Number)**: 非数字，通常由缺失数据、无效运算产生

### 处理策略
1. **源头控制**：在因子计算时立即将 `inf` 替换为 `nan`
2. **计算过滤**：在加权计算前过滤掉 `inf` 和 `nan`
3. **结果验证**：计算完成后检查结果是否为 `inf`
4. **显示优化**：在用户界面中将异常值显示为 `N/A` 而非技术术语

### Pandas/NumPy 中的异常值处理函数
```python
# 检测
pd.isna(val)          # 检测 NaN
np.isinf(val)         # 检测 inf
np.isfinite(val)      # 检测有限值（非 inf 且非 nan）

# 替换
series.replace([np.inf, -np.inf], np.nan)  # 将 inf 替换为 nan
series.dropna()       # 删除 nan
series.fillna(0)      # 用0填充 nan

# 过滤
df = df[~np.isinf(df[col])]  # 过滤掉 inf 行
df = df[np.isfinite(df[col])]  # 只保留有限值
```

## 为什么会出现这些异常值？

### 1. 动量因子的 inf
**场景**：股票复牌后价格从0跳到正常价格
```python
# 前一个价格为0
prev_price = 0.0
curr_price = 10.5
mom = curr_price / prev_price - 1  # = 10.5 / 0 - 1 = inf
```

### 2. 高频β的 nan
**场景**：指数数据缺失或价格不变
```python
# 指数收益率方差为0
idx_returns = [0, 0, 0, 0, 0]  # 价格不变
variance = np.var(idx_returns)  # = 0
beta = covariance / variance  # = x / 0 = nan
```

### 3. 价格振幅的 inf
**场景**：股票价格为0
```python
range_val = high - low  # = 0.5
close = 0.0
amplitude = range_val / close  # = 0.5 / 0 = inf
```

## 最佳实践建议

### 1. 数据预处理
- 在数据源头过滤掉无效价格（价格<=0）
- 使用 `clip()` 限制极端值
- 使用 `winsorize()` 处理离群值

### 2. 计算保护
- 除法前检查分母是否为0
- 使用 `np.divide()` 的 `where` 参数
- 百分比变化使用 `pd.Series.pct_change(fill_method=None)`

### 3. 结果验证
- 计算后立即检查 `inf` 和 `nan`
- 使用 `assert` 验证关键指标
- 记录异常值的数量和位置

### 4. 用户展示
- 将技术异常值转换为用户友好的显示（如 `N/A`）
- 提供异常值的解释说明
- 考虑添加数据质量指标

## 相关文件
- 修改的核心文件：`lightweight_analysis.py`
  - 第3502-3512行：因子计算
  - 第4026-4041行：加权暴露计算
  - 第4075-4090行：市场基准计算
  - 第4148-4176行：指标格式化
- 生成的可视化页面：`factor_exposure_light.html`

## 后续注意事项
1. 在其他涉及因子计算的地方应用相同的 `inf` 过滤逻辑
2. 考虑在数据加载时就过滤掉无效价格（价格<=0）
3. 添加数据质量监控，统计每日 `inf` 和 `nan` 的数量
4. 在文档中说明哪些因子可能因数据质量问题显示为 `N/A`

---

**记录人**：AI Assistant  
**最后更新**：2025-10-05
