先下结论：

* **如果你的目的是评估“择时能力 / 信号质量”本身**，现在这种只看价格在窗口高低点之间位置的 EntryRank / ExitRank，是一个**合理且标准**的做法。
* **但如果你想用它来“解释策略的盈利能力”**（尤其是解释 PnL、资金曲线），**只看 Rank 而不看“仓位大小 / 资金规模”确实不够**，你的顾虑是成立的，不过问题主要出在**聚合方式**而不是 Rank 定义本身。

下面我从量化研究的视角，把“哪里确实有问题”、“应该怎么补”讲清楚，并结合你现有的 `orders.parquet` / `paired_trades_fifo.parquet` 给一些具体可落地的改造建议。

---

## 1. 先分清两个层面：择时 vs 盈利

对单个持仓周期 (i)（多头）：

[
\text{PnL}_i \approx \text{size}_i \times (\text{卖价}_i - \text{买价}_i) - \text{fee}_i
]

其中“赚钱/亏钱”由三块组成：

1. **方向/择时好不好**：

   * 买得靠低（EntryRank 小）、卖得靠高（ExitRank 小）；
2. **行情有多大**：

   * 同样 Rank=0.2，如果窗口振幅只有 0.5%，赚不多；振幅 5%，就很香；
3. **你下了多大仓位 size**：

   * 小仓位赢 100 次，大仓位输 1 次，总 PnL 可能还是负的。

你现在的 EntryRank / ExitRank：

* **已经很好地刻画了第 1 点（择时）**：在给定窗口内，你的成交价格相对区间高低点的位置；
* **完全不看第 2 点和第 3 点**：

  * 没有刻画“这个窗口本身振幅多大”；
  * 聚合分布时，把 1 手的小单和 1,000 万的大仓位视为同等样本。

> 严格地说，每个周期内部你的成交均价 `open_price` / `close_price` 已经是自己成交量加权的，所以**“周期内的成交量”是被考虑了的**；
> 真正缺的是：在**不同持仓周期之间**，你现在是“每周期一票”平均，没有用成交量 / 资金 / PnL 做权重。

所以：

* **用现在的 Rank 分布来回答：“策略择时有没有 alpha？” → OK 的**；
* **用它来解释：“为什么整体赚/亏这么多钱？” → 信息明显不够，你的担心是对的。**

---

## 2. 基于现有数据，建议新增的指标 & 字段

以下都可以在你已经描述的 `paired_trades_fifo.parquet` + “持仓周期合并结果”上直接做，不需要额外外部数据。

### 2.1 先把“持仓周期级别”的基本量补齐

在你已经按 Code + FIFO 合并成 “0→非 0→0 的持仓周期”之后，对每个周期 (i) 建议至少添加：

* `position_id`：周期唯一 ID；
* `code`, `side`, `open_ts_i`, `close_ts_i`, `holding_minutes_trading_i`（已有口径即可）；
* **规模相关：**

  * `qty_i`：该周期全部成交的净持仓股数（long 用正，short 用绝对值）；
  * `notional_in_i`：建仓总买入金额（多头）/ 卖出金额（空头）；
  * `notional_out_i`：平仓总卖出金额 / 买入金额；
* **收益相关：**

  * `pnl_i = notional_out_i - notional_in_i - fee_i`
  * `ret_i = pnl_i / notional_in_i`（单周期收益率）
* **价格窗口相关（你已经在算的那些）：**

  * `P_E_min_i, P_E_max_i`：Entry 窗口内 min / max price
  * `P_X_min_i, P_X_max_i`：Exit 窗口内 min / max price
  * `EntryRank_i, ExitRank_i`（现有定义）

额外再加两个“振幅”：

* `range_E_i = P_E_max_i - P_E_min_i`
* `range_X_i = P_X_max_i - P_X_min_i`

这样你就有了：**{择时好不好、行情有多大、你下多大} 三板斧**，可以同时拿来解释 PnL。

---

## 3. 让 Rank 真正“对上”盈利：三种加权视角

### 3.1 成交金额加权的 Rank（volume-aware）

现在的 KPI 一般是：

* (\bar{E} = \text{mean}(EntryRank_i))
* (\bar{X} = \text{mean}(ExitRank_i))

这是“**每个持仓周期一票**”。

建议新增两组：

1. **成交金额加权的平均 Rank：**

[
\bar{E}^{(\text{notional})}
= \frac{\sum_i EntryRank_i \cdot notional_in_i}{\sum_i notional_in_i}
]

[
\bar{X}^{(\text{notional})}
= \frac{\sum_i ExitRank_i \cdot notional_in_i}{\sum_i notional_in_i}
]

含义：**“按真实资金规模看的平均择时水平”**。
如果你在小仓位上择时很好、大仓位上择时很差，这个数字会明显变坏。

2. **PnL 加权的平均 Rank（更直接贴合盈利）：**

[
\bar{E}^{(\text{pnl})}
= \frac{\sum_i EntryRank_i \cdot pnl_i}{\sum_i pnl_i^+}
]

其中 (\sum pnl_i^+) 只对盈利周期求和（亏损可以单独看一版）。
这个指标体现的是：**利润主要来自哪些 Rank 区间的交易**。

> 可视化上，你可以在原 EntryRank / ExitRank 分布图旁边，再放两条曲线：
>
> * “按笔数计”分布（当前的直方图）；
> * “按成交金额计”的分布；
> * “按 PnL 计”的分布。
>   三条一对比，很容易看出：赚钱的是不是集中在低 Rank 的大单上。

### 3.2 做一个“收益捕获率（edge capture）”指标

单独看 Entry 和 Exit Rank，还是很“局部”；为了直接贴 PnL，可以加一个综合指标，把“买在哪 + 卖在哪 + 振幅有多大”整合在一起。

对多头仓位，在统一窗口（比如 Entry 窗口并上 Exit 窗口，或者用持仓周期 [open_ts_i, close_ts_i]）上定义：

* (P_{\min,i} =) 该窗口内最低价
* (P_{\max,i} =) 该窗口内最高价
* (P_{\text{buy},i}, P_{\text{sell},i}) 为你的均价

定义单周期“收益捕获率”：

[
Edge_i =
\frac{P_{\text{sell},i} - P_{\text{buy},i}}{P_{\max,i} - P_{\min,i} + \epsilon}
]

* (Edge_i \approx 1)：几乎买在最低、卖在最高，把区间涨幅几乎全吃了；
* (Edge_i \approx 0)：基本没赚到区间波动；
* (Edge_i < 0)：区间总体是有机会的，但你反着做了。

然后再给 `Edge_i` 做同样的三种聚合：

* 按笔数平均：(\mathbb{E}[Edge_i])；
* 按成交金额加权平均：(\mathbb{E}[Edge_i; w = notional_in_i])；
* 按 PnL 加权平均：(\mathbb{E}[Edge_i; w = pnl_i])。

这样你就可以非常直接地说：

> “从 ex-post 的行情来看，本策略平均只捕获了可用波动的大约 X%，其中真正赚钱的那部分资金，捕获率大约是 Y%。”

这比单独展示 EntryRank / ExitRank 要更接近“盈利能力”的语言。

---

## 4. 把“仓位大小”这个维度单独剥离出来看

单纯做全局加权还不够，强烈建议再做一件事：**按仓位规模分组看 Rank → 这直接告诉你“大仓位是不是在帮你赚钱”。**

具体做法：

1. 在 position-level 表里新增：

   * `size_bucket`：

     * 例如按 `notional_in_i` 做分位数分箱：

       * S：0–40%（小仓）
       * M：40–80%（中仓）
       * L：80–95%（大仓）
       * XL：95–100%（超大单）
2. 对每个 `size_bucket` 分别画：

   * EntryRank / ExitRank 的分布；
   * `Edge_i` 分布；
   * 平均 `ret_i`、平均 `pnl_i`。

你能一眼看出一些典型模式，比如：

* 小仓位 EntryRank 很好、Edge 很高，但 PnL 贡献有限；
* 大仓位 EntryRank/ExitRank 明显变差，是不是“信号越强越敢上大仓”，但模型在高 `pred` 区间的校准不够好；
* 某些 bucket 上是亏钱的，说明 size 分配方案有问题。

这类分析对回答“为什么策略盈利能力不如预期”会很有帮助。

---

## 5. 如果你也关心“市场成交量”的维度，可以再多走一步

上面说的“成交量”都只是**你自己的成交规模**。如果你愿意用 Tushare 的分钟行情里的 `vol`，可以再加一个“相对市场成交量/参与率”的维度：

对每个持仓周期 (i)：

1. 计算你在开仓/平仓附近窗口内的市场总成交量：

   * 例如用 `minute_ohlcv`：在 Entry 窗口内求和 `volume` 得到 `vol_mkt_E_i`；
   * 在 Exit 窗口内求和 `volume` 得到 `vol_mkt_X_i`。
2. 定义你的“参与率”：

[
part_E_i = \frac{qty_i}{vol_mkt_E_i}, \quad
part_X_i = \frac{qty_i}{vol_mkt_X_i}
]

3. 再看 EntryRank / Edge_i 随 `part_E_i` 的关系：

   * 高参与率时，Rank 是否显著变差？
   * 低流动性（市场量很小）的区间里，是否更容易买在高位 / 卖在低位？

这类分析可以告诉你：**策略的择时 alpha 在多大的资金体量下还能保持，现在的资金规模是否已经开始“吃掉自己”。**

---

## 6. 总结一下回答你的核心问题

**Q1：只用 EntryRank / ExitRank（价格相对高低点的位置）而不看交易量，能不能解释盈利能力？**

* **不能完整解释**。

  * 它只能说明“你平均买/卖在本区间的什么位置”，
  * 但不能告诉你“钱主要是在哪些单、哪些 Rank 区间赚到或亏掉的”。
* 所以，如果你用它来证明“策略有择时 alpha”，是 OK 的；
* 如果你想用它来“对 PnL 做归因”，就必须把**仓位规模 / PnL / 波动幅度**拉进来。

**Q2：在现有数据集的基础上，具体应该做哪些补充 / 修改？**

核心建议可以浓缩成三条：

1. **在 position-level 数据上补齐规模 & 收益字段：**

   * `qty_i, notional_in_i, pnl_i, ret_i, range_E_i / range_X_i` 等；
2. **对 EntryRank / ExitRank / Edge 做多种加权聚合：**

   * 笔数、成交金额、PnL 三种视角；
   * 用这些视角来对照策略整体盈利；
3. **显式地把“仓位大小”和“相对市场成交量”作为分组维度看 Rank & PnL：**

   * size_bucket 分箱 + Rank/Edge/PnL；
   * 可选：参与率 `part_E/part_X` 分组，看大资金是否显著拖累择时。

只要把这几层补齐，你这个 Entry/ExitRank 页面就从一个“纯择时质量”看板，升级成了一个**能同时解释“为什么赚钱 / 为什么没赚到该赚的钱”**的分析面板，而不需要大改你当前那套基于 (T) 窗口和随机基准的整体设计。

如果你愿意，我也可以帮你把这些新指标塞进你现有的 HTML Dashboard 结构里，大致画一下前端布局 & 字段结构。
