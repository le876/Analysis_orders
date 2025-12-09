#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量使用 baostock 5 分钟行情计算 EntryRank / ExitRank（全量标的），带行情缓存 + 结果缓存
- 全体交易: Tα_global=234 分钟
- 超短单(持仓<=10分钟): Tα_short=5 分钟
- 行情缓存: data/cache/baostock_5min/{code}.parquet（存在则复用，不再拉取）
- 结果缓存: data/cache/entry_exit_rank_baostock_result.json（存在则直接生成页面；如算法或参数改动，请 --recompute）
输出:
- docs/entry_exit_rank_baostock_full.html (直方图页面，用于发布)
- docs/entry_exit_rank_baostock_full.txt  (样本计数)
运行方式（在仓库根目录）:
HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= /home/ubuntu/.conda/envs/quant_env/bin/python scripts/run_entry_exit_rank_baostock.py
强制重算（忽略结果缓存）:
HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= /home/ubuntu/.conda/envs/quant_env/bin/python scripts/run_entry_exit_rank_baostock.py --recompute
"""
import argparse
import baostock as bs
import json
import pandas as pd
from datetime import timedelta
from pathlib import Path
import plotly.graph_objects as go
import uuid
import numpy as np

T_GLOBAL = 234      # 全体交易窗口（分钟）
T_SHORT = 5         # 超短单窗口（分钟）
PAIRS_PATH = Path('data/paired_trades_fifo.parquet')
REPORT_HTML = Path('docs/entry_exit_rank_baostock_full.html')
REPORT_TXT = Path('docs/entry_exit_rank_baostock_full.txt')
COPY_HTML_TARGETS = []
CACHE_DIR = Path('data/cache/baostock_5min')
RESULT_CACHE = Path('data/cache/entry_exit_rank_baostock_result.json')
RNG = np.random.default_rng(42)  # 经验基准抽样用，保证可复现

parser = argparse.ArgumentParser(description='计算 Entry/ExitRank (baostock 5min)')
parser.add_argument('--recompute', action='store_true', help='忽略结果缓存，重新计算')
parser.add_argument('--workers', type=int, default=1, help='并行 worker 数，仅缓存齐全时有效（建议 4-6）')
args = parser.parse_args()
use_result_cache = RESULT_CACHE.exists() and (not args.recompute)


def weighted_percentile(arr, weights, q):
    arr = np.asarray(arr, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(arr) & np.isfinite(weights) & (weights >= 0)
    arr = arr[mask]; weights = weights[mask]
    if arr.size == 0 or weights.sum() == 0:
        return np.nan
    sorter = np.argsort(arr)
    arr_sorted = arr[sorter]
    w_sorted = weights[sorter]
    cum_w = np.cumsum(w_sorted)
    cutoff = q / 100.0 * cum_w[-1]
    idx = np.searchsorted(cum_w, cutoff, side='left')
    idx = min(idx, len(arr_sorted) - 1)
    return arr_sorted[idx]


def summarize_hist(data, key, title, bins=30, weights=None, baseline_data=None, baseline_weights=None):
    arr = np.asarray(data, dtype=float)
    mask = np.isfinite(arr)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        mask = mask & np.isfinite(w) & (w >= 0)
        w = w[mask]
    else:
        w = None
    arr = arr[mask]
    if arr.size == 0:
        return None
    counts, edges = np.histogram(arr, bins=bins, range=(0, 1), weights=w)
    if w is None:
        stats = {
            "size": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }
    else:
        w_sum = w.sum()
        stats = {
            "size": int(w_sum),
            "mean": float((arr * w).sum() / w_sum),
            "median": float(weighted_percentile(arr, w, 50)),
            "p25": float(weighted_percentile(arr, w, 25)),
            "p75": float(weighted_percentile(arr, w, 75)),
        }
    baseline_probs = None
    if baseline_data is not None:
        base_arr = np.asarray(baseline_data, dtype=float)
        base_mask = np.isfinite(base_arr)
        if baseline_weights is not None:
            bw = np.asarray(baseline_weights, dtype=float)
            base_mask = base_mask & np.isfinite(bw) & (bw >= 0)
            bw = bw[base_mask]
        else:
            bw = None
        base_arr = base_arr[base_mask]
        if base_arr.size > 0:
            base_counts, _ = np.histogram(base_arr, bins=bins, range=(0, 1), weights=bw)
            total = base_counts.sum()
            if total > 0:
                baseline_probs = (base_counts / total).tolist()

    return {
        "key": key,
        "title": title,
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "stats": stats,
        "baseline_probs": baseline_probs,
    }


def paired_hist_fig(title, entry_hist, exit_hist, colors=None):
    colors = colors or {"entry": "#10b981", "exit": "#f43f5e"}
    edges = np.asarray(entry_hist["edges"], dtype=float)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    widths = (edges[1:] - edges[:-1]).tolist()

    e_counts = np.asarray(entry_hist["counts"], dtype=float)
    x_counts = np.asarray(exit_hist["counts"], dtype=float)
    e_probs_arr = e_counts / e_counts.sum() if e_counts.sum() > 0 else e_counts
    x_probs_arr = x_counts / x_counts.sum() if x_counts.sum() > 0 else x_counts
    e_probs = e_probs_arr.tolist()
    x_probs = x_probs_arr.tolist()

    fig = go.Figure()
    fig.add_bar(name="Entry", x=centers, y=e_probs, width=widths, marker_color=colors["entry"], opacity=0.78)
    fig.add_bar(name="Exit", x=centers, y=x_probs, width=widths, marker_color=colors["exit"], opacity=0.66)

    y_max = 0.0
    for label, hist, color in [
        ("Entry median", entry_hist.get("stats", {}), colors["entry"]),
        ("Exit median", exit_hist.get("stats", {}), colors["exit"]),
    ]:
        if hist and hist.get("median") is not None:
            fig.add_vline(
                x=hist["median"],
                line_dash="dot",
                line_color=color,
                opacity=0.55,
            )
    if len(e_probs) > 0:
        y_max = max(y_max, max(e_probs))
    if len(x_probs) > 0:
        y_max = max(y_max, max(x_probs))

    base_probs = entry_hist.get("baseline_probs") or exit_hist.get("baseline_probs")
    if base_probs and len(base_probs) == len(centers):
        y_max = max(y_max, max(base_probs))
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=base_probs,
                mode="lines",
                name="经验基准",
                line=dict(color="#9ca3af", width=2, dash="dot"),
                hovertemplate="Baseline: %{y:.3f}<extra></extra>",
            )
        )
    elif len(widths) > 0:
        base_level = 1 / len(widths)
        y_max = max(y_max, base_level)
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=[base_level] * len(centers),
                mode="lines",
                name="随机均匀基准",
                line=dict(color="#9ca3af", width=2, dash="dot"),
                hovertemplate="Uniform: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Rank（0=更优）",
        yaxis_title="比例",
        barmode="group",
        bargap=0.08,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=20, t=48, b=50),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Helvetica, Arial, sans-serif", color="#374151"),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, (y_max * 1.25) if y_max > 0 else 1], tickformat=".3f"),
    )
    return fig


def single_hist_fig(title, hist, color="#2563eb"):
    edges = np.asarray(hist["edges"], dtype=float)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    widths = (edges[1:] - edges[:-1]).tolist()
    counts = np.asarray(hist["counts"], dtype=float)
    probs_arr = counts / counts.sum() if counts.sum() > 0 else counts
    probs = probs_arr.tolist()
    y_max = probs_arr.max() if probs_arr.size else 0
    fig = go.Figure()
    fig.add_bar(name="Edge", x=centers, y=probs, width=widths, marker_color=color, opacity=0.78)
    base_probs = hist.get("baseline_probs")
    if base_probs and len(base_probs) == len(centers):
        y_max = max(y_max, max(base_probs))
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=base_probs,
                mode="lines",
                name="经验基准",
                line=dict(color="#9ca3af", width=2, dash="dot"),
                hovertemplate="Baseline: %{y:.3f}<extra></extra>",
            )
        )
    elif len(widths) > 0:
        base_level = 1 / len(widths)
        y_max = max(y_max, base_level)
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=[base_level] * len(centers),
                mode="lines",
                name="随机均匀基准",
                line=dict(color="#9ca3af", width=2, dash="dot"),
                hovertemplate="Uniform: %{y:.3f}<extra></extra>",
            )
        )
    st = hist["stats"]
    fig.update_layout(
        title=title,
        xaxis_title="Edge（0=未捕获波动，1=吃满区间）",
        yaxis_title="比例",
        bargap=0.08,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=48, b=50),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Helvetica, Arial, sans-serif", color="#374151"),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, (y_max * 1.25) if y_max > 0 else 1], tickformat=".3f"),
        annotations=[
            dict(
                x=0.98,
                y=0.92,
                xref="paper",
                yref="paper",
                text=f"样本 {st['size']:,} | 均值 {st['mean']:.3f}",
                showarrow=False,
                font=dict(size=12, color="#475467"),
                align="right",
                bgcolor="rgba(255,255,255,0.6)",
            )
        ],
    )
    return fig


def _to_plain(obj):
    """递归把 numpy / pandas 对象转成原生 Python，避免 to_json 生成 typed array"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_plain(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def fig_to_div_script(fig, mode_key, visible=True, config=None):
    """把图对象序列化为纯 JSON + div + Plotly.newPlot 脚本，避免 typed array 兼容问题"""
    div_id = f"fig-{uuid.uuid4().hex}"
    config = config or {"responsive": True, "displayModeBar": False}
    payload = _to_plain(fig.to_plotly_json())  # 转成纯 Python 可 JSON 化对象
    data_json = json.dumps(payload.get("data", []))
    layout_json = json.dumps(payload.get("layout", {}))
    div_html = f"<div class=\"mode-chart\" data-mode=\"{mode_key}\" id=\"{div_id}\" style=\"display:{'block' if visible else 'none'}\"></div>"
    script = (
        f"try {{\n"
        f"  Plotly.newPlot('{div_id}', {data_json}, {layout_json}, {json.dumps(config)});\n"
        f"}} catch (e) {{\n"
        f"  console.error('Plotly render error', e);\n"
        f"  const el = document.getElementById('{div_id}');\n"
        f"  if (el) el.innerText = '图表渲染失败: ' + e;\n"
        f"}}"
    )
    return div_html, script


def perfect_share(hist, threshold=0.1):
    if not hist:
        return None
    edges = np.asarray(hist["edges"], dtype=float)
    counts = np.asarray(hist["counts"], dtype=float)
    total = counts.sum()
    if total <= 0:
        return None
    acc = 0.0
    for i, c in enumerate(counts):
        left, right = edges[i], edges[i + 1]
        if left >= threshold:
            break
        if right <= threshold:
            acc += c
        else:
            # 线性近似分摊跨阈值的桶
            acc += c * (threshold - left) / (right - left)
            break
    return float(acc / total)


def format_stats(stats):
    if not stats or stats.get("size", 0) == 0:
        return "无数据"
    return f"样本 {stats['size']:,} | 均值 {stats['mean']:.3f} | 中位数 {stats['median']:.3f} | P25/P75 {stats['p25']:.3f}/{stats['p75']:.3f}"


def format_pct(p):
    if p is None:
        return "--"
    return f"{p*100:.1f}%"


def format_two_decimals(v):
    if v is None:
        return "--"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "--"


def trading_minutes(o, c):
    open_ts = pd.Timestamp(o); close_ts = pd.Timestamp(c)
    open_date = open_ts.date(); close_date = close_ts.date()
    M1, M2, A1, A2 = 570, 690, 780, 900
    open_min = open_ts.hour * 60 + open_ts.minute
    close_min = close_ts.hour * 60 + close_ts.minute
    if open_date == close_date:
        return max(0, min(M2, close_min) - max(M1, open_min)) + max(0, min(A2, close_min) - max(A1, open_min))
    open_m = max(0, M2 - max(M1, open_min)) + max(0, A2 - max(A1, open_min))
    close_m = max(0, min(M2, close_min) - M1) + max(0, min(A2, close_min) - A1)
    middle_days = np.busday_count(np.array(open_date, dtype='datetime64[D]') + np.timedelta64(1, 'D'), np.array(close_date, dtype='datetime64[D]'), weekmask='1111100')
    return open_m + close_m + int(middle_days) * 240


sample_counts = {}
stats_map = {}
hists = []
plot_scripts = []
T_GLOBAL_USE = T_GLOBAL
T_SHORT_USE = T_SHORT

if use_result_cache:
    print('🗂️ 检测到结果缓存，直接生成页面（如需重算请加 --recompute）', flush=True)
    payload = json.loads(RESULT_CACHE.read_text(encoding='utf-8'))
    meta = payload.get('meta', {})
    hists = payload.get('hists', [])
    T_GLOBAL_USE = meta.get('t_global', T_GLOBAL)
    T_SHORT_USE = meta.get('t_short', T_SHORT)
    sample_counts = meta.get('sample_counts', {})
    stats_map = {h['key']: h.get('stats', {}) for h in hists}
else:
    print('✅ 加载配对交易数据...', flush=True)
    pairs = pd.read_parquet(PAIRS_PATH, columns=['code', 'trade_type', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price', 'matched_qty', 'buy_fee', 'sell_fee'])
    for col in ['buy_timestamp', 'sell_timestamp']:
        if not pd.api.types.is_datetime64_any_dtype(pairs[col]):
            pairs[col] = pd.to_datetime(pairs[col])

    short_mask = pairs['trade_type'] == 'short'
    pairs['open_timestamp'] = pairs['buy_timestamp'].where(~short_mask, pairs['sell_timestamp'])
    pairs['close_timestamp'] = pairs['sell_timestamp'].where(~short_mask, pairs['buy_timestamp'])
    pairs['open_price'] = pairs['buy_price'].where(~short_mask, pairs['sell_price'])
    pairs['close_price'] = pairs['sell_price'].where(~short_mask, pairs['buy_price'])
    pairs['holding_minutes_trading'] = [trading_minutes(o, c) for o, c in zip(pairs['open_timestamp'], pairs['close_timestamp'])]

    # 计算同一标的的前一平仓 / 下一开仓，用于窗口裁剪
    pairs = pairs.sort_values(['code', 'open_timestamp']).reset_index(drop=True)
    pairs['prev_close_ts'] = pairs.groupby('code')['close_timestamp'].shift(1)
    pairs['next_open_ts'] = pairs.groupby('code')['open_timestamp'].shift(-1)
    pairs_group = {c: g for c, g in pairs.groupby('code')}

    # 按标的构建日期范围，按交易条数排序
    code_ranges = {}
    for code, g in pairs.groupby('code'):
        start = g[['open_timestamp', 'close_timestamp']].min().min().date()
        end = g[['open_timestamp', 'close_timestamp']].max().max().date()
        code_ranges[code] = (start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    codes_sorted = sorted(code_ranges.keys(), key=lambda x: len(pairs[pairs['code'] == x]), reverse=True)
    print(f'📈 标的数量: {len(codes_sorted)}', flush=True)

    entries_g = []; exits_g = []; entries_s = []; exits_s = []
    entries_g_base = []; exits_g_base = []; entries_s_base = []; exits_s_base = []
    entries_g_notional = []; exits_g_notional = []; entries_s_notional = []; exits_s_notional = []
    entries_g_base_notional = []; exits_g_base_notional = []; entries_s_base_notional = []; exits_s_base_notional = []
    entries_g_pnl = []; exits_g_pnl = []; entries_s_pnl = []; exits_s_pnl = []
    entries_g_pnl_all = []; exits_g_pnl_all = []; entries_s_pnl_all = []; exits_s_pnl_all = []
    entries_g_base_pnl = []; exits_g_base_pnl = []; entries_s_base_pnl = []; exits_s_base_pnl = []
    entries_g_base_pnl_all = []; exits_g_base_pnl_all = []; entries_s_base_pnl_all = []; exits_s_base_pnl_all = []
    edges_g = []; edges_g_notional = []; edges_g_pnl = []; edges_g_pnl_all = []
    edges_s = []; edges_s_notional = []; edges_s_pnl = []; edges_s_pnl_all = []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_files = {code: CACHE_DIR / f"{code.replace('.', '_')}.parquet" for code in codes_sorted}
    missing_codes = [c for c in codes_sorted if not cache_files[c].exists()]

    lg = None
    if missing_codes:
        print('🔌 登录 baostock...', flush=True)
        lg = bs.login()
        if lg.error_code != '0':
            raise SystemExit('baostock login failed: ' + lg.error_msg)
        worker_use = 1
    else:
        print('🗄️ 缓存齐全，跳过行情下载，直接读取本地。', flush=True)
        worker_use = max(1, args.workers)

    def process_code(code):
        start, end = code_ranges[code]
        mkt = code.split('.')[-1].lower(); base = code.split('.')[0]
        bs_code = f"{mkt}.{base}"
        cache_file = cache_files[code]

        if cache_file.exists():
            md = pd.read_parquet(cache_file)
            md.index = pd.to_datetime(md.index)
        elif lg is not None:
            rs = bs.query_history_k_data_plus(bs_code, 'date,time,open,high,low,close,volume',
                                              start_date=start, end_date=end, frequency='5')
            data = []
            while rs.error_code == '0' and rs.next():
                data.append(rs.get_row_data())
            if not data:
                return {}
            cdf = pd.DataFrame(data, columns=rs.fields)
            cdf['datetime'] = pd.to_datetime(cdf['date'] + ' ' + cdf['time'].str[8:10] + ':' + cdf['time'].str[10:12] + ':' + cdf['time'].str[12:14])
            cdf[['open', 'high', 'low', 'close']] = cdf[['open', 'high', 'low', 'close']].astype(float)
            cdf.set_index('datetime', inplace=True)
            md = cdf[['high', 'low', 'open', 'close']]
            md.to_parquet(cache_file, index=True)
        else:
            return {}

        trades = pairs_group.get(code)
        if trades is None or trades.empty:
            return {}

        res = {
            "entries_g": [], "exits_g": [], "entries_s": [], "exits_s": [],
            "entries_g_base": [], "exits_g_base": [], "entries_s_base": [], "exits_s_base": [],
            "entries_g_notional": [], "exits_g_notional": [], "entries_s_notional": [], "exits_s_notional": [],
            "entries_g_base_notional": [], "exits_g_base_notional": [], "entries_s_base_notional": [], "exits_s_base_notional": [],
            "entries_g_pnl": [], "exits_g_pnl": [], "entries_s_pnl": [], "exits_s_pnl": [],
            "entries_g_pnl_all": [], "exits_g_pnl_all": [], "entries_s_pnl_all": [], "exits_s_pnl_all": [],
            "entries_g_base_pnl": [], "exits_g_base_pnl": [], "entries_s_base_pnl": [], "exits_s_base_pnl": [],
            "entries_g_base_pnl_all": [], "exits_g_base_pnl_all": [], "entries_s_base_pnl_all": [], "exits_s_base_pnl_all": [],
            "edges_g": [], "edges_g_notional": [], "edges_g_pnl": [], "edges_g_pnl_all": [],
            "edges_s": [], "edges_s_notional": [], "edges_s_pnl": [], "edges_s_pnl_all": [],
        }

        for _, row in trades.iterrows():
            qty = float(row.get('matched_qty', 0) or 0)
            fees = float((row.get('buy_fee', 0) or 0) + (row.get('sell_fee', 0) or 0))
            notional_in = row['open_price'] * qty
            pnl = ((row['close_price'] - row['open_price']) * qty if row['trade_type'] != 'short' else (row['open_price'] - row['close_price']) * qty) - fees
            pnl_abs = abs(pnl)
            # 边界裁剪：避免跨越前一平仓和下一开仓
            prev_close = row.get('prev_close_ts')
            next_open = row.get('next_open_ts')

            def apply_bounds(start_ts, end_ts):
                if pd.notna(prev_close):
                    start_ts = max(start_ts, prev_close)
                if pd.notna(next_open):
                    end_ts = min(end_ts, next_open)
                return start_ts, end_ts

            # Entry 窗口：开仓 -> 平仓，全程内的相对位置
            e_start = row['open_timestamp']
            e_end = row['close_timestamp']
            e_start, e_end = apply_bounds(e_start, e_end)
            es = md.loc[(md.index >= e_start) & (md.index <= e_end)]
            if not es.empty:
                lo, hi = es['low'].min(), es['high'].max()
                er = 0.5 if hi == lo else ((hi - row['open_price']) / (hi - lo) if row['trade_type'] == 'short' else (row['open_price'] - lo) / (hi - lo))
                res["entries_g"].append(er)
                res["entries_g_notional"].append((er, notional_in))
                res["entries_g_pnl"].append((er, max(pnl, 0)))
                res["entries_g_pnl_all"].append((er, pnl_abs))
                # 经验基准：随机抽取窗口内一根K线的随机价
                rand_idx = RNG.integers(0, len(es))
                low_r, high_r = es['low'].iloc[rand_idx], es['high'].iloc[rand_idx]
                if np.isfinite(low_r) and np.isfinite(high_r) and hi != lo:
                    rand_price = RNG.uniform(low_r, high_r)
                    er_base = (hi - rand_price) / (hi - lo) if row['trade_type'] == 'short' else (rand_price - lo) / (hi - lo)
                    er_base = max(0.0, min(1.0, er_base))
                    res["entries_g_base"].append(er_base)
                    res["entries_g_base_notional"].append((er_base, notional_in))
                    res["entries_g_base_pnl"].append((er_base, max(pnl, 0)))
                    res["entries_g_base_pnl_all"].append((er_base, pnl_abs))

            # Exit 窗口：开仓 -> 平仓后 T/2（物理分钟）
            x_start = row['open_timestamp']
            x_end = row['close_timestamp'] + timedelta(minutes=T_GLOBAL / 2)
            x_start, x_end = apply_bounds(x_start, x_end)
            xs = md.loc[(md.index >= x_start) & (md.index <= x_end)]
            if not xs.empty:
                lo2, hi2 = xs['low'].min(), xs['high'].max()
                xr = 0.5 if hi2 == lo2 else ((row['close_price'] - lo2) / (hi2 - lo2) if row['trade_type'] == 'short' else (hi2 - row['close_price']) / (hi2 - lo2))
                res["exits_g"].append(xr)
                res["exits_g_notional"].append((xr, notional_in))
                res["exits_g_pnl"].append((xr, max(pnl, 0)))
                res["exits_g_pnl_all"].append((xr, pnl_abs))
                rand_idx = RNG.integers(0, len(xs))
                low_r2, high_r2 = xs['low'].iloc[rand_idx], xs['high'].iloc[rand_idx]
                if np.isfinite(low_r2) and np.isfinite(high_r2) and hi2 != lo2:
                    rand_price2 = RNG.uniform(low_r2, high_r2)
                    xr_base = (rand_price2 - lo2) / (hi2 - lo2) if row['trade_type'] == 'short' else (hi2 - rand_price2) / (hi2 - lo2)
                    xr_base = max(0.0, min(1.0, xr_base))
                    res["exits_g_base"].append(xr_base)
                    res["exits_g_base_notional"].append((xr_base, notional_in))
                    res["exits_g_base_pnl"].append((xr_base, max(pnl, 0)))
                    res["exits_g_base_pnl_all"].append((xr_base, pnl_abs))

            hold_slice = md.loc[(md.index >= row['open_timestamp']) & (md.index <= row['close_timestamp'])]
            if not hold_slice.empty:
                lo_h, hi_h = hold_slice['low'].min(), hold_slice['high'].max()
                denom_h = hi_h - lo_h
                if denom_h <= 0:
                    edge = 0.0
                else:
                    edge = ((row['close_price'] - row['open_price']) / denom_h) if row['trade_type'] != 'short' else ((row['open_price'] - row['close_price']) / denom_h)
                edge = max(0.0, min(1.0, edge))
                res["edges_g"].append(edge)
                res["edges_g_notional"].append((edge, notional_in))
                res["edges_g_pnl"].append((edge, max(pnl, 0)))
                res["edges_g_pnl_all"].append((edge, pnl_abs))

            if row['holding_minutes_trading'] <= 10:
                e_start_s = row['open_timestamp']
                e_end_s = row['close_timestamp']
                e_start_s, e_end_s = apply_bounds(e_start_s, e_end_s)
                es_s = md.loc[(md.index >= e_start_s) & (md.index <= e_end_s)]
                if not es_s.empty:
                    loS, hiS = es_s['low'].min(), es_s['high'].max()
                    er_s = 0.5 if hiS == loS else ((hiS - row['open_price']) / (hiS - loS) if row['trade_type'] == 'short' else (row['open_price'] - loS) / (hiS - loS))
                    res["entries_s"].append(er_s)
                    res["entries_s_notional"].append((er_s, notional_in))
                    res["entries_s_pnl"].append((er_s, max(pnl, 0)))
                    res["entries_s_pnl_all"].append((er_s, pnl_abs))
                    rand_idx_s = RNG.integers(0, len(es_s))
                    low_rs, high_rs = es_s['low'].iloc[rand_idx_s], es_s['high'].iloc[rand_idx_s]
                    if np.isfinite(low_rs) and np.isfinite(high_rs) and hiS != loS:
                        rand_price_s = RNG.uniform(low_rs, high_rs)
                        er_s_base = (hiS - rand_price_s) / (hiS - loS) if row['trade_type'] == 'short' else (rand_price_s - loS) / (hiS - loS)
                        er_s_base = max(0.0, min(1.0, er_s_base))
                        res["entries_s_base"].append(er_s_base)
                        res["entries_s_base_notional"].append((er_s_base, notional_in))
                        res["entries_s_base_pnl"].append((er_s_base, max(pnl, 0)))
                        res["entries_s_base_pnl_all"].append((er_s_base, pnl_abs))
                x_start_s = row['open_timestamp']
                x_end_s = row['close_timestamp'] + timedelta(minutes=T_SHORT / 2)
                x_start_s, x_end_s = apply_bounds(x_start_s, x_end_s)
                xs_s = md.loc[(md.index >= x_start_s) & (md.index <= x_end_s)]
                if not xs_s.empty:
                    loS2, hiS2 = xs_s['low'].min(), xs_s['high'].max()
                    xr_s = 0.5 if hiS2 == loS2 else ((row['close_price'] - loS2) / (hiS2 - loS2) if row['trade_type'] == 'short' else (hiS2 - row['close_price']) / (hiS2 - loS2))
                    res["exits_s"].append(xr_s)
                    res["exits_s_notional"].append((xr_s, notional_in))
                    res["exits_s_pnl"].append((xr_s, max(pnl, 0)))
                    res["exits_s_pnl_all"].append((xr_s, pnl_abs))
                    rand_idx_s2 = RNG.integers(0, len(xs_s))
                    low_rs2, high_rs2 = xs_s['low'].iloc[rand_idx_s2], xs_s['high'].iloc[rand_idx_s2]
                    if np.isfinite(low_rs2) and np.isfinite(high_rs2) and hiS2 != loS2:
                        rand_price_s2 = RNG.uniform(low_rs2, high_rs2)
                        xr_s_base = (rand_price_s2 - loS2) / (hiS2 - loS2) if row['trade_type'] == 'short' else (hiS2 - rand_price_s2) / (hiS2 - loS2)
                        xr_s_base = max(0.0, min(1.0, xr_s_base))
                        res["exits_s_base"].append(xr_s_base)
                        res["exits_s_base_notional"].append((xr_s_base, notional_in))
                        res["exits_s_base_pnl"].append((xr_s_base, max(pnl, 0)))
                        res["exits_s_base_pnl_all"].append((xr_s_base, pnl_abs))
                if not hold_slice.empty:
                    res["edges_s"].append(edge)
                    res["edges_s_notional"].append((edge, notional_in))
                    res["edges_s_pnl"].append((edge, max(pnl, 0)))
                    res["edges_s_pnl_all"].append((edge, pnl_abs))

        return res

    all_results = []
    if worker_use > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        worker_use = min(worker_use, max(1, multiprocessing.cpu_count() - 1))
        print(f'⚙️ 缓存齐全，启用多进程计算 workers={worker_use}', flush=True)
        # 预先把不可序列化的对象移除
        shared_args = {
            "T_GLOBAL": T_GLOBAL,
            "T_SHORT": T_SHORT,
        }
        def wrapper(code):
            return process_code(code)

        with ProcessPoolExecutor(max_workers=worker_use) as ex:
            futures = {ex.submit(wrapper, code): code for code in codes_sorted}
            for idx, fut in enumerate(as_completed(futures), 1):
                if idx % 100 == 0:
                    print(f'进度 {idx}/{len(codes_sorted)} ...', flush=True)
                res = fut.result()
                if res:
                    all_results.append(res)
    else:
        if missing_codes:
            print('⚠️ 存在缺失行情，需要串行下载+计算', flush=True)
        for idx, code in enumerate(codes_sorted, 1):
            if idx % 200 == 0:
                print(f'进度 {idx}/{len(codes_sorted)} ...', flush=True)
            res = process_code(code)
            if res:
                all_results.append(res)

    if lg is not None:
        bs.logout()

    # 聚合并行结果
    agg_keys = [
        "entries_g", "exits_g", "entries_s", "exits_s",
        "entries_g_base", "exits_g_base", "entries_s_base", "exits_s_base",
        "entries_g_notional", "exits_g_notional", "entries_s_notional", "exits_s_notional",
        "entries_g_base_notional", "exits_g_base_notional", "entries_s_base_notional", "exits_s_base_notional",
        "entries_g_pnl", "exits_g_pnl", "entries_s_pnl", "exits_s_pnl",
        "entries_g_pnl_all", "exits_g_pnl_all", "entries_s_pnl_all", "exits_s_pnl_all",
        "entries_g_base_pnl", "exits_g_base_pnl", "entries_s_base_pnl", "exits_s_base_pnl",
        "entries_g_base_pnl_all", "exits_g_base_pnl_all", "entries_s_base_pnl_all", "exits_s_base_pnl_all",
        "edges_g", "edges_g_notional", "edges_g_pnl", "edges_g_pnl_all",
        "edges_s", "edges_s_notional", "edges_s_pnl", "edges_s_pnl_all",
    ]
    merged = {k: [] for k in agg_keys}
    for res in all_results:
        for k in agg_keys:
            merged[k].extend(res.get(k, []))

    entries_g = merged["entries_g"]; exits_g = merged["exits_g"]
    entries_s = merged["entries_s"]; exits_s = merged["exits_s"]
    entries_g_base = merged["entries_g_base"]; exits_g_base = merged["exits_g_base"]
    entries_s_base = merged["entries_s_base"]; exits_s_base = merged["exits_s_base"]
    entries_g_notional = merged["entries_g_notional"]; exits_g_notional = merged["exits_g_notional"]
    entries_s_notional = merged["entries_s_notional"]; exits_s_notional = merged["exits_s_notional"]
    entries_g_base_notional = merged["entries_g_base_notional"]; exits_g_base_notional = merged["exits_g_base_notional"]
    entries_s_base_notional = merged["entries_s_base_notional"]; exits_s_base_notional = merged["exits_s_base_notional"]
    entries_g_pnl = merged["entries_g_pnl"]; exits_g_pnl = merged["exits_g_pnl"]
    entries_s_pnl = merged["entries_s_pnl"]; exits_s_pnl = merged["exits_s_pnl"]
    entries_g_pnl_all = merged["entries_g_pnl_all"]; exits_g_pnl_all = merged["exits_g_pnl_all"]
    entries_s_pnl_all = merged["entries_s_pnl_all"]; exits_s_pnl_all = merged["exits_s_pnl_all"]
    entries_g_base_pnl = merged["entries_g_base_pnl"]; exits_g_base_pnl = merged["exits_g_base_pnl"]
    entries_s_base_pnl = merged["entries_s_base_pnl"]; exits_s_base_pnl = merged["exits_s_base_pnl"]
    entries_g_base_pnl_all = merged["entries_g_base_pnl_all"]; exits_g_base_pnl_all = merged["exits_g_base_pnl_all"]
    entries_s_base_pnl_all = merged["entries_s_base_pnl_all"]; exits_s_base_pnl_all = merged["exits_s_base_pnl_all"]
    edges_g = merged["edges_g"]; edges_g_notional = merged["edges_g_notional"]; edges_g_pnl = merged["edges_g_pnl"]
    edges_g_pnl_all = merged["edges_g_pnl_all"]
    edges_s = merged["edges_s"]; edges_s_notional = merged["edges_s_notional"]; edges_s_pnl = merged["edges_s_pnl"]
    edges_s_pnl_all = merged["edges_s_pnl_all"]

    print('✅ 行情抓取与计算完成', flush=True)
    print('样本数: global entry/exit =', len(entries_g), len(exits_g), '; short entry/exit =', len(entries_s), len(exits_s))

    def unpack_weighted(lst):
        if not lst:
            return [], []
        vals, ws = zip(*lst)
        return list(vals), list(ws)

    hists = []
    def add_hist(key, title, data, weights=None, baseline_data=None, baseline_weights=None):
        h = summarize_hist(data, key, title, weights=weights, baseline_data=baseline_data, baseline_weights=baseline_weights)
        if h is not None:
            hists.append(h)

    add_hist('entries_g', f'全体交易 EntryRank (Tα={T_GLOBAL}分钟, 5min行情, 全量)', entries_g, baseline_data=entries_g_base)
    add_hist('exits_g', f'全体交易 ExitRank (Tα={T_GLOBAL}分钟, 5min行情, 全量)', exits_g, baseline_data=exits_g_base)
    add_hist('entries_s', f'超短单 EntryRank (持仓<=10分钟, Tα={T_SHORT}分钟, 5min行情)', entries_s, baseline_data=entries_s_base)
    add_hist('exits_s', f'超短单 ExitRank (持仓<=10分钟, Tα={T_SHORT}分钟, 5min行情)', exits_s, baseline_data=exits_s_base)

    ev, ew = unpack_weighted(entries_g_notional); evb, ewb = unpack_weighted(entries_g_base_notional); add_hist('entries_g_notional', '全体交易 EntryRank（成交金额加权）', ev, weights=ew, baseline_data=evb, baseline_weights=ewb)
    xv, xw = unpack_weighted(exits_g_notional); xvb, xwb = unpack_weighted(exits_g_base_notional); add_hist('exits_g_notional', '全体交易 ExitRank（成交金额加权）', xv, weights=xw, baseline_data=xvb, baseline_weights=xwb)
    evp, ewp = unpack_weighted(entries_g_pnl); evpb, ewpb = unpack_weighted(entries_g_base_pnl); add_hist('entries_g_pnl', '全体交易 EntryRank（PnL加权，盈利部分）', evp, weights=ewp, baseline_data=evpb, baseline_weights=ewpb)
    xvp, xwp = unpack_weighted(exits_g_pnl); xvpb, xwpb = unpack_weighted(exits_g_base_pnl); add_hist('exits_g_pnl', '全体交易 ExitRank（PnL加权，盈利部分）', xvp, weights=xwp, baseline_data=xvpb, baseline_weights=xwpb)
    evp_all, ewp_all = unpack_weighted(entries_g_pnl_all); evpb_all, ewpb_all = unpack_weighted(entries_g_base_pnl_all); add_hist('entries_g_pnl_all', '全体交易 EntryRank（PnL加权，含亏损）', evp_all, weights=ewp_all, baseline_data=evpb_all, baseline_weights=ewpb_all)
    xvp_all, xwp_all = unpack_weighted(exits_g_pnl_all); xvpb_all, xwpb_all = unpack_weighted(exits_g_base_pnl_all); add_hist('exits_g_pnl_all', '全体交易 ExitRank（PnL加权，含亏损）', xvp_all, weights=xwp_all, baseline_data=xvpb_all, baseline_weights=xwpb_all)

    evs, ews = unpack_weighted(entries_s_notional); evsb, ewsb = unpack_weighted(entries_s_base_notional); add_hist('entries_s_notional', '超短单 EntryRank（成交金额加权）', evs, weights=ews, baseline_data=evsb, baseline_weights=ewsb)
    xvs, xws = unpack_weighted(exits_s_notional); xvbs, xwbs = unpack_weighted(exits_s_base_notional); add_hist('exits_s_notional', '超短单 ExitRank（成交金额加权）', xvs, weights=xws, baseline_data=xvbs, baseline_weights=xwbs)
    evsp, ewsp = unpack_weighted(entries_s_pnl); evspb, ewspb = unpack_weighted(entries_s_base_pnl); add_hist('entries_s_pnl', '超短单 EntryRank（PnL加权，盈利部分）', evsp, weights=ewsp, baseline_data=evspb, baseline_weights=ewspb)
    xvsp, xwsp = unpack_weighted(exits_s_pnl); xvspb, xwspb = unpack_weighted(exits_s_base_pnl); add_hist('exits_s_pnl', '超短单 ExitRank（PnL加权，盈利部分）', xvsp, weights=xwsp, baseline_data=xvspb, baseline_weights=xwspb)
    evsp_all, ewsp_all = unpack_weighted(entries_s_pnl_all); evspb_all, ewspb_all = unpack_weighted(entries_s_base_pnl_all); add_hist('entries_s_pnl_all', '超短单 EntryRank（PnL加权，含亏损）', evsp_all, weights=ewsp_all, baseline_data=evspb_all, baseline_weights=ewspb_all)
    xvsp_all, xwsp_all = unpack_weighted(exits_s_pnl_all); xvspb_all, xwspb_all = unpack_weighted(exits_s_base_pnl_all); add_hist('exits_s_pnl_all', '超短单 ExitRank（PnL加权，含亏损）', xvsp_all, weights=xwsp_all, baseline_data=xvspb_all, baseline_weights=xwspb_all)

    add_hist('edge_g', '全体交易 Edge 捕获率（笔数）', edges_g)
    ev_edge, ew_edge = unpack_weighted(edges_g_notional); add_hist('edge_g_notional', '全体交易 Edge 捕获率（成交金额加权）', ev_edge, ew_edge)
    ev_edgep, ew_edgep = unpack_weighted(edges_g_pnl); add_hist('edge_g_pnl', '全体交易 Edge 捕获率（PnL加权，盈利部分）', ev_edgep, ew_edgep)
    ev_edgep_all, ew_edgep_all = unpack_weighted(edges_g_pnl_all); add_hist('edge_g_pnl_all', '全体交易 Edge 捕获率（PnL加权，含亏损）', ev_edgep_all, ew_edgep_all)

    add_hist('edge_s', '超短单 Edge 捕获率（笔数）', edges_s)
    ev_edge_s, ew_edge_s = unpack_weighted(edges_s_notional); add_hist('edge_s_notional', '超短单 Edge 捕获率（成交金额加权）', ev_edge_s, ew_edge_s)
    ev_edge_sp, ew_edge_sp = unpack_weighted(edges_s_pnl); add_hist('edge_s_pnl', '超短单 Edge 捕获率（PnL加权，盈利部分）', ev_edge_sp, ew_edge_sp)
    ev_edge_sp_all, ew_edge_sp_all = unpack_weighted(edges_s_pnl_all); add_hist('edge_s_pnl_all', '超短单 Edge 捕获率（PnL加权，含亏损）', ev_edge_sp_all, ew_edge_sp_all)

    sample_counts = {
        'entries_g': len(entries_g),
        'exits_g': len(exits_g),
        'entries_s': len(entries_s),
        'exits_s': len(exits_s),
    }
    payload = {
        "meta": {
            "t_global": T_GLOBAL,
            "t_short": T_SHORT,
            "sample_counts": sample_counts,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
        "hists": hists,
    }
    RESULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'💾 已写入结果缓存: {RESULT_CACHE}')

    stats_map = {h['key']: h.get('stats', {}) for h in hists}

hist_map = {h["key"]: h for h in hists}

stats_e_g = stats_map.get("entries_g", {})
stats_x_g = stats_map.get("exits_g", {})
perfect_e = perfect_share(hist_map.get("entries_g"))
perfect_x = perfect_share(hist_map.get("exits_g"))


def build_mode_figs(is_short=False):
    modes = []
    prefix = "short" if is_short else "overall"
    tg = T_SHORT_USE if is_short else T_GLOBAL_USE
    def add_mode(mode_key, label, fig):
        visible = len(modes) == 0
        div_html, script = fig_to_div_script(fig, mode_key, visible=visible)
        plot_scripts.append(script)
        modes.append((mode_key, label, div_html))
    # rank - 笔数
    key_e = "entries_s" if is_short else "entries_g"
    key_x = "exits_s" if is_short else "exits_g"
    if key_e in hist_map and key_x in hist_map:
        fig = paired_hist_fig(
            f"{'超短单' if is_short else '全体交易'} Entry / Exit Rank（窗口 Tα={tg} 分钟，5min 行情）",
            hist_map[key_e],
            hist_map[key_x],
        )
        add_mode(f"{prefix}_rank_counts", "Rank·笔数", fig)
    # rank - 金额加权
    key_e_n = "entries_s_notional" if is_short else "entries_g_notional"
    key_x_n = "exits_s_notional" if is_short else "exits_g_notional"
    if key_e_n in hist_map and key_x_n in hist_map:
        fig = paired_hist_fig(
            f"{'超短单' if is_short else '全体交易'} Entry / Exit Rank（成交金额加权，Tα={tg} 分钟）",
            hist_map[key_e_n],
            hist_map[key_x_n],
        )
        add_mode(f"{prefix}_rank_notional", "Rank·金额权重", fig)
    # rank - PnL加权（盈利部分）
    key_e_p = "entries_s_pnl" if is_short else "entries_g_pnl"
    key_x_p = "exits_s_pnl" if is_short else "exits_g_pnl"
    if key_e_p in hist_map and key_x_p in hist_map:
        fig = paired_hist_fig(
            f"{'超短单' if is_short else '全体交易'} Entry / Exit Rank（PnL加权，盈利部分，Tα={tg} 分钟）",
            hist_map[key_e_p],
            hist_map[key_x_p],
        )
        add_mode(f"{prefix}_rank_pnl", "Rank·PnL权重", fig)
    # rank - PnL加权（含亏损）
    key_e_p_all = "entries_s_pnl_all" if is_short else "entries_g_pnl_all"
    key_x_p_all = "exits_s_pnl_all" if is_short else "exits_g_pnl_all"
    if key_e_p_all in hist_map and key_x_p_all in hist_map:
        fig = paired_hist_fig(
            f"{'超短单' if is_short else '全体交易'} Entry / Exit Rank（PnL加权，含亏损，Tα={tg} 分钟）",
            hist_map[key_e_p_all],
            hist_map[key_x_p_all],
        )
        add_mode(f"{prefix}_rank_pnl_all", "Rank·PnL含亏损", fig)
    # Edge
    key_edge = "edge_s" if is_short else "edge_g"
    if key_edge in hist_map:
        fig = single_hist_fig(
            f"{'超短单' if is_short else '全体交易'} Edge 捕获率（笔数，持仓窗口内波动覆盖度）",
            hist_map[key_edge],
            color="#6366f1",
        )
        add_mode(f"{prefix}_edge_counts", "Edge·笔数", fig)
    key_edge_n = "edge_s_notional" if is_short else "edge_g_notional"
    if key_edge_n in hist_map:
        fig = single_hist_fig(
            f"{'超短单' if is_short else '全体交易'} Edge 捕获率（成交金额加权）",
            hist_map[key_edge_n],
            color="#4338ca",
        )
        add_mode(f"{prefix}_edge_notional", "Edge·金额权重", fig)
    key_edge_p = "edge_s_pnl" if is_short else "edge_g_pnl"
    if key_edge_p in hist_map:
        fig = single_hist_fig(
            f"{'超短单' if is_short else '全体交易'} Edge 捕获率（PnL加权，盈利部分）",
            hist_map[key_edge_p],
            color="#1f2937",
        )
        add_mode(f"{prefix}_edge_pnl", "Edge·PnL权重", fig)
    key_edge_p_all = "edge_s_pnl_all" if is_short else "edge_g_pnl_all"
    if key_edge_p_all in hist_map:
        fig = single_hist_fig(
            f"{'超短单' if is_short else '全体交易'} Edge 捕获率（PnL加权，含亏损）",
            hist_map[key_edge_p_all],
            color="#111827",
        )
        add_mode(f"{prefix}_edge_pnl_all", "Edge·PnL含亏损", fig)
    return modes


def render_modes_block(title, subtitle, modes, stats_entry, stats_exit, block_id):
    if not modes:
        return ""
    buttons = "\n".join(
        [
            f"<button class=\"mode-btn{' active' if i==0 else ''}\" data-target=\"{block_id}\" data-mode=\"{m[0]}\">{m[1]}</button>"
            for i, m in enumerate(modes)
        ]
    )
    charts = "\n".join(
        [
            m[2]
            for _, m in enumerate(modes)
        ]
    )
    return f"""
      <div class="space-y-3 rounded-2xl border border-slate-200 bg-white/90 shadow-sm p-4">
        <div class="flex items-start justify-between">
          <div>
            <div class="text-base font-semibold text-slate-900">{title}</div>
            <div class="text-sm text-slate-500">{subtitle}</div>
          </div>
        </div>
        <div class="flex flex-wrap gap-2 mb-2">{buttons}</div>
        <div class="chart" id="{block_id}">{charts}</div>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div class="rounded-xl border border-emerald-100 bg-emerald-50/60 px-3 py-2">
            <div class="text-xs font-semibold text-emerald-700">Entry</div>
            <div class="text-sm text-slate-900">{stats_entry}</div>
          </div>
          <div class="rounded-xl border border-rose-100 bg-rose-50/60 px-3 py-2">
            <div class="text-xs font-semibold text-rose-700">Exit</div>
            <div class="text-sm text-slate-900">{stats_exit}</div>
          </div>
        </div>
      </div>
    """


modes_overall = build_mode_figs(is_short=False)
modes_short = build_mode_figs(is_short=True)

fig_blocks = []
fig_blocks.append(
    render_modes_block(
        "全体交易",
        f"窗口 Tα={T_GLOBAL_USE} 分钟，Rank / Edge 支持多种权重视角（按钮切换）",
        modes_overall,
        format_stats(stats_map.get("entries_g")),
        format_stats(stats_map.get("exits_g")),
        "chart-overall",
    )
)
fig_blocks.append(
    render_modes_block(
        "超短单（持仓≤10 分钟）",
        f"窗口 Tα={T_SHORT_USE} 分钟，聚焦撮合速度与极短期行情偏移（按钮切换权重视角）",
        modes_short,
        format_stats(stats_map.get("entries_s")),
        format_stats(stats_map.get("exits_s")),
        "chart-short",
    )
)
charts_html = "\n".join(fig_blocks)

plot_scripts_js = "\n".join(plot_scripts)

script_block = f"""
  <script>
    {plot_scripts_js}
    window.addEventListener('load', () => {{
      if (!location.hash) {{
        const el = document.getElementById('charts');
        if (el) el.scrollIntoView({{behavior:'auto', block:'start'}});
      }}
      const initGroups = () => {{
        const groups = new Set();
        document.querySelectorAll('.mode-btn').forEach(btn => groups.add(btn.getAttribute('data-target')));
        groups.forEach(g => {{
          let active = document.querySelector(".mode-btn[data-target='" + g + "'].active");
          if (!active) {{
            active = document.querySelector(".mode-btn[data-target='" + g + "']");
            if (active) active.classList.add('active');
          }}
          const mode = active ? active.getAttribute('data-mode') : null;
          document.querySelectorAll('#' + g + ' .mode-chart').forEach(div => {{
            const show = div.getAttribute('data-mode') === mode;
            div.style.display = show ? 'block' : 'none';
            if (show && window.Plotly && div.id) {{
              window.Plotly.Plots.resize(div);
            }}
          }});
        }});
      }};
      initGroups();
      document.querySelectorAll('.mode-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
          const target = btn.getAttribute('data-target');
          const mode = btn.getAttribute('data-mode');
          document.querySelectorAll(".mode-btn[data-target='" + target + "']").forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          document.querySelectorAll('#' + target + ' .mode-chart').forEach(div => {{
            const show = div.getAttribute('data-mode') === mode;
            div.style.display = show ? 'block' : 'none';
            if (show && window.Plotly && div.id) {{
              window.Plotly.Plots.resize(div);
            }}
          }});
        }});
      }});
      window.addEventListener('resize', () => {{
        document.querySelectorAll('.mode-chart').forEach(div => {{
          if (div.style.display !== 'none' && window.Plotly && div.id) {{
            window.Plotly.Plots.resize(div);
          }}
        }});
      }});
      // 初始强制 resize 确保首次渲染
      if (window.Plotly) {{
        document.querySelectorAll('.mode-chart').forEach(div => {{
          if (div.style.display !== 'none' && div.id) {{
            window.Plotly.Plots.resize(div);
          }}
        }});
      }}
    }});
  </script>
"""

REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)



html_template = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>择时能力分布（baostock 5min，全量）</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.27.1.min.js"></script>
  <!-- MathJax for LaTeX rendering -->
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true
      },
      options: {
        ignoreHtmlClass: 'tex2jax_ignore',
        processHtmlClass: 'tex2jax_process'
      }
    };
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    body { font-family: "Noto Sans SC", "Inter", "Helvetica", "Arial", sans-serif; }
    .mode-btn { padding: 6px 10px; border-radius: 999px; font-size: 13px; border: 1px solid #e2e8f0; background: #f8fafc; color: #475569; cursor: pointer; }
    .mode-btn:hover { background: #eef2ff; }
    .mode-btn.active { background: #eef2ff; color: #4338ca; border-color: #c7d2fe; }
    .mode-chart { width: 100%; }
    .math-block { text-align: left; margin: 6px 0; }
    /* Fix MathJax font size in Tailwind context */
    mjx-container { font-size: 1.1em !important; }
  </style>
</head>
<body>
  <div class="bg-slate-50 min-h-screen">
    <header class="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div class="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        <div>
          <div class="text-xs font-semibold text-indigo-600">交易执行分析</div>
          <h1 class="text-xl font-bold text-slate-900">择时能力分布（Entry/Exit Rank）</h1>
        </div>
        <div class="text-[11px] text-slate-500 text-right leading-tight">行情：baostock 5min<br/>窗口：全体 Tα=__T_GLOBAL__ 分钟｜超短 Tα=__T_SHORT__ 分钟</div>
      </div>
    </header>

    <div class="max-w-7xl mx-auto p-6 space-y-6">
      <div class="rounded-2xl border border-slate-200 bg-white shadow-sm p-5">
        <div class="text-sm text-slate-700 leading-relaxed">
          - 目标：评估买入/卖出点在后续 Tα 分钟行情窗口内的相对位置，诊断择时优劣。<br/>
          - Rank∈[0,1]：越接近 0 表示更优（买得更低 / 卖得更高），空头已镜像为可比方向。<br/>
          - 口径：全体交易窗口 Tα=__T_GLOBAL__ 分钟；超短单（持仓≤10 分钟）窗口 Tα=__T_SHORT__ 分钟。
        </div>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="rounded-2xl border border-green-200 bg-green-50 shadow-sm p-4">
          <div class="text-xs font-semibold text-green-700 uppercase tracking-wide">全体 Entry 均值</div>
          <div class="text-2xl font-bold text-green-800 mt-1">__STATS_E_G_MEAN__</div>
          <div class="text-sm text-green-700 mt-1">越低越好，随机基准 ≈ 0.50</div>
        </div>
        <div class="rounded-2xl border border-red-200 bg-red-50 shadow-sm p-4">
          <div class="text-xs font-semibold text-red-700 uppercase tracking-wide">全体 Exit 均值</div>
          <div class="text-2xl font-bold text-red-800 mt-1">__STATS_X_G_MEAN__</div>
          <div class="text-sm text-red-700 mt-1">越低越好，随机基准 ≈ 0.50</div>
        </div>
        <div class="rounded-2xl border border-green-200 bg-white shadow-sm p-4">
          <div class="text-xs font-semibold text-green-700 uppercase tracking-wide">完美买入占比 (Rank&lt;0.1)</div>
          <div class="text-2xl font-bold text-slate-900 mt-1">__PERFECT_E__</div>
          <div class="text-sm text-slate-500 mt-1">随机基准 ≈ 10%</div>
        </div>
        <div class="rounded-2xl border border-red-200 bg-white shadow-sm p-4">
          <div class="text-xs font-semibold text-red-700 uppercase tracking-wide">完美卖出占比 (Rank&lt;0.1)</div>
          <div class="text-2xl font-bold text-slate-900 mt-1">__PERFECT_X__</div>
          <div class="text-sm text-slate-500 mt-1">随机基准 ≈ 10%</div>
        </div>
      </div>

      <a id="charts"></a>
      <div class="space-y-4">
        __CHARTS_HTML__
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div class="rounded-xl border border-slate-200 bg-white px-3 py-2">
            <div class="text-xs text-slate-500">全体 Entry 样本</div>
            <div class="text-lg font-semibold text-slate-900">__SAMPLE_E_G__</div>
          </div>
          <div class="rounded-xl border border-slate-200 bg-white px-3 py-2">
            <div class="text-xs text-slate-500">全体 Exit 样本</div>
            <div class="text-lg font-semibold text-slate-900">__SAMPLE_X_G__</div>
          </div>
          <div class="rounded-xl border border-slate-200 bg-white px-3 py-2">
            <div class="text-xs text-slate-500">超短 Entry 样本</div>
            <div class="text-lg font-semibold text-slate-900">__SAMPLE_E_S__</div>
          </div>
          <div class="rounded-xl border border-slate-200 bg-white px-3 py-2">
            <div class="text-xs text-slate-500">超短 Exit 样本</div>
            <div class="text-lg font-semibold text-slate-900">__SAMPLE_X_S__</div>
          </div>
        </div>
        <p class="text-sm text-slate-600 leading-relaxed">
          说明：Rank 以交易后续 5 分钟行情区间的相对位置度量；全体/超短分别使用 Tα=__T_GLOBAL__/__T_SHORT__ 分钟窗口，空头已镜像，便于与多头可比。
        </p>
      </div>

      <div class="rounded-2xl border border-slate-200 bg-white shadow-sm p-4 space-y-4">
        <div>
            <div class="text-base font-semibold text-slate-900 mb-2">页面目的</div>
            <p class="text-slate-700 leading-relaxed text-sm">
                本页面旨在定量评估策略在交易执行层面的择时能力（Market Timing Ability）。通过统计模型在实际交易发生后的一定时间窗口内，成交价格相对于该窗口内市场极值（最高价与最低价）的位置分布，来判断策略是否具备捕捉短期市场拐点的能力。
                <br/><br/>
                指标 <strong>Rank</strong> 越接近 0，表明买入点越接近局部最低价（或卖出点越接近局部最高价），择时能力越强；反之，若 Rank 分布接近 0.5 或服从均匀分布，则说明策略在执行层面不具备显著的择时优势（即类似于随机入场）。
            </p>
        </div>
        
        <div>
            <div class="text-base font-semibold text-slate-900 mb-2">实现方式</div>
            <div class="text-slate-700 text-sm leading-relaxed space-y-3">
                <p>核心指标 <strong>Entry Rank</strong>（开仓择时得分）与 <strong>Exit Rank</strong>（平仓择时得分）使用统一窗口与极值位置计算：</p>
                <ul class="list-disc pl-5 space-y-3">
                  <li><strong>时间窗口（入场/出场分开评估）</strong>：Entry 窗口 \([t_{\text{open}},\, t_{\text{close}}]\)，Exit 窗口 \([t_{\text{open}},\, t_{\text{close}} + 0.5\,T_{\alpha}]\)；\(t_{\text{open}}/t_{\text{close}}\) 为开/平仓时间，\(T_{\alpha}\) 取自持仓时长分位数（全体 \(=__T_GLOBAL__\,\text{min}\)，超短 \(=__T_SHORT__\,\text{min}\)），同一标的按前一笔平仓时刻（prev_close）与下一笔开仓时刻（next_open）裁剪窗口，避免跨越相邻持仓。</li>
                  <li><strong>Rank 公式（多头）</strong>：
                    <div class="math-block">$$
                    \begin{aligned}
                    \text{EntryRank} &= \frac{P_{\text{buy}} - P_{\text{low}}}{P_{\text{high}} - P_{\text{low}}},\\\\
                    \text{ExitRank}  &= \frac{P_{\text{high}} - P_{\text{sell}}}{P_{\text{high}} - P_{\text{low}}}
                    \end{aligned}
                    $$</div>
                    其中 \(P_{\text{high}}, P_{\text{low}}\) 取自对应窗口（Entry/Exit）内的极值。空头镜像（卖出开仓与买入平仓对调）：
                    <div class="math-block">$$
                    \begin{aligned}
                    \text{EntryRank}_{\text{short}} &= \frac{P_{\text{high}} - P_{\text{sell}}}{P_{\text{high}} - P_{\text{low}}},\\\\
                    \text{ExitRank}_{\text{short}}  &= \frac{P_{\text{buy}} - P_{\text{low}}}{P_{\text{high}} - P_{\text{low}}}
                    \end{aligned}
                    $$</div>
                    若 \(P_{\text{high}} = P_{\text{low}}\) 则取 0.5；所有结果裁剪到 \([0,1]\)。
                  </li>
                  <li><strong>Edge 捕获率</strong>：实际持仓区间 \([t_{\text{open}},\, t_{\text{close}}]\) 上，
                    <div class="math-block">$$
                    \text{Edge} = \frac{P_{\text{close}} - P_{\text{open}}}{P^{\text{hold}}_{\text{high}} - P^{\text{hold}}_{\text{low}}}
                    $$</div>
                    空头镜像后裁剪到 \([0,1]\)，衡量吃到的波动占比。
                  </li>
                  <li><strong>加权视角</strong>：直方图支持笔数、成交金额、PnL 两种口径（<em>盈利部分</em>：max(PnL,0)，<em>含亏损</em>：|PnL|）权重，按钮切换；基准线优先使用<strong>经验基准</strong>（在同一窗口随机抽取一根 5 分钟 K 线，并在其 High/Low 间随机取价计算 Rank，累积成分布），若样本不足则回退为均匀基准 \(1/\text{bins}\)；同时叠加中位数虚线，便于对照是否优于随机择时。</li>
                  <li><strong>行情口径</strong>：全部使用 5 分钟 K 线提取窗口内的 High/Low 极值，空头已镜像为“买低卖高”方向以便可比。</li>
                </ul>
            </div>
        </div>

        <div class="pt-2 border-t border-slate-100">
            <div class="text-base font-semibold text-slate-900 mb-2">相关页面</div>
            <ul class="text-sm text-indigo-700 leading-relaxed list-disc pl-5 space-y-1">
              <li><a class="underline hover:text-indigo-500" target="_blank" rel="noopener noreferrer" href="daily_returns_comparison_light.html">日收益率对比</a></li>
              <li><a class="underline hover:text-indigo-500" target="_blank" rel="noopener noreferrer" href="pred_real_relationship_light.html">预测值与实际收益关系分析</a></li>
              <li><a class="underline hover:text-indigo-500" target="_blank" rel="noopener noreferrer" href="intraday_avg_holding_time_light.html">交易平均持仓时间</a></li>
            </ul>
        </div>
      </div>
    </div>
  </div>
  __SCRIPT_BLOCK__

</body>
</html>
"""

html_text = html_template.replace("__T_GLOBAL__", str(T_GLOBAL_USE)) \
    .replace("__T_SHORT__", str(T_SHORT_USE)) \
    .replace("__STATS_E_G_MEAN__", format_two_decimals(stats_e_g.get('mean') if stats_e_g else None)) \
    .replace("__STATS_X_G_MEAN__", format_two_decimals(stats_x_g.get('mean') if stats_x_g else None)) \
    .replace("__PERFECT_E__", format_pct(perfect_e)) \
    .replace("__PERFECT_X__", format_pct(perfect_x)) \
    .replace("__CHARTS_HTML__", charts_html) \
    .replace("__SAMPLE_E_G__", f"{sample_counts.get('entries_g', 0):,}") \
    .replace("__SAMPLE_X_G__", f"{sample_counts.get('exits_g', 0):,}") \
    .replace("__SAMPLE_E_S__", f"{sample_counts.get('entries_s', 0):,}") \
    .replace("__SAMPLE_X_S__", f"{sample_counts.get('exits_s', 0):,}") \
    .replace("__SCRIPT_BLOCK__", script_block)

REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
REPORT_HTML.write_text(html_text, encoding='utf-8')
REPORT_TXT.write_text(
    f'global_entry={sample_counts.get("entries_g",0)}, global_exit={sample_counts.get("exits_g",0)}, short_entry={sample_counts.get("entries_s",0)}, short_exit={sample_counts.get("exits_s",0)}\n',
    encoding='utf-8'
)
print('🎯 输出:', REPORT_HTML, REPORT_TXT)
# 复制到可视化/发布目录，方便 iframe 引用
for tgt in COPY_HTML_TARGETS:
    try:
        tgt.parent.mkdir(parents=True, exist_ok=True)
        tgt.write_text(REPORT_HTML.read_text(encoding='utf-8'), encoding='utf-8')
    except Exception as e:
        print(f'⚠️ 拷贝到 {tgt} 失败: {e}')
