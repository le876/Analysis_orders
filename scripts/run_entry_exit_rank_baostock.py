#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量使用 baostock 5 分钟行情计算 EntryRank / ExitRank（全量标的），带行情缓存 + 结果缓存
- 全体交易: Tα_global=234 分钟
- 超短单(持仓<=10分钟): Tα_short=5 分钟
- 行情缓存: data/cache/baostock_5min/{code}.parquet（存在则复用，不再拉取）
- 结果缓存: data/cache/entry_exit_rank_baostock_result.json（存在则直接生成页面；如算法或参数改动，请 --recompute）
输出:
- reports/entry_exit_rank_baostock_full.html (直方图页面)
- reports/entry_exit_rank_baostock_full.txt  (样本计数)
运行方式（在仓库根目录）:
HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= /home/ubuntu/.conda/envs/quant_env/bin/python scripts/run_entry_exit_rank_baostock.py
强制重算（忽略结果缓存）:
HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= /home/ubuntu/.conda/envs/quant_env/bin/python scripts/run_entry_exit_rank_baostock.py --recompute
"""
import argparse
import baostock as bs
import json
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import plotly.graph_objects as go

T_GLOBAL = 234      # 全体交易窗口（分钟）
T_SHORT = 5         # 超短单窗口（分钟）
PAIRS_PATH = Path('data/paired_trades_fifo.parquet')
REPORT_HTML = Path('reports/entry_exit_rank_baostock_full.html')
REPORT_TXT = Path('reports/entry_exit_rank_baostock_full.txt')
COPY_HTML_TARGETS = [
    Path('reports/visualization_analysis/entry_exit_rank_baostock_full.html'),
    Path('docs/entry_exit_rank_baostock_full.html'),
]
CACHE_DIR = Path('data/cache/baostock_5min')
RESULT_CACHE = Path('data/cache/entry_exit_rank_baostock_result.json')

parser = argparse.ArgumentParser(description='计算 Entry/ExitRank (baostock 5min)')
parser.add_argument('--recompute', action='store_true', help='忽略结果缓存，重新计算')
args = parser.parse_args()
use_result_cache = RESULT_CACHE.exists() and (not args.recompute)


def summarize_hist(data, key, title, bins=30):
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    counts, edges = np.histogram(arr, bins=bins, range=(0, 1))
    stats = {
        "size": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }
    return {
        "key": key,
        "title": title,
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "stats": stats,
    }


def fig_from_hist(hist):
    edges = np.asarray(hist["edges"], dtype=float)
    counts = np.asarray(hist["counts"], dtype=float)
    total = counts.sum()
    probs = counts / total if total > 0 else counts
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    st = hist["stats"]
    stats_str = f"样本: {st['size']:,} | 均值: {st['mean']:.3f} | 中位数: {st['median']:.3f} | P25/P75: {st['p25']:.3f}/{st['p75']:.3f}"
    fig = go.Figure(go.Bar(x=centers, y=probs, width=widths, marker=dict(color='teal')))
    fig.update_layout(
        title=f"{hist['title']}<br><sub>{stats_str}</sub>",
        xaxis_title='Rank (0=好)',
        yaxis_title='比例',
        bargap=0.05,
    )
    return fig


def format_stats(stats):
    if not stats or stats.get("size", 0) == 0:
        return "无数据"
    return f"样本 {stats['size']:,} | 均值 {stats['mean']:.3f} | 中位数 {stats['median']:.3f} | P25/P75 {stats['p25']:.3f}/{stats['p75']:.3f}"


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
figs = []
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
    figs = [fig_from_hist(h) for h in hists]
else:
    print('✅ 加载配对交易数据...', flush=True)
    pairs = pd.read_parquet(PAIRS_PATH, columns=['code', 'trade_type', 'buy_timestamp', 'sell_timestamp', 'buy_price', 'sell_price'])
    for col in ['buy_timestamp', 'sell_timestamp']:
        if not pd.api.types.is_datetime64_any_dtype(pairs[col]):
            pairs[col] = pd.to_datetime(pairs[col])

    short_mask = pairs['trade_type'] == 'short'
    pairs['open_timestamp'] = pairs['buy_timestamp'].where(~short_mask, pairs['sell_timestamp'])
    pairs['close_timestamp'] = pairs['sell_timestamp'].where(~short_mask, pairs['buy_timestamp'])
    pairs['open_price'] = pairs['buy_price'].where(~short_mask, pairs['sell_price'])
    pairs['close_price'] = pairs['sell_price'].where(~short_mask, pairs['buy_price'])
    pairs['holding_minutes_trading'] = [trading_minutes(o, c) for o, c in zip(pairs['open_timestamp'], pairs['close_timestamp'])]

    # 按标的构建日期范围，按交易条数排序
    code_ranges = {}
    for code, g in pairs.groupby('code'):
        start = g[['open_timestamp', 'close_timestamp']].min().min().date()
        end = g[['open_timestamp', 'close_timestamp']].max().max().date()
        code_ranges[code] = (start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    codes_sorted = sorted(code_ranges.keys(), key=lambda x: len(pairs[pairs['code'] == x]), reverse=True)
    print(f'📈 标的数量: {len(codes_sorted)}', flush=True)

    entries_g = []; exits_g = []; entries_s = []; exits_s = []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_files = {code: CACHE_DIR / f"{code.replace('.', '_')}.parquet" for code in codes_sorted}
    missing_codes = [c for c in codes_sorted if not cache_files[c].exists()]

    lg = None
    if missing_codes:
        print('🔌 登录 baostock...', flush=True)
        lg = bs.login()
        if lg.error_code != '0':
            raise SystemExit('baostock login failed: ' + lg.error_msg)
    else:
        print('🗄️ 缓存齐全，跳过行情下载，直接读取本地。', flush=True)

    for idx, code in enumerate(codes_sorted, 1):
        if idx % 200 == 0:
            print(f'进度 {idx}/{len(codes_sorted)} ...', flush=True)
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
                continue
            cdf = pd.DataFrame(data, columns=rs.fields)
            cdf['datetime'] = pd.to_datetime(cdf['date'] + ' ' + cdf['time'].str[8:10] + ':' + cdf['time'].str[10:12] + ':' + cdf['time'].str[12:14])
            cdf[['open', 'high', 'low', 'close']] = cdf[['open', 'high', 'low', 'close']].astype(float)
            cdf.set_index('datetime', inplace=True)
            md = cdf[['high', 'low', 'open', 'close']]
            md.to_parquet(cache_file, index=True)
        else:
            continue

        trades = pairs[pairs['code'] == code]
        for _, row in trades.iterrows():
            # 全体窗口
            es = md.loc[(md.index >= row['open_timestamp']) & (md.index <= row['open_timestamp'] + timedelta(minutes=T_GLOBAL))]
            if not es.empty:
                lo, hi = es['low'].min(), es['high'].max()
                er = 0.5 if hi == lo else ((hi - row['open_price']) / (hi - lo) if row['trade_type'] == 'short' else (row['open_price'] - lo) / (hi - lo))
                entries_g.append(er)
            xs = md.loc[(md.index >= row['close_timestamp']) & (md.index <= row['close_timestamp'] + timedelta(minutes=T_GLOBAL))]
            if not xs.empty:
                lo2, hi2 = xs['low'].min(), xs['high'].max()
                xr = 0.5 if hi2 == lo2 else ((row['close_price'] - lo2) / (hi2 - lo2) if row['trade_type'] == 'short' else (hi2 - row['close_price']) / (hi2 - lo2))
                exits_g.append(xr)
            # 超短单窗口
            if row['holding_minutes_trading'] <= 10:
                es_s = md.loc[(md.index >= row['open_timestamp']) & (md.index <= row['open_timestamp'] + timedelta(minutes=T_SHORT))]
                if not es_s.empty:
                    loS, hiS = es_s['low'].min(), es_s['high'].max()
                    er_s = 0.5 if hiS == loS else ((hiS - row['open_price']) / (hiS - loS) if row['trade_type'] == 'short' else (row['open_price'] - loS) / (hiS - loS))
                    entries_s.append(er_s)
                xs_s = md.loc[(md.index >= row['close_timestamp']) & (md.index <= row['close_timestamp'] + timedelta(minutes=T_SHORT))]
                if not xs_s.empty:
                    loS2, hiS2 = xs_s['low'].min(), xs_s['high'].max()
                    xr_s = 0.5 if hiS2 == loS2 else ((row['close_price'] - loS2) / (hiS2 - loS2) if row['trade_type'] == 'short' else (hiS2 - row['close_price']) / (hiS2 - loS2))
                    exits_s.append(xr_s)

    if lg is not None:
        bs.logout()
    print('✅ 行情抓取与计算完成', flush=True)
    print('样本数: global entry/exit =', len(entries_g), len(exits_g), '; short entry/exit =', len(entries_s), len(exits_s))

    hists = []
    for key, title, data in [
        ('entries_g', f'全体交易 EntryRank (Tα={T_GLOBAL}分钟, 5min行情, 全量)', entries_g),
        ('exits_g', f'全体交易 ExitRank (Tα={T_GLOBAL}分钟, 5min行情, 全量)', exits_g),
        ('entries_s', f'超短单 EntryRank (持仓<=10分钟, Tα={T_SHORT}分钟, 5min行情)', entries_s),
        ('exits_s', f'超短单 ExitRank (持仓<=10分钟, Tα={T_SHORT}分钟, 5min行情)', exits_s),
    ]:
        h = summarize_hist(data, key, title)
        if h is not None:
            hists.append(h)

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
    figs = [fig_from_hist(h) for h in hists]

fig_html_parts = [
    f.to_html(full_html=False, include_plotlyjs='cdn', default_width='100%', default_height='420px')
    for f in figs
]
charts_html = "\n".join(f"<div class='chart'>{h}</div>" for h in fig_html_parts)

REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)

html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>择时能力分布（baostock 5min，全量）</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    :root {{ --bg: #f6f7fb; --card: #fff; --text: #1f2937; --muted: #6b7280; --shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    body {{ margin: 0; padding: 0; font-family: "Helvetica", "Arial", sans-serif; background: var(--bg); color: var(--text); }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 20px; display: grid; gap: 14px; }}
    .card {{ background: var(--card); border-radius: 12px; padding: 14px 16px; box-shadow: var(--shadow); }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
    h2 {{ margin: 0 0 10px 0; font-size: 17px; color: var(--text); }}
    p {{ margin: 6px 0; line-height: 1.6; color: #374151; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }}
    .badge {{ display: inline-flex; align-items: center; padding: 2px 10px; border-radius: 999px; background: #e0f2fe; color: #1d4ed8; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .stat {{ background: #f9fafb; border-radius: 10px; padding: 10px 12px; border: 1px solid #e5e7eb; }}
    .stat .title {{ font-size: 13px; color: var(--muted); margin-bottom: 4px; }}
    .stat .value {{ font-size: 20px; font-weight: 700; color: var(--text); }}
    .stat .small {{ font-size: 13px; color: #374151; font-weight: 500; line-height: 1.5; }}
    .chart {{ margin-top: 10px; }}
    .section-title {{ font-weight: 700; margin-bottom: 6px; }}
    .note {{ font-size: 13px; color: #4b5563; margin-top: 6px; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1>⚡ 交易执行分析｜择时能力分布</h1>
      <div class="badges">
        <span class="badge">Entry/ExitRank</span>
        <span class="badge">baostock 5min</span>
        <span class="badge">Tα 全体 {T_GLOBAL_USE} 分钟</span>
        <span class="badge">Tα 超短 {T_SHORT_USE} 分钟</span>
      </div>
      <p>口径：全体交易使用窗口 Tα={T_GLOBAL_USE} 分钟；超短单（持仓≤10 分钟）使用窗口 Tα={T_SHORT_USE} 分钟。Rank∈[0,1]，越接近 0 说明择时越好；空头已镜像处理。</p>
    </div>

    <div class="card">
      <h2>样本与概览</h2>
      <div class="grid">
        <div class="stat"><div class="title">全体 Entry 样本</div><div class="value">{sample_counts.get('entries_g', 0):,}</div></div>
        <div class="stat"><div class="title">全体 Exit 样本</div><div class="value">{sample_counts.get('exits_g', 0):,}</div></div>
        <div class="stat"><div class="title">超短 Entry 样本</div><div class="value">{sample_counts.get('entries_s', 0):,}</div></div>
        <div class="stat"><div class="title">超短 Exit 样本</div><div class="value">{sample_counts.get('exits_s', 0):,}</div></div>
      </div>
      <div class="grid" style="margin-top:10px;">
        <div class="stat"><div class="title">全体 Entry 统计</div><div class="small">{format_stats(stats_map.get('entries_g'))}</div></div>
        <div class="stat"><div class="title">全体 Exit 统计</div><div class="small">{format_stats(stats_map.get('exits_g'))}</div></div>
        <div class="stat"><div class="title">超短 Entry 统计</div><div class="small">{format_stats(stats_map.get('entries_s'))}</div></div>
        <div class="stat"><div class="title">超短 Exit 统计</div><div class="small">{format_stats(stats_map.get('exits_s'))}</div></div>
      </div>
      <p class="note">说明：指标基于交易时段分钟数；空头价格已镜像，保证 Rank 可比。</p>
    </div>

    <div class="card">
      <h2>分布直方图</h2>
      <p class="note">采用预聚合分箱，页面轻量可直接嵌入 iframe。</p>
      {charts_html}
    </div>

    <div class="card">
      <h2 class="section-title">实现方式</h2>
      <p>数据来源：订单配对 data/paired_trades_fifo.parquet；行情来源：baostock 5min，缓存目录 data/cache/baostock_5min（若文件存在则复用，不再请求）。</p>
      <p>计算口径：全体交易窗口 Tα={T_GLOBAL_USE} 分钟；超短单（持仓≤10 分钟）窗口 Tα={T_SHORT_USE} 分钟。Entry/ExitRank ∈[0,1]，0=择时佳、1=择时差，空头方向已镜像。</p>
      <h2 class="section-title">制作目的</h2>
      <p>用于“交易执行分析”板块诊断全体与超短单的择时分布，支持后续与指数或基准盘面横向对比。</p>
      <p class="note">如需更新，运行 scripts/run_entry_exit_rank_baostock.py（自动生成并复制到 reports/visualization_analysis/ 与 docs/；若算法或窗口改动请加 --recompute 或删除结果缓存 {RESULT_CACHE}）。</p>
    </div>
  </div>
</body>
</html>
"""

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
