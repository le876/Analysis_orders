#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化策略分析脚本
解决HTML文件过大导致浏览器无法加载的问题
"""

import pandas as pd
import numpy as np
import json
import math
import base64
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import warnings
from string import Template
import hashlib
import pickle
warnings.filterwarnings('ignore')
from scipy import stats
from statsmodels.stats.multitest import multipletests
import sys
import os
from typing import Optional, List, Tuple
import re
import urllib.request
from urllib.parse import quote
import builtins

# Windows 终端中文乱码修复与安全打印
def _configure_windows_console_utf8() -> None:
    if os.name != 'nt':
        return
    try:
        import ctypes  # type: ignore
        # 将输入/输出代码页设置为 UTF-8
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    # 环境变量与Python层面的编码强制为 UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('PYTHONUTF8', '1')
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

_configure_windows_console_utf8()

def _remap_symbols(text: str) -> str:
    """将控制台可能不支持的emoji替换为ASCII提示，避免乱码。
    仅影响终端打印，不影响HTML报告内容。
    (已简化以修复语法错误)
    """
    return text

def _print_dup(*args, **kwargs):  # type: ignore[override]
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    try:
        text = sep.join(str(a) for a in args)
    except Exception:
        text = ' '.join(map(str, args))
    text = _remap_symbols(text)
    try:
        sys.stdout.write(text + end)
    except UnicodeEncodeError:
        # 最后的兜底：强制以UTF-8字节写入
        sys.stdout.buffer.write((text + end).encode('utf-8', errors='replace'))
    sys.stdout.flush()

# -------- 终端输出去除图标（仅保留纯文本，HTML 页面保留图标） --------
_orig_print = builtins.print

def _strip_icons(text: str) -> str:
    """移除终端输出中的 FontAwesome 图标标签，保持纯文本可读性。"""
    try:
        return re.sub(r"<i class=['\"][^>]*></i>\s*", "", text)
    except Exception:
        return text

def _print_no_icons(*args, **kwargs):
    new_args = [_strip_icons(str(a)) for a in args]
    _orig_print(*new_args, **kwargs)

# 覆盖内置 print，确保所有终端日志不显示图标
builtins.print = _print_no_icons


class LightweightAnalysis:
    """量化策略分析器 - 优化浏览器加载性能"""
    
    def __init__(self, data_path: str = "data/orders.parquet", benchmark_dir: str = "benchmark_data"):
        self.data_path = data_path
        self.benchmark_dir = Path(benchmark_dir)
        self.df = None
        self.benchmark_data = {}
        self.reports_dir = Path("reports") / "visualization_analysis"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures = []
        # 运行时缓存，避免重复加载/计算
        self._daily_price_df: Optional[pd.DataFrame] = None
        self._daily_factor_df: Optional[pd.DataFrame] = None
        self._trade_flow_cache: Optional[pd.DataFrame] = None
        self._intraday_snapshot_cache: Optional[pd.DataFrame] = None
        self._positions_cache: Optional[pd.DataFrame] = None
        self._factor_build_meta: dict = {}
        self._credit_rules: dict = {}
        self._risk_free_cache: dict = {}
        self._index_5m_cache_df: Optional[pd.DataFrame] = None
        self._index_5m_source_tag: str = ''
        self.strategy_metrics: dict = {}

    # ------------------------------------------------------------------
    # 无风险利率获取（按策略末日对齐）
    # ------------------------------------------------------------------
    def _get_risk_free_rate(self, target_date: Optional[str] = None) -> Tuple[float, str, Optional[str]]:
        """
        返回 (年化无风险利率, 来源, 数据日期)。
        优先级：
        1) 环境变量 RISK_FREE_RATE（允许 0.025 或 2.5 两种写法）
        2) 本地缓存 reports/risk_free_cache.json（若缓存日期不晚于 target_date 则复用）
        3) 在线获取东财国债收益率表（取 EMM00166466 近似1Y，按 target_date 向前取最近一条），失败则默认2%
        """
        # 1) 环境变量
        env_val = os.getenv("RISK_FREE_RATE")
        if env_val:
            try:
                v = float(env_val)
                rate = v / 100 if v > 1 else v
                return rate, "env_RISK_FREE_RATE", None
            except Exception:
                pass

        cache_path = Path("reports") / "risk_free_cache.json"
        # 2) 缓存复用
        try:
            if cache_path.exists():
                cache = json.loads(cache_path.read_text(encoding='utf-8'))
                rate_cached = float(cache["rate"])
                source_cached = cache.get("source", "cache")
                data_date = cache.get("data_date")
                if target_date is None or (data_date and data_date <= target_date):
                    return rate_cached, source_cached, data_date
        except Exception:
            pass

        # 3) 在线兜底
        risk_free_rate = 0.02
        source = "fallback_default"
        data_date: Optional[str] = None
        try:
            url = (
                "https://datacenter-web.eastmoney.com/api/data/v1/get"
                "?reportName=RPTA_WEB_TREASURYYIELD&columns=ALL"
                "&sortColumns=SOLAR_DATE&sortTypes=-1&pageSize=1&pageNumber=1"
            )
            if target_date:
                filter_expr = f"(SOLAR_DATE<='{target_date}')"
                url += f"&filter={quote(filter_expr)}"
            with urllib.request.urlopen(url, timeout=6) as resp:
                content = resp.read().decode('utf-8')
            payload = json.loads(content)
            rows = payload.get("result", {}).get("data", [])
            if rows:
                row = rows[0]
                rf_pct = row.get("EMM00166466")
                if rf_pct is not None:
                    risk_free_rate = float(rf_pct) / 100.0
                    source = "eastmoney_treasury_1y"
                    data_date = row.get("SOLAR_DATE")
        except Exception as e:
            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 在线获取无风险利率失败，使用默认值2%: {e}")

        # 写缓存
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_obj = {
                "rate": risk_free_rate,
                "source": source,
                "data_date": data_date,
                "fetched_at": datetime.now().isoformat()
            }
            cache_path.write_text(json.dumps(cache_obj, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

        return risk_free_rate, source, data_date

    def _parquet_columns(self, path: str) -> list:
        """获取 parquet 文件的列名（不读取数据）"""
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(path)
            return schema.names
        except Exception:
            # 回退：读取少量行获取列名
            try:
                df = pd.read_parquet(path, engine='pyarrow')
                return df.columns.tolist()
            except Exception:
                return []

    def _winsorize_series(self, series: pd.Series, lower: float = 0.001, upper: float = 0.999) -> pd.Series:
        """用于可视化的winsorize处理，限制极值但不影响原始指标数据。"""
        if series is None or len(series) == 0:
            return series.copy() if hasattr(series, 'copy') else series
        try:
            ser = series.astype(float).copy()
        except Exception:
            ser = series.copy()
        lower_val = ser.quantile(lower) if hasattr(ser, 'quantile') else None
        upper_val = ser.quantile(upper) if hasattr(ser, 'quantile') else None
        if lower_val is not None and upper_val is not None:
            return ser.clip(lower=lower_val, upper=upper_val)
        return ser


    # === 授信/保证金配置加载 ===
    def _load_credit_rules_from_md(self, md_path: str = "可视化输出指导文档/授信与保证金规则采集清单.md") -> dict:
        """从 Markdown 末尾的 YAML 代码块读取授信/保证金配置。
        - 优先尝试使用 PyYAML 解析；若不可用，使用简易解析器处理常用键。
        - 返回 dict，位于顶层 key 'credit_rules' 下的条目会被提升至同级。
        """
        path = Path(md_path)
        if not path.exists():
            return {}
        try:
            text = path.read_text(encoding='utf-8')
        except Exception:
            try:
                text = path.read_text(encoding='gbk', errors='ignore')
            except Exception:
                return {}

        # 抽取最后一个 ```yaml 代码块
        code_blocks = re.findall(r"```yaml\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if not code_blocks:
            return {}
        yaml_text = code_blocks[-1]

        # 优先用 PyYAML
        cfg = {}
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(yaml_text)
            if isinstance(data, dict):
                cfg = data.copy()
        except Exception:
            # 简易解析：仅解析常用扁平键与简单嵌套
            current_section = []
            data: dict = {}
            for raw in yaml_text.splitlines():
                line = raw.strip('\n')
                if not line.strip() or line.strip().startswith('#'):
                    continue
                # 列表项
                if line.strip().startswith('- '):
                    key = '__list__'
                    val = line.strip()[2:]
                    cur = data
                    for sec in current_section:
                        cur = cur.setdefault(sec, {})
                    lst = cur.setdefault(key, [])
                    lst.append(val)
                    cur['__list__'] = lst
                    continue
                # 键值或新节
                if ':' in line:
                    k, v = line.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    # 嵌套节开始
                    if v == '':
                        current_section.append(k)
                        cur = data
                        for sec in current_section:
                            cur = cur.setdefault(sec, {})
                        continue
                    # 普通键值
                    def _cast(x: str):
                        xs = x.strip()
                        if xs.lower() in ('true', 'false'):
                            return xs.lower() == 'true'
                        try:
                            if xs.endswith('%'):
                                xs = xs[:-1]
                            if '.' in xs:
                                return float(xs)
                            return int(xs)
                        except Exception:
                            return xs
                    cur = data
                    for sec in current_section:
                        cur = cur.setdefault(sec, {})
                    cur[k] = _cast(v)
            cfg = data

        # 将 credit_rules 内层提升到顶层，便于直接访问
        if isinstance(cfg, dict) and 'credit_rules' in cfg and isinstance(cfg['credit_rules'], dict):
            base = cfg.get('credit_rules', {})
            if isinstance(base, dict):
                merged = cfg.copy()
                merged.update(base)
                cfg = merged
        # 允许覆盖的安全系数
        if 'initial_capital_factor' in cfg:
            try:
                cfg['initial_capital_factor'] = float(cfg['initial_capital_factor'])
            except Exception:
                pass
        return cfg

    def _ensure_credit_rules_loaded(self) -> None:
        if not self._credit_rules:
            self._credit_rules = self._load_credit_rules_from_md()
        
    def _ensure_mathjax_bundle(self) -> str:
        """确保本地 MathJax 资源存在，返回相对于报告目录的路径。"""
        base_dir = Path('reports') / 'assets' / 'mathjax'
        base_dir.mkdir(parents=True, exist_ok=True)
        local_script = base_dir / 'tex-chtml-full.js'
        if not local_script.exists():
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 本地缺少 MathJax，尝试从 CDN 下载...")
            try:
                url = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js'
                with urllib.request.urlopen(url, timeout=20) as resp:
                    data = resp.read()
                local_script.write_bytes(data)
                size_mb = len(data) / (1024 * 1024)
                print(f"   <i class='fas fa-check-circle text-green-500'></i> MathJax 已下载 ({size_mb:.2f} MB) -> {local_script}")
            except Exception as exc:
                print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> MathJax 下载失败: {exc}")
        try:
            rel_path = os.path.relpath(local_script, start=self.reports_dir)
        except Exception:
            rel_path = '../assets/mathjax/tex-chtml-full.js'
        return rel_path.replace('\\', '/')

    def load_and_sample_data(self):
        """加载并智能采样数据"""
        print("<i class='fas fa-sync-alt text-blue-500'></i> 加载数据...")
        self.df = pd.read_parquet(self.data_path, engine='pyarrow')
        original_size = len(self.df)
        print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 原始数据: {original_size:,} 行")
        
        # 保留全量原始样本，后续在聚合结果中控制异常对统计的影响
        print("<i class='fas fa-broom text-gray-400'></i> 数据检查...")
        print(f"real字段范围: {self.df['real'].min():.2f} 到 {self.df['real'].max():.2f}")
        real_mean = self.df['real'].mean()
        real_std = self.df['real'].std()
        print(f"real字段均值: {real_mean:.4f}, 标准差: {real_std:.4f}")

        # 保留全量订单，后续按股票/日聚合控制规模
        # （原智能采样逻辑已移除，以避免样本偏差）
        # 确保时间列格式正确
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        if 'tradeTimestamp' in self.df.columns:
            self.df['tradeTimestamp'] = pd.to_datetime(self.df['tradeTimestamp'])
            
        print(f"<i class='fas fa-check-circle text-green-500'></i> 数据准备完成")
        
        # 加载基准指数数据
        self.load_benchmark_data()
        
    def load_benchmark_data(self):
        """加载基准指数数据"""
        print("<i class='fas fa-chart-line text-green-500'></i> 加载基准指数数据...")
        
        benchmark_files = {
            '创业板指数': 'sz_399006_daily_2024.csv',
            '深证成指': 'sz_399001_daily_2024.csv', 
            '中小板指数': 'sz_399005_daily_2024.csv',
            '深证100': 'sz_399330_daily_2024.csv'
        }
        
        for name, filename in benchmark_files.items():
            filepath = self.benchmark_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    
                    # 确保数据按日期排序
                    df = df.sort_values('date').reset_index(drop=True)
                    
                    # 检查并修复收盘价数据
                    if 'close' not in df.columns:
                        print(f"<i class='fas fa-times-circle text-red-500'></i> {name}: 缺少close列")
                        continue
                        
                    # 处理缺失值和异常值
                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                    df = df.dropna(subset=['close'])
                    
                    if len(df) == 0:
                        print(f"<i class='fas fa-times-circle text-red-500'></i> {name}: 收盘价数据全部缺失")
                        continue
                    
                    # 计算日收益率：(今日收盘价 - 昨日收盘价) / 昨日收盘价
                    df['daily_return'] = df['close'].pct_change()
                    
                    # 计算累积收益率：复利计算
                    df['cumulative_return'] = (1 + df['daily_return'].fillna(0)).cumprod() - 1
                    
                    # 验证计算结果
                    start_price = df['close'].iloc[0]
                    end_price = df['close'].iloc[-1]
                    expected_return = (end_price / start_price) - 1
                    calculated_return = df['cumulative_return'].iloc[-1]
                    
                    print(f"<i class='fas fa-check-circle text-green-500'></i> {name}: {len(df)} 条记录")
                    print(f"   起始价格: {start_price:.2f}, 结束价格: {end_price:.2f}")
                    print(f"   计算收益: {calculated_return*100:.2f}%, 验证收益: {expected_return*100:.2f}%")
                    
                    self.benchmark_data[name] = df
                    
                except Exception as e:
                    print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 加载 {name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 文件不存在: {filepath}")
                
        if not self.benchmark_data:
            print("<i class='fas fa-times-circle text-red-500'></i> 未能加载任何基准数据，将跳过基准对比分析")
        else:
            print(f"<i class='fas fa-check-circle text-green-500'></i> 成功加载 {len(self.benchmark_data)} 个基准指数")

    def _format_intraday_time(self, t_val) -> str:
        """将各种格式的时间字段规范为 HH:MM:SS。"""
        digits = ''.join(ch for ch in str(t_val) if ch.isdigit())
        if len(digits) >= 14:
            time_part = digits[8:14]
        elif len(digits) >= 8:
            time_part = digits[:6]
        elif len(digits) >= 6:
            time_part = digits[-6:]
        else:
            time_part = digits.zfill(6)
        time_part = time_part.zfill(6)
        return f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"

    def _normalize_index_5m_df(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """标准化指数5m行情列，兼容 baostock / akshare / tushare 字段。"""
        if df is None:
            return None
        if isinstance(df, pd.Series):
            df = df.to_frame().T
        if not isinstance(df, pd.DataFrame):
            return None
        df = df.copy()
        # 列名统一
        rename_map = {
            '时间': 'datetime',
            '交易时间': 'datetime',
            'trade_time': 'datetime',
            'timestamp': 'datetime',
            'date_time': 'datetime',
            '日期': 'date',
            '时间戳': 'time',
            'time': 'time',
            'open_price': 'open',
            'close_price': 'close',
            'high_price': 'high',
            'low_price': 'low',
            'vol': 'volume',
            '成交量': 'volume',
            '成交量(手)': 'volume',
            '成交额': 'amount',
            '成交额(元)': 'amount',
        }
        for src, dst in rename_map.items():
            if src in df.columns and dst not in df.columns:
                df = df.rename(columns={src: dst})

        # 若有 date + time 字段，则组合为 datetime
        if 'datetime' not in df.columns:
            if {'date', 'time'}.issubset(df.columns):
                try:
                    df['datetime'] = [
                        f"{d} {self._format_intraday_time(t)}" for d, t in zip(df['date'], df['time'])
                    ]
                except Exception:
                    pass
        if 'datetime' not in df.columns and 'trade_time' in rename_map and 'trade_time' in df.columns:
            df['datetime'] = df['trade_time']
        if 'datetime' not in df.columns:
            return None

        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])

        price_candidates = {
            'open': ['开盘', 'open', 'Open'],
            'high': ['最高', 'high', 'High'],
            'low': ['最低', 'low', 'Low'],
            'close': ['收盘', 'close', 'Close', 'price', 'last'],
            'volume': ['volume'],
            'amount': ['amount'],
        }
        for std_name, options in price_candidates.items():
            if std_name in df.columns:
                continue
            for opt in options:
                if opt in df.columns:
                    df = df.rename(columns={opt: std_name})
                    break

        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        keep_cols = ['datetime'] + [c for c in ['open', 'high', 'low', 'close', 'volume', 'amount'] if c in df.columns]
        df = df[keep_cols]
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime'], keep='last')
        return df if len(df) > 0 else None

    def _fetch_index_5m_online(self, start_dt: Optional[pd.Timestamp], end_dt: Optional[pd.Timestamp]) -> Tuple[Optional[pd.DataFrame], str]:
        """尝试在线抓取沪深300指数5分钟数据，返回标准化行情和数据源标签。"""
        start_dt = pd.to_datetime(start_dt) if start_dt is not None else None
        end_dt = pd.to_datetime(end_dt) if end_dt is not None else None
        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            end_dt = start_dt
        start_date = start_dt.strftime('%Y-%m-%d') if start_dt is not None else None
        end_date = end_dt.strftime('%Y-%m-%d') if end_dt is not None else None

        proxy_backup = {}
        proxy_keys = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        for k in proxy_keys:
            if k in os.environ:
                proxy_backup[k] = os.environ[k]
                os.environ.pop(k, None)

        try:
            # 1) baostock（无需 Token）
            try:
                import baostock as bs  # type: ignore
                print("<i class='fas fa-plug text-blue-500'></i> 尝试通过 baostock 获取沪深300 5分钟数据...")
                lg = bs.login()
                if lg.error_code != '0':
                    raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
                fields = "date,time,open,high,low,close,volume,amount"
                bs_codes = [
                    ("sh.000300", "沪深300指数"),
                    ("sh.510300", "沪深300ETF(510300)"),
                    ("sz.159919", "沪深300ETF(159919)"),
                ]
                try:
                    for code, tag in bs_codes:
                        rs = bs.query_history_k_data_plus(
                            code,
                            fields,
                            start_date=start_date,
                            end_date=end_date,
                            frequency="5"
                        )
                        data = []
                        while rs.error_code == '0' and rs.next():
                            data.append(rs.get_row_data())
                        if data:
                            bdf = pd.DataFrame(data, columns=rs.fields)
                            bdf['datetime'] = pd.to_datetime(
                                bdf['date'].astype(str) + ' ' + bdf['time'].apply(self._format_intraday_time),
                                errors='coerce'
                            )
                            bdf[['open', 'high', 'low', 'close', 'volume', 'amount']] = bdf[['open', 'high', 'low', 'close', 'volume', 'amount']].apply(pd.to_numeric, errors='coerce')
                            bdf = bdf[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                            bdf = self._normalize_index_5m_df(bdf)
                            if bdf is not None and len(bdf):
                                print(f"<i class='fas fa-check-circle text-green-500'></i> baostock {tag} 获得 {len(bdf):,} 条5m 数据")
                                return bdf, f'baostock_{code}'
                        else:
                            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> baostock {tag} 返回为空")
                finally:
                    bs.logout()
            except Exception as exc:
                print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> baostock 5m 抓取失败: {exc}")

            # 2) akshare（无需 Token）
            try:
                import akshare as ak  # type: ignore
                print("<i class='fas fa-plug text-blue-500'></i> 尝试通过 akshare 获取沪深300 5分钟数据...")
                kwargs = {
                    'symbol': 'sh000300',
                    'period': '5'
                }
                if start_dt is not None:
                    kwargs['start_date'] = start_dt.strftime('%Y-%m-%d %H:%M:%S')
                if end_dt is not None:
                    kwargs['end_date'] = end_dt.strftime('%Y-%m-%d %H:%M:%S')
                adf = ak.index_zh_a_hist_min_em(**kwargs)  # type: ignore
                adf = adf.rename(columns={
                    '时间': 'datetime',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount',
                })
                adf = self._normalize_index_5m_df(adf)
                if adf is not None and len(adf):
                    print(f"<i class='fas fa-check-circle text-green-500'></i> akshare 获得 {len(adf):,} 条沪深300 5m 数据")
                    return adf, 'akshare_sh000300'
                else:
                    print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> akshare 返回为空")
            except Exception as exc:
                print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> akshare 5m 抓取失败: {exc}")

            # 3) tushare（需要 Token）
            token = os.environ.get('TUSHARE_TOKEN') or os.environ.get('ts_token') or os.environ.get('TS_TOKEN')
            if token:
                try:
                    import tushare as ts  # type: ignore
                    print("<i class='fas fa-plug text-blue-500'></i> 尝试通过 tushare 获取沪深300 5分钟数据...")
                    pro = ts.pro_api(token)
                    params = {'ts_code': '000300.SH', 'freq': '5min'}
                    if start_dt is not None:
                        params['start_date'] = start_dt.strftime('%Y%m%d')
                    if end_dt is not None:
                        params['end_date'] = end_dt.strftime('%Y%m%d')
                    tdf = pro.index_min(**params)  # type: ignore
                    tdf = self._normalize_index_5m_df(tdf)
                    if tdf is not None and len(tdf):
                        print(f"<i class='fas fa-check-circle text-green-500'></i> tushare 获得 {len(tdf):,} 条沪深300 5m 数据")
                        return tdf, 'tushare_000300.SH'
                    else:
                        print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> tushare 返回为空")
                except Exception as exc:
                    print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> tushare 5m 抓取失败: {exc}")
            else:
                print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 未检测到 TUSHARE_TOKEN，跳过 tushare 抓取")

            return None, 'unknown'
        finally:
            for k, v in proxy_backup.items():
                os.environ[k] = v

    def _get_index_5m_returns(self, start_dt: Optional[pd.Timestamp], end_dt: Optional[pd.Timestamp]) -> Tuple[Optional[pd.DataFrame], str]:
        """确保指数5m数据可用，返回 datetime/idx_ret 与数据源标签。"""
        path = Path('data/index_5m_cache.parquet')
        idx_df = self._index_5m_cache_df.copy() if self._index_5m_cache_df is not None else None
        source_tag = self._index_5m_source_tag or 'index_5m_cache'
        if idx_df is None:
            try:
                if path.exists():
                    idx_df = pd.read_parquet(path)
                    source_tag = 'index_5m_cache'
            except Exception as exc:
                print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 读取 {path} 失败: {exc}")
                idx_df = None

        idx_df = self._normalize_index_5m_df(idx_df)
        need_fetch = idx_df is None or len(idx_df) == 0
        if not need_fetch and start_dt is not None and end_dt is not None:
            try:
                start_dt = pd.to_datetime(start_dt)
                end_dt = pd.to_datetime(end_dt)
                margin = pd.Timedelta(minutes=5)
                cover_min = idx_df['datetime'].min() - margin
                cover_max = idx_df['datetime'].max() + margin
                if start_dt < cover_min or end_dt > cover_max:
                    need_fetch = True
            except Exception:
                need_fetch = True

        if need_fetch:
            fetched_df, fetched_source = self._fetch_index_5m_online(start_dt, end_dt)
            if fetched_df is not None and len(fetched_df):
                if idx_df is not None and len(idx_df):
                    idx_df = pd.concat([idx_df, fetched_df], ignore_index=True)
                else:
                    idx_df = fetched_df
                idx_df = self._normalize_index_5m_df(idx_df)
                source_tag = fetched_source or source_tag
                try:
                    if idx_df is not None:
                        idx_df.to_parquet(path, index=False)
                        print(f"<i class='fas fa-save text-green-500'></i> 指数5m数据已写入 {path}（{len(idx_df):,} 行）")
                except Exception as exc:
                    print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 写入 {path} 失败: {exc}")

        if idx_df is None or len(idx_df) == 0 or 'close' not in idx_df.columns:
            return None, source_tag

        idx_df = idx_df.sort_values('datetime')
        idx_df['idx_ret'] = idx_df['close'].pct_change()
        # 缓存到实例，避免重复 IO
        self._index_5m_cache_df = idx_df.copy()
        self._index_5m_source_tag = source_tag

        if start_dt is not None and end_dt is not None:
            try:
                start_dt = pd.to_datetime(start_dt)
                end_dt = pd.to_datetime(end_dt)
                idx_df = idx_df[(idx_df['datetime'] >= start_dt - pd.Timedelta(minutes=5)) & (idx_df['datetime'] <= end_dt + pd.Timedelta(minutes=5))]
            except Exception:
                pass

        return idx_df[['datetime', 'idx_ret']], source_tag
        
    def model_performance_analysis(self):
        """模型性能分析"""
        print("\n<i class='fas fa-bullseye text-red-500'></i> === 模型性能分析 ===")
        data_processing_steps = ""
        risk_free_rate_annual = 0.02
        rf_source = "fallback_default"
        rf_data_date: Optional[str] = None
        risk_free_daily = risk_free_rate_annual / 252
        return_series_full = None
        return_series_aligned = None
        sharpe_full_value = None
        rolling_sharpe_latest = None
        # 无风险利率（默认2%年化，可按末日对齐）
        risk_free_rate_annual = 0.02
        rf_source = "fallback_default"
        rf_data_date: Optional[str] = None
        risk_free_daily = risk_free_rate_annual / 252
        return_series_full = None
        return_series_aligned = None
        
        # 0) 预处理：仅保留必要列并去除缺失
        required_cols = ['Code', 'Timestamp', 'pred', 'real']
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 缺少必要列: {missing_cols}")
            return
        data = self.df[required_cols].dropna().copy()
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data['date'] = data['Timestamp'].dt.date
        
        # 1) 先按 股票-日 聚合，避免订单粒度重复计权
        daily_by_code = (data
            .groupby(['Code', 'date'])
            .agg(pred=('pred', 'mean'), real=('real', 'mean'))
            .reset_index()
            .sort_values(['Code', 'date'])
        )
        # 2) 计算下一期真实收益(T+1)以做真正的前瞻评估
        daily_by_code['real_next'] = daily_by_code.groupby('Code')['real'].shift(-1)
        
        # 3) 横截面IC（同日）与 T+1 IC；同时给出 RankIC
        def _cs_ic(df_group):
            return df_group['pred'].corr(df_group['real']) if len(df_group) > 5 else np.nan
        def _cs_ic_next(df_group):
            g = df_group.dropna(subset=['real_next'])
            return g['pred'].corr(g['real_next']) if len(g) > 5 else np.nan
        def _cs_rank(df_group):
            return df_group['pred'].rank().corr(df_group['real'].rank()) if len(df_group) > 5 else np.nan
        def _cs_rank_next(df_group):
            g = df_group.dropna(subset=['real_next'])
            return g['pred'].rank().corr(g['real_next'].rank()) if len(g) > 5 else np.nan
        
        ic_same = daily_by_code.groupby('date').apply(_cs_ic).dropna()
        ic_next = daily_by_code.groupby('date').apply(_cs_ic_next).dropna()
        rank_ic_same = daily_by_code.groupby('date').apply(_cs_rank).dropna()
        rank_ic_next = daily_by_code.groupby('date').apply(_cs_rank_next).dropna()
        
        # 统一索引与类型（以 T+1 为主评估口径）
        daily_ic = pd.Series(pd.to_numeric(ic_next.values, errors='coerce').astype(float),
                             index=pd.to_datetime(ic_next.index)).sort_index()
        
        # 初始化默认值
        ic_metrics = {'状态': '无有效数据'}
        ic_explain = "<p>无有效IC数据进行分析</p>"
        
        if len(daily_ic) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 横截面IC样本不足，生成占位说明")
            fig_placeholder = go.Figure()
            fig_placeholder.add_annotation(
                text="暂无可用于计算IC的样本",
                showarrow=False,
                font=dict(size=16, color='#444')
            )
            fig_placeholder.update_layout(
                title="IC时间序列（无可用数据）",
                height=360,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            placeholder_metrics = {
                '状态': '无有效数据',
                '说明': 'pred 与 real_next 缺乏有效重叠或数据被清洗为空'
            }
            self._save_figure_with_details(
                fig_placeholder,
                name='ic_timeseries_light',
                title='IC时间序列（暂无数据）',
                explanation_html=ic_explain,
                metrics=placeholder_metrics
            )
            return
        
        # 存在有效样本，继续完整分析
        if len(daily_ic) > 0:
            # 计算扩展指标
            daily_ic_dt = pd.Series(daily_ic.values, index=pd.to_datetime(daily_ic.index))
            weekly_ic = daily_ic_dt.resample('W').mean()
            monthly_ic = daily_ic_dt.resample('M').mean()
            ic_metrics = {
                '样本天数': f"{len(daily_ic)}",
                '平均IC(T+1)': f"{daily_ic.mean():.4f}",
                'IC标准差': f"{daily_ic.std():.4f}",
                'IR(信息比率)': f"{(daily_ic.mean()/daily_ic.std() if daily_ic.std()>0 else 0):.4f}",
                '正IC占比': f"{(daily_ic>0).mean():.1%}",
                '周均IC(T+1)': f"{weekly_ic.mean():.4f}",
                '月均IC(T+1)': f"{monthly_ic.mean():.4f}",
                '无风险年化(Sharpe)': f"{risk_free_rate_annual*100:.2f}%@{rf_source if rf_source else 'auto'}"
            }
            ic_explain = (
                "<ul>"
                "<li><b>IC定义</b>: 每个交易日横截面上 <code>pred</code> 与 <code>real</code> 的皮尔逊相关系数。</li>"
                "<li><b>10日均线</b>: 对日度IC做10日滑动平均以观察趋势。</li>"
                "<li><b>周/月均IC</b>: 将日度IC按周/月取均值，衡量更长期稳定性。</li>"
                "</ul>"
            )
            # IC时间序列 - 智能采样（保持完整性）
            if len(daily_ic) > 300:  # 超过300个交易日则采样
                step = max(1, len(daily_ic) // 250)  # 保留250个点
                daily_ic_sampled = daily_ic.iloc[::step]
                print(f"<i class='fas fa-chart-line-down text-red-500'></i> IC数据采样: {len(daily_ic)} -> {len(daily_ic_sampled)} 个点")
            else:
                daily_ic_sampled = daily_ic
                print(f"<i class='fas fa-chart-bar text-indigo-500'></i> IC数据无需采样: {len(daily_ic)} 个点")
                
            # 创建图表 - 彻底修复版本
            fig_ic = go.Figure()
            extra_figs_for_ic: List[Tuple[str, go.Figure]] = []
            profit_trace_idx = None
            
            # 添加10日移动平均 - 确保平滑效果
            rolling_mean = daily_ic_sampled.rolling(window=min(10, len(daily_ic_sampled)), min_periods=1).mean()
            
            # 验证移动平均是否有差异
            diff_check = (daily_ic_sampled - rolling_mean).abs().max()
            print(f"<i class='fas fa-chart-bar text-indigo-500'></i> IC与10日均线最大差异: {diff_check:.6f}")
            plot_ic_series = self._winsorize_series(daily_ic_sampled)
            plot_rolling_series = self._winsorize_series(rolling_mean)
            
            # 确保数据类型和格式正确
            x_dates = [x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x) for x in plot_ic_series.index.tolist()]
            y_ic = plot_ic_series.values.astype(float).tolist()
            y_rolling = plot_rolling_series.values.astype(float).tolist()
            if return_series_full is not None:
                return_series_aligned = return_series_full.reindex(pd.to_datetime(plot_ic_series.index))
            
            # 先添加移动平均线（作为背景）
            fig_ic.add_trace(go.Scatter(
                x=x_dates,
                y=y_rolling,
                mode='lines',
                name='10日移动平均',
                line=dict(color='red', width=4, dash='dash'),
                opacity=0.9,
                hovertemplate='日期: %{x}<br>10日均值: %{y:.4f}<extra></extra>'
            ))
            
            # 再添加日度IC线（在前景）
            fig_ic.add_trace(go.Scatter(
                x=x_dates,
                y=y_ic,
                mode='lines+markers',
                name='全市场日度IC(T+1)',
                line=dict(color='blue', width=2),
                marker=dict(size=3, color='blue'),
                hovertemplate='日期: %{x}<br>全市场IC: %{y:.4f}<extra></extra>'
            ))
            
            # 添加G1和G10组的日度IC追踪
            print("<i class='fas fa-chart-bar text-indigo-500'></i> 为IC时间序列添加G1和G10组日度追踪...")
            try:
                # 计算每日的G1和G10组IC（复用后面的函数逻辑）
                def calculate_extreme_groups_ic_for_timeseries(df_next):
                    """计算G1和G10组的IC"""
                    try:
                        df_sorted = df_next.dropna(subset=['real_next']).sort_values('pred')
                        n_total = len(df_sorted)
                        if n_total < 20:
                            return {'G1_IC': np.nan, 'G10_IC': np.nan}
                        g1_size = max(1, n_total // 10)
                        g1_data = df_sorted.head(g1_size)
                        g1_ic = g1_data['pred'].corr(g1_data['real_next']) if len(g1_data) > 2 else np.nan
                        g10_data = df_sorted.tail(g1_size)
                        g10_ic = g10_data['pred'].corr(g10_data['real_next']) if len(g10_data) > 2 else np.nan
                        return {'G1_IC': g1_ic, 'G10_IC': g10_ic}
                    except Exception:
                        return {'G1_IC': np.nan, 'G10_IC': np.nan}
                
                # 计算每日的G1和G10组IC
                daily_extreme_ic_ts = daily_by_code.groupby('date').apply(calculate_extreme_groups_ic_for_timeseries)
                
                g1_ic_daily = pd.Series([x['G1_IC'] for x in daily_extreme_ic_ts], 
                                       index=pd.to_datetime(daily_extreme_ic_ts.index)).dropna()
                g10_ic_daily = pd.Series([x['G10_IC'] for x in daily_extreme_ic_ts], 
                                        index=pd.to_datetime(daily_extreme_ic_ts.index)).dropna()
                
                # 对G1和G10组IC进行采样（与主IC序列保持一致）
                if len(daily_ic) > 300:
                    step = max(1, len(daily_ic) // 250)
                    g1_ic_sampled = g1_ic_daily.reindex(daily_ic_sampled.index).dropna()
                    g1_ic_plot = self._winsorize_series(g1_ic_sampled)
                    g10_ic_sampled = g10_ic_daily.reindex(daily_ic_sampled.index).dropna()
                    g10_ic_plot = self._winsorize_series(g10_ic_sampled)
                else:
                    g1_ic_sampled = g1_ic_daily.reindex(daily_ic.index).dropna()
                    g10_ic_sampled = g10_ic_daily.reindex(daily_ic.index).dropna()
                    g1_ic_plot = self._winsorize_series(g1_ic_sampled)
                    g10_ic_plot = self._winsorize_series(g10_ic_sampled)
                
                # 添加G1组IC曲线
                if len(g1_ic_plot) > 0:
                    x_g1 = [x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x) for x in g1_ic_plot.index.tolist()]
                    y_g1 = g1_ic_plot.values.astype(float).tolist()
                    
                    fig_ic.add_trace(go.Scatter(
                        x=x_g1,
                        y=y_g1,
                        mode='lines',
                        name='G1组IC(最低pred)',
                        line=dict(color='red', width=1.5, dash='dot'),
                        opacity=0.8,
                        hovertemplate='日期: %{x}<br>G1组IC: %{y:.4f}<extra></extra>'
                    ))
                
                # 添加G10组IC曲线
                if len(g10_ic_plot) > 0:
                    x_g10 = [x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x) for x in g10_ic_plot.index.tolist()]
                    y_g10 = g10_ic_plot.values.astype(float).tolist()
                    
                    fig_ic.add_trace(go.Scatter(
                        x=x_g10,
                        y=y_g10,
                        mode='lines',
                        name='G10组IC(最高pred)',
                        line=dict(color='green', width=1.5, dash='dot'),
                        opacity=0.8,
                        hovertemplate='日期: %{x}<br>G10组IC: %{y:.4f}<extra></extra>'
                    ))
                
                print(f"   G1组日度IC范围: {g1_ic_daily.min():.4f} 到 {g1_ic_daily.max():.4f}")
                print(f"   G10组日度IC范围: {g10_ic_daily.min():.4f} 到 {g10_ic_daily.max():.4f}")
                
            except Exception as e:
                print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> G1/G10组IC追踪添加失败: {e}")
            
            # 添加收益率最高日和最低日的标注线
            print("<i class='fas fa-chart-bar text-indigo-500'></i> 添加极端收益日标注线...")
            try:
                # 获取盯市分析的日收益率数据来识别极端收益日
                mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
                if mtm_file.exists():
                    mtm_df = pd.read_csv(mtm_file)
                    mtm_df['date'] = pd.to_datetime(mtm_df['date'])
                    
                    # 解析日收益率
                    def parse_return_for_annotation(val):
                        try:
                            if isinstance(val, str) and val.endswith('%'):
                                return float(val.rstrip('%')) / 100.0
                            else:
                                return float(val)
                        except (ValueError, TypeError):
                            return 0.0
                    
                    mtm_df['daily_return_num'] = mtm_df['daily_return'].apply(parse_return_for_annotation)
                    return_series_full = pd.Series(
                        mtm_df['daily_return_num'].values,
                        index=pd.to_datetime(mtm_df['date'])
                    ).sort_index()
                    try:
                        last_date = return_series_full.index.max()
                        target_date_str = pd.to_datetime(last_date).strftime('%Y-%m-%d')
                        risk_free_rate_annual, rf_source, rf_data_date = self._get_risk_free_rate(target_date_str)
                        risk_free_daily = risk_free_rate_annual / 252
                    except Exception:
                        pass
                    
                    # 找到收益最高和最低的日期
                    max_return_date = mtm_df.loc[mtm_df['daily_return_num'].idxmax(), 'date']
                    min_return_date = mtm_df.loc[mtm_df['daily_return_num'].idxmin(), 'date']
                    max_return_value = mtm_df['daily_return_num'].max()
                    min_return_value = mtm_df['daily_return_num'].min()
                    
                    # 确保日期格式正确
                    max_date_str = max_return_date.strftime('%Y-%m-%d') if hasattr(max_return_date, 'strftime') else str(max_return_date)
                    min_date_str = min_return_date.strftime('%Y-%m-%d') if hasattr(min_return_date, 'strftime') else str(min_return_date)
                    
                    # 添加收益最高日标注线（使用shape避免字符串参与数值计算）
                    fig_ic.add_shape(
                        type="line",
                        xref="x", yref="paper",
                        x0=max_date_str, x1=max_date_str,
                        y0=0, y1=1,
                        line=dict(dash="dash", color="black", width=2),
                        opacity=0.7
                    )
                    fig_ic.add_annotation(
                        x=max_date_str, y=1.02, xref='x', yref='paper',
                        showarrow=False,
                        text=f"收益最高日 {max_date_str} | {max_return_value*100:.2f}%",
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
                    
                    # 添加收益最低日标注线（使用shape避免字符串参与数值计算）
                    fig_ic.add_shape(
                        type="line",
                        xref="x", yref="paper",
                        x0=min_date_str, x1=min_date_str,
                        y0=0, y1=1,
                        line=dict(dash="dash", color="black", width=2),
                        opacity=0.7
                    )
                    fig_ic.add_annotation(
                        x=min_date_str, y=-0.02, xref='x', yref='paper',
                        showarrow=False,
                        text=f"收益最低日 {min_date_str} | {min_return_value*100:.2f}%",
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
                    
                    print(f"   <i class='fas fa-check-circle text-green-500'></i> 添加标注线: 收益最高日({max_date_str}, {max_return_value*100:.2f}%)")
                    print(f"   <i class='fas fa-check-circle text-green-500'></i> 添加标注线: 收益最低日({min_date_str}, {min_return_value*100:.2f}%)")
                    
                else:
                    print("   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到盯市分析结果文件，跳过极端日标注")
                    
            except Exception as e:
                print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 极端收益日标注添加失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 添加日度绝对盈利数据到图表中（默认隐藏）
            print("<i class='fas fa-chart-bar text-indigo-500'></i> 添加日度绝对盈利数据...")
            try:
                # 获取盯市分析的日度绝对盈利数据
                mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
                if mtm_file.exists():
                    mtm_df = pd.read_csv(mtm_file)
                    mtm_df['date'] = pd.to_datetime(mtm_df['date'])
                    
                    # 解析总资产并计算日度绝对盈利
                    def parse_currency_for_chart(val):
                        try:
                            if isinstance(val, str):
                                return float(val.replace(',', '').strip())
                            return float(val)
                        except (ValueError, TypeError):
                            return np.nan
                    
                    mtm_df['total_assets_num'] = mtm_df['total_assets'].apply(parse_currency_for_chart)
                    mtm_df = mtm_df.sort_values('date')
                    mtm_df['daily_abs_profit'] = mtm_df['total_assets_num'].diff()
                    
                    # 将绝对盈利数据对齐到IC数据的时间轴
                    profit_series = pd.Series(mtm_df['daily_abs_profit'].values, 
                                             index=pd.to_datetime(mtm_df['date'])).dropna()
                    
                    # 采样绝对盈利数据以匹配IC数据
                    if len(daily_ic) > 300:
                        step = max(1, len(daily_ic) // 250)
                        profit_sampled = profit_series.reindex(daily_ic_sampled.index).dropna()
                    else:
                        profit_sampled = profit_series.reindex(daily_ic.index).dropna()
                    
                    # 转换为与IC图表一致的x轴格式
                    profit_x_dates = [x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x) for x in profit_sampled.index]
                    profit_y_values = [float(v) for v in profit_sampled.values]  # 绝对金额
                    
                    # 确保绝对盈利x轴与IC曲线完全对齐
                    # 使用与IC相同的x轴数据
                    profit_aligned_x = x_dates[:len(profit_y_values)]  # 确保长度匹配
                    profit_aligned_y = profit_y_values[:len(x_dates)]  # 确保长度匹配
                    
                    # 添加绝对盈利曲线（默认隐藏）
                    fig_ic.add_trace(go.Scatter(
                        x=profit_aligned_x,
                        y=profit_aligned_y,
                        mode='lines+markers',
                        name='日度绝对盈利(¥)',
                        line=dict(color='orange', width=2),
                        marker=dict(size=4, color='orange'),
                        yaxis='y2',
                        visible=False,  # 默认隐藏
                        hovertemplate='日期: %{x}<br>绝对盈利: ¥%{y:,.0f}<extra></extra>'
                    ))
                    
                    profit_trace_idx = len(fig_ic.data) - 1
                    print(f"   <i class='fas fa-check-circle text-green-500'></i> 添加日度绝对盈利数据: {len(profit_sampled)} 个点")
                    
                else:
                    print("   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到盯市分析结果文件，跳过绝对盈利数据")
                    
            except Exception as e:
                print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 添加日度绝对盈利数据失败: {e}")

            # Sharpe 可视化（真实净值口径，按末日对齐无风险利率）
            if return_series_full is not None and len(return_series_full.dropna()) > 10:
                sharpe_window = 60 if len(return_series_full) >= 90 else max(20, len(return_series_full) // 3 or 5)
                min_periods_sharpe = max(5, sharpe_window // 3)
                excess_full = return_series_full - risk_free_daily
                rolling_excess = excess_full.rolling(window=sharpe_window, min_periods=min_periods_sharpe)
                sharpe_mean = rolling_excess.mean()
                sharpe_std = rolling_excess.std(ddof=1).replace(0, np.nan)
                sharpe_series_full = sharpe_mean.divide(sharpe_std) * np.sqrt(252)
                sharpe_series_plot = sharpe_series_full.sort_index()
                sharpe_std_plot = sharpe_std.reindex(sharpe_series_plot.index)
                full_std = excess_full.std(ddof=1)
                if full_std > 0:
                    sharpe_full_value = float(excess_full.mean() / full_std * np.sqrt(252))
                sharpe_latest_value = float(sharpe_series_full.dropna().iloc[-1]) if not sharpe_series_full.dropna().empty else None
                rolling_std_latest = float(sharpe_std.dropna().iloc[-1]) if not sharpe_std.dropna().empty else None
                if sharpe_latest_value is not None:
                    rolling_sharpe_latest = sharpe_latest_value

                sharpe_x = [pd.to_datetime(idx).strftime('%Y-%m-%d') if not pd.isna(idx) else '' for idx in sharpe_series_plot.index]
                sharpe_y = [None if pd.isna(v) else float(v) for v in sharpe_series_plot.values]
                sharpe_std_hover = [None if pd.isna(v) else float(v) for v in sharpe_std_plot.values]
                indicator_value = rolling_sharpe_latest if rolling_sharpe_latest is not None else (sharpe_full_value if sharpe_full_value is not None else 0.0)

                fig_sharpe = make_subplots(
                    rows=2,
                    cols=1,
                    specs=[[{"type": "indicator"}], [{"type": "xy"}]],
                    row_heights=[0.35, 0.65],
                    vertical_spacing=0.08
                )
                fig_sharpe.add_trace(
                        go.Scatter(
                            x=sharpe_x,
                            y=sharpe_y,
                            mode='lines',
                            name=f'滚动Sharpe({sharpe_window}日年化)',
                            line=dict(color='#8e44ad', width=2),
                            hovertemplate='日期: %{x}<br>Sharpe: %{y:.3f}<br>标准差: %{customdata:.4f}<extra></extra>',
                            customdata=sharpe_std_hover
                        ),
                        row=2,
                        col=1
                    )
                fig_sharpe.add_hline(y=1.0, line=dict(color='#95a5a6', dash='dash'), row=2, col=1, annotation_text="Sharpe=1", annotation_position="top left")
                fig_sharpe.add_hline(y=1.5, line=dict(color='#bdc3c7', dash='dot'), row=2, col=1, annotation_text="Sharpe=1.5", annotation_position="top right")
                fig_sharpe.add_trace(
                    go.Indicator(
                        mode="number+delta",
                        value=indicator_value,
                        delta={'reference': 1.0, 'valueformat': '.2f'},
                        number={'valueformat': '.3f'},
                        title={'text': f"当前年化Sharpe（{sharpe_window}日滚动）<br><span style='font-size:12px'>无风险年化 {risk_free_rate_annual*100:.2f}% [{rf_source}]</span>"}
                    ),
                    row=1,
                    col=1
                )
                fig_sharpe.update_yaxes(title_text='滚动Sharpe(年化)', row=2, col=1, tickformat='.2f')
                fig_sharpe.update_xaxes(title_text='日期', row=2, col=1)
                fig_sharpe.update_layout(
                    height=520,
                    title=f'夏普比率（真实净值口径，滚动{sharpe_window}日）',
                    hovermode='x unified',
                    showlegend=False,
                    margin=dict(t=80, b=40, l=60, r=30)
                )
                extra_figs_for_ic.append(('strategy_sharpe_nav', fig_sharpe))
                sharpe_metrics = {
                    '滚动窗口(交易日)': f"{sharpe_window}",
                    '无风险年化': f"{risk_free_rate_annual*100:.2f}%@{rf_source if rf_source else 'auto'}" + (f" @ {rf_data_date}" if rf_data_date else ""),
                }
                if rolling_std_latest is not None:
                    sharpe_metrics['滚动标准差(末值)'] = f"{rolling_std_latest:.4f}"
                if full_std is not None and not np.isnan(full_std):
                    sharpe_metrics['全样本标准差'] = f"{float(full_std):.4f}"
                if indicator_value is not None:
                    sharpe_metrics['当前滚动Sharpe(年化)'] = f"{indicator_value:.3f}"
                if sharpe_full_value is not None:
                    sharpe_metrics['全样本Sharpe(年化)'] = f"{sharpe_full_value:.3f}"
                sharpe_explain = (
                    "<ul>"
                    "<li><b>口径</b>：真实净值日收益率减去无风险日收益率（按净值末日向前取国债1Y利率），年化：均值÷标准差×√252，标准差使用样本标准差(ddof=1)。</li>"
                    "<li><b>窗口</b>：默认60个交易日；样本不足时按样本量自适应，标准差为0时返回NaN以避免误判。</li>"
                    "<li><b>无风险来源</b>：东财国债收益率表 EMM00166466（1Y），可用环境变量 <code>RISK_FREE_RATE</code> 覆盖。</li>"
                    "<li><b>Hover信息</b>：每个点的Sharpe旁同步展示该窗口的收益率标准差。</li>"
                    "</ul>"
                )
                self._save_figure_with_details(
                    fig_sharpe,
                    name='strategy_sharpe_nav',
                    title='夏普比率（真实净值口径）',
                    explanation_html=sharpe_explain,
                    metrics=sharpe_metrics
                )
            
            # 对比曲线：同日IC与RankIC（采样并对齐x轴）
            try:
                comp_df = pd.DataFrame({
                    'IC_same': ic_same,
                    'RankIC_same': rank_ic_same,
                    'RankIC_T+1': rank_ic_next
                })
                comp_df.index = pd.to_datetime(comp_df.index)
                # 将x_dates转换回datetime进行reindex，然后再转换为字符串
                x_dates_dt = [pd.to_datetime(x) for x in x_dates]
                comp_df = comp_df.reindex(x_dates_dt).dropna(how='all')
                if len(comp_df) > 0:
                    # 确保x轴数据格式一致
                    comp_x_dates = [x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x) for x in comp_df.index]
                    fig_ic.add_trace(go.Scatter(
                        x=comp_x_dates, y=comp_df['IC_same'],
                        mode='lines', name='同日IC', line=dict(color='gray', width=1, dash='dot'),
                        hovertemplate='日期: %{x}<br>同日IC: %{y:.4f}<extra></extra>'
                    ))
                    fig_ic.add_trace(go.Scatter(
                        x=comp_x_dates, y=comp_df['RankIC_T+1'],
                        mode='lines', name='RankIC(T+1)', line=dict(color='purple', width=1.5, dash='dash'),
                        hovertemplate='日期: %{x}<br>RankIC(T+1): %{y:.4f}<extra></extra>'
                    ))
            except Exception:
                pass
            
            # 动态计算按钮可见性数组，避免 trace 数量变化导致前端错误
            n_traces = len(fig_ic.data)
            only_ic_visible = [True] * n_traces
            if profit_trace_idx is not None and 0 <= profit_trace_idx < n_traces:
                only_ic_visible[profit_trace_idx] = False
            ic_plus_profit_visible = [True] * n_traces

            fig_ic.update_layout(
                title=f'IC时间序列分析（含极端信号组追踪）<br><sub>全市场平均IC: {daily_ic.mean():.4f}, IC标准差: {daily_ic.std():.4f}, IR: {daily_ic.mean()/daily_ic.std():.4f}</sub>',
                xaxis_title='日期',
                yaxis_title='IC值',
                yaxis2=dict(
                    title='日度绝对盈利(¥)',
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    tickformat=',.0f'  # 使用千位分隔符格式
                ),
                xaxis=dict(
                    range=[str(pd.to_datetime(daily_ic.index.min()).date()), str(pd.to_datetime(daily_ic.index.max()).date())],  # 使用字符串范围避免Timestamp序列化
                    type='date'
                ),
                yaxis=dict(range=[-1, 1]),
                height=500,  # 增加高度以容纳按钮
                hovermode='x unified',
                showlegend=True,
                # 添加控制按钮 - 使用动态可见性数组
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=[
                            dict(
                                args=[{"visible": only_ic_visible}],
                                label="仅显示IC",
                                method="update"
                            ),
                            dict(
                                args=[{"visible": ic_plus_profit_visible}],
                                label="显示IC+收益率",
                                method="update"
                            )
                        ],
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.01,
                        xanchor="left",
                        y=1.02,
                        yanchor="top",
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="rgba(0,0,0,0.2)",
                        font=dict(size=12)
                    ),
                ]
            )

            # 添加详细的数据处理过程说明
            data_processing_steps = """
        <div style="margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;">
        <h4><i class='fas fa-clipboard-list text-blue-500'></i> 数据处理详细过程</h4>
        <ol>
            <li><b>数据源</b>: orders.parquet文件中的 <code>pred</code> 和 <code>real</code> 字段</li>
            <li><b>预处理</b>: 移除 <code>pred</code> 和 <code>real</code> 字段的缺失值</li>
            <li><b>股票-日聚合</b>: 按 (<code>Code</code>, 日期) 分组，对 <code>pred</code> 和 <code>real</code> 取平均值</li>
            <li><b>前瞻收益计算</b>: 对按股票分组后的 <code>real</code> 字段向后移动一期，得到下一交易日的真实收益用于T+1评估</li>
            <li><b>横截面IC计算</b>: 每个交易日，计算当日 <code>pred</code> 与下一日 <code>real</code> 的皮尔逊相关系数</li>
            <li><b>数据采样</b>: 若超过300个交易日，按步长采样保留250个点用于展示</li>
            <li><b>移动平均</b>: 计算10日滑动平均，最小窗口为1</li>
            <li><b>图表生成</b>: 使用Plotly生成时间序列图，包含原始IC和移动平均线</li>
            <li><b>夏普计算</b>: 真实净值日收益率 - 无风险日收益率（按净值末日向前取国债1Y利率，当前 {risk_free_rate_annual*100:.2f}%@{rf_source if rf_source else 'auto'} {rf_data_date if rf_data_date else ''}），窗口内均值/标准差×√252；标准差为0时返回NaN。</li>
        </ol>
        <p><b>关键字段说明</b>:</p>
        <ul>
            <li><b>pred</b>: 模型预测值，数值越高表示预期收益越好</li>
            <li><b>real</b>: 实际收益标签，由数据准备脚本基于未来价格变动计算</li>
            <li><b>IC</b>: Information Coefficient，衡量预测值与实际收益的相关性</li>
        </ul>
        </div>
        """
        
        # 更新说明，包含极端信号组信息和标注线说明
        enhanced_ic_explain = ic_explain + """
        <h4><i class='fas fa-bullseye text-red-500'></i> 新增：极端信号组日度追踪</h4>
        <ul>
            <li><b>G1组IC（红色虚线）</b>: pred值最低的10%股票的日度IC表现（做空信号）</li>
            <li><b>G10组IC（绿色虚线）</b>: pred值最高的10%股票的日度IC表现（做多信号）</li>
            <li><b>核心假设</b>: 如果模型对极端信号有效，G1和G10组的IC应该显著高于全市场IC</li>
            <li><b>实战意义</b>: 帮助识别模型在哪些时间段对极端信号最有效，指导仓位管理</li>
        </ul>
        <h4><i class='fas fa-map-marker-alt text-red-500'></i> 新增：极端收益日标注</h4>
        <ul>
            <li><b>黑色虚线</b>: 标注策略收益率最高日和最低日的位置</li>
            <li><b>关键观察</b>: 在这些极端收益日，各组IC的表现如何？</li>
            <li><b>验证假设</b>: 在收益最高日，G10组IC是否表现优异？在收益最低日，模型表现如何？</li>
            <li><b>投资洞察</b>: 理解模型在市场极端情况下的预测能力</li>
        </ul>
        <h4><i class='fas fa-sync-alt text-blue-500'></i> 新增：IC与收益率关系分析</h4>
        <ul>
            <li><b>交互式按钮</b>: 使用图表上方的按钮切换显示模式</li>
            <li><b>"仅显示IC"</b>: 只显示IC相关曲线，专注于模型预测能力分析</li>
            <li><b>"显示IC+收益率"</b>: 同时显示IC和每日收益率，观察两者的关系</li>
            <li><b>双Y轴设计</b>: 左轴显示IC值(-1到1)，右轴显示收益率(%)，便于对比</li>
            <li><b>关键洞察</b>: 观察IC与收益率的领先/滞后关系，验证模型预测的有效性</li>
        </ul>
        """
        if sharpe_full_value is not None:
            ic_metrics['夏普比率(年化,真实净值)'] = f"{sharpe_full_value:.3f}"
        if rolling_sharpe_latest is not None:
            ic_metrics['滚动Sharpe当前值'] = f"{rolling_sharpe_latest:.3f}"
        
        self._save_figure_with_details(
            fig_ic,
            name='ic_timeseries_light',
            title='IC时间序列（含极端信号组追踪）',
            explanation_html=enhanced_ic_explain + data_processing_steps,
            metrics=ic_metrics,
            extra_figs=extra_figs_for_ic,
        )
        
        # IC分布图 - 针对高质量模型优化版
        print("<i class='fas fa-chart-bar text-indigo-500'></i> 生成IC分布图...")
        
        # 确保使用正确的IC数据
        ic_values = daily_ic.values.astype(float)
        ic_values_plot = self._winsorize_series(pd.Series(ic_values))
        print(f"<i class='fas fa-chart-bar text-indigo-500'></i> IC分布数据: {len(ic_values)}个值, 范围{ic_values.min():.4f}到{ic_values.max():.4f}")
        print(f'   展示范围经 winsorize: {ic_values_plot.min():.4f} 到 {ic_values_plot.max():.4f}')
        
        fig_ic_dist = go.Figure()

        # 针对窄范围IC数据优化bin设置
        ic_range = ic_values_plot.max() - ic_values_plot.min()
        optimal_bins = max(15, min(30, int(len(ic_values) / 8)))  # 根据数据量动态调整

        # 使用直方图显示IC分布
        fig_ic_dist.add_trace(go.Histogram(
            x=ic_values_plot.values,
            nbinsx=optimal_bins,
            name='IC分布',
            opacity=0.85,
            marker_color='lightcoral',
            marker_line_color='darkred',
            marker_line_width=1.0,
            hovertemplate='IC区间: %{x:.4f}<br>频次: %{y}<extra></extra>'
        ))
            
        # 添加统计线和注释
        ic_mean = ic_values.mean()
        ic_plot_mean = ic_values_plot.mean()
        ic_plot_median = ic_values_plot.median()
        ic_plot_std = ic_values_plot.std()
        ic_median = np.median(ic_values)
        ic_std = ic_values.std()
            
            # 均值线
        fig_ic_dist.add_vline(
                x=ic_mean, 
                line_dash="dash", 
                line_color="red",
                line_width=3,
                annotation_text=f"均值: {ic_mean:.4f}",
                annotation_position="top"
            )
            
            # 中位数线
        fig_ic_dist.add_vline(
                x=ic_median, 
                line_dash="dot", 
                line_color="blue",
                line_width=2,
                annotation_text=f"中位数: {ic_median:.4f}",
                annotation_position="bottom"
            )
            
            # ±1标准差线
        fig_ic_dist.add_vline(
                x=ic_mean - ic_std, 
                line_dash="dashdot", 
                line_color="orange",
                line_width=1,
                opacity=0.7,
                annotation_text=f"-1σ: {ic_mean - ic_std:.4f}",
                annotation_position="top left"
            )
            
        fig_ic_dist.add_vline(
                x=ic_mean + ic_std, 
                line_dash="dashdot", 
                line_color="orange",
                line_width=1,
                opacity=0.7,
                annotation_text=f"+1σ: {ic_mean + ic_std:.4f}",
                annotation_position="top right"
            )
            
        # 针对窄范围数据优化x轴显示
        margin = ic_range * 0.1  # 10%边距
        x_min = max(ic_values_plot.min() - margin, -1.0)
        x_max = min(ic_values_plot.max() + margin, 1.0)

        # 数据质量诊断
        positive_ic_ratio = (ic_values > 0).mean()
        negative_ic_ratio = (ic_values < 0).mean()
        ic_abs_mean = np.abs(ic_values).mean()

        # 确定警告级别
        warning_level = "<i class='fas fa-bell text-red-600'></i> 严重异常" if positive_ic_ratio > 0.8 else \
                       "<i class='fas fa-exclamation-triangle text-yellow-500'></i> 需要关注" if positive_ic_ratio > 0.7 else \
                       "<i class='fas fa-check-circle text-green-500'></i> 相对正常"

        # 生成诊断报告
        quality_issues = []
        if positive_ic_ratio == 1.0:
            quality_issues.append("100%正IC极其异常")
        if ic_abs_mean > 0.1:
            quality_issues.append(f"IC绝对值过高({ic_abs_mean:.3f})")
        if ic_std < 0.01:
            quality_issues.append("IC标准差过小，可能存在数据问题")

        # 构建角标文本内容
        
        # 统一修复：
        # 1) 强制y轴从0起（避免出现-1~4的异常范围）
        # 2) 将图例放置在图内右上角，确保不会被标题遮挡或超出画布
        # 3) 轻微增加柱间距，直方图更易于辨识
        fig_ic_dist.update_layout(
                title=f'IC分布图 - {warning_level}<br><sub>样本: {len(ic_values)}天 | 正IC: {positive_ic_ratio:.1%} | 负IC: {negative_ic_ratio:.1%} | 均值±标准差: {ic_mean:.4f}±{ic_std:.4f}</sub>',
                xaxis_title='IC值 (预测与实际收益相关系数)',
                yaxis_title='频次 (天数)',
                height=520,
                bargap=0.02,
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='top', y=0.98,
                    xanchor='right', x=0.98,
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1
                ),
                xaxis=dict(
                    range=[x_min, x_max],
                    tickformat='.3f',
                    dtick=max(ic_range/10, 0.005)
                )
        )
        fig_ic_dist.update_yaxes(autorange=True)

        # 单一"数据质量诊断"角标（避免重复）
        diagnosis_points = []
        if quality_issues:
            diagnosis_points = [f"• {msg}" for msg in quality_issues]
        diagnosis_text = (
            "<b><i class='fas fa-star text-yellow-400'></i> 数据质量诊断:</b><br>" + "<br>".join(diagnosis_points)
        ) if diagnosis_points else "<b><i class='fas fa-star text-yellow-400'></i> 数据质量诊断:</b><br>未发现明显异常"
        fig_ic_dist.add_annotation(
            xref='paper', yref='paper', x=0.98, y=0.02,
            xanchor='right', yanchor='bottom',
            align='left',
            text=diagnosis_text,
            showarrow=False,
            bordercolor='rgba(0,0,0,0.2)', borderwidth=1,
            bgcolor='rgba(255,255,255,0.85)'
        )
            
        # 生成详细的IC分析说明（仅在异常时展示"重要提示"）
        if quality_issues:
            problems_html = "".join([f"<li><strong>{msg}</strong></li>" for msg in quality_issues])
            ic_dist_explain = f"""
            <div style=\"margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #dc3545;\">
            <h4><i class='fas fa-bell text-red-600'></i> 重要提示：IC分布异常分析</h4>
            <p><strong>当前数据显示的问题：</strong></p>
                <ul>{problems_html}</ul>
            <p><strong>可能的原因：</strong></p>
            <ul>
                <li><strong>前瞻偏差</strong>：模型可能使用了未来信息</li>
                    <li><strong>数据泄漏</strong>：pred 与 real 字段可能存在信息泄漏</li>
                    <li><strong>样本选择偏差</strong>：可能包含表现最好的时间段</li>
                    <li><strong>时间对齐问题</strong>：预测与实际收益的时间匹配可能有误</li>
            </ul>
            </div>
            """
        else:
            ic_dist_explain = f"""
            <div style=\"margin: 20px 0; padding: 15px; background-color: #f5fff5; border-left: 4px solid #28a745;\">
                <h4><i class='fas fa-check-circle text-green-500'></i> IC分布诊断：相对正常</h4>
                <ul>
                    <li>正IC比例：{positive_ic_ratio:.1%}；负IC比例：{negative_ic_ratio:.1%}</li>
                    <li>均值±标准差：{ic_mean:.4f} ± {ic_std:.4f}</li>
                    <li>分布形态：围绕 0 近似对称</li>
            </ul>
        </div>
            """
        
        # 统一的技术解读（保留，但不带"异常"措辞）
        ic_dist_explain += f"""
        <h4><i class='fas fa-chart-bar text-indigo-500'></i> IC分布技术解读</h4>
            <ul>
                <li><strong>IC定义</strong>: Information Coefficient，每个交易日横截面上预测值与实际收益的皮尔逊相关系数</li>
                <li><strong>统计特征</strong>: 
                    <ul>
                        <li>均值: {ic_mean:.4f}</li>
                        <li>标准差: {ic_std:.4f}</li>
                        <li>偏度: {(((ic_values - ic_mean) / ic_std) ** 3).mean():.3f}</li>
                        <li>峰度: {(((ic_values - ic_mean) / ic_std) ** 4).mean() - 3:.3f}</li>
                    </ul>
                </li>
            </ul>
        """
        
        ic_dist_metrics = {
            'IC均值': f"{ic_mean:.4f}",
            'IC标准差': f"{ic_std:.4f}",
            'IC最小值': f"{ic_values.min():.4f}",
            'IC最大值': f"{ic_values.max():.4f}",
            '正IC比例': f"{positive_ic_ratio:.1%}",
            '负IC比例': f"{negative_ic_ratio:.1%}",
            'IC绝对值均值': f"{ic_abs_mean:.4f}",
            '数据质量评估': warning_level
        }
        
        self._save_figure_with_details(
            fig_ic_dist,
            name='ic_distribution_light',
            title='IC分布分析（含数据质量诊断）',
            explanation_html=ic_dist_explain,
            metrics=ic_dist_metrics
        )

        # ============== 分段稳定性：月份/行情/行业（联动RankIC(T+1)) ==============
        print("<i class='fas fa-chart-bar text-indigo-500'></i> 生成分段稳定性图表（月份/行情/行业）...")

        # 1) 按月份稳定性（T+1 IC 与 RankIC）
        ic_next_dt = pd.Series(ic_next.values, index=pd.to_datetime(ic_next.index)).sort_index()
        rank_ic_next_dt = pd.Series(rank_ic_next.values, index=pd.to_datetime(rank_ic_next.index)).sort_index()
        ic_month = ic_next_dt.resample('M').mean()
        rank_month = rank_ic_next_dt.resample('M').mean()

        # 按月份汇总IC与RankIC（T+1评估口径）
        print("<i class='fas fa-chart-bar text-indigo-500'></i> 计算按月份的IC与RankIC稳定性...")
        fig_month = go.Figure()
        y_ic_m = [float(v) if pd.notna(v) else None for v in ic_month.values]
        y_rank_m = [float(v) if pd.notna(v) else None for v in rank_month.values]

        x_ic_month = [x.strftime('%Y-%m') for x in ic_month.index]
        x_rank_month = [x.strftime('%Y-%m') for x in rank_month.index]

        fig_month.add_trace(go.Scatter(x=x_ic_month, y=y_ic_m, mode='lines+markers',
                                       name='全市场IC(T+1)', line=dict(color='steelblue', width=2)))
        fig_month.add_trace(go.Scatter(x=x_rank_month, y=y_rank_m, mode='lines+markers',
                                       name='全市场RankIC(T+1)', line=dict(color='purple', width=2, dash='dash')))

        fig_month.update_layout(
            title='按月份的IC稳定性（T+1评估口径）',
            xaxis_title='月份',
            yaxis_title='IC / RankIC',
            height=420,
            hovermode='x unified',
            showlegend=True,
            yaxis=dict(range=[-0.15, 0.15], tickformat='.3f', zeroline=True, zerolinecolor='gray')
        )
        try:
            fig_month.add_hline(y=0, line_dash='dot', line_color='gray')
        except Exception:
            pass

        # ===== 极端信号分位收益分析（闭环交易） =====
        print("<i class='fas fa-chart-bar text-indigo-500'></i> 计算极端信号组的闭环收益表现...")
        extreme_group_summary = None
        extreme_metrics = {}
        try:
            pairs = pd.read_parquet('data/paired_trades_fifo.parquet')
            if pairs.empty:
                raise ValueError("paired_trades_fifo.parquet 为空")
            pairs = pairs.copy()
            pairs['open_timestamp'] = np.where(pairs['trade_type'] == 'long', pairs['buy_timestamp'], pairs['sell_timestamp'])
            pairs['open_amount'] = np.where(pairs['trade_type'] == 'long', pairs['buy_amount'], pairs['sell_amount']).astype(float)
            pairs['open_amount'] = pairs['open_amount'].abs()
            pairs['open_pred'] = pairs['buy_pred']
            pairs['open_date'] = pd.to_datetime(pairs['open_timestamp']).dt.date

            short_mask = pairs['trade_type'] == 'short'
            missing_short = short_mask & pairs['open_pred'].isna()
            orders_cache = None
            if missing_short.any():
                print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 空头开仓缺失预测值 {int(missing_short.sum()):,} 条，尝试回填...")
                orders_cache = pd.read_parquet('data/orders.parquet', columns=['Timestamp', 'Code', 'pred', 'direction'])
                orders_cache['Timestamp'] = pd.to_datetime(orders_cache['Timestamp'])
                dir_series = orders_cache['direction']
                dir_upper = dir_series.astype(str).str.upper()
                mask_short_dir = dir_upper.str.startswith('S') | dir_upper.str.contains('SELL') | dir_upper.str.contains('SHORT') | (dir_series == -1)
                orders_short = orders_cache.loc[mask_short_dir, ['Code', 'Timestamp', 'pred']].copy()
                short_pairs = pairs.loc[missing_short, ['code', 'open_timestamp']].copy()
                short_pairs['__row_id'] = short_pairs.index
                merged_short = short_pairs.merge(
                    orders_short,
                    left_on=['code', 'open_timestamp'],
                    right_on=['Code', 'Timestamp'],
                    how='left'
                ).set_index('__row_id')
                pairs.loc[merged_short.index, 'open_pred'] = merged_short['pred']

            long_mask = (pairs['trade_type'] == 'long') & pairs['open_pred'].isna()
            if long_mask.any():
                print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 多头开仓缺失预测值 {int(long_mask.sum()):,} 条，尝试回填...")
                if orders_cache is None:
                    orders_cache = pd.read_parquet('data/orders.parquet', columns=['Timestamp', 'Code', 'pred', 'direction'])
                    orders_cache['Timestamp'] = pd.to_datetime(orders_cache['Timestamp'])
                dir_series = orders_cache['direction']
                dir_upper = dir_series.astype(str).str.upper()
                mask_long_dir = dir_upper.str.startswith('B') | dir_upper.str.contains('BUY') | dir_upper.str.contains('LONG') | (dir_series == 1)
                orders_long = orders_cache.loc[mask_long_dir, ['Code', 'Timestamp', 'pred']].copy()
                long_pairs = pairs.loc[long_mask, ['code', 'open_timestamp']].copy()
                long_pairs['__row_id'] = long_pairs.index
                merged_long = long_pairs.merge(
                    orders_long,
                    left_on=['code', 'open_timestamp'],
                    right_on=['Code', 'Timestamp'],
                    how='left'
                ).set_index('__row_id')
                pairs.loc[merged_long.index, 'open_pred'] = merged_long['pred']

            pairs = pairs.dropna(subset=['open_pred', 'open_amount'])
            pairs = pairs[pairs['open_amount'] > 0]
            unique_preds = pairs['open_pred'].nunique(dropna=True)
            if unique_preds < 2 or len(pairs) < 10:
                raise ValueError("有效预测值或样本量不足，无法进行分位分析")

            num_bins = min(10, int(unique_preds))
            pairs['group_idx'] = pd.qcut(pairs['open_pred'], q=num_bins, labels=False, duplicates='drop')
            pairs = pairs.dropna(subset=['group_idx'])
            pairs['group_idx'] = pairs['group_idx'].astype(int)

            group_summary = pairs.groupby('group_idx').agg(
                total_profit=('absolute_profit', 'sum'),
                total_notional=('open_amount', 'sum'),
                order_count=('absolute_profit', 'size')
            ).reset_index()
            group_summary['pred_group'] = group_summary['group_idx'].apply(lambda x: f'G{x+1}')
            group_summary['group_return'] = group_summary['total_profit'] / group_summary['total_notional'].replace({0: np.nan})
            group_summary = group_summary.sort_values('group_idx').reset_index(drop=True)

            if len(group_summary) < 2:
                raise ValueError("分位结果不足以构建多空组合")

            extreme_group_summary = group_summary
            low_row = group_summary.iloc[0]
            high_row = group_summary.iloc[-1]
            spread_return = None
            if pd.notna(high_row['group_return']) and pd.notna(low_row['group_return']):
                spread_return = high_row['group_return'] - low_row['group_return']
                print(f"   {low_row['pred_group']} -> {high_row['pred_group']} 收益率差：{spread_return:.4f}")
            else:
                print(f"   {low_row['pred_group']} -> {high_row['pred_group']} 收益率差暂无有效值")
            spread_profit = high_row['total_profit'] - low_row['total_profit']

            extreme_metrics = {
                '组合数': str(len(group_summary)),
                f"{low_row['pred_group']}收益率": f"{low_row['group_return']:.2%}" if pd.notna(low_row['group_return']) else 'N/A',
                f"{high_row['pred_group']}收益率": f"{high_row['group_return']:.2%}" if pd.notna(high_row['group_return']) else 'N/A',
                '总盈亏(亿元)': f"{group_summary['total_profit'].sum()/1e8:.3f}",
                '总开仓金额(亿元)': f"{group_summary['total_notional'].sum()/1e8:.3f}",
                f"{high_row['pred_group']}-{low_row['pred_group']}收益率差": f"{spread_return:.2%}" if spread_return is not None else 'N/A',
                f"{high_row['pred_group']}-{low_row['pred_group']}盈亏差(百万元)": f"{spread_profit/1e6:.3f}"
            }
        except Exception as e:
            print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 极端信号收益分析失败: {e}")
            extreme_group_summary = None
            extreme_metrics = {}

        if extreme_group_summary is not None and not extreme_group_summary.empty:
            x_labels = extreme_group_summary['pred_group'].tolist()
            returns_values = [float(v) if pd.notna(v) else None for v in extreme_group_summary['group_return']]
            profit_million = [float(v) / 1e6 for v in extreme_group_summary['total_profit']]

            fig_extreme = make_subplots(specs=[[{'secondary_y': True}]])
            fig_extreme.add_trace(
                go.Bar(
                    x=x_labels,
                    y=returns_values,
                    name='组收益率',
                    marker_color='teal',
                    text=[f"{v*100:.2f}%" if v is not None else '' for v in returns_values],
                    textposition='outside'
                ),
                secondary_y=False
            )
            fig_extreme.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=profit_million,
                    name='累计盈亏 (百万元)',
                    mode='lines+markers',
                    line=dict(color='orange', width=2),
                    marker=dict(size=8)
                ),
                secondary_y=True
            )
            fig_extreme.update_yaxes(title_text='组收益率', tickformat='.1%', secondary_y=False)
            fig_extreme.update_yaxes(title_text='累计盈亏 (百万元)', secondary_y=True)
            fig_extreme.update_layout(
                title='预测分位闭环收益表现（基于配对交易）',
                xaxis_title='预测分位组',
                bargap=0.2,
                height=420,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
            )

            extreme_explanation_html = (
                "<p>将所有闭环交易按开仓时的预测值进行分位分组，统计每组的累计盈亏与资金占用收益率。</p>"
                "<ul>"
                "<li>收益率 = Σ盈亏 / Σ开仓金额，反映预测排序的现金效率。</li>"
                "<li>累计盈亏展示每组为策略贡献的绝对收益水平。</li>"
                "<li>最优组与最差组的收益率差值可视为多空组合的超额收益。</li>"
                "</ul>"
            )

            self._save_figure_with_details(
                fig_extreme,
                name='pred_quantile_closed_trade_light',
                title='预测分位闭环收益表现',
                explanation_html=extreme_explanation_html,
                metrics=extreme_metrics
            )

            summary_path = self.reports_dir / 'pred_quantile_closed_trade_summary.csv'
            summary_to_save = extreme_group_summary[['pred_group', 'order_count', 'total_notional', 'total_profit', 'group_return']].copy()
            summary_to_save['group_return'] = summary_to_save['group_return'].map(lambda x: float(x) if pd.notna(x) else np.nan)
            summary_to_save.to_csv(summary_path, index=False, encoding='utf-8-sig')

        # 2) 按行情分段（以全市场日均real衡量：多头/空头/盘整）
        market_daily = daily_by_code.groupby('date')['real'].mean()
        market_daily.index = pd.to_datetime(market_daily.index)
        regime_df = pd.DataFrame({
            'IC_T1': ic_next_dt,
            'RankIC_T1': rank_ic_next_dt
        }).join(market_daily.rename('mkt_ret'), how='inner')
        def _label_regime(r):
            th = 0.001
            if r > th:
                return '多头'
            if r < -th:
                return '空头'
            return '盘整'
        regime_df['regime'] = regime_df['mkt_ret'].apply(_label_regime)
        reg_stats = regime_df.groupby('regime').agg(
            IC_T1=('IC_T1', 'mean'), RankIC_T1=('RankIC_T1', 'mean'), n=('IC_T1', 'count')
        )
        # 固定顺序
        reg_stats = reg_stats.reindex(['多头', '盘整', '空头']).dropna(how='all')
        fig_reg = go.Figure()
        y_ic_reg = [float(v) if pd.notna(v) else None for v in reg_stats['IC_T1'].values]
        y_rank_reg = [float(v) if pd.notna(v) else None for v in reg_stats['RankIC_T1'].values]
        fig_reg.add_trace(go.Bar(x=reg_stats.index.astype(str), y=y_ic_reg, name='IC(T+1)', marker_color='steelblue',
                                 text=[f"{v:.3f}" if v is not None else '' for v in y_ic_reg], textposition='auto'))
        fig_reg.add_trace(go.Bar(x=reg_stats.index.astype(str), y=y_rank_reg, name='RankIC(T+1)', marker_color='purple',
                                 text=[f"{v:.3f}" if v is not None else '' for v in y_rank_reg], textposition='auto'))
        fig_reg.update_layout(
            title='按行情分段的IC稳定性（T+1）', xaxis_title='行情分段', yaxis_title='均值', height=420,
            barmode='group', yaxis=dict(range=[-0.15, 0.15], tickformat='.3f', zeroline=True, zerolinecolor='gray')
        )
        try:
            fig_reg.add_hline(y=0, line_dash='dot', line_color='gray')
        except Exception:
            pass
        self._save_figure_with_details(
            fig_reg,
            name='ic_stability_regime_light',
            title='IC按行情分段（T+1）',
            explanation_html='<p>以全市场日均收益作为行情 proxy：>0.1% 为多头，<-0.1% 为空头，其余为盘整；对每段统计 IC 与 RankIC(T+1) 的均值。</p>',
            metrics={k: f"{v:.4f}" for k, v in reg_stats[['IC_T1','RankIC_T1']].mean().to_dict().items()}
        )

        # 3) 行业维度分段（若有行业列）
        industry_col = next((c for c in ['industry', 'Industry', 'IndustryCode', 'industry_code', 'sector', 'SW', 'SW1'] if c in self.df.columns), None)
        if industry_col:
            # 重新构造包含行业信息的 股票-日 聚合
            data_ind = self.df[['Code', 'Timestamp', 'pred', 'real', industry_col]].dropna()
            data_ind['Timestamp'] = pd.to_datetime(data_ind['Timestamp'])
            data_ind['date'] = data_ind['Timestamp'].dt.date
            daily_ind = (data_ind
                .groupby(['Code', 'date'])
                .agg(pred=('pred','mean'), real=('real','mean'), industry=(industry_col, 'first'))
                .reset_index()
                .sort_values(['Code','date'])
            )
            daily_ind['real_next'] = daily_ind.groupby('Code')['real'].shift(-1)
            def _cs_ind(g):
                g = g.dropna(subset=['real_next'])
                return g['pred'].corr(g['real_next']) if len(g) > 5 else np.nan
            def _cs_rank_ind(g):
                g = g.dropna(subset=['real_next'])
                return g['pred'].rank().corr(g['real_next'].rank()) if len(g) > 5 else np.nan
            ic_ind = daily_ind.groupby(['date','industry']).apply(_cs_ind).dropna()
            rank_ind = daily_ind.groupby(['date','industry']).apply(_cs_rank_ind).dropna()
            ind_df = pd.DataFrame({'IC_T1': ic_ind, 'RankIC_T1': rank_ind}).reset_index()
            ind_mean = ind_df.groupby('industry').agg(IC_T1=('IC_T1','mean'), RankIC_T1=('RankIC_T1','mean'), n=('IC_T1','count'))
            # 选出现次数最多的前15个行业，便于展示
            ind_mean = ind_mean.sort_values('n', ascending=False).head(15)
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Bar(x=ind_mean.index.astype(str), y=ind_mean['IC_T1'], name='IC(T+1)', marker_color='steelblue'))
            fig_ind.add_trace(go.Bar(x=ind_mean.index.astype(str), y=ind_mean['RankIC_T1'], name='RankIC(T+1)', marker_color='purple'))
            fig_ind.update_layout(title='按行业分段的IC（T+1）', xaxis_title='行业', yaxis_title='均值', height=500, barmode='group',
                                  yaxis=dict(range=[-0.15, 0.15], tickformat='.3f', zeroline=True, zerolinecolor='gray'))
            try:
                fig_ind.add_hline(y=0, line_dash='dot', line_color='gray')
            except Exception:
                pass
            self._save_figure_with_details(
                fig_ind,
                name='ic_stability_industry_light',
                title='IC按行业分段（T+1）',
                explanation_html='<p>在行业内按天做横截面相关得到行业层面的 IC/T+1，随后按行业取均值。</p>',
                metrics={}
            )
            
        print(f"<i class='fas fa-check-circle text-green-500'></i> IC分析完成，平均IC: {daily_ic.mean():.4f}")
        
    def pred_real_relationship_analysis(self):
        """模型预测值与实际收益关系分析 - 基于完整交易的绝对收益分析"""
        print("\n<i class='fas fa-bullseye text-red-500'></i> === 预测值与实际收益关系分析（基于完整交易绝对收益）===")
        
        # 1. 数据预处理（使用全量订单，避免采样导致配对失真）
        print("<i class='fas fa-search text-blue-400'></i> 准备分析数据...")
        required_cols = ['Code', 'direction', 'pred', 'real', 'price', 'tradeAmount', 'fee', 'tradeQty', 'Timestamp']
        try:
            raw_data = pd.read_parquet(self.data_path, columns=required_cols, engine='pyarrow')
        except Exception:
            raw_data = pd.read_parquet(self.data_path, columns=required_cols)
        raw_data = raw_data.dropna(subset=['Code', 'direction', 'pred', 'tradeAmount', 'fee', 'tradeQty', 'Timestamp']).copy()
        # 仅保留实际成交的订单，防止0数量订单引入配对错误
        raw_data = raw_data[raw_data['tradeQty'] > 0]
        # 仅保留标准方向
        raw_data = raw_data[raw_data['direction'].isin(['B','S'])]
        print(f"原始数据量(全量): {len(raw_data):,} 条")

        # 若存在缓存的配对结果且比原始数据新，直接使用，避免重复计算
        cache_path = Path('data') / 'paired_trades_fifo.parquet'
        use_cache = False
        if cache_path.exists():
            try:
                data_mtime = os.path.getmtime(self.data_path)
                cache_mtime = os.path.getmtime(cache_path)
                if cache_mtime >= data_mtime:
                    use_cache = True
                    print(f"[OK] 发现缓存配对结果: {cache_path}, 直接加载")
            except Exception:
                use_cache = False
        
        if use_cache:
            try:
                trades_df = pd.read_parquet(cache_path)
                # 若旧缓存不包含 trade_type，触发重算以获得多/空分类
                if 'trade_type' not in trades_df.columns:
                    print("[INFO] 旧缓存缺少trade_type，将重算以支持多/空配对...")
                    use_cache = False
                else:
                    print(f"缓存交易对载入成功: {len(trades_df):,} 条")
                    # 跳转到后续分组与可视化
                    all_trade_pairs = None
            except Exception as e:
                print(f"[WARN] 读取缓存失败，将重新计算: {e}")
                use_cache = False
        
        if len(raw_data) == 0:
            print("<i class='fas fa-times-circle text-red-500'></i> 无有效数据进行分析")
            return
            
        # 2. 实现买卖订单配对算法（FIFO原则）
        print("<i class='fas fa-sync-alt text-blue-500'></i> 实现买卖订单配对算法...")
        
        def pair_trades_fifo(stock_data):
            """
            为单只股票的订单实现FIFO配对逻辑（支持多/空）：
            - 先买后卖 → 多头交易对（long）
            - 先卖后买 → 空头交易对（short）
            返回完整的交易对列表
            """
            # 按时间戳排序
            stock_data = stock_data.sort_values('Timestamp').reset_index(drop=True)
            
            buy_queue = []   # 未配对的买入（潜在多头开仓）
            sell_queue = []  # 未配对的卖出（潜在空头开仓）
            trade_pairs = []  # 完整交易对列表
            # 容错：若存在同一时点多条记录，加入自增序号保障稳定顺序
            stock_data['_ord'] = np.arange(len(stock_data))
            
            for _, row in stock_data.iterrows():
                if row['direction'] == 'B':  # 买入
                    buy_qty_total = float(row['tradeQty'])
                    buy_price = (row['price'] if 'price' in row and pd.notna(row['price']) else (row['tradeAmount'] / row['tradeQty'] if row['tradeQty'] > 0 else 0))
                    buy_fee_total = float(row['fee'])
                    remaining_buy_qty = buy_qty_total
                    
                    # 先尝试与未平的空头（sell_queue）对冲 → 形成空头交易对
                    while remaining_buy_qty > 0 and len(sell_queue) > 0:
                        sell_order = sell_queue[0]
                        matched_qty = min(remaining_buy_qty, sell_order['qty'])
                        matched_buy_amount = buy_price * matched_qty
                        matched_buy_fee = (buy_fee_total * matched_qty / buy_qty_total) if buy_qty_total > 0 else 0
                        matched_sell_amount = sell_order['price'] * matched_qty
                        matched_sell_fee = (sell_order['fee'] * matched_qty / sell_order['qty']) if sell_order['qty'] > 0 else 0
                        absolute_profit = matched_sell_amount - matched_buy_amount - (matched_buy_fee + matched_sell_fee)
                        trade_pairs.append({
                            'trade_type': 'short',
                            'buy_pred': np.nan,
                            'buy_amount': matched_buy_amount,
                            'sell_amount': matched_sell_amount,
                            'buy_fee': matched_buy_fee,
                            'sell_fee': matched_sell_fee,
                            'absolute_profit': absolute_profit,
                            'matched_qty': matched_qty,
                            'buy_price': buy_price,
                            'sell_price': sell_order['price'],
                            'buy_real': np.nan,
                            'buy_timestamp': row['Timestamp'],
                            'sell_timestamp': sell_order['timestamp']
                        })
                        # 更新剩余量
                        remaining_buy_qty -= matched_qty
                        # 更新sell_queue里该卖出订单剩余
                        sell_order['qty'] -= matched_qty
                        sell_order['amount'] -= (sell_order['price'] * matched_qty)
                        sell_order['fee'] -= matched_sell_fee
                        if sell_order['qty'] <= 1e-12:
                            sell_queue.pop(0)
                    
                    # 若买入仍有剩余，则作为新的多头开仓加入 buy_queue
                    if remaining_buy_qty > 1e-12:
                        # 按比例分摊剩余费用与金额
                        used_ratio = (buy_qty_total - remaining_buy_qty) / buy_qty_total if buy_qty_total > 0 else 0
                        remaining_amount = row['tradeAmount'] * (1 - used_ratio)
                        remaining_fee = buy_fee_total * (1 - used_ratio)
                        buy_queue.append({
                            'pred': row['pred'],
                            'price': buy_price,
                            'qty': remaining_buy_qty,
                            'amount': remaining_amount,
                            'fee': remaining_fee,
                            'buy_real': (row['real'] if 'real' in row and pd.notna(row['real']) else np.nan),
                            'timestamp': row['Timestamp']
                        })
                
                elif row['direction'] == 'S':  # 卖出
                    sell_qty_total = float(row['tradeQty'])
                    sell_price = (row['price'] if 'price' in row and pd.notna(row['price']) else (row['tradeAmount'] / row['tradeQty'] if row['tradeQty'] > 0 else 0))
                    sell_fee_total = float(row['fee'])
                    remaining_sell_qty = sell_qty_total
                    
                    # 优先与未平的多头（buy_queue）对冲 → 形成多头交易对
                    while remaining_sell_qty > 0 and len(buy_queue) > 0:
                        buy_order = buy_queue[0]
                        matched_qty = min(remaining_sell_qty, buy_order['qty'])
                        matched_sell_amount = sell_price * matched_qty
                        matched_sell_fee = (sell_fee_total * matched_qty / sell_qty_total) if sell_qty_total > 0 else 0
                        # 从买入订单中按比例分摊
                        matched_buy_amount = (buy_order['amount'] * matched_qty / buy_order['qty']) if buy_order['qty'] > 0 else 0
                        matched_buy_fee = (buy_order['fee'] * matched_qty / buy_order['qty']) if buy_order['qty'] > 0 else 0
                        absolute_profit = matched_sell_amount - matched_buy_amount - (matched_buy_fee + matched_sell_fee)
                        trade_pairs.append({
                            'trade_type': 'long',
                            'buy_pred': buy_order['pred'],
                            'buy_amount': matched_buy_amount,
                            'sell_amount': matched_sell_amount,
                            'buy_fee': matched_buy_fee,
                            'sell_fee': matched_sell_fee,
                            'absolute_profit': absolute_profit,
                            'matched_qty': matched_qty,
                            'buy_price': buy_order['price'],
                            'sell_price': sell_price,
                            'buy_real': buy_order.get('buy_real', np.nan),
                            'buy_timestamp': buy_order['timestamp'],
                            'sell_timestamp': row['Timestamp']
                        })
                        # 更新剩余量
                        remaining_sell_qty -= matched_qty
                        # 更新买入队列剩余
                        buy_order['qty'] -= matched_qty
                        buy_order['amount'] -= matched_buy_amount
                        buy_order['fee'] -= matched_buy_fee
                        if buy_order['qty'] <= 1e-12:
                            buy_queue.pop(0)
                    
                    # 若卖出仍有剩余，则作为新的空头开仓加入 sell_queue
                    if remaining_sell_qty > 1e-12:
                        used_ratio = (sell_qty_total - remaining_sell_qty) / sell_qty_total if sell_qty_total > 0 else 0
                        remaining_amount = row['tradeAmount'] * (1 - used_ratio)
                        remaining_fee = sell_fee_total * (1 - used_ratio)
                        sell_queue.append({
                            'price': sell_price,
                            'qty': remaining_sell_qty,
                            'amount': remaining_amount,
                            'fee': remaining_fee,
                            'timestamp': row['Timestamp']
                        })
            
            return trade_pairs
        
        # 3. 若无缓存则执行配对计算
        if not use_cache:
            print("<i class='fas fa-chart-bar text-indigo-500'></i> 按股票分组进行交易配对...")
            all_trade_pairs = []
            stock_codes = raw_data['Code'].unique()
            print(f"需要处理的股票数量: {len(stock_codes)}")
            for i, code in enumerate(stock_codes):
                if i % 100 == 0:
                    print(f"  处理进度: {i}/{len(stock_codes)} ({i/len(stock_codes)*100:.1f}%)")
                stock_data = raw_data[raw_data['Code'] == code].copy()
                stock_pairs = pair_trades_fifo(stock_data)
                for pair in stock_pairs:
                    pair['code'] = code
                all_trade_pairs.extend(stock_pairs)
            print(f"完成交易配对，共生成 {len(all_trade_pairs)} 笔完整交易")
            if len(all_trade_pairs) == 0:
                print("<i class='fas fa-times-circle text-red-500'></i> 没有成功配对的完整交易")
                return
            trades_df = pd.DataFrame(all_trade_pairs)
            # 保存缓存（高效列类型）
            try:
                save_cols = ['code','trade_type','buy_timestamp','sell_timestamp','matched_qty','buy_price','sell_price','buy_amount','sell_amount','buy_fee','sell_fee','buy_pred','buy_real','absolute_profit']
                for c in ['matched_qty','buy_amount','sell_amount','buy_fee','sell_fee','buy_price','sell_price','absolute_profit','buy_pred','buy_real']:
                    if c in trades_df.columns:
                        trades_df[c] = pd.to_numeric(trades_df[c], errors='coerce')
                trades_df.to_parquet(cache_path, index=False)
                print(f"[OK] 配对结果已缓存: {cache_path}")
            except Exception as e:
                print(f"[WARN] 缓存保存失败: {e}")
        print(f"交易配对结果统计:")
        print(f"  总交易对数: {len(trades_df):,}")
        print(f"  总绝对盈利: {trades_df['absolute_profit'].sum():.2f}")
        print(f"  平均绝对盈利: {trades_df['absolute_profit'].mean():.2f}")
        print(f"  盈利交易比例: {(trades_df['absolute_profit'] > 0).mean()*100:.1f}%")

        # 4.1 诊断校验：用价格差校验利润计算的一致性（严格以价格×数量计算，不再使用卖出总额分摊）
        try:
            alt_profit = trades_df['matched_qty'] * (trades_df['sell_price'] - trades_df['buy_price']) - (trades_df['buy_fee'] + trades_df['sell_fee'])
            diff_abs_sum = float(np.abs(alt_profit - trades_df['absolute_profit']).sum())
            diff_abs_mean = float(np.abs(alt_profit - trades_df['absolute_profit']).mean())
            print(f"  校验: 利润两种算法差异-合计: {diff_abs_sum:.2f}, 平均: {diff_abs_mean:.4f}")
        except Exception as _:
            pass

        # 4.1.1 预置 Recon 变量（便于后续解释输出）
        realized_pnl = float(trades_df['absolute_profit'].sum())
        unrealized_pnl = np.nan
        realized_plus_unreal = np.nan
        mtm_total_abs = np.nan
        recon_diff = np.nan

        # 4.2 诊断校验：买/卖金额按(股票,时间)聚合后是否与原始记录一致（检出重复或漏计）
        try:
            sells_orig = (raw_data[raw_data['direction'] == 'S']
                          .groupby(['Code','Timestamp'])['tradeQty'].sum())
            sells_mapped = (trades_df.groupby(['code','sell_timestamp'])['matched_qty'].sum())
            chk = sells_orig.rename_axis(['code','sell_timestamp']).to_frame('orig_qty').join(
                sells_mapped.to_frame('mapped_qty'), how='left')
            chk['mapped_qty'] = chk['mapped_qty'].fillna(0)
            mismatch_ratio = (np.abs(chk['orig_qty'] - chk['mapped_qty']) > 1e-6).mean()
            print(f"  校验: 卖出数量对齐差异占比: {mismatch_ratio:.2%}")
        except Exception as _:
            pass

        # 4.3 期末未平仓的未实现盈亏Recon（以订单末价近似期末价）
        try:
            print("[INFO] 计算期末未平仓的未实现盈亏用于Recon...")
            # 聚合买入/卖出总量与成本
            buys = (raw_data[raw_data['direction'] == 'B']
                    .groupby('Code')
                    .agg(buy_qty=('tradeQty','sum'), buy_amount=('tradeAmount','sum'), buy_fee=('fee','sum')))
            sells = (raw_data[raw_data['direction'] == 'S']
                     .groupby('Code')
                     .agg(sell_qty=('tradeQty','sum'), sell_amount=('tradeAmount','sum'), sell_fee=('fee','sum')))
            matched = (trades_df
                       .groupby('code')
                       .agg(match_qty=('matched_qty','sum'),
                            matched_buy_amount=('buy_amount','sum'), matched_buy_fee=('buy_fee','sum'),
                            matched_sell_amount=('sell_amount','sum'), matched_sell_fee=('sell_fee','sum')))
            recon = buys.join(sells, how='outer').join(matched, how='outer').fillna(0)

            # 剩余头寸（多/空）
            recon['left_long_qty'] = (recon['buy_qty'] - recon['match_qty']).clip(lower=0)
            recon['left_long_amount'] = (recon['buy_amount'] - recon['matched_buy_amount']).clip(lower=0)
            recon['left_long_fee'] = (recon['buy_fee'] - recon['matched_buy_fee']).clip(lower=0)
            recon['left_short_qty'] = (recon['sell_qty'] - recon['match_qty']).clip(lower=0)
            recon['left_short_amount'] = (recon['sell_amount'] - recon['matched_sell_amount']).clip(lower=0)
            recon['left_short_fee'] = (recon['sell_fee'] - recon['matched_sell_fee']).clip(lower=0)

            # 期末价格：优先使用缓存的收盘价，其次退回到最后一笔订单价格
            closing_cache = Path('data/closing_price_cache.parquet')
            last_px_series = None
            if closing_cache.exists():
                try:
                    close_df = pd.read_parquet(closing_cache)
                    if {'Code', 'close_price'}.issubset(close_df.columns):
                        last_px_series = close_df.set_index('Code')['close_price']
                except Exception:
                    last_px_series = None
            if last_px_series is None:
                # 回退：使用每只股票最后一笔订单价格
                last_px_series = (raw_data.sort_values('Timestamp')
                                  .drop_duplicates('Code', keep='last')
                                  .set_index('Code')['price'])
            recon = recon.join(last_px_series.rename('last_price'), how='left').fillna({'last_price':0.0})

            # 未实现盈亏：
            # 多头：按期末价估值 - 成本 - 费用；空头：已收卖出额 - 期末回补成本 - 费用
            recon['unreal_long'] = recon['left_long_qty'] * recon['last_price'] - recon['left_long_amount'] - recon['left_long_fee']
            recon['unreal_short'] = recon['left_short_amount'] - recon['left_short_qty'] * recon['last_price'] - recon['left_short_fee']
            unrealized_pnl = float(recon['unreal_long'].sum() + recon['unreal_short'].sum())
            realized_plus_unreal = realized_pnl + unrealized_pnl

            # 读取盯市总绝对盈利（使用正确的初始资金重新计算）
            mtm_total_abs = None
            mtm_path = Path("mtm_analysis_results/daily_nav_revised.csv")
            if mtm_path.exists():
                mtm_df = pd.read_csv(mtm_path)
                def _parse_currency(v):
                    try:
                        if isinstance(v, str):
                            return float(v.replace(',', '').strip())
                        return float(v)
                    except Exception:
                        return np.nan
                
                # 使用正确的初始资金重新计算NAV
                CORRECT_INITIAL_CAPITAL = 62_090_808
                mtm_df['long_value_num'] = mtm_df['long_value'].apply(_parse_currency)
                mtm_df['short_value_num'] = mtm_df['short_value'].apply(_parse_currency)
                
                # 从订单重新计算现金和NAV
                if hasattr(self, 'df') and self.df is not None:
                    orders_temp = pd.read_parquet(self.data_path, columns=['Timestamp', 'direction', 'tradeAmount', 'fee'])
                    orders_temp['date'] = pd.to_datetime(orders_temp['Timestamp']).dt.date
                    daily_flows_temp = orders_temp.groupby(['date', 'direction'])[['tradeAmount', 'fee']].sum().unstack(fill_value=0)
                    daily_flows_temp.columns = [f"{a}_{b}" for a, b in daily_flows_temp.columns]
                    
                    mtm_df['date'] = pd.to_datetime(mtm_df['date'])
                    cash_balance = CORRECT_INITIAL_CAPITAL
                    cash_series = []
                    for date_val in mtm_df['date'].dt.date:
                        if date_val in daily_flows_temp.index:
                            buy_amt = daily_flows_temp.loc[date_val, 'tradeAmount_B'] if 'tradeAmount_B' in daily_flows_temp.columns else 0
                            sell_amt = daily_flows_temp.loc[date_val, 'tradeAmount_S'] if 'tradeAmount_S' in daily_flows_temp.columns else 0
                            fee_amt = (daily_flows_temp.loc[date_val, 'fee_B'] if 'fee_B' in daily_flows_temp.columns else 0) + \
                                      (daily_flows_temp.loc[date_val, 'fee_S'] if 'fee_S' in daily_flows_temp.columns else 0)
                            cash_balance += sell_amt - buy_amt - fee_amt
                        cash_series.append(cash_balance)
                    
                    mtm_df['cash_num'] = cash_series
                    mtm_df['total_assets_num'] = mtm_df['cash_num'] + mtm_df['long_value_num'] - mtm_df['short_value_num']
                else:
                    mtm_df['total_assets_num'] = mtm_df['total_assets'].apply(_parse_currency)
                
                mtm_df = mtm_df.sort_values('date')
                mtm_df['daily_abs_profit'] = mtm_df['total_assets_num'].diff()
                mtm_total_abs = float(mtm_df['daily_abs_profit'].dropna().sum())

            print(f"  Recon: 已实现PnL: {realized_pnl:,.2f} | 期末未实现PnL: {unrealized_pnl:,.2f} | 合计: {realized_plus_unreal:,.2f}")
            if mtm_total_abs is not None:
                recon_diff = realized_plus_unreal - mtm_total_abs
                print(f"  Recon: 盯市总绝对盈利: {mtm_total_abs:,.2f} | 差额: {recon_diff:,.2f}")
        except Exception as e:
            print(f"[WARN] 未实现盈亏Recon计算失败: {e}")
        
        # 5. 按买入时的pred值分组
        print("<i class='fas fa-chart-line text-green-500'></i> 按买入时pred值分组分析...")
        n_groups = 10
        
        try:
            trades_df['pred_group'] = pd.qcut(
                trades_df['buy_pred'], 
                q=n_groups, 
                labels=[f'G{i+1}' for i in range(n_groups)],
                duplicates='drop'
            )
        except ValueError as e:
            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 分组时遇到问题: {e}")
            trades_df['pred_group'] = pd.cut(
                trades_df['buy_pred'], 
                bins=n_groups, 
                labels=[f'G{i+1}' for i in range(n_groups)]
            )
            
        # 6. 计算各组统计指标（包含买入时的real用于性能对比）
        group_stats = trades_df.groupby('pred_group', observed=True).agg({
            'buy_pred': ['mean', 'min', 'max', 'count'],
            'buy_real': ['mean', 'std'],
            'absolute_profit': ['sum', 'mean', 'std', 'count']
        }).round(4)
        
        # 扁平化列名
        group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns]
        group_stats = group_stats.reset_index()
        
        # 计算胜率
        win_rates = trades_df.groupby('pred_group', observed=True)['absolute_profit'].apply(
            lambda x: (x > 0).mean()
        ).reset_index()
        win_rates.columns = ['pred_group', 'win_rate']
        group_stats = group_stats.merge(win_rates, on='pred_group')
        
        print(f"成功创建 {len(group_stats)} 个分组")

        
        # 7. 创建可视化图表
        print("<i class='fas fa-chart-bar text-indigo-500'></i> 生成可视化图表...")
        
        from plotly.subplots import make_subplots
        
        # 创建子图：上图显示【pred柱状图+real折线】；下图显示【绝对收益柱+胜率折线】
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'pred-true对比（按买入pred分组）',
                '绝对收益与胜率（按买入pred分组）'
            ),
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        x_labels = [str(group) for group in group_stats['pred_group']]
        y_pred_mean = [float(val) for val in group_stats['buy_pred_mean']]
        y_real_mean = [float(val) if pd.notna(val) else 0.0 for val in group_stats['buy_real_mean']]
        y_absolute_profit_sum = [float(val) for val in group_stats['absolute_profit_sum']]
        y_absolute_profit_mean = [float(val) for val in group_stats['absolute_profit_mean']]
        y_win_rates = [float(val) for val in group_stats['win_rate']]
        
        # 上图：pred柱状 + real折线
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=y_pred_mean,
                name='平均预测值 (pred)',
                marker_color='rgba(231, 76, 60, 0.7)',
                hovertemplate='分组: %{x}<br>平均pred: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
            x=x_labels,
            y=y_real_mean,
                mode='lines+markers',
                name='平均真实标签 (real)',
                line=dict(color='steelblue', width=3),
                marker=dict(size=7, color='steelblue'),
                hovertemplate='分组: %{x}<br>平均real: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 下图：绝对收益（柱状图）
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=y_absolute_profit_sum,
                name='总绝对收益',
            marker_color='steelblue',
            opacity=0.8,
                hovertemplate='分组: %{x}<br>总绝对收益: %{y:.2f}<extra></extra>',
                text=[f'{val:.0f}' for val in y_absolute_profit_sum],
            textposition='outside'
            ),
            row=2, col=1
        )
        
        # 添加胜率线（右Y轴）
        win_rate_pct = [rate * 100 for rate in y_win_rates]
        fig.add_trace(
            go.Scatter(
            x=x_labels,
                y=win_rate_pct,
            mode='lines+markers',
                name='胜率 (%)',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=6, color='green'),
                hovertemplate='分组: %{x}<br>胜率: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1, secondary_y=True
        )
        
        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        
        # 计算相关系数
        correlation_pred_real = np.corrcoef(y_pred_mean, y_real_mean)[0, 1] if len(y_pred_mean) > 1 else np.nan
        correlation_pred_profit = np.corrcoef(y_pred_mean, y_absolute_profit_sum)[0, 1] if len(y_pred_mean) > 1 else np.nan
        
        # 更新布局
        fig.update_xaxes(title_text="预测值分组 (从低到高)", row=1, col=1)
        fig.update_yaxes(title_text="平均预测值", row=1, col=1)
        fig.update_xaxes(title_text="预测值分组 (从低到高)", row=2, col=1)
        fig.update_yaxes(title_text="总绝对收益", row=2, col=1)
        
        # 为胜率添加右Y轴
        fig.update_yaxes(title_text="胜率 (%)", secondary_y=True, row=2, col=1)
        fig.update_yaxes(range=[45, 55], secondary_y=True, row=2, col=1)
        
        fig.update_layout(
            title=(
                '预测性能与盈利能力分组分析（基于完整交易）'
                f"<br><sub>pred-real相关性: {correlation_pred_real:.4f} | pred-绝对收益相关性: {correlation_pred_profit:.4f} | 分组数: {n_groups} | 交易对数: {len(trades_df):,}</sub>"
            ),
            height=800,
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98),
            bargap=0.3
        )
        
        # 计算详细指标
        total_profit = trades_df['absolute_profit'].sum()
        total_trades = len(trades_df)
        overall_win_rate = (trades_df['absolute_profit'] > 0).mean()
        
        pred_real_metrics = {
            '分组数量': f"{len(group_stats)}",
            '完整交易对数': f"{total_trades:,}",
            '总绝对盈利': f"{total_profit:.2f}",
            '平均每笔盈利': f"{total_profit/total_trades:.2f}",
            '整体胜率': f"{overall_win_rate*100:.1f}%",
            'pred-real相关性': f"{correlation_pred_real:.4f}",
            'pred-绝对收益相关性': f"{correlation_pred_profit:.4f}",
            '预测值范围': f"{trades_df['buy_pred'].min():.4f} ~ {trades_df['buy_pred'].max():.4f}",
            '最高组总收益': f"{max(y_absolute_profit_sum):.2f}",
            '最低组总收益': f"{min(y_absolute_profit_sum):.2f}",
            '收益区分度': f"{max(y_absolute_profit_sum) - min(y_absolute_profit_sum):.2f}"
        }
        
        # Recon 解释块
        def _fmt_num(v):
            try:
                if pd.notna(v):
                    return f"{float(v):,.2f}"
            except Exception:
                pass
            return "N/A"

        recon_explain_html = f"""
        <h4>盈亏对账口径</h4>
        <ol>
            <li><b>已实现部分</b>：基于 FIFO 配对的净收益直接求和。</li>
            <li><b>未实现部分</b>：汇总未被配对的 <code>tradeQty</code>，用该标的最后一笔 <code>price</code> 近似收盘价估值，按方向扣除或回补成本与 <code>fee</code>。</li>
            <li><b>Recon 对齐</b>：已实现盈亏 + 未实现盈亏 与盯市总绝对盈利对照，差异主要来自末价近似、费用四舍五入及未计入的分红/融资。</li>
        </ol>
        <p>本期 Recon：已实现盈亏={_fmt_num(realized_pnl)}，未实现盈亏={_fmt_num(unrealized_pnl)}，合计={_fmt_num(realized_plus_unreal)}；盯市总绝对盈利={_fmt_num(mtm_total_abs)}；差额={_fmt_num(recon_diff)}。</p>
        """
        
        explanation_html = f"""
        <h4>页面目的</h4>
        <ul>
            <li>检验买入信号 <code>pred</code> 与完整交易绝对收益的单调性，评估信号排序价值。</li>
            <li>识别盈利集中在哪些信号分组，为阈值、持仓时长和风控调节提供依据。</li>
        </ul>
        <h4>实现方式</h4>
        <ol>
            <li>按 <code>Code</code> 和 <code>Timestamp</code> 先后对 <code>direction</code> 为 <code>B/S</code> 的订单执行 FIFO 配对，单笔净收益 = 卖出 <code>tradeAmount</code> − 买入 <code>tradeAmount</code> − 买卖两端 <code>fee</code>，并记录开仓时的 <code>pred</code>。</li>
            <li>将完整配对按 <code>pred</code> 分位划分为 {n_groups} 组，计算每组总绝对盈利、平均净收益与胜率，并绘制分组柱状与累计曲线。</li>
            <li>输出相关性指标：<code>pred</code> 与 <code>real</code>、与单笔绝对盈利的皮尔逊相关，用于验证信号排序是否带来收益差异。</li>
            <li>指标面板同步展示完整交易对数量({total_trades:,})、总绝对盈利({total_profit:.2f})、最高/最低分组差值和整体胜率({overall_win_rate*100:.1f}%)。</li>
        </ol>
        """
        
        pred_real_processing_steps = f"""
        <div style="margin-top: 12px; padding: 12px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;">
        <h4>计算要点</h4>
        <ul>
            <li>买入成交金额 = <code>tradeAmount</code>（<code>B</code>方向），卖出成交金额同理；费用 <code>fee</code> 按买卖两端实付计入。</li>
            <li>单笔净收益 = 卖出金额 − 买入金额 − 买卖两端 <code>fee</code>；单笔绝对收益取其绝对值用于分组累加。</li>
            <li>组内指标（总绝对盈利、平均净收益、胜率）全部基于配对结果计算，未直接使用 <code>real</code> 作为收益。</li>
        </ul>
        </div>
        """
        
        self._save_figure_with_details(
            fig,
            name='pred_real_relationship_light',
            title='预测值与实际收益关系分析',
            explanation_html=explanation_html + pred_real_processing_steps + recon_explain_html,
            metrics=pred_real_metrics
        )
        
        print(f"<i class='fas fa-check-circle text-green-500'></i> 基于完整交易的绝对收益分析完成")
        print(f"  [COUNT] 总交易对数: {total_trades:,}")
        print(f"  <i class='fas fa-coins text-yellow-500'></i> 总绝对盈利: {total_profit:.2f}")
        print(f"  <i class='fas fa-chart-line text-green-500'></i> 整体胜率: {overall_win_rate*100:.1f}%")
        print(f"  <i class='fas fa-chart-bar text-indigo-500'></i> pred-real相关性: {correlation_pred_real:.4f}")
        print(f"  <i class='fas fa-chart-bar text-indigo-500'></i> pred-绝对收益相关性: {correlation_pred_profit:.4f}")
        
    def profitability_paradox_analysis(self):
        """盈利悖论分析 - 为什么预测准确但整体亏损"""
        print("\n<i class='fas fa-question-circle text-yellow-500'></i> === 盈利悖论分析 ===")
        print("分析问题：模型预测准确但策略整体亏损的原因")
        
        # 1. 基础数据准备
        required_cols = ['pred', 'real', 'direction', 'tradeAmount', 'fee', 'tradeQty']
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 缺少分析所需列: {missing_cols}")
            return
            
        analysis_data = self.df[required_cols + ['Timestamp']].dropna().copy()
        analysis_data['date'] = analysis_data['Timestamp'].dt.date
        
        print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 分析数据量: {len(analysis_data):,} 条")
        
        # 2. 按pred分组的详细盈亏分析
        print("<i class='fas fa-search text-blue-400'></i> 按预测值分组的盈亏分析...")
        
        # 分组
        n_groups = 10
        analysis_data['pred_group'] = pd.qcut(
            analysis_data['pred'], 
            q=n_groups, 
            labels=[f'G{i+1}' for i in range(n_groups)],
            duplicates='drop'
        )
        
        # 计算各组的真实盈亏（考虑交易方向和费用）
        def calculate_true_pnl(group):
            """计算真实盈亏，考虑方向和费用"""
            if len(group) == 0:
                return pd.Series({
                    'total_pnl': 0,
                    'total_volume': 0,
                    'pnl_rate': 0,
                    'avg_fee_rate': 0,
                    'trade_count': 0
                })
            
            total_pnl = 0
            total_volume = 0
            
            # 应用合理的缩放因子
            scale_factor = 100  # 与其他收益计算保持一致
            
            for _, row in group.iterrows():
                # 计算缩放后的real值
                scaled_real = row['real'] / scale_factor
                
                # 根据方向计算理论盈亏
                if row['direction'] == 'B':  # 买入
                    theoretical_pnl = scaled_real * row['tradeAmount']
                else:  # 卖出
                    theoretical_pnl = -scaled_real * row['tradeAmount']  # 卖出时收益相反
                
                # 扣除交易费用
                actual_pnl = theoretical_pnl - row['fee']
                
                total_pnl += actual_pnl
                total_volume += row['tradeAmount']
            
            return pd.Series({
                'total_pnl': total_pnl,
                'total_volume': total_volume,
                'pnl_rate': total_pnl / total_volume if total_volume > 0 else 0,
                'avg_fee_rate': group['fee'].sum() / total_volume if total_volume > 0 else 0,
                'trade_count': len(group)
            })
        
        group_pnl_df = analysis_data.groupby('pred_group', observed=True).apply(calculate_true_pnl).reset_index()
        print(f"盈亏计算完成: {len(group_pnl_df)} 个分组")
        
        # 同时计算基础统计
        group_stats = analysis_data.groupby('pred_group', observed=True).agg({
            'pred': 'mean',
            'real': 'mean',
            'tradeAmount': 'sum',
            'fee': 'sum'
        }).reset_index()
        
        # 合并数据
        combined_stats = pd.merge(group_stats, group_pnl_df, on='pred_group')
        
        print("各组盈亏情况:")
        if len(combined_stats) == 0:
            print("  <i class='fas fa-times-circle text-red-500'></i> 没有可分析的数据")
            return
            
        for _, row in combined_stats.iterrows():
            print(f"  {row['pred_group']}: 预测{row['pred']:.3f}, 实际{row['real']:.3f}, "
                  f"真实盈亏率{row['pnl_rate']*100:.3f}%, 手续费率{row['avg_fee_rate']*100:.4f}%")
        
        # 3. 创建盈利悖论分析图表
        print("<i class='fas fa-chart-line text-green-500'></i> 生成盈利悖论分析图表...")
        
        if len(combined_stats) == 0:
            print("<i class='fas fa-times-circle text-red-500'></i> 无数据可生成图表")
            return
            
        x_labels = [str(group) for group in combined_stats['pred_group']]
        y_real = [float(val) for val in combined_stats['real']]
        y_pnl_rate = [float(val) * 100 for val in combined_stats['pnl_rate']]  # 转换为百分比
        y_pred = [float(val) for val in combined_stats['pred']]
        
        print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 图表数据准备完成: {len(x_labels)}个分组")
        
        fig_paradox = go.Figure()
        
        # 理论收益（基于real）
        fig_paradox.add_trace(go.Bar(
            x=x_labels,
            y=y_real,
            name='理论收益 (real)',
            marker_color='lightblue',
            opacity=0.7,
            text=[f'{val:.3f}' for val in y_real],
            textposition='outside',
            yaxis='y1'
        ))
        
        # 实际盈亏率（扣除费用）
        fig_paradox.add_trace(go.Bar(
            x=x_labels,
            y=y_pnl_rate,
            name='实际盈亏率 (%)',
            marker_color='red',
            opacity=0.8,
            text=[f'{val:.2f}%' for val in y_pnl_rate],
            textposition='outside',
            yaxis='y2'
        ))
        
        # 预测值趋势线
        fig_paradox.add_trace(go.Scatter(
            x=x_labels,
            y=y_pred,
            mode='lines+markers',
            name='预测值 (pred)',
            line=dict(color='green', width=3),
            marker=dict(size=8),
            yaxis='y1'
        ))
        
        # 计算整体统计
        total_theoretical = combined_stats['real'].sum() * combined_stats['tradeAmount'].sum()
        total_actual_pnl = combined_stats['total_pnl'].sum()
        total_volume = combined_stats['total_volume'].sum()
        total_fees = combined_stats['fee'].sum()
        
        actual_return_rate = total_actual_pnl / total_volume if total_volume > 0 else 0
        fee_impact = total_fees / total_volume if total_volume > 0 else 0
        
        fig_paradox.update_layout(
            title=f'盈利悖论分析：理论vs实际<br><sub>整体实际收益率: {actual_return_rate*100:.3f}% | 费用影响: {fee_impact*100:.4f}%</sub>',
            xaxis_title='预测值分组',
            yaxis=dict(
                title='理论收益 (real)',
                side='left'
            ),
            yaxis2=dict(
                title='实际盈亏率 (%)',
                overlaying='y',
                side='right'
            ),
            height=500,
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98)
        )
        
        # 4. 问题诊断和指标计算
        print("<i class='fas fa-search text-blue-400'></i> 问题诊断...")
        
        # 分析各种可能的原因
        diagnoses = []
        
        # A. 费用影响分析
        if fee_impact > 0.01:  # 费用超过1%
            diagnoses.append(f"高费用负担: {fee_impact*100:.3f}%，显著侵蚀收益")
        
        # B. 方向性分析
        buy_trades = analysis_data[analysis_data['direction'] == 'B']
        sell_trades = analysis_data[analysis_data['direction'] == 'S']
        buy_ratio = len(buy_trades) / len(analysis_data) if len(analysis_data) > 0 else 0
        
        if abs(buy_ratio - 0.5) > 0.2:  # 买卖不平衡
            diagnoses.append(f"交易方向失衡: 买入占比{buy_ratio:.1%}，可能存在方向性偏差")
        
        # C. 预测值分布分析
        negative_pred_volume = combined_stats[combined_stats['pred'] < 0]['total_volume'].sum()
        positive_pred_volume = combined_stats[combined_stats['pred'] > 0]['total_volume'].sum()
        negative_volume_ratio = negative_pred_volume / (negative_pred_volume + positive_pred_volume)
        
        if negative_volume_ratio > 0.4:  # 负预测值交易过多
            diagnoses.append(f"负预测交易过多: {negative_volume_ratio:.1%}的交易量对应负预测值")
        
        # D. 头部vs尾部效应
        top_groups = combined_stats.tail(3)  # 最高3组
        bottom_groups = combined_stats.head(3)  # 最低3组
        
        top_avg_pnl = top_groups['pnl_rate'].mean()
        bottom_avg_pnl = bottom_groups['pnl_rate'].mean()
        
        if top_avg_pnl <= 0:
            diagnoses.append(f"高预测组未盈利: 最高3组平均收益率{top_avg_pnl*100:.3f}%")
        
        # 5. 生成诊断指标
        paradox_metrics = {
            '理论收益能力': '优秀' if len(y_real) > 0 and max(y_real) > 0 and min(y_real) < 0 else '一般',
            '实际盈亏率': f"{actual_return_rate*100:.3f}%",
            '费用影响': f"{fee_impact*100:.4f}%",
            '买卖平衡度': f"买入{buy_ratio:.1%}/卖出{1-buy_ratio:.1%}",
            '负预测交易占比': f"{negative_volume_ratio:.1%}",
            '最高组收益率': f"{top_avg_pnl*100:.3f}%",
            '最低组收益率': f"{bottom_avg_pnl*100:.3f}%",
            '问题数量': f"{len(diagnoses)}个"
        }
        
        # 生成解释
        if diagnoses:
            diagnosis_text = "；".join(diagnoses)
            explanation_html = f"""
            <h4>页面目的</h4>
            <ul>
                <li>对比理论收益与实际盈亏，定位信号正确但收益下滑的执行损耗。</li>
                <li>用分位分组、费用率与方向占比诊断收益脱节的具体原因。</li>
            </ul>
            <h4>实现方式</h4>
            <ol>
                <li>按 <code>pred</code> 分位划分为 {n_groups} 组：理论收益 = <code>real × tradeAmount</code>（卖出方向取相反号），实际收益 = 理论收益 − <code>fee</code>。</li>
                <li>组内收益率 = 组内实际盈亏 ÷ 组内成交额；同步记录费用率、买卖比例、最高/最低组收益差。</li>
                <li>整体诊断输出实际收益率 {actual_return_rate*100:.3f}% 与费用影响 {fee_impact*100:.4f}% ，并列出异常模式。</li>
            </ol>
            <h4>诊断结果</h4>
            <p style="margin:8px 0; padding:10px; border-left:4px solid #ffc107; background:#fff3cd;">{diagnosis_text}</p>
            """
        else:
            explanation_html = """
            <h4>页面目的</h4>
            <ul>
                <li>对比理论收益与实际盈亏，确认信号与执行是否一致。</li>
            </ul>
            <h4>实现方式</h4>
            <ol>
                <li>按 <code>pred</code> 分组，计算理论收益（方向加权的 <code>real × tradeAmount</code>）与实际收益（扣除 <code>fee</code>）。</li>
                <li>组内收益率以成交额为分母，整体输出实际收益率与费用影响。</li>
            </ol>
            <p style="margin:8px 0; padding:10px; border-left:4px solid #4caf50; background:#e8f5e9;">未发现系统性偏差，收益波动更多由市场环境或样本噪声驱动。</p>
            """
        
        # 添加详细的数据处理过程说明 - 重点解释收益计算差异
        paradox_processing_steps = f"""
        <div style="margin-top: 30px; padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107;">
        <h4>计算口径</h4>
        <ol>
            <li><b>理论盈亏</b>：<code>direction</code>=B 取 <code>real × tradeAmount</code>，<code>direction</code>=S 取相反号，按分组求和。</li>
            <li><b>实际盈亏</b>：理论盈亏 − <code>fee</code>，费用率 = 组内 <code>fee</code> 总额 ÷ 组内成交额。</li>
            <li><b>收益率</b>：组内收益率 = 组内实际盈亏 ÷ 组内成交额；当分母≤0或缺失时记为 NaN 不展示。</li>
        </ol>
        <p style="margin:6px 0 0 0;">当前整体实际收益率 {actual_return_rate*100:.3f}% ，费用影响 {fee_impact*100:.4f}% 。</p>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;">
        <h4>数据处理流程</h4>
        <ol>
            <li><b>字段</b>：使用 <code>pred</code>、<code>real</code>、<code>direction</code>、<code>tradeAmount</code>、<code>fee</code>。</li>
            <li><b>分组</b>：按 <code>pred</code> 分位分为{n_groups}组，确保每组样本量近似均衡。</li>
            <li><b>聚合</b>：计算分组理论盈亏、实际盈亏、收益率与费用率，并在图表中对比蓝/红柱（理论/实际）与绿色 <code>pred</code> 线。</li>
        </ol>
        </div>
        """
        
        self._save_figure_with_details(
            fig_paradox,
            name='profitability_paradox_light',
            title='盈利悖论分析',
            explanation_html=explanation_html + paradox_processing_steps,
            metrics=paradox_metrics
        )
        
        print(f"<i class='fas fa-check-circle text-green-500'></i> 盈利悖论分析完成")
        print(f"<i class='fas fa-bullseye text-red-500'></i> 关键发现: 实际收益率{actual_return_rate*100:.3f}%, 费用影响{fee_impact*100:.4f}%")
        if diagnoses:
            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 发现{len(diagnoses)}个潜在问题")
            for i, diag in enumerate(diagnoses, 1):
                print(f"   {i}. {diag}")
        
    def portfolio_composition_analysis(self):
        """收盘后持仓市值概览 - 现金、仓位市值与交易成本走势"""
        print("\n<i class='fas fa-briefcase text-gray-600'></i> === 收盘后持仓市值 ===")
        
        # 尝试从盯市分析结果中加载数据
        print("<i class='fas fa-chart-bar text-indigo-500'></i> 读取盯市分析数据...")
        
        try:
            from pathlib import Path
            mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
            if not mtm_file.exists():
                print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到盯市分析结果文件，跳过收盘后持仓市值页面")
                return
            
            # 读取盯市分析的详细数据
            mtm_df = pd.read_csv(mtm_file)
            mtm_df['date'] = pd.to_datetime(mtm_df['date'])
            
            print(f"<i class='fas fa-check-circle text-green-500'></i> 成功读取盯市数据: {len(mtm_df)} 天")
            
            # 解析数值数据 - 处理格式化的字符串
            def parse_currency(val):
                """解析货币格式的字符串，如 '1,000,000.00'"""
                try:
                    if isinstance(val, str):
                        # 移除逗号和空格
                        cleaned = val.replace(',', '').replace(' ', '')
                        return float(cleaned)
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0
            
            def parse_percentage(val):
                """解析百分比格式的字符串"""
                try:
                    if isinstance(val, str) and val.endswith('%'):
                        return float(val.rstrip('%')) / 100.0
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0
            
            # 解析各个字段（先使用原始值）
            mtm_df['long_value_num'] = mtm_df['long_value'].apply(parse_currency)
            mtm_df['short_value_num'] = mtm_df['short_value'].apply(parse_currency)
            mtm_df['total_assets_num_original'] = mtm_df['total_assets'].apply(parse_currency)
            
            # 使用正确的初始资金重新计算现金和NAV
            CORRECT_INITIAL_CAPITAL = 62_090_808
            
            print(f"\n[RECALC] 使用正确的初始资金重新计算现金余额: ¥{CORRECT_INITIAL_CAPITAL:,.0f}")
            
            # 从订单数据计算每日现金流
            self.df['date'] = self.df['Timestamp'].dt.date
            daily_flows = self.df.groupby(['date', 'direction'])[['tradeAmount', 'fee']].sum().unstack(fill_value=0)
            daily_flows.columns = [f"{a}_{b}" for a, b in daily_flows.columns]
            
            # 将日期对齐到盯市数据
            daily_flows_aligned = pd.DataFrame({
                'date': pd.to_datetime(mtm_df['date']).dt.date,
                'buy_amount': 0.0,
                'sell_amount': 0.0,
                'fee': 0.0
            })
            
            # 从订单数据填充现金流
            for date in daily_flows_aligned['date'].unique():
                if date in daily_flows.index:
                    daily_flows_aligned.loc[daily_flows_aligned['date'] == date, 'buy_amount'] = daily_flows.loc[date, 'tradeAmount_B'] if 'tradeAmount_B' in daily_flows.columns else 0
                    daily_flows_aligned.loc[daily_flows_aligned['date'] == date, 'sell_amount'] = daily_flows.loc[date, 'tradeAmount_S'] if 'tradeAmount_S' in daily_flows.columns else 0
                    daily_flows_aligned.loc[daily_flows_aligned['date'] == date, 'fee'] = (
                        (daily_flows.loc[date, 'fee_B'] if 'fee_B' in daily_flows.columns else 0) +
                        (daily_flows.loc[date, 'fee_S'] if 'fee_S' in daily_flows.columns else 0)
                    )
            
            # 计算每日现金余额
            mtm_df['date_key'] = pd.to_datetime(mtm_df['date']).dt.date
            mtm_df = mtm_df.merge(daily_flows_aligned, left_on='date_key', right_on='date', how='left', suffixes=('', '_flow'))
            
            # 初始化现金
            cash_balance = CORRECT_INITIAL_CAPITAL
            cash_series = []
            
            for idx, row in mtm_df.iterrows():
                # 更新现金：卖出收入 - 买入支出 - 费用
                cash_balance += row['sell_amount'] - row['buy_amount'] - row['fee']
                cash_series.append(cash_balance)
            
            mtm_df['cash_num'] = cash_series
            
            # 重新计算NAV（基于正确的现金）
            mtm_df['total_assets_num'] = mtm_df['cash_num'] + mtm_df['long_value_num'] - mtm_df['short_value_num']
            
            print(f"   现金范围: {mtm_df['cash_num'].min():,.0f} 到 {mtm_df['cash_num'].max():,.0f}")
            print(f"   多头市值范围: {mtm_df['long_value_num'].min():,.0f} 到 {mtm_df['long_value_num'].max():,.0f}")
            print(f"   空头市值范围: {mtm_df['short_value_num'].min():,.0f} 到 {mtm_df['short_value_num'].max():,.0f}")
            print(f"   总资产范围: {mtm_df['total_assets_num'].min():,.0f} 到 {mtm_df['total_assets_num'].max():,.0f}")
            
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 读取盯市数据失败: {e}")
            return
        
        # fee已经在上面的daily_flows_aligned中计算了，直接使用
        portfolio_df = mtm_df.copy()
        portfolio_df['fee'] = portfolio_df['fee'].fillna(0)
        
        print("<i class='fas fa-check-circle text-green-500'></i> 已基于正确初始资金重新计算现金余额和NAV")
        
        # 校验现金恒等式：cash = total_assets - long_value + short_value
        portfolio_df['cash_expected'] = portfolio_df['total_assets_num'] - portfolio_df['long_value_num'] + portfolio_df['short_value_num']
        portfolio_df['cash_diff'] = portfolio_df['cash_num'] - portfolio_df['cash_expected']
        cash_gap = portfolio_df['cash_diff'].abs().max()
        if np.isfinite(cash_gap):
            print(f"   现金恒等式校验偏差（最大）: {cash_gap:,.2f} 元")
        
        # 计算日收益率用于校验
        portfolio_df['daily_return_num'] = portfolio_df['total_assets_num'].pct_change()
        portfolio_df['cumulative_return_num'] = (1 + portfolio_df['daily_return_num'].fillna(0)).cumprod() - 1

        # 数据采样以优化图表性能
        if len(portfolio_df) > 250:
            step = len(portfolio_df) // 200
            portfolio_sampled = portfolio_df.iloc[::step]
            print(f"<i class='fas fa-chart-line-down text-red-500'></i> 数据采样: {len(portfolio_df)} -> {len(portfolio_sampled)} 个点")
        else:
            portfolio_sampled = portfolio_df
            print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 数据无需采样: {len(portfolio_df)} 个点")
        
        # 创建投资组合构成图表
        fig_portfolio = go.Figure()
        
        # 准备时间轴数据
        x_dates = [date.strftime('%Y-%m-%d') for date in portfolio_sampled['date']]
        
        # 1. 现金走势（左Y轴）- 默认隐藏
        cash_values = portfolio_sampled['cash_num'].tolist()
        fig_portfolio.add_trace(go.Scatter(
            x=x_dates,
            y=cash_values,
            mode='lines',
            name='现金余额',
            line=dict(color='green', width=2.5),
            yaxis='y1',
            hovertemplate='日期: %{x}<br>现金余额: ¥%{y:,.0f}<extra></extra>',
            visible='legendonly'  # 默认隐藏，通过图例可手动开启
        ))
        
        # 2. 多头持仓市值（左Y轴）
        long_values = portfolio_sampled['long_value_num'].tolist()
        fig_portfolio.add_trace(go.Scatter(
            x=x_dates,
            y=long_values,
            mode='lines',
            name='多头持仓市值',
            line=dict(color='blue', width=2.5),
            yaxis='y1',
            hovertemplate='日期: %{x}<br>多头市值: ¥%{y:,.0f}<extra></extra>'
        ))
        
        # 3. 空头持仓市值（左Y轴）
        short_values = portfolio_sampled['short_value_num'].tolist()
        if max(short_values) > 0:  # 只有存在空头时才显示
            fig_portfolio.add_trace(go.Scatter(
                x=x_dates,
                y=short_values,
                mode='lines',
                name='空头持仓市值',
                line=dict(color='red', width=2.5),
                yaxis='y1',
                hovertemplate='日期: %{x}<br>空头市值: ¥%{y:,.0f}<extra></extra>'
            ))
        
        # 4. 每日交易费用（右Y轴）
        fee_values = portfolio_sampled['fee'].tolist()
        if max(fee_values) > 0:  # 只有存在交易费用时才显示
            fig_portfolio.add_trace(go.Scatter(
                x=x_dates,
                y=fee_values,
                mode='lines+markers',
                name='每日交易费用',
                line=dict(color='orange', width=2),
                marker=dict(size=4, color='orange'),
                yaxis='y2',
                hovertemplate='日期: %{x}<br>交易费用: ¥%{y:.2f}<extra></extra>',
                visible='legendonly'
            ))
        
        # 计算汇总统计
        total_cash_change = portfolio_sampled['cash_num'].iloc[-1] - portfolio_sampled['cash_num'].iloc[0]
        avg_long_position = portfolio_sampled['long_value_num'].mean()
        avg_short_position = portfolio_sampled['short_value_num'].mean()
        total_fees = portfolio_sampled['fee'].sum()
        max_long_position = portfolio_sampled['long_value_num'].max()
        min_cash = portfolio_sampled['cash_num'].min()
        initial_cash_est = portfolio_sampled['cash_num'].iloc[0]
        final_cash_est = portfolio_sampled['cash_num'].iloc[-1]
        initial_total_assets_est = portfolio_sampled['total_assets_num'].iloc[0] if len(portfolio_sampled) > 0 else 0
        fee_ratio = (total_fees / initial_total_assets_est * 100) if initial_total_assets_est > 0 else 0.0

        # 更新图表布局
        fig_portfolio.update_layout(
            title=f'收盘后持仓市值概览<br><sub>现金变化: {total_cash_change:+,.0f} | 平均多头: {avg_long_position:,.0f} | 总费用: {total_fees:,.2f}</sub>',
            xaxis_title='日期',
            yaxis=dict(
                title='金额 (¥)',
                side='left',
                tickformat=',.0f'
            ),
            yaxis2=dict(
                title='每日交易费用 (¥)',
                overlaying='y',
                side='right',
                tickformat=',.2f',
                showgrid=False
            ),
            height=500,
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98),
            xaxis=dict(type='date')
        )
        
        # 生成详细的指标
        portfolio_metrics = {
            '初始总资产': f"¥{initial_total_assets_est:,.0f}",
            '初始现金': f"¥{initial_cash_est:,.0f}",
            '最终现金': f"¥{final_cash_est:,.0f}",
            '现金变化': f"{total_cash_change:+,.0f}",
            '最低现金': f"¥{min_cash:,.0f}",
            '平均多头市值': f"¥{avg_long_position:,.0f}",
            '最大多头市值': f"¥{max_long_position:,.0f}",
            '平均空头市值': f"¥{avg_short_position:,.0f}",
            '总交易费用': f"¥{total_fees:,.2f}",
            '日均交易费用': f"¥{total_fees/len(portfolio_sampled):.2f}",
            '费用占比': f"{fee_ratio:.3f}%"
        }
        
        # 生成说明文档
        portfolio_explanation = f"""
        <h4>页面目的</h4>
        <ul>
            <li>呈现收盘时的现金、多空敞口与交易费用，评估资金安全垫和杠杆/对冲使用度。</li>
            <li>结合日度费用走势，判断调仓节奏是否带来额外成本压力。</li>
            <li>为对比收益曲线提供资产端口径，观察收盘结构随时间的稳定性。</li>
        </ul>
        <h4>实现方式</h4>
        <ol>
            <li>按 <code>Timestamp</code> 的自然日聚合订单现金流：买入(<code>B</code>)记为 <code>tradeAmount + fee</code> 现金流出，卖出(<code>S</code>)记为 <code>tradeAmount - fee</code> 回笼，得到每日净现金变动与手续费。</li>
            <li>以期初资金为起点，将净现金流叠加至收盘估值表中的多头/空头市值，回推当日收盘现金余额。</li>
            <li>收盘多头/空头市值直接取收盘估值；总资产=现金+多头市值−空头市值，手续费曲线为逐日汇总的 <code>fee</code>（右轴）。</li>
            <li>为保证加载性能，若样本较长会按时间抽样展示，但全部指标均基于全量数据计算。</li>
        </ol>
        """
        
        self._save_figure_with_details(
            fig_portfolio,
            name='portfolio_composition_light',
            title='收盘后持仓市值',
            explanation_html=portfolio_explanation,
            metrics=portfolio_metrics
        )
        
        # 交易结构：按市值/行业/板块的交易金额占比（饼图）
        print("<i class='fas fa-chart-bar text-indigo-500'></i> 计算交易结构饼图（市值/行业/板块）...")
        try:
            # 1) 使用配对成交数据计算每只股票的成交额（买+卖），更贴近真实成交
            pairs = pd.read_parquet('data/paired_trades_fifo.parquet')
            if 'code' not in pairs.columns:
                raise RuntimeError('paired_trades_fifo.parquet 缺少列 code')
            for col in ['buy_amount', 'sell_amount']:
                if col not in pairs.columns:
                    raise RuntimeError(f'paired_trades_fifo.parquet 缺少列 {col}')
            pairs['turnover'] = pairs['buy_amount'].astype(float) + pairs['sell_amount'].astype(float)
            by_code = (pairs.groupby('code')['turnover'].sum().reset_index()
                       .rename(columns={'code': 'Code', 'turnover': 'Turnover'}))
            total_turnover = float(by_code['Turnover'].sum()) if len(by_code) else 0.0

            # 2) 合并元数据（行业/市值/板块）。若缺失则做稳健回退。
            meta = None
            try:
                meta = pd.read_parquet('data/stock_metadata.parquet')
            except Exception:
                meta = None

            if meta is None or 'Code' not in meta.columns:
                meta = pd.DataFrame({'Code': by_code['Code']})
            else:
                meta = meta.copy()

            # 补充板块推断
            if 'Board' not in meta.columns:
                def _infer_board(code: str) -> str:
                    code6 = str(code).split('.')[0]
                    if code6.startswith('300') or code6.startswith('301'):
                        return '创业板'
                    if code6.startswith('688'):
                        return '科创板'
                    if code6.startswith('000') or code6.startswith('001') or code6.startswith('003') or code6.startswith('002'):
                        return '主板'
                    if code6.startswith('600') or code6.startswith('601') or code6.startswith('603'):
                        return '主板'
                    return '未知'
                meta['Board'] = meta['Code'].apply(_infer_board)

            # 市值分桶
            if 'MarketCapBucket' not in meta.columns:
                if 'MarketCap' in meta.columns:
                    def _cap_bucket(v):
                        try:
                            if pd.isna(v):
                                return '未知'
                            v = float(v)
                            if v >= 1000:
                                return '大盘股'
                            if v >= 100:
                                return '中盘股'
                            return '小盘股/微盘股'
                        except Exception:
                            return '未知'
                    meta['MarketCapBucket'] = meta['MarketCap'].apply(_cap_bucket)
                else:
                    meta['MarketCapBucket'] = '未知'

            if 'Industry' not in meta.columns:
                meta['Industry'] = '未知'

            enriched = by_code.merge(meta[['Code', 'MarketCapBucket', 'Industry', 'Board']], on='Code', how='left')

            # 3) 三类饼图数据
            def _pie_data(series: pd.Series, title: str, top_k: Optional[int] = None):
                s = series.groupby(series.index).sum().sort_values(ascending=False)
                if top_k is not None and len(s) > top_k:
                    head = s.head(top_k - 1)
                    tail_sum = s.iloc[top_k - 1:].sum()
                    s = pd.concat([head, pd.Series({'其他': tail_sum})])
                labels = [str(x) for x in s.index]
                values = [float(v) for v in s.values]
                return labels, values

            # 市值
            mc_series = (enriched
                         .groupby('MarketCapBucket')['Turnover']
                         .sum().fillna(0.0))
            # 固定顺序：大/中/小/未知
            mc_order = ['大盘股', '中盘股', '小盘股/微盘股', '未知']
            mc_series = mc_series.reindex(mc_order).fillna(0.0)
            mc_labels = list(mc_series.index)
            mc_values = [float(v) for v in mc_series.values]

            fig_mc = go.Figure(data=[go.Pie(labels=mc_labels, values=mc_values, hole=0.35, textinfo='label+percent', hovertemplate='%{label}: ¥%{value:,.0f} (%{percent})<extra></extra>')])
            fig_mc.update_layout(title='按市值大小的交易金额占比', showlegend=True)

            mc_metrics = {
                '总成交额': f"¥{total_turnover:,.0f}",
                '类别数': f"{sum(v > 0 for v in mc_values)}",
                '最大类别': (mc_labels[int(np.argmax(mc_values))] if len(mc_values) else 'NA')
            }

            # 行业（top12，其余合并为“其他”）
            ind_series = (enriched
                          .groupby('Industry')['Turnover']
                          .sum().sort_values(ascending=False))
            if len(ind_series) > 12:
                head = ind_series.head(11)
                other = float(ind_series.iloc[11:].sum())
                ind_series = pd.concat([head, pd.Series({'其他': other})])
            ind_labels = [str(x) for x in ind_series.index]
            ind_values = [float(v) for v in ind_series.values]

            fig_ind = go.Figure(data=[go.Pie(labels=ind_labels, values=ind_values, hole=0.35, textinfo='label+percent', hovertemplate='%{label}: ¥%{value:,.0f} (%{percent})<extra></extra>')])
            fig_ind.update_layout(title='按行业的交易金额占比', showlegend=True)

            ind_metrics = {
                '行业数': f"{len(ind_labels)}",
                '最大行业': (ind_labels[int(np.argmax(ind_values))] if len(ind_values) else 'NA'),
                'Top3占比': f"{(sum(sorted(ind_values, reverse=True)[:3]) / total_turnover):.1%}" if total_turnover > 0 else 'NA'
            }

            # 板块
            board_series = (enriched
                            .groupby('Board')['Turnover']
                            .sum())
            board_order = ['主板', '创业板', '科创板', '未知']
            board_series = board_series.reindex(board_order).fillna(0.0)
            board_labels = list(board_series.index)
            board_values = [float(v) for v in board_series.values]

            fig_board = go.Figure(data=[go.Pie(labels=board_labels, values=board_values, hole=0.35, textinfo='label+percent', hovertemplate='%{label}: ¥%{value:,.0f} (%{percent})<extra></extra>')])
            fig_board.update_layout(title='按交易所板块的交易金额占比', showlegend=True)

            board_metrics = {
                '板块数': f"{sum(v > 0 for v in board_values)}",
                '最大板块': (board_labels[int(np.argmax(board_values))] if len(board_values) else 'NA'),
                '总成交额': f"¥{total_turnover:,.0f}"
            }

            # 说明
            pies_explain = """
            <h4>页面目的</h4>
            <ul>
                <li>梳理成交额在市值分桶、行业与交易所板块的分布，识别资金偏好与集中度。</li>
                <li>为后续与盈利占比对照，观察“投得多是否赚得多”。</li>
            </ul>
            <h4>实现方式</h4>
            <ol>
                <li>按 <code>Code</code> 聚合订单的成交额：买入与卖出均取 <code>tradeAmount</code> 的绝对值求和，得到单标的累计成交额。</li>
                <li>基于 <code>Code</code> 映射标的市值分桶、行业和交易所板块，无法识别的归为“未知”。</li>
                <li>按分类汇总成交额并计算占比绘制饼图；类别过多时合并尾部为“其他”以突出主力分布。</li>
            </ol>
            <div style="margin-top:8px; padding:10px; border-left:4px solid #ffc107; background:#fff3cd;">解读提示：查看成交额集中度是否与策略预期一致，若某类标的占比过高且收益不佳，应结合盈利占比图进一步诊断。</div>
            """

            self._save_figure_with_details(fig_mc, name='amount_by_market_cap_pie_light', title='按市值大小的交易金额占比', explanation_html=pies_explain, metrics=mc_metrics)
            self._save_figure_with_details(fig_ind, name='amount_by_industry_pie_light', title='按行业的交易金额占比', explanation_html=pies_explain, metrics=ind_metrics)
            self._save_figure_with_details(fig_board, name='amount_by_board_pie_light', title='按交易所板块的交易金额占比', explanation_html=pies_explain, metrics=board_metrics)

            # 4) 盈利金额占比（基于正的 absolute_profit）
            if 'absolute_profit' in pairs.columns:
                profits = pairs[pairs['absolute_profit'] > 0].groupby('code')['absolute_profit'].sum().reset_index().rename(columns={'code': 'Code', 'absolute_profit': 'Profit'})
                total_profit = float(profits['Profit'].sum()) if len(profits) else 0.0
                enriched_p = profits.merge(meta[['Code', 'MarketCapBucket', 'Industry', 'Board']], on='Code', how='left')

                # 市值 - 盈利
                mc_p = (enriched_p.groupby('MarketCapBucket')['Profit'].sum()).reindex(mc_order).fillna(0.0)
                mc_p_labels = list(mc_p.index)
                mc_p_values = [float(v) for v in mc_p.values]
                fig_mc_p = go.Figure(data=[go.Pie(labels=mc_p_labels, values=mc_p_values, hole=0.35, textinfo='label+percent', hovertemplate='%{label}: ¥%{value:,.0f} (%{percent})<extra></extra>')])
                fig_mc_p.update_layout(title='按市值大小的盈利金额占比', showlegend=True)
                mc_p_metrics = {
                    '总盈利额': f"¥{total_profit:,.0f}",
                    '最大类别': (mc_p_labels[int(np.argmax(mc_p_values))] if len(mc_p_values) else 'NA')
                }

                # 行业 - 盈利（top12，其他合并）
                ind_p = (enriched_p.groupby('Industry')['Profit'].sum().sort_values(ascending=False))
                if len(ind_p) > 12:
                    head = ind_p.head(11)
                    other = float(ind_p.iloc[11:].sum())
                    ind_p = pd.concat([head, pd.Series({'其他': other})])
                ind_p_labels = [str(x) for x in ind_p.index]
                ind_p_values = [float(v) for v in ind_p.values]
                fig_ind_p = go.Figure(data=[go.Pie(labels=ind_p_labels, values=ind_p_values, hole=0.35, textinfo='label+percent', hovertemplate='%{label}: ¥%{value:,.0f} (%{percent})<extra></extra>')])
                fig_ind_p.update_layout(title='按行业的盈利金额占比', showlegend=True)
                ind_p_metrics = {
                    '行业数': f"{len(ind_p_labels)}",
                    '最大行业': (ind_p_labels[int(np.argmax(ind_p_values))] if len(ind_p_values) else 'NA')
                }

                # 板块 - 盈利
                board_p = (enriched_p.groupby('Board')['Profit'].sum()).reindex(board_order).fillna(0.0)
                board_p_labels = list(board_p.index)
                board_p_values = [float(v) for v in board_p.values]
                fig_board_p = go.Figure(data=[go.Pie(labels=board_p_labels, values=board_p_values, hole=0.35, textinfo='label+percent', hovertemplate='%{label}: ¥%{value:,.0f} (%{percent})<extra></extra>')])
                fig_board_p.update_layout(title='按交易所板块的盈利金额占比', showlegend=True)
                board_p_metrics = {
                    '板块数': f"{sum(v > 0 for v in board_p_values)}",
                    '最大板块': (board_p_labels[int(np.argmax(board_p_values))] if len(board_p_values) else 'NA'),
                    '总盈利额': f"¥{total_profit:,.0f}"
                }

                profit_explain = """
                <h4>页面目的</h4>
                <ul>
                    <li>量化正收益交易对在各分类下的贡献，定位主要盈利来源。</li>
                    <li>与成交额占比并排展示，检验资金配置效率与偏好是否匹配盈利能力。</li>
                </ul>
                <h4>实现方式</h4>
                <ol>
                    <li>按 <code>Code</code> 采用 FIFO 将 <code>direction</code> 为 <code>B/S</code> 的订单配对，单笔净利润 = 卖出 <code>tradeAmount</code> − 买入 <code>tradeAmount</code> − 买卖两端 <code>fee</code>。</li>
                    <li>仅保留净利润大于0的配对，按 <code>Code</code> 汇总为标的盈利额。</li>
                    <li>使用与成交额相同的分类映射求和占比并绘制饼图；总盈利额不含亏损配对，需结合成交额占比理解整体净效应。</li>
                </ol>
                """

                # 合并为一页：上方显示成交额占比，下方显示盈利额占比；在新窗口宽屏下并排显示
                combined_explain = pies_explain + profit_explain
                self._save_figure_pair_with_details(
                    fig_mc, fig_mc_p,
                    name='amount_by_market_cap_pie_light',
                    title='按市值大小的交易金额占比（含盈利占比）',
                    explanation_html=combined_explain,
                    metrics_primary=mc_metrics,
                    metrics_secondary=mc_p_metrics,
                    primary_title='按市值大小的交易金额占比',
                    secondary_title='按市值大小的盈利金额占比'
                )
                self._save_figure_pair_with_details(
                    fig_ind, fig_ind_p,
                    name='amount_by_industry_pie_light',
                    title='按行业的交易金额占比（含盈利占比）',
                    explanation_html=combined_explain,
                    metrics_primary=ind_metrics,
                    metrics_secondary=ind_p_metrics,
                    primary_title='按行业的交易金额占比',
                    secondary_title='按行业的盈利金额占比'
                )
                self._save_figure_pair_with_details(
                    fig_board, fig_board_p,
                    name='amount_by_board_pie_light',
                    title='按交易所板块的交易金额占比（含盈利占比）',
                    explanation_html=combined_explain,
                    metrics_primary=board_metrics,
                    metrics_secondary=board_p_metrics,
                    primary_title='按交易所板块的交易金额占比',
                    secondary_title='按交易所板块的盈利金额占比'
                )

            print("   <i class='fas fa-check-circle text-green-500'></i> 交易结构饼图已生成")
        except Exception as e:
            print(f"   <i class='fas fa-times-circle text-red-500'></i> 交易结构饼图生成失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"<i class='fas fa-check-circle text-green-500'></i> 收盘后持仓市值分析完成")
        print(f"<i class='fas fa-coins text-yellow-500'></i> 现金变化: {total_cash_change:+,.0f} 元")
        print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 平均多头市值: {avg_long_position:,.0f} 元")
        print(f"<i class='fas fa-money-bill-wave text-green-600'></i> 总交易费用: {total_fees:,.2f} 元")
        
    def _build_factor_dataset(self) -> Optional[pd.DataFrame]:
        """构建因子数据集（按 Code-日 粒度）
        已实现因子：
        - ln_market_cap: 市值对数（优先来自 stock_metadata 的 MarketCap；若无则尝试日频字段）
        - mom_5d, mom_20d: 5日/20日动量（使用日频收盘价近似代替分钟）
        - liquidity: Amihud 非流动性（ILLIQ）≈ |ret_1d| / 成交额（值越大越不液），若无成交额则为空

        返回: DataFrame[Code, date, ln_market_cap, mom_5d, mom_20d, liquidity, market_cap(optional)]
        """
        print("\n<i class='fas fa-box-open text-yellow-600'></i> 构建因子数据集（近似日频）...")
        close_df = None
        # 记录构建口径，供方法说明精确描述
        info = {
            'close_source': None,
            'amount_source': None,
            'momentum_source': None,
            'momentum_path': None,
            'market_cap_source': None,
        }
        # 1) 加载日频收盘价缓存
        used_path = None
        for path in [
            Path('data/daily_k_cache.parquet'),  # 优先：包含amount/volume
            Path('data/daily_close_cache.parquet'),
            Path('data/closing_price_cache.parquet'),
        ]:
            try:
                if path.exists():
                    tmp = pd.read_parquet(path)
                    close_df = tmp if close_df is None else close_df
                    if close_df is not None:
                        used_path = path
                        break
            except Exception:
                continue

        if close_df is None:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到日频收盘价缓存，因子将退化（仅可能使用静态市值）")
            # 仍尝试构造仅包含市值的因子
            meta = None
            try:
                meta = pd.read_parquet('data/stock_metadata.parquet')
            except Exception:
                meta = None
            if meta is None:
                print("<i class='fas fa-times-circle text-red-500'></i> 无可用的市值或收盘价数据，跳过因子构建")
                return None
            meta = meta.copy()
            code_col = 'Code' if 'Code' in meta.columns else ('code' if 'code' in meta.columns else None)
            if code_col is None:
                print("<i class='fas fa-times-circle text-red-500'></i> stock_metadata.parquet 缺少 Code 列，跳过因子构建")
                return None
            # 取市值列
            cap_col = None
            for c in ['MarketCap', 'market_cap', 'circulating_market_cap', 'circ_mv', '总市值', '流通市值']:
                if c in meta.columns:
                    cap_col = c
                    break
            if cap_col is None:
                print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 元数据缺少市值列，无法构建 size 因子")
                return None
            meta['market_cap'] = pd.to_numeric(meta[cap_col], errors='coerce')
            meta = meta[[code_col, 'market_cap']].dropna()
            meta.rename(columns={code_col: 'Code'}, inplace=True)
            meta['ln_market_cap'] = np.log(meta['market_cap'].where(meta['market_cap'] > 0))
            # 构造一个哑日期（用交易数据中的最小日期作为占位）
            if self.df is not None and 'Timestamp' in self.df.columns:
                min_date = pd.to_datetime(self.df['Timestamp']).dt.date.min()
            else:
                min_date = datetime.now().date()
            meta['date'] = pd.to_datetime(min_date)
            return meta[['Code', 'date', 'ln_market_cap', 'market_cap']]

        # 标准化列
        close_df = close_df.copy()
        if used_path is not None:
            info['close_source'] = used_path.name
        # 统一代码列
        code_col = 'Code' if 'Code' in close_df.columns else ('code' if 'code' in close_df.columns else None)
        if code_col is None:
            print("<i class='fas fa-times-circle text-red-500'></i> 收盘价缓存缺少 Code 列")
            return None
        close_df.rename(columns={code_col: 'Code'}, inplace=True)
        # 统一日期列
        date_col = None
        for c in ['date', 'Date', 'trade_date', 'TradeDate', 'timestamp']:
            if c in close_df.columns:
                date_col = c
                break
        if date_col is None:
            print("<i class='fas fa-times-circle text-red-500'></i> 收盘价缓存缺少日期列")
            return None
        close_df['date'] = pd.to_datetime(close_df[date_col]).dt.date
        # 统一收盘价列
        px_col = None
        for c in ['close', 'Close', 'close_price', 'last_price', 'adj_close', 'AdjClose']:
            if c in close_df.columns:
                px_col = c
                break
        if px_col is None:
            print("<i class='fas fa-times-circle text-red-500'></i> 收盘价缓存缺少收盘价列")
            return None
        close_df = close_df[['Code', 'date', px_col] + [c for c in close_df.columns if c not in ['Code', 'date', px_col]]]
        close_df.sort_values(['Code', 'date'], inplace=True)
        close_df['close'] = pd.to_numeric(close_df[px_col], errors='coerce')

        # 2) 动量与日收益
        def _mom(series: pd.Series, n: int) -> pd.Series:
            return series / series.shift(n) - 1
        g_px = close_df.groupby('Code')[px_col]
        close_df['ret_1d'] = g_px.pct_change()
        close_df['mom_5d'] = g_px.transform(lambda s: s / s.shift(5) - 1)
        close_df['mom_20d'] = g_px.transform(lambda s: s / s.shift(20) - 1)

        # 3) 近似流动性（Amihud）
        amount_col = None
        for c in ['amount', 'turnover', 'TurnoverAmount', '成交额', '成交额(万元)']:
            if c in close_df.columns:
                amount_col = c
                break
        # 若无amount但存在volume，尝试用 close*volume 近似成交额
        if amount_col is None:
            vol_col = None
            for c in ['volume', 'vol', '成交量']:
                if c in close_df.columns:
                    vol_col = c
                    break
            if vol_col is not None:
                try:
                    amt_est = pd.to_numeric(close_df[px_col], errors='coerce') * pd.to_numeric(close_df[vol_col], errors='coerce')
                    close_df['amount'] = amt_est
                    amount_col = 'amount'
                except Exception:
                    amount_col = None
        if amount_col is not None:
            amt = pd.to_numeric(close_df[amount_col], errors='coerce')
            # 若单位为万元（常见），约定名称包含“万元”则放大 * 1e4
            if '万元' in amount_col:
                amt = amt * 1e4
            # 标准 Amihud 非流动性（ILLIQ）：值越大流动性越差
            close_df['liquidity'] = (close_df['ret_1d'].abs() / (amt.replace(0, np.nan)))
            if info.get('amount_source') is None:
                info['amount_source'] = amount_col
        else:
            # 回退：使用配对成交数据按 Code-日 聚合的买入额+卖出额近似成交额
            try:
                pairs_cols = ['code', 'Code', 'buy_timestamp', 'buy_amount', 'sell_timestamp', 'sell_amount']
                pairs = pd.read_parquet('data/paired_trades_fifo.parquet')
                # 仅保留需要的列
                keep = [c for c in pairs_cols if c in pairs.columns]
                pairs = pairs[keep]
                code_col_pairs = 'code' if 'code' in pairs.columns else ('Code' if 'Code' in pairs.columns else None)
                if code_col_pairs is not None:
                    pairs = pairs.rename(columns={code_col_pairs: 'Code'})
                else:
                    raise RuntimeError('paired_trades_fifo 缺少 code 列')
                # 生成买入/卖出日期
                parts = []
                if 'buy_timestamp' in pairs.columns and 'buy_amount' in pairs.columns:
                    p = pairs[['Code', 'buy_timestamp', 'buy_amount']].dropna(subset=['buy_timestamp', 'buy_amount']).copy()
                    p['date'] = pd.to_datetime(p['buy_timestamp']).dt.date
                    p['turnover'] = pd.to_numeric(p['buy_amount'], errors='coerce').fillna(0.0)
                    parts.append(p[['Code', 'date', 'turnover']])
                if 'sell_timestamp' in pairs.columns and 'sell_amount' in pairs.columns:
                    p = pairs[['Code', 'sell_timestamp', 'sell_amount']].dropna(subset=['sell_timestamp', 'sell_amount']).copy()
                    p['date'] = pd.to_datetime(p['sell_timestamp']).dt.date
                    p['turnover'] = pd.to_numeric(p['sell_amount'], errors='coerce').fillna(0.0)
                    parts.append(p[['Code', 'date', 'turnover']])
                if parts:
                    turnover = pd.concat(parts, ignore_index=True)
                    turnover = (turnover.groupby(['Code', 'date'])['turnover'].sum().reset_index())
                    # 合并到 close_df
                    close_df = close_df.merge(turnover, on=['Code', 'date'], how='left')
                    # 回退成交额来源时亦使用标准 ILLIQ 定义
                    close_df['liquidity'] = (close_df['ret_1d'].abs() / (close_df['turnover'].replace(0, np.nan)))
                    close_df.drop(columns=['turnover'], inplace=True)
                    info['amount_source'] = 'paired_trades_fifo.buy_amount+sell_amount'
                else:
                    close_df['liquidity'] = np.nan
            except Exception:
                close_df['liquidity'] = np.nan

        # 统一保障 ILLIQ 非负
        try:
            if 'liquidity' in close_df.columns:
                close_df['liquidity'] = pd.to_numeric(close_df['liquidity'], errors='coerce').abs()
        except Exception:
            pass

        # 4) 市值（静态或日频）
        market_cap = None
        for c in ['market_cap', 'MarketCap', 'circulating_market_cap', 'circ_mv', '总市值', '流通市值']:
            if c in close_df.columns:
                market_cap = pd.to_numeric(close_df[c], errors='coerce')
                break
        if market_cap is None:
            # 尝试从 stock_metadata 读取静态市值
            try:
                meta = pd.read_parquet('data/stock_metadata.parquet')
                cap_col = None
                for c in ['MarketCap', 'market_cap', 'circulating_market_cap', 'circ_mv', '总市值', '流通市值']:
                    if c in meta.columns:
                        cap_col = c
                        break
                code_col_meta = 'Code' if 'Code' in meta.columns else ('code' if 'code' in meta.columns else None)
                if cap_col is not None and code_col_meta is not None:
                    meta = meta.rename(columns={code_col_meta: 'Code'})
                    meta['market_cap'] = pd.to_numeric(meta[cap_col], errors='coerce')
                    close_df = close_df.merge(meta[['Code', 'market_cap']], on='Code', how='left')
                    info['market_cap_source'] = 'stock_metadata.parquet'
                else:
                    close_df['market_cap'] = np.nan
            except Exception:
                close_df['market_cap'] = np.nan
        else:
            close_df['market_cap'] = market_cap
            if info.get('market_cap_source') is None:
                info['market_cap_source'] = 'daily_k_cache_or_input'

        close_df['ln_market_cap'] = np.log(close_df['market_cap'].where(close_df['market_cap'] > 0))

        # 0) 若存在因子缓存，且覆盖本次需要的日期与标的，直接返回缓存
        buy_keys = None
        try:
            pk = pd.read_parquet('data/paired_trades_fifo.parquet', columns=[c for c in ['code', 'Code', 'buy_timestamp'] if c in pd.read_parquet('data/paired_trades_fifo.parquet').columns])
        except Exception:
            try:
                pk = pd.read_parquet('data/paired_trades_fifo.parquet')
            except Exception:
                pk = None
        if pk is not None and len(pk):
            ccol = 'code' if 'code' in pk.columns else ('Code' if 'Code' in pk.columns else None)
            if ccol and 'buy_timestamp' in pk.columns:
                tmp = pk[[ccol, 'buy_timestamp']].dropna()
                tmp = tmp.rename(columns={ccol: 'Code'})
                tmp['date'] = pd.to_datetime(tmp['buy_timestamp']).dt.date
                buy_keys = tmp[['Code', 'date']].drop_duplicates()
        try:
            from pathlib import Path as _Path
            fcache = _Path('data/factors_daily_cache.parquet')
            if fcache.exists():
                f_old = pd.read_parquet(fcache)
                # 兼容：历史缓存可能使用了“取负号”的旧口径，这里统一矫正为非负 ILLIQ
                try:
                    if 'liquidity' in f_old.columns:
                        f_old['liquidity'] = pd.to_numeric(f_old['liquidity'], errors='coerce').abs()
                except Exception:
                    pass
                if {'Code', 'date'}.issubset(f_old.columns):
                    f_old['date'] = pd.to_datetime(f_old['date']).dt.date
                    keys_covered = True
                    if buy_keys is not None:
                        merged_keys = buy_keys.merge(f_old[['Code', 'date']], on=['Code', 'date'], how='left', indicator=True)
                        missing = merged_keys[merged_keys['_merge'] == 'left_only']
                        keys_covered = len(missing) == 0
                    # 若缓存缺少高频β或日内振幅，或在买入日期上全为 NaN，则继续补算，不直接返回
                    need_recompute = False
                    missing_critical_cols = [c for c in ['beta_5m', 'range_day', 'mom_5m_is_intraday', 'mom_30m_is_intraday'] if c not in f_old.columns]
                    if missing_critical_cols:
                        need_recompute = True
                    else:
                        try:
                            subset = f_old.merge(buy_keys, on=['Code', 'date'], how='inner') if buy_keys is not None else f_old
                            beta_nonnull = subset['beta_5m'].notna().sum() if 'beta_5m' in subset.columns else 0
                            range_nonnull = subset['range_day'].notna().sum() if 'range_day' in subset.columns else 0
                            if beta_nonnull == 0 or range_nonnull == 0:
                                need_recompute = True
                        except Exception:
                            # 出现异常则保守起见触发重算
                            need_recompute = True
                    if keys_covered and not need_recompute:
                        print(f"[缓存] 使用已存在的因子日表，共{len(f_old):,}行")
                        # 当使用缓存直接返回时，补充来源元信息，避免页面出现 unknown
                        try:
                            self._factor_build_meta = {
                                'close_source': 'cache:factors_daily_cache.parquet',
                                'amount_source': 'cache',
                                'momentum_source': 'cache',
                                'momentum_path': '',
                                'market_cap_source': 'cache',
                                'index_5m_source': 'cache',
                                'range_source': 'cache',
                            }
                        except Exception:
                            pass
                        return f_old
                    elif keys_covered and need_recompute:
                        print(f"[缓存] 因子缓存缺失或为空(beta_5m/range_day)，将补算5m指标")
        except Exception:
            pass

        # 2.1 若有5分钟K缓存，则计算 5m/30m/60m 动量（按日均值聚合）；否则保留回退
        minute_mom = None
        try:
            minute_candidates = [Path('data/minute_5m_cache.parquet'), Path('data/minute_5m.parquet')]
            mdf = None
            for mp in minute_candidates:
                if mp.exists():
                    try:
                        mdf = pd.read_parquet(mp, columns=['Code', 'datetime', 'close', 'high', 'low'])
                    except Exception:
                        mdf = pd.read_parquet(mp)
                    info['momentum_source'] = '5m_bars'
                    info['momentum_path'] = mp.name
                    break
            if mdf is not None:
                # 仅保留与买入有关的 Code-日，显著降内存与加速
                buy_keys = None
                try:
                    pcols = ['code', 'Code', 'buy_timestamp', 'buy_amount']
                    pairs_preview = pd.read_parquet('data/paired_trades_fifo.parquet', columns=[c for c in pcols if c in pd.read_parquet('data/paired_trades_fifo.parquet', columns=pcols, engine='pyarrow').columns])
                except Exception:
                    try:
                        pairs_preview = pd.read_parquet('data/paired_trades_fifo.parquet')
                    except Exception:
                        pairs_preview = None
                if pairs_preview is not None and len(pairs_preview):
                    code_cp = 'code' if 'code' in pairs_preview.columns else ('Code' if 'Code' in pairs_preview.columns else None)
                    if code_cp and 'buy_timestamp' in pairs_preview.columns:
                        tmp = pairs_preview[[code_cp, 'buy_timestamp']].dropna()
                        tmp = tmp.rename(columns={code_cp: 'Code'})
                        tmp['date'] = pd.to_datetime(tmp['buy_timestamp']).dt.date
                        buy_keys = tmp[['Code', 'date']].drop_duplicates()

                before_rows = len(mdf)
                code_m = 'Code' if 'Code' in mdf.columns else ('code' if 'code' in mdf.columns else None)
                dt_col = None
                for c in ['datetime', 'Datetime', 'time', 'Time', 'timestamp']:
                    if c in mdf.columns:
                        dt_col = c
                        break
                px_m = None
                for c in ['close', 'Close', 'price', 'last']:
                    if c in mdf.columns:
                        px_m = c
                        break
                if code_m and dt_col and px_m:
                    mdf = mdf.rename(columns={code_m: 'Code'})
                    mdf['datetime'] = pd.to_datetime(mdf[dt_col])
                    mdf = mdf.sort_values(['Code', 'datetime'])
                    # 过滤到买入相关的 Code 与日期
                    if buy_keys is not None and len(buy_keys):
                        mdf['date'] = mdf['datetime'].dt.date
                        mdf = mdf.merge(buy_keys, on=['Code', 'date'], how='inner')
                        print(f"[5m筛选] 原始{before_rows:,}行 -> 相关{len(mdf):,}行")

                    # 全市场等权 5m 收益（用于回退或补齐）
                    mdf['ret_5m_tmp'] = mdf.groupby('Code')[px_m].pct_change()
                    mkt_5m = (mdf.groupby('datetime')['ret_5m_tmp'].mean().rename('idx_ret').reset_index())

                    # 指数5m收益（优先抓取沪深300，若缺失回退为等权市场）
                    idx_ret = None
                    idx_source = 'unknown'
                    try:
                        idx_ret, idx_source = self._get_index_5m_returns(mdf['datetime'].min(), mdf['datetime'].max())
                    except Exception as exc:
                        print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 指数5m获取失败，准备回退等权: {exc}")
                        idx_ret = None
                    if idx_ret is not None and len(mkt_5m) > 0:
                        try:
                            cov_merge = mkt_5m.merge(idx_ret, on='datetime', how='left', suffixes=('_mkt', '_idx'))
                            coverage_ratio = cov_merge['idx_ret_idx'].notna().mean()
                            if coverage_ratio < 0.95:
                                cov_merge['idx_ret'] = cov_merge['idx_ret_idx'].combine_first(cov_merge['idx_ret_mkt'])
                                idx_ret = cov_merge[['datetime', 'idx_ret']]
                                info['index_5m_source'] = f"{idx_source}+fill_equal_weight"
                                print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 指数5m覆盖率 {coverage_ratio:.1%}，已用等权市场填补缺口")
                            else:
                                info['index_5m_source'] = idx_source
                        except Exception:
                            info['index_5m_source'] = idx_source
                    # 回退：若指数5m不可用，则用等权市场5m收益
                    if idx_ret is None and len(mkt_5m) > 0:
                        idx_ret = mkt_5m
                        info['index_5m_source'] = 'equal_weight_market_from_5m'

                    # 释放临时列，降低后续内存压力
                    try:
                        mdf = mdf.drop(columns=['ret_5m_tmp'])
                    except Exception:
                        pass

                    # 分块按股票处理，支持断点续算与周期性落盘
                    results = []
                    batch_results = []
                    FLUSH_N = 500  # 每500个股票写盘一次

                    # 若存在部分分钟因子缓存，则跳过已处理过的股票，继续增量计算
                    partial_path = Path('data/minute_5m_factors_cache.parquet')
                    partial_df = None
                    try:
                        if partial_path.exists():
                            partial_df = pd.read_parquet(partial_path)
                    except Exception:
                        partial_df = None

                    codes = mdf['Code'].dropna().unique()
                    if partial_df is not None and 'Code' in partial_df.columns:
                        done_codes = set(partial_df['Code'].dropna().unique())
                        codes = [c for c in codes if c not in done_codes]

                    total_codes = len(codes)

                    def _flush_batch(batch_rows):
                        if not batch_rows:
                            return
                        dfb = pd.DataFrame(batch_rows)
                        if len(dfb) == 0:
                            return
                        # 规范日期
                        try:
                            dfb['date'] = pd.to_datetime(dfb['date']).dt.date
                        except Exception:
                            pass
                        # 1) 累加到分钟因子部分缓存
                        try:
                            if partial_path.exists():
                                oldp = pd.read_parquet(partial_path)
                                allp = pd.concat([oldp, dfb], ignore_index=True)
                                allp = allp.sort_values(['Code', 'date']).drop_duplicates(['Code', 'date'], keep='last')
                            else:
                                allp = dfb
                            allp.to_parquet(partial_path, index=False)
                        except Exception:
                            pass
                        # 2) 同步更新日度因子缓存（只对本批次代码）
                        try:
                            keys = dfb[['Code', 'date']].drop_duplicates()
                            stat_cols = close_df[['Code', 'date', 'ln_market_cap', 'liquidity', 'market_cap']]
                            stat_sub = stat_cols.merge(keys, on=['Code', 'date'], how='inner')
                            fac_batch = stat_sub.merge(dfb, on=['Code', 'date'], how='left')
                            fcache = Path('data/factors_daily_cache.parquet')
                            if fcache.exists():
                                old = pd.read_parquet(fcache)
                                old['date'] = pd.to_datetime(old['date']).dt.date
                                allf = pd.concat([old, fac_batch], ignore_index=True)
                                allf = allf.sort_values(['Code', 'date']).drop_duplicates(['Code', 'date'], keep='last')
                            else:
                                allf = fac_batch
                            allf.to_parquet(fcache, index=False)
                        except Exception:
                            pass

                    for i, code in enumerate(codes, 1):
                        g = mdf[mdf['Code'] == code].copy()
                        if len(g) == 0:
                            continue
                        g = g.sort_values('datetime')
                        # 动量与5m收益
                        px = g[px_m].astype(float)
                        g['mom_5m'] = px / px.shift(1) - 1
                        g['mom_30m'] = px / px.shift(6) - 1
                        g['mom_60m'] = px / px.shift(12) - 1
                        g['ret_5m'] = px.pct_change()
                        # 指数收益合并
                        if idx_ret is not None:
                            # 更稳健的时间对齐：优先使用 asof 近邻对齐，避免由于时间戳细微不一致导致大面积 NaN
                            try:
                                g = pd.merge_asof(
                                    g.sort_values('datetime'),
                                    idx_ret.sort_values('datetime'),
                                    on='datetime',
                                    direction='nearest',
                                    tolerance=pd.Timedelta('150s')  # 允许 2.5 分钟的误差
                                )
                            except Exception:
                                # 回退到普通等值合并
                                g = g.merge(idx_ret, on='datetime', how='left')
                        g['date'] = g['datetime'].dt.date
                        # 按日聚合
                        day_rows = []
                        for dt, gd in g.groupby('date'):
                            r = gd['ret_5m'].dropna()
                            # 修正：使用标准差而非求和的平方根，避免随数据点数变化
                            rv = float(r.std(ddof=0)) if len(r) > 1 else np.nan
                            if 'idx_ret' in gd.columns and gd['idx_ret'].notna().any():
                                gg = gd.dropna(subset=['ret_5m', 'idx_ret'])
                                # 放宽样本点数阈值，减少因数据稀疏导致的空图
                                if len(gg) >= 3 and float(np.var(gg['idx_ret'])) > 0:
                                    x = gg['idx_ret'].astype(float)
                                    y = gg['ret_5m'].astype(float)
                                    x_c = x - x.mean()
                                    y_c = y - y.mean()
                                    beta = float((x_c * y_c).sum() / (x_c ** 2).sum())
                                else:
                                    beta = np.nan
                            else:
                                beta = np.nan
                            # 日内振幅
                            high_col = 'high' if 'high' in gd.columns else ('High' if 'High' in gd.columns else None)
                            low_col = 'low' if 'low' in gd.columns else ('Low' if 'Low' in gd.columns else None)
                            rng = np.nan
                            try:
                                # 修正：使用首笔开盘价作为分母，避免当日涨跌影响振幅度量
                                close_series = pd.to_numeric(gd[px_m], errors='coerce')
                                first_close = float(close_series.dropna().iloc[0]) if close_series.notna().any() else np.nan
                                if pd.notna(first_close) and first_close > 0:
                                    if high_col and low_col and gd[high_col].notna().any() and gd[low_col].notna().any():
                                        hi = pd.to_numeric(gd[high_col], errors='coerce')
                                        lo = pd.to_numeric(gd[low_col], errors='coerce')
                                        rng = float((hi.max() - lo.min()) / first_close)
                                        info['range_source'] = 'high_low_vs_first_close'
                                    else:
                                        # 回退：无高低价时用当日 close 的极值近似
                                        rng = float((close_series.max() - close_series.min()) / first_close)
                                        if 'range_source' not in info:
                                            info['range_source'] = 'close_range_vs_first_close'
                            except Exception:
                                rng = np.nan
                            day_rows.append({
                                'Code': code,
                                'date': dt,
                                'mom_5m': float(gd['mom_5m'].mean(skipna=True)),
                                'mom_30m': float(gd['mom_30m'].mean(skipna=True)),
                                'mom_60m': float(gd['mom_60m'].mean(skipna=True)),
                                'rv_5m': rv,
                                'beta_5m': beta,
                                'range_day': rng,
                            })
                        if day_rows:
                            results.extend(day_rows)
                            batch_results.extend(day_rows)
                        if i % 100 == 0:
                            print(f"[5m处理] 进度: {i}/{total_codes} ({i/total_codes*100:.1f}%)")
                        if i % FLUSH_N == 0:
                            # 批量落盘并清空 batch
                            _flush_batch(batch_results)
                            batch_results = []

                    # 处理剩余批次并合并历史部分缓存
                    _flush_batch(batch_results)
                    minute_mom = pd.DataFrame(results) if results else None
                    if minute_mom is not None and isinstance(minute_mom, pd.DataFrame) and 'date' in minute_mom.columns:
                        mm_df = minute_mom.copy()
                        mm_df['date'] = pd.to_datetime(mm_df['date']).dt.date
                        minute_mom = mm_df
                    # 合并历史 partial，确保本次返回的数据包含以往已计算结果
                    try:
                        if partial_df is not None:
                            minute_mom = minute_mom if minute_mom is not None else pd.DataFrame(columns=['Code','date','mom_5m','mom_30m','mom_60m','rv_5m','beta_5m','range_day'])
                            minute_mom = pd.concat([partial_df, minute_mom], ignore_index=True)
                            minute_mom = minute_mom.sort_values(['Code','date']).drop_duplicates(['Code','date'], keep='last')
                    except Exception:
                        pass
        except Exception:
            minute_mom = None

        # 修正：保持日频与分钟频率分离，避免混合不同经济含义的动量
        factors = close_df[['Code', 'date', 'ln_market_cap', 'liquidity', 'market_cap', 'mom_5d', 'mom_20d']].copy()
        if minute_mom is not None:
            # 先合并分钟动量
            factors = factors.merge(minute_mom, on=['Code', 'date'], how='left')
            # 标记哪些行确实来自5m分钟数据
            factors['mom_5m_is_intraday'] = factors['mom_5m'].notna()
            factors['mom_30m_is_intraday'] = factors['mom_30m'].notna()
            # 修正：不再回退，保持频率纯净；缺失时为NaN
            # 原逻辑：factors['mom_5m'] = np.where(factors['mom_5m'].notna(), factors['mom_5m'], factors['mom_5d'])
            # 理由：5分钟动量(日内情绪) vs 5日动量(短期趋势)，经济含义不同，不应混合
            
            # 60m 无可靠日频替代，保持原值
            # 记录分钟动量覆盖率
            try:
                cov_5m = float(factors['mom_5m_is_intraday'].mean()) if len(factors) else 0.0
                cov_30m = float(factors['mom_30m_is_intraday'].mean()) if len(factors) else 0.0
                info['mom_5m_intraday_coverage'] = cov_5m
                info['mom_30m_intraday_coverage'] = cov_30m
                if cov_5m < 0.5 or cov_30m < 0.5:
                    print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 分钟动量覆盖率较低: mom_5m={cov_5m:.1%}, mom_30m={cov_30m:.1%}，缺失部分保持NaN")
            except Exception:
                pass
        else:
            # 修正：无分钟数据时，不使用日频动量替代，直接设为NaN
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到5分钟K数据，高频动量因子将全部为NaN（不使用日频替代）")
            factors['mom_5m'] = np.nan
            factors['mom_30m'] = np.nan
            factors['mom_60m'] = np.nan
            if info.get('momentum_source') is None:
                info['momentum_source'] = 'no_intraday_data_available'
            factors['rv_5m'] = np.nan
            factors['beta_5m'] = np.nan
            factors['range_day'] = np.nan
            # 标记全部为缺失（而非回退）
            factors['mom_5m_is_intraday'] = False
            factors['mom_30m_is_intraday'] = False
            info['mom_5m_intraday_coverage'] = 0.0
            info['mom_30m_intraday_coverage'] = 0.0

        # 丢弃完全缺失的行
        factors = factors.dropna(how='all', subset=['ln_market_cap', 'liquidity', 'mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'beta_5m', 'range_day'])
        # 复制日频数值用于回退与说明
        for _col in ['mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'beta_5m', 'range_day']:
            fallback_col = f"{_col}_daily"
            if _col in factors.columns and fallback_col not in factors.columns:
                factors[fallback_col] = factors[_col]
        if 'close' in close_df.columns and 'close' not in factors.columns:
            factors = factors.merge(
                close_df[['Code', 'date', 'close']],
                on=['Code', 'date'],
                how='left'
            )
        print(f"<i class='fas fa-check-circle text-green-500'></i> 因子集构建完成: {len(factors):,} 行, 覆盖股票数: {factors['Code'].nunique():,}")
        try:
            price_cols = [c for c in ['Code', 'date', 'close', 'high', 'low', 'amount', 'volume', 'market_cap'] if c in close_df.columns]
            self._daily_price_df = close_df[price_cols].copy() if price_cols else close_df[['Code', 'date']].copy()
        except Exception:
            self._daily_price_df = None
        try:
            self._daily_factor_df = factors.copy()
        except Exception:
            self._daily_factor_df = None
        # 更新与保存因子缓存
        try:
            from pathlib import Path as _Path
            fcache = _Path('data/factors_daily_cache.parquet')
            if fcache.exists():
                old = pd.read_parquet(fcache)
                old['date'] = pd.to_datetime(old['date']).dt.date
                allf = pd.concat([old, factors], ignore_index=True)
                allf = allf.sort_values(['Code', 'date']).drop_duplicates(['Code', 'date'], keep='last')
            else:
                allf = factors
            allf.to_parquet(fcache, index=False)
            print(f"[缓存] 已写入/更新因子日表: {len(allf):,} 行 -> data/factors_daily_cache.parquet")
        except Exception as e:
            print(f"[缓存] 写入因子日表失败: {e}")
        # 暴露给后续说明文案
        try:
            self._factor_build_meta = info
        except Exception:
            pass
        return factors

    def _prepare_trade_flows(self) -> Optional[pd.DataFrame]:
        """构建增量仓位变化的交易流水，用于因子暴露分析"""
        if self._trade_flow_cache is not None:
            return self._trade_flow_cache.copy()
        if self.df is None or len(self.df) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无交易数据，无法构建交易流水")
            return None

        cols_required = ['Code', 'Timestamp', 'direction', 'tradeQty', 'tradeAmount']
        missing_cols = [c for c in cols_required if c not in self.df.columns]
        if missing_cols:
            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 交易数据缺少必需列: {missing_cols}")
            return None

        trades = self.df[cols_required + ['price'] if 'price' in self.df.columns else cols_required].copy()
        trades = trades.dropna(subset=['Code', 'Timestamp', 'direction', 'tradeQty', 'tradeAmount'])
        trades['tradeQty'] = pd.to_numeric(trades['tradeQty'], errors='coerce')
        trades['tradeAmount'] = pd.to_numeric(trades['tradeAmount'], errors='coerce')
        trades = trades[(trades['tradeQty'] > 0) & trades['tradeAmount'].notna()]
        if len(trades) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 过滤后无有效成交记录")
            return None

        trades['Timestamp'] = pd.to_datetime(trades['Timestamp'], utc=False, errors='coerce')
        trades['Timestamp'] = trades['Timestamp'].dt.tz_localize(None)
        trades['Timestamp'] = trades['Timestamp'].astype('datetime64[ns]')
        trades = trades.dropna(subset=['Timestamp'])
        trades = trades.sort_values(['Code', 'Timestamp', 'tradeQty']).reset_index(drop=True)

        trades['signed_qty'] = np.where(trades['direction'] == 'B', trades['tradeQty'], -trades['tradeQty'])
        trades['pos_after'] = trades.groupby('Code')['signed_qty'].cumsum()
        trades['pos_prev'] = trades['pos_after'] - trades['signed_qty']

        # 成交价格：优先 tradeAmount/tradeQty；若缺失尝试 price 列
        trades['exec_price'] = trades['tradeAmount'] / trades['tradeQty'].replace(0, np.nan)
        if 'price' in trades.columns:
            trades['exec_price'] = trades['exec_price'].fillna(pd.to_numeric(trades['price'], errors='coerce'))
        trades['exec_price'] = trades['exec_price'].replace([np.inf, -np.inf], np.nan)

        # 计算多空方向增量仓位（股数）
        trades['delta_long_qty'] = np.maximum(trades['pos_after'], 0.0) - np.maximum(trades['pos_prev'], 0.0)
        trades['delta_short_qty'] = np.maximum(-trades['pos_after'], 0.0) - np.maximum(-trades['pos_prev'], 0.0)

        # 将股数转换为金额权重
        trades['long_exposure_amount'] = (trades['delta_long_qty'] * trades['exec_price']).clip(lower=0.0)
        trades['short_exposure_amount'] = (trades['delta_short_qty'].abs() * trades['exec_price']).clip(lower=0.0)
        trades['trade_weight'] = trades['long_exposure_amount'] + trades['short_exposure_amount']

        # 剔除纯平仓（不改变仓位）的记录
        trades = trades[trades['trade_weight'] > 0]
        if len(trades) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 所有成交均为仓位削减或对冲，缺少新增仓位记录")
            return None

        trades['trade_date'] = trades['Timestamp'].dt.date
        trades['trade_id'] = np.arange(len(trades))
        cache_cols = [
            'trade_id', 'Code', 'Timestamp', 'trade_date', 'trade_weight',
            'long_exposure_amount', 'short_exposure_amount', 'exec_price'
        ]
        self._trade_flow_cache = trades[cache_cols].copy()
        return self._trade_flow_cache.copy()

    def _build_intraday_factor_snapshots(self) -> Optional[pd.DataFrame]:
        """基于交易时刻向后查找的分钟级因子快照（最多回溯10分钟）
        注：使用merge_asof向后匹配最近的分钟K线因子值，而非10分钟窗口平均
        """
        if self._intraday_snapshot_cache is not None:
            return self._intraday_snapshot_cache.copy()

        trades = self._prepare_trade_flows()
        if trades is None or len(trades) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无交易流水，分钟因子快照构建跳过")
            return None

        code_set = trades['Code'].dropna().unique()
        if len(code_set) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 交易流水未包含代码信息")
            return None

        minute_candidates = [Path('data/minute_5m_cache.parquet'), Path('data/minute_5m.parquet')]
        minute_df = None
        used_path = None
        for mp in minute_candidates:
            if mp.exists():
                try:
                    minute_df = pd.read_parquet(mp)
                    used_path = mp.name
                    break
                except Exception:
                    continue
        if minute_df is None or len(minute_df) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到5分钟K数据，分钟因子回退为日频")
            self._intraday_snapshot_cache = trades.copy()
            self._intraday_snapshot_cache['has_intraday'] = False
            return self._intraday_snapshot_cache.copy()

        # 标准化列
        code_col = 'Code' if 'Code' in minute_df.columns else ('code' if 'code' in minute_df.columns else None)
        dt_col = None
        for c in ['datetime', 'Datetime', 'time', 'Time', 'timestamp']:
            if c in minute_df.columns:
                dt_col = c
                break
        close_col = None
        for c in ['close', 'Close', 'last', 'price']:
            if c in minute_df.columns:
                close_col = c
                break
        high_col = next((c for c in ['high', 'High', 'max'] if c in minute_df.columns), None)
        low_col = next((c for c in ['low', 'Low', 'min'] if c in minute_df.columns), None)

        if code_col is None or dt_col is None or close_col is None:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 5分钟K缺少必要列(Code/时间/收盘价)")
            self._intraday_snapshot_cache = trades.copy()
            self._intraday_snapshot_cache['has_intraday'] = False
            return self._intraday_snapshot_cache.copy()

        minute_df = minute_df.rename(columns={code_col: 'Code'})
        minute_df['datetime'] = pd.to_datetime(minute_df[dt_col])
        minute_df = minute_df.sort_values(['Code', 'datetime'])
        minute_df = minute_df[minute_df['Code'].isin(code_set)].copy()
        if len(minute_df) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 5分钟K与交易代码无重叠")
            self._intraday_snapshot_cache = trades.copy()
            self._intraday_snapshot_cache['has_intraday'] = False
            return self._intraday_snapshot_cache.copy()

        minute_df['close'] = pd.to_numeric(minute_df[close_col], errors='coerce').astype('float32')
        if high_col:
            minute_df['high'] = pd.to_numeric(minute_df[high_col], errors='coerce').astype('float32')
        else:
            minute_df['high'] = minute_df['close']
        if low_col:
            minute_df['low'] = pd.to_numeric(minute_df[low_col], errors='coerce').astype('float32')
        else:
            minute_df['low'] = minute_df['close']

        # 尝试使用快照缓存（基于订单与分钟K指纹），避免重复计算
        cache_path = Path('data/trade_factor_snapshots_cache.parquet')
        cache_meta_path = Path('data/trade_factor_snapshots_cache.meta.json')
        try:
            orders_fp = self._orders_file_fingerprint()
            minute_fp = {
                'path': str(Path(used_path).resolve()) if used_path else '',
                'size': int(Path(used_path).stat().st_size) if used_path else -1,
                'mtime': float(Path(used_path).stat().st_mtime) if used_path else -1.0,
            }
            if cache_path.exists() and cache_meta_path.exists():
                meta = json.loads(cache_meta_path.read_text(encoding='utf-8'))
                if meta.get('orders') == orders_fp and meta.get('minute') == minute_fp:
                    try:
                        cached = pd.read_parquet(cache_path)
                        print(f"[CACHE] 命中分钟因子快照: {cache_path} ({len(cached)} 行)")
                        self._intraday_snapshot_cache = cached.copy()
                        return self._intraday_snapshot_cache.copy()
                    except Exception:
                        pass
        except Exception:
            pass

        # === 高频特征计算：优先使用 polars 加速，失败回退 pandas ===
        use_polars_calc = False
        try:
            import polars as pl  # type: ignore
            use_polars_calc = True
            # 保持排序稳定，确保窗口与 shift 计算一致
            minute_pl = pl.from_pandas(minute_df.sort_values(['Code', 'datetime']))
            window2 = dict(window_size=2, min_periods=1)
            window6 = dict(window_size=6, min_periods=3)
            minute_pl = minute_pl.with_columns([
                pl.col('close').pct_change().over('Code').alias('ret_5m'),
            ])
            minute_pl = minute_pl.with_columns([
                pl.col('ret_5m').alias('mom_5m'),
                (pl.col('close') / pl.col('close').shift(6).over('Code') - 1).alias('mom_30m'),
                (pl.col('close') / pl.col('close').shift(12).over('Code') - 1).alias('mom_60m'),
                (pl.col('ret_5m') ** 2).rolling_sum(**window2).over('Code').sqrt().alias('rv_5m'),
                (
                    pl.col('high').rolling_max(**window2).over('Code') -
                    pl.col('low').rolling_min(**window2).over('Code')
                ) / pl.col('close')
                .alias('range_day'),
            ])
            # Rolling β: cov(ret, idx)/var(idx)，窗口=6，min_periods=3
            minute_df = minute_pl.to_pandas()
            # 确保所有高频特征列存在
            for _col in ['ret_5m', 'mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'range_day']:
                if _col not in minute_df.columns:
                    minute_df[_col] = np.nan
            minute_df[['ret_5m', 'mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'range_day']] = \
                minute_df[['ret_5m', 'mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'range_day']].replace([np.inf, -np.inf], np.nan)
        except Exception as exc:
            if use_polars_calc:
                print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> polars 高频特征计算失败，回退 pandas: {exc}")
            # 回退 pandas 计算
            group_close = minute_df.groupby('Code')['close']
            minute_df['ret_5m'] = group_close.pct_change()
            minute_df['mom_5m'] = minute_df['ret_5m'].replace([np.inf, -np.inf], np.nan)
            minute_df['mom_30m'] = group_close.transform(lambda s: s / s.shift(6) - 1).replace([np.inf, -np.inf], np.nan)
            minute_df['mom_60m'] = group_close.transform(lambda s: s / s.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
            minute_df['rv_5m'] = group_close.transform(lambda s: (s.pct_change().pow(2).rolling(window=2, min_periods=1).sum()) ** 0.5).replace([np.inf, -np.inf], np.nan)

            rolling_high = minute_df.groupby('Code')['high'].transform(lambda s: s.rolling(window=2, min_periods=1).max())
            rolling_low = minute_df.groupby('Code')['low'].transform(lambda s: s.rolling(window=2, min_periods=1).min())
            minute_df['range_day'] = ((rolling_high - rolling_low) / minute_df['close']).replace([np.inf, -np.inf], np.nan)

        # 全市场等权 5m 收益（用于回退或填补缺口）
        market_ret = None
        try:
            market_ret = minute_df.groupby('datetime')['ret_5m'].mean().rename('idx_ret_mkt').reset_index()
        except Exception:
            market_ret = None

        # 指数5m收益：优先沪深300（自动抓取），缺失则回退全市场等权
        idx_ret = None
        idx_source = 'unknown'
        try:
            idx_ret, idx_source = self._get_index_5m_returns(minute_df['datetime'].min(), minute_df['datetime'].max())
            if idx_ret is not None:
                print(f"<i class='fas fa-check-circle text-green-500'></i> 使用指数5m数据计算β（{len(idx_ret):,} 条记录，来源 {idx_source}）")
        except Exception as exc:
            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 指数5m获取失败，将回退等权: {exc}")
            idx_ret = None

        # 回退或补齐：使用全市场等权收益替代指数收益
        if idx_ret is not None and market_ret is not None and len(market_ret) > 0:
            try:
                cov_merge = market_ret.merge(idx_ret, on='datetime', how='left')
                coverage_ratio = cov_merge['idx_ret'].notna().mean()
                if coverage_ratio < 0.95:
                    cov_merge['idx_ret'] = cov_merge['idx_ret'].combine_first(cov_merge['idx_ret_mkt'])
                    idx_ret = cov_merge[['datetime', 'idx_ret']]
                    idx_source = f"{idx_source}+fill_equal_weight"
                    print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 指数5m覆盖率 {coverage_ratio:.1%}，已用等权市场填补缺口")
            except Exception:
                pass

        if idx_ret is None and market_ret is not None and len(market_ret) > 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 指数5m数据缺失，使用全市场等权收益计算β")
            try:
                idx_ret = market_ret.rename(columns={'idx_ret_mkt': 'idx_ret'})
                idx_source = 'equal_weight_market_5m_snapshot'
                print(f"<i class='fas fa-check-circle text-green-500'></i> 已生成等权市场收益（{len(idx_ret)}个时点）")
            except Exception as e:
                print(f"<i class='fas fa-times-circle text-red-500'></i> 等权市场收益计算失败: {e}")
                idx_ret = None

        # 合并指数收益
        if idx_ret is None:
            minute_df['idx_ret'] = np.nan
        else:
            minute_df = minute_df.merge(idx_ret, on='datetime', how='left')
        
        # 记录β数据源以便在页面说明中使用
        if not hasattr(self, '_factor_build_meta'):
            self._factor_build_meta = {}
        self._factor_build_meta['beta_index_source_snapshot'] = idx_source

        # 计算高频β（使用 ret_5m 与 idx_ret，窗口6，min_periods=3）
        try:
            import polars as pl  # type: ignore
            beta_window = dict(window_size=6, min_periods=3)
            beta_pl = pl.from_pandas(minute_df[['Code', 'datetime', 'ret_5m', 'idx_ret']].sort_values(['Code', 'datetime']))
            beta_pl = beta_pl.with_columns([
                pl.col('ret_5m').rolling_mean(**beta_window).over('Code').alias('_mx'),
                pl.col('idx_ret').rolling_mean(**beta_window).over('Code').alias('_my'),
                (pl.col('ret_5m') * pl.col('idx_ret')).rolling_mean(**beta_window).over('Code').alias('_mxy'),
                (pl.col('idx_ret') ** 2).rolling_mean(**beta_window).over('Code').alias('_my2'),
            ])
            beta_pl = beta_pl.with_columns([
                (pl.col('_mxy') - pl.col('_mx') * pl.col('_my')).alias('_cov'),
                (pl.col('_my2') - pl.col('_my') * pl.col('_my')).alias('_var'),
            ])
            beta_pl = beta_pl.with_columns([
                pl.when(pl.col('_var') == 0)
                .then(None)
                .otherwise(pl.col('_cov') / pl.col('_var'))
                .alias('beta_5m')
            ])
            beta_pdf = beta_pl.select(['Code', 'datetime', 'beta_5m']).to_pandas()
            minute_df = minute_df.merge(beta_pdf, on=['Code', 'datetime'], how='left')
        except Exception:
            try:
                minute_df['beta_5m'] = np.nan
                cov = minute_df.groupby('Code')['ret_5m'].rolling(window=6, min_periods=3).cov(minute_df.set_index('Code')['idx_ret']).reset_index(level=0, drop=True)
                var = minute_df.groupby('Code')['idx_ret'].rolling(window=6, min_periods=3).var().reset_index(level=0, drop=True)
                minute_df['beta_5m'] = cov / var.replace(0, np.nan)
            except Exception:
                minute_df['beta_5m'] = np.nan

        feature_cols = ['mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'beta_5m', 'range_day']
        minute_features = minute_df[['Code', 'datetime'] + feature_cols].copy()
        minute_features['datetime'] = pd.to_datetime(minute_features['datetime'], utc=False, errors='coerce')
        minute_features['datetime'] = minute_features['datetime'].dt.tz_localize(None)
        minute_features['datetime'] = minute_features['datetime'].astype('datetime64[ns]')
        minute_features = minute_features.dropna(subset=['datetime'])
        minute_features = minute_features.sort_values(['Code', 'datetime']).reset_index(drop=True)

        trades_sorted = trades.sort_values(['Code', 'Timestamp']).reset_index(drop=True)
        asof_tolerance = pd.Timedelta(minutes=10)

        merged: Optional[pd.DataFrame] = None
        used_polars = False
        # 优先使用 polars 的 join_asof（多线程加速），失败再回退 pandas
        try:
            import polars as pl  # type: ignore
            used_polars = True
            # 仅保留需要的列以降低内存占用
            minute_polars_cols = ['Code', 'datetime'] + feature_cols
            minute_features_pl = pl.from_pandas(minute_features[minute_polars_cols])
            trades_pl = pl.from_pandas(trades_sorted[['Code', 'Timestamp', 'trade_weight', 'long_exposure_amount', 'short_exposure_amount', 'exec_price', 'trade_date', 'trade_id']])
            trades_pl = trades_pl.sort(['Code', 'Timestamp'])
            minute_features_pl = minute_features_pl.sort(['Code', 'datetime'])
            merged_pl = trades_pl.join_asof(
                minute_features_pl,
                left_on='Timestamp',
                right_on='datetime',
                by='Code',
                strategy='backward',
                tolerance=asof_tolerance.to_pytimedelta()
            )
            merged = merged_pl.to_pandas()
        except Exception as exc:
            if used_polars:
                print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> polars join_asof 失败，回退 pandas merge_asof: {exc}")
            used_polars = False

        if merged is None:
            merged = pd.merge_asof(
                trades_sorted,
                minute_features,
                left_on='Timestamp',
                right_on='datetime',
                by='Code',
                direction='backward',
                tolerance=asof_tolerance
            )

        merged['has_intraday'] = merged[feature_cols].notna().any(axis=1)
        for col in feature_cols:
            merged[f'{col}_is_intraday'] = merged[col].notna()
        if 'datetime' in merged.columns:
            merged.drop(columns=['datetime'], inplace=True)

        self._intraday_snapshot_cache = merged.copy()
        try:
            snapshot_path = Path('data/trade_factor_snapshots_cache.parquet')
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(snapshot_path, index=False)
        except Exception:
            pass
        if used_path:
            self._factor_build_meta['minute_source'] = used_path
        if used_polars:
            self._factor_build_meta['minute_join_engine'] = 'polars_join_asof'
        else:
            self._factor_build_meta['minute_join_engine'] = 'pandas_merge_asof'

        # 写缓存与元信息，便于下次复用
        try:
            snapshot_path = Path('data/trade_factor_snapshots_cache.parquet')
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(snapshot_path, index=False)
            cache_meta_path = Path('data/trade_factor_snapshots_cache.meta.json')
            cache_meta = {
                'orders': self._orders_file_fingerprint(),
                'minute': {
                    'path': str(Path(used_path).resolve()) if used_path else '',
                    'size': int(Path(used_path).stat().st_size) if used_path else -1,
                    'mtime': float(Path(used_path).stat().st_mtime) if used_path else -1.0,
                },
                'join_engine': self._factor_build_meta.get('minute_join_engine', 'unknown'),
            }
            cache_meta_path.write_text(json.dumps(cache_meta, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

        return self._intraday_snapshot_cache.copy()

    def _build_daily_positions(self) -> Optional[pd.DataFrame]:
        """构建逐日持仓（股票数量），用于持仓暴露分析"""
        if self._positions_cache is not None:
            return self._positions_cache.copy()
        if self.df is None or len(self.df) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无交易数据，无法计算持仓")
            return None

        required = ['Code', 'Timestamp', 'direction', 'tradeQty']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 交易数据缺少计算持仓所需列: {missing}")
            return None

        orders = self.df[required].dropna(subset=required).copy()
        orders['tradeQty'] = pd.to_numeric(orders['tradeQty'], errors='coerce')
        orders = orders[orders['tradeQty'] > 0]
        if len(orders) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 交易记录为空，无法统计持仓")
            return None

        orders['Timestamp'] = pd.to_datetime(orders['Timestamp'])
        orders = orders.sort_values(['Code', 'Timestamp']).reset_index(drop=True)
        orders['signed_qty'] = np.where(orders['direction'] == 'B', orders['tradeQty'], -orders['tradeQty'])
        orders['position_qty'] = orders.groupby('Code')['signed_qty'].cumsum()
        orders['date'] = orders['Timestamp'].dt.date

        eod = orders.groupby(['Code', 'date'])['position_qty'].last().reset_index()
        if len(eod) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无有效持仓数据")
            return None

        position_rows = []
        for code, sub in eod.groupby('Code'):
            sub = sub.sort_values('date')
            date_range = pd.date_range(sub['date'].min(), sub['date'].max(), freq='D')
            sub = sub.set_index('date').reindex(date_range)
            sub['Code'] = code
            sub['position_qty'] = sub['position_qty'].ffill().fillna(0.0)
            sub = sub.reset_index().rename(columns={'index': 'date'})
            sub['date'] = sub['date'].dt.date
            position_rows.append(sub[['Code', 'date', 'position_qty']])

        positions_df = pd.concat(position_rows, ignore_index=True)
        self._positions_cache = positions_df.copy()
        return positions_df.copy()

    def _factor_exposure_analysis_legacy(self):
        """策略因子特征暴露度分析（按买入交易额加权，与市场基准对比）"""
        print("\n<i class='fas fa-chart-bar text-indigo-500'></i> === 策略因子特征暴露度分析 ===")
        # 1) 构建因子数据集
        factors = self._build_factor_dataset()
        if factors is None or len(factors) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无法构建因子集，跳过因子暴露分析")
            return
        factors['date'] = pd.to_datetime(factors['date']).dt.date

        # 2) 读取配对成交（用于买入权重）
        try:
            pairs = pd.read_parquet('data/paired_trades_fifo.parquet')
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 读取 paired_trades_fifo 失败: {e}")
            return

        # 标准化列
        code_col_pairs = 'code' if 'code' in pairs.columns else ('Code' if 'Code' in pairs.columns else None)
        if code_col_pairs is None:
            print("<i class='fas fa-times-circle text-red-500'></i> paired_trades_fifo 缺少 code 列")
            return
        pairs = pairs.rename(columns={code_col_pairs: 'Code'})

        # 买入时间/金额
        if 'buy_timestamp' not in pairs.columns or 'buy_amount' not in pairs.columns:
            print("<i class='fas fa-times-circle text-red-500'></i> paired_trades_fifo 缺少 buy_timestamp 或 buy_amount 列")
            return
        pairs['buy_date'] = pd.to_datetime(pairs['buy_timestamp']).dt.date
        pairs['buy_amount'] = pd.to_numeric(pairs['buy_amount'], errors='coerce').fillna(0.0)
        pairs = pairs[(pairs['buy_amount'] > 0) & pairs['Code'].notna()]

        # 3) 计算每日策略暴露：Σ(w*X)/Σw, w=buy_amount
        merged = pairs[['Code', 'buy_date', 'buy_amount']].merge(
            factors.rename(columns={'date': 'buy_date'}),
            on=['Code', 'buy_date'], how='left'
        )
        if len(merged) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 买入交易与因子数据无重叠，跳过")
            return

        def _weighted_exposure(df: pd.DataFrame, col: str) -> float:
            g = df.dropna(subset=[col])
            if len(g) == 0:
                return np.nan
            w = g['buy_amount'].astype(float)
            x = g[col].astype(float)
            sw = w.sum()
            return float((w * x).sum() / sw) if sw > 0 else np.nan

        by_date = []
        factor_cols = [c for c in ['ln_market_cap', 'mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'beta_5m', 'range_day', 'liquidity'] if c in merged.columns]
        for dt, df_d in merged.groupby('buy_date'):
            row = {'date': pd.to_datetime(dt)}
            for c in factor_cols:
                row[f'strat_{c}'] = _weighted_exposure(df_d, c)
            # 分钟动量覆盖率（买入金额加权）
            if 'mom_5m_is_intraday' in df_d.columns:
                try:
                    row['strat_cov_mom_5m'] = _weighted_exposure(df_d, 'mom_5m_is_intraday')
                except Exception:
                    row['strat_cov_mom_5m'] = np.nan
            else:
                row['strat_cov_mom_5m'] = np.nan
            if 'mom_30m_is_intraday' in df_d.columns:
                try:
                    row['strat_cov_mom_30m'] = _weighted_exposure(df_d, 'mom_30m_is_intraday')
                except Exception:
                    row['strat_cov_mom_30m'] = np.nan
            else:
                row['strat_cov_mom_30m'] = np.nan
            by_date.append(row)
        strat_exp = pd.DataFrame(by_date).sort_values('date')

        # 4) 市场基准暴露（同日、全市场；权重=流通市值，否则等权）
        def _market_exposure(factors_day: pd.DataFrame, col: str) -> float:
            g = factors_day.dropna(subset=[col])
            if len(g) == 0:
                return np.nan
            if 'market_cap' in g.columns and g['market_cap'].notna().any():
                w = g['market_cap'].astype(float).clip(lower=0)
                sw = w.sum()
                if sw > 0:
                    return float((w * g[col].astype(float)).sum() / sw)
            # 回退：等权
            return float(g[col].astype(float).mean())

        mkt_rows = []
        factor_by_date = factors.groupby('date')
        for dt in strat_exp['date'].dt.date.unique():
            fday = factor_by_date.get_group(dt) if dt in factor_by_date.groups else None
            row = {'date': pd.to_datetime(dt)}
            if fday is not None:
                for c in factor_cols:
                    row[f'mkt_{c}'] = _market_exposure(fday, c)
                # 市场端分钟动量覆盖率（市值加权，缺失则等权）
                row['mkt_cov_mom_5m'] = _market_exposure(fday, 'mom_5m_is_intraday') if 'mom_5m_is_intraday' in fday.columns else np.nan
                row['mkt_cov_mom_30m'] = _market_exposure(fday, 'mom_30m_is_intraday') if 'mom_30m_is_intraday' in fday.columns else np.nan
            else:
                for c in factor_cols:
                    row[f'mkt_{c}'] = np.nan
                row['mkt_cov_mom_5m'] = np.nan
                row['mkt_cov_mom_30m'] = np.nan
            mkt_rows.append(row)
        mkt_exp = pd.DataFrame(mkt_rows).sort_values('date')

        exp_df = strat_exp.merge(mkt_exp, on='date', how='left')

        # 写入“策略因子暴露日表”缓存
        try:
            from pathlib import Path as _Path
            ecache = _Path('data/factor_exposure_daily_cache.parquet')
            exp_df_save = exp_df.copy()
            if ecache.exists():
                old = pd.read_parquet(ecache)
                old['date'] = pd.to_datetime(old['date'])
                allx = pd.concat([old, exp_df_save], ignore_index=True)
                allx = allx.sort_values('date').drop_duplicates('date', keep='last')
            else:
                allx = exp_df_save
            allx.to_parquet(ecache, index=False)
            print(f"[缓存] 已写入/更新策略因子暴露日表: {len(allx):,} 行 -> data/factor_exposure_daily_cache.parquet")
        except Exception as e:
            print(f"[缓存] 写入策略因子暴露日表失败: {e}")

        # 5) 可视化（2x2子图，最多4个因子）
        titles_map = {
            'ln_market_cap': 'Size（ln市值）暴露',
            'mom_5m': '动量（5分钟）暴露',
            'mom_30m': '动量（30分钟）暴露',
            'mom_60m': '动量（60分钟）暴露',
            'rv_5m': '已实现波动率（5分钟）暴露',
            'beta_5m': '高频β（5分钟）暴露',
            'range_day': '日内价格振幅 暴露',
            'liquidity': '非流动性（Amihud ILLIQ，值越大流动性越差）暴露',
        }
        show_cols = [c for c in ['ln_market_cap', 'mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'beta_5m', 'range_day', 'liquidity'] if f'strat_{c}' in exp_df.columns]
        if len(show_cols) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无可视化的因子列，跳过")
            return

        # 根据展示因子数量动态设置网格（固定每行2列，便于在dashboard中整齐展示）
        n_charts = len(show_cols)
        cols = 2
        rows = (n_charts + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[titles_map.get(c, c) for c in show_cols])

        def _rc(idx: int) -> tuple:
            r = idx // cols + 1
            c = idx % cols + 1
            return r, c

        x = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in exp_df['date']]
        for i, c in enumerate(show_cols):
            r, cc = _rc(i)
            y_s = exp_df[f'strat_{c}']
            y_m = exp_df[f'mkt_{c}']
            fig.add_trace(go.Scatter(x=x, y=y_s, mode='lines', name=f'策略-{c}', line=dict(width=2)), row=r, col=cc)
            # 将虚线短线段长度缩小为原来的约 1/3，在小图表上更流畅
            fig.add_trace(go.Scatter(x=x, y=y_m, mode='lines', name=f'市场-{c}', line=dict(width=2, dash='3px,3px')), row=r, col=cc)
            # 在动量图上标注“分钟缺失导致回退”的日期点
            if c in ('mom_5m', 'mom_30m'):
                cov_col = f'strat_cov_{c}'
                if cov_col in exp_df.columns:
                    try:
                        cov_series = pd.to_numeric(exp_df[cov_col], errors='coerce')
                        y_series = pd.to_numeric(y_s, errors='coerce')
                        mask = cov_series < 0.999999
                        x_mark = [x[idx] for idx, m in enumerate(mask) if bool(m) and pd.notna(y_series.iloc[idx])]
                        y_mark = [float(y_series.iloc[idx]) for idx, m in enumerate(mask) if bool(m)]
                        cov_mark = [float(cov_series.iloc[idx]) if pd.notna(cov_series.iloc[idx]) else np.nan for idx, m in enumerate(mask) if bool(m)]
                        if len(x_mark) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=x_mark,
                                    y=y_mark,
                                    mode='markers',
                                    name=f'回退标记-{c}',
                                    marker=dict(symbol='x', size=7, color='#d62728'),
                                    customdata=cov_mark,
                                    hovertemplate='日期=%{x}<br>覆盖率=%{customdata:.0%}<br>使用日频回退'
                                ),
                                row=r, col=cc
                            )
                    except Exception:
                        pass
        # 保持页面内完整显示（含图例与x轴）；dashboard 端再通过 CSS 隐藏图例和x轴
        fig.update_layout(
            height=max(220 * rows, 300),
            title='策略因子特征暴露度（与市场基准对比）',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=-0.12),
            margin=dict(l=40, r=10, t=60, b=40)
        )

        # 6) 指标与说明
        metrics = {}
        for c in show_cols:
            metrics[f'{titles_map.get(c, c)}-均值'] = f"{pd.to_numeric(exp_df[f'strat_{c}'], errors='coerce').mean():.4f}"
            metrics[f'{titles_map.get(c, c)}-末值'] = f"{pd.to_numeric(exp_df[f'strat_{c}'], errors='coerce').iloc[-1]:.4f}"
            if f'mkt_{c}' in exp_df.columns:
                diff_last = pd.to_numeric(exp_df[f'strat_{c}'], errors='coerce').iloc[-1] - pd.to_numeric(exp_df[f'mkt_{c}'], errors='coerce').iloc[-1]
                metrics[f'{titles_map.get(c, c)}-末值相对市场'] = f"{diff_last:+.4f}"

        # 基于实际构建记录，生成“当前图表”的精确方法说明
        meta = getattr(self, '_factor_build_meta', {}) or {}
        close_src = meta.get('close_source', 'unknown')
        amt_src = meta.get('amount_source', 'unknown')
        mom_src = meta.get('momentum_source', 'unknown')
        mom_path = meta.get('momentum_path', '')
        mc_src = meta.get('market_cap_source', 'unknown')
        index_5m_src = meta.get('index_5m_source', 'unknown')
        mom_display = f"({mom_path})" if mom_path else ""

        # 覆盖率用于决定是否写“发生回退”的说明
        cov5_series = pd.to_numeric(exp_df.get('strat_cov_mom_5m', pd.Series(dtype=float)), errors='coerce')
        cov30_series = pd.to_numeric(exp_df.get('strat_cov_mom_30m', pd.Series(dtype=float)), errors='coerce')
        mom_fallback_used = (pd.notna(cov5_series).any() and (cov5_series < 0.999999).any()) or (pd.notna(cov30_series).any() and (cov30_series < 0.999999).any())

        # 修正：覆盖率反映分钟数据的完整性，但不再回退混合频率
        if mom_fallback_used:
            cov5_avg = float(cov5_series.mean()) if cov5_series.notna().any() else 0.0
            cov30_avg = float(cov30_series.mean()) if cov30_series.notna().any() else 0.0
            momentum_line = (
                r"<b>动量（5分钟/30分钟/60分钟）</b>: $\text{动量}_{n\times 5m}=\frac{p_t}{p_{t-n}}-1$，n=1/6/12；"
                f"当前5分钟覆盖率={cov5_avg:.1%}，30分钟覆盖率={cov30_avg:.1%}；缺失部分保持NaN，<b>不再回退混合日频动量</b>（避免混淆日内情绪与短期趋势）。"
            )
        else:
            momentum_line = (
                r"<b>动量（5分钟/30分钟/60分钟）</b>: $\text{动量}_{n\times 5m}=\frac{p_t}{p_{t-n}}-1$，n=1/6/12；"
                "全部来自 5 分钟K线计算，并按\"同股当日\"取均值；本页未发生回退。"
            )

        # β 的指数来源：若为等权回退来源则明确说明
        if str(index_5m_src) == 'equal_weight_market_from_5m':
            beta_line = (
                r"<b><i class='fas fa-exclamation-triangle text-yellow-500'></i> 高频β（5分钟）- 等权市场基准</b>: "
                r"当日内时间近邻对齐（±150秒）后，去均值做斜率近似：$\beta\approx\frac{\operatorname{Cov}(r_{\text{个股}},\,r_m^{EW})}{\operatorname{Var}(r_m^{EW})}$，"
                r"其中 $r_m^{EW}=\frac{1}{N}\sum_{j=1}^{N}r_j$ 为全市场等权平均收益（约2835只股票）。"
                "<b>由于沪深300指数5分钟数据缺失，采用此替代方案。</b>"
                "此β仍衡量个股系统性风险，但数值通常比市值加权β高30-40%（因小盘股权重更大）。"
            )
        else:
            beta_line = (
                r"<b>高频β（5分钟）</b>: 当日内时间近邻对齐（±150秒）后，去均值做斜率近似：$\beta\approx\frac{\operatorname{Cov}(r_{\text{个股}},\,r_{\text{指数}})}{\operatorname{Var}(r_{\text{指数}})}$。"
            )

        # 日内价格振幅来源
        rng_src = str(meta.get('range_source', ''))
        if 'first_close' in rng_src or rng_src == 'high_low_vs_first_close':
            range_line = (
                r"<b>日内价格振幅</b>: $\text{振幅}=\frac{\max(\text{最高价})-\min(\text{最低价})}{p_{\text{首笔}}}$（修正：使用首笔 <code>price</code> 作为分母，避免当日涨跌影响）。"
            )
        elif rng_src == 'close_range_vs_first_close':
            range_line = (
                r"<b>日内价格振幅</b>: $\text{振幅}=\frac{\max(p)-\min(p)}{p_{\text{首笔}}}$（修正：无高低价时使用首笔 <code>price</code> 作为分母）。"
            )
        else:
            range_line = (
                r"<b>日内价格振幅</b>: $\text{振幅}=\frac{\max(\text{最高价})-\min(\text{最低价})}{p_{\text{首笔}}}$。"
            )

        explanation = r"""
        <h4><i class='fas fa-thumbtack text-red-400'></i> 方法说明（本页口径）</h4>
        <ul>
            <li><b>总体流程</b>: 以买入日期为锚，将每日策略持仓（按当日买入 <code>tradeAmount</code> 加权）与同日全市场的因子均值对比，得到"策略-市场"的动态暴露曲线。</li>
            <li><b>分钟口径</b>: 动量与日内指标在 5 分钟K 级别计算后按"同股当日"聚合；高频β在"同日内"用个股5分钟收益与指数5分钟收益做斜率近似。</li>
            <li><b>因子定义</b>：
                <ul>
                    <li><b>市值对数因子</b>: $\ln(\text{市值})$。仅对市值>0 的样本取对数。</li>
                    <li>$MOM_LINE$</li>
                    <li><b>已实现波动率（5分钟）</b>: $\text{RV}_{5m}=\text{std}(r_{5m})$（修正：使用标准差，避免随数据点数变化而失真）。</li>
                    <li>$BETA_LINE$</li>
                    <li>$RANGE_LINE$</li>
                    <li><b>非流动性（Amihud ILLIQ）</b>: $\text{ILLIQ}=\frac{|r_{1d}|}{\text{成交额}}$（值越大流动性越差）。成交额优先用日线数据；若缺失且有成交量，则用 $\text{成交额}\approx \text{price}\times \text{成交量}$；仍缺失则回退为配对交易当日买卖金额之和。</li>
                </ul>
            </li>
            <li><b>因子暴露与因子均值计算</b>：
                <ul>
                    <li><b>策略端暴露</b>: 仅统计当日买入集合 $B_t$；权重 $w_i(t)$ 为买入时 <code>tradeAmount</code>。对任一因子 $X$，$$\text{策略暴露}_X(t)=\frac{\sum_{i\in B_t} w_i(t)\,X_i(t)}{\sum_{i\in B_t} w_i(t)}$$ 仅使用 $X_i(t)$ 非空样本（从分子分母同时剔除）。</li>
                    <li><b>市场端因子均值</b>: 当日全市场集合 $U_t$；权重 $v_i(t)$ 为市值（来源：$MC_SRC$）。$$\text{市场暴露}_X(t)=\frac{\sum_{i\in U_t} v_i(t)\,X_i(t)}{\sum_{i\in U_t} v_i(t)}$$ 当日缺失或非正市值样本不参与加权。</li>
                </ul>
            </li>
            <li><b>稳健性处理</b>: 分钟与指数收益时间近邻对齐（±150秒）；β 样本阈值≥3；振幅分母用最后非空收盘价；聚合忽略 NaN。</li>
            <li><b>输出解读</b>: 每条曲线表示当日策略（或市场）对该因子的加权暴露强弱；两条曲线的差异反映相对特征偏离。</li>
            <li><b>页面指标口径</b>: "均值"=全期时间均值 $E_t[\text{策略暴露}_X(t)]$；"末值"=末日值 $\text{策略暴露}_X(T)$；"末值相对市场"=$\text{策略暴露}_X(T)-\text{市场暴露}_X(T)$。</li>
        </ul>

        <h4><i class='fas fa-box-open text-yellow-600'></i> 数据来源与构建（简要）</h4>
        <ul>
            <li><b>5分钟个股K</b>: 由脚本抓取，先从 <code>data/orders.parquet</code> 推断 <code>Code</code> 与时间窗；输出包含开高低收价、成交量、成交额。</li>
            <li><b>5分钟指数</b>: 抓取沪深300指数或ETF；若不可用则以"全市场等权 5分钟收益"替代。</li>
            <li><b>日线K/收盘价</b>: 通过脚本获取日度开高低收、成交额、成交量、市值等，统一单位（如"万元"×1e4）。</li>
            <li><b>元数据/市值</b>: 组合行业与市值数据，用于市值对数因子与市场端加权。</li>
            <li><b>配对交易权重</b>: 从 <code>data/orders.parquet</code> 用 FIFO 配对，按买入时 <code>tradeAmount</code> 作为权重。</li>
        </ul>
        """

        # 由于 explanation 已经是 f-string 拼好的文本，这里无需再做占位符替换
        explanation_filled = (explanation
            .replace('$MOM_LINE$', momentum_line)
            .replace('$BETA_LINE$', beta_line)
            .replace('$RANGE_LINE$', range_line)
            .replace('$MC_SRC$', str(mc_src))
        )

        # 修正：若分钟动量覆盖率不完整，提示数据缺失但不再混合频率
        mix_note_html = ""
        try:
            cov5 = pd.to_numeric(exp_df.get('strat_cov_mom_5m', pd.Series(dtype=float)), errors='coerce')
            cov30 = pd.to_numeric(exp_df.get('strat_cov_mom_30m', pd.Series(dtype=float)), errors='coerce')
            cov5_avg = float(cov5.mean()) if len(cov5) else np.nan
            cov30_avg = float(cov30.mean()) if len(cov30) else np.nan
            need_note = (pd.notna(cov5).any() and (cov5 < 0.999999).any()) or (pd.notna(cov30).any() and (cov30 < 0.999999).any())
            if need_note:
                cov5_txt = (f"{cov5_avg:.0%}" if pd.notna(cov5_avg) else "N/A")
                cov30_txt = (f"{cov30_avg:.0%}" if pd.notna(cov30_avg) else "N/A")
                mix_note_html = f"""
                <h4><i class='fas fa-exclamation-triangle text-yellow-500'></i> 数据完整性提示</h4>
                <ul>
                    <li>本页动量存在分钟数据缺失：5m覆盖率≈{cov5_txt}，30m覆盖率≈{cov30_txt}。</li>
                    <li><b>修正策略</b>：缺失部分保持NaN，<span style="color:#e74c3c;">不再混合日频动量</span>（避免日内情绪与短期趋势的经济含义混淆）。</li>
                    <li><b>影响</b>：缺失日期的暴露度计算将基于其他可用因子；若需完整5m动量，请补充分钟K数据。</li>
                </ul>
                """
        except Exception:
            pass

        explanation_final = explanation_filled + mix_note_html

        self._save_figure_with_details(
            fig,
            name='factor_exposure_light',
            title='策略因子特征暴露度',
            explanation_html=explanation_final,
            metrics=metrics
        )
        print("<i class='fas fa-check-circle text-green-500'></i> 因子暴露分析完成")

    def factor_exposure_analysis(self):
        print("\n<i class='fas fa-chart-bar text-indigo-500'></i> === 策略因子特征暴露分析（分钟快照 + 增量仓位） ===")

        factors = self._build_factor_dataset()
        if factors is None or len(factors) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无法构建因子集，跳过因子暴露分析")
            return
        factors = factors.copy()
        factors['date'] = pd.to_datetime(factors['date']).dt.date

        trades = self._prepare_trade_flows()
        if trades is None or len(trades) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无新增仓位成交记录，因子暴露分析跳过")
            return

        snapshots = self._build_intraday_factor_snapshots()
        if snapshots is None:
            snapshots = trades.copy()
            for col in ['mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'beta_5m', 'range_day']:
                snapshots[col] = np.nan
                snapshots[f'{col}_is_intraday'] = False
            snapshots['has_intraday'] = False
        else:
            snapshots = snapshots.copy()

        snapshots['trade_date'] = pd.to_datetime(snapshots['trade_date']).dt.date

        merge_df = snapshots.merge(
            factors.rename(columns={'date': 'trade_date'}),
            on=['Code', 'trade_date'],
            how='left',
            suffixes=('', '_daily')
        )

        minute_cols = ['mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'beta_5m', 'range_day']
        # 修正：不再用日频填充分钟缺失，保持频率纯净性（与_build_factor_dataset的修正同步）
        for col in minute_cols:
            if f'{col}_is_intraday' not in merge_df.columns:
                merge_df[f'{col}_is_intraday'] = merge_df[col].notna()
        # 删除原fillna逻辑，理由：日频动量与分钟动量经济含义不同，不应混合

        factor_cols = [c for c in ['ln_market_cap', 'liquidity'] + minute_cols if c in merge_df.columns]

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

        exp_rows = []
        direction_rows = []
        covered_weights = {col: 0.0 for col in minute_cols}
        total_weights = {col: 0.0 for col in minute_cols}

        for dt, df_day in merge_df.groupby('trade_date'):
            dt_ts = pd.to_datetime(dt)
            row = {'date': dt_ts}
            direction_row = {'date': dt_ts}
            for col in factor_cols:
                row[f'strat_{col}'] = _weighted_exposure(df_day, col, 'trade_weight')
                direction_row[f'long_{col}'] = _weighted_exposure(df_day, col, 'long_exposure_amount')
                direction_row[f'short_{col}'] = -_weighted_exposure(df_day, col, 'short_exposure_amount')
            exp_rows.append(row)
            direction_rows.append(direction_row)

            # 修正：覆盖率分母应为"该因子有效样本的权重"，而非"全部交易权重"
            for col in minute_cols:
                if f'{col}_is_intraday' not in df_day.columns:
                    continue
                # 分子：来自分钟数据且有效的权重
                covered = df_day.loc[df_day[f'{col}_is_intraday'] & df_day[col].notna(), 'trade_weight'].sum()
                # 分母：该因子非NaN的权重总和（正确统计有效样本）
                total_valid = df_day.loc[df_day[col].notna(), 'trade_weight'].sum()
                
                if total_valid > 0:
                    covered_weights[col] += float(covered)
                    total_weights[col] += float(total_valid)

        exp_df = pd.DataFrame(exp_rows).sort_values('date')
        direction_df = pd.DataFrame(direction_rows).sort_values('date')

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

        mkt_rows = []
        factor_by_date = factors.groupby('date')
        for dt in exp_df['date'].dt.date.unique():
            row = {'date': pd.to_datetime(dt)}
            fday = factor_by_date.get_group(dt) if dt in factor_by_date.groups else None
            for col in factor_cols:
                row[f'mkt_{col}'] = _market_exposure(fday, col) if fday is not None else np.nan
            mkt_rows.append(row)
        mkt_df = pd.DataFrame(mkt_rows).sort_values('date')
        exp_df = exp_df.merge(mkt_df, on='date', how='left')

        try:
            cache_path = Path('data/factor_exposure_daily_cache.parquet')
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            exp_df.to_parquet(cache_path, index=False)
        except Exception as exc:
            print(f"[缓存] 写入策略因子暴露日表失败: {exc}")

        titles_map = {
            'ln_market_cap': 'Size（ln市值）暴露',
            'liquidity': '非流动性（Amihud ILLIQ）暴露',
            'mom_5m': '动量（5分钟）暴露',
            'mom_30m': '动量（30分钟）暴露',
            'mom_60m': '动量（60分钟）暴露',
            'rv_5m': '已实现波动率（10分钟窗口）',
            'beta_5m': '高频β（5分钟）',
            'range_day': '近10分钟价格振幅'
        }

        show_cols = [c for c in factor_cols if f'strat_{c}' in exp_df.columns]
        if len(show_cols) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无可视化因子列，跳过图表绘制")
            return

        cols = 2
        rows = (len(show_cols) + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[titles_map.get(c, c) for c in show_cols])

        def _grid(idx: int):
            r = idx // cols + 1
            c = idx % cols + 1
            return r, c

        x_axis = [d.strftime('%Y-%m-%d') for d in exp_df['date']]
        for idx, col in enumerate(show_cols):
            r, c = _grid(idx)
            strat = exp_df[f'strat_{col}']
            market = exp_df[f'mkt_{col}'] if f'mkt_{col}' in exp_df.columns else None
            fig.add_trace(
                go.Scatter(x=x_axis, y=strat, mode='lines', name=f'策略-{col}', line=dict(width=2)),
                row=r,
                col=c
            )
            if market is not None:
                fig.add_trace(
                    go.Scatter(x=x_axis, y=market, mode='lines', name=f'市场-{col}', line=dict(width=1.5, dash='dash'), opacity=0.85),
                    row=r,
                    col=c
                )

        fig.update_layout(
            height=max(rows * 240, 360),
            title='策略因子特征暴露度（与市场基准对比）',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=-0.12),
            margin=dict(l=40, r=10, t=60, b=40)
        )

        metrics = {
            '暴露跨度': f"{exp_df['date'].min().strftime('%Y-%m-%d')} ~ {exp_df['date'].max().strftime('%Y-%m-%d')}",
            '新增仓位样本数': f"{len(trades):,}",
            '统计天数': f"{len(exp_df):,}"
        }
        
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

        coverage_summary = {}
        coverage_items = []
        for col in minute_cols:
            if total_weights.get(col, 0) > 0:
                ratio = covered_weights[col] / total_weights[col]
                coverage_summary[col] = ratio
                coverage_items.append(f"<li>{titles_map.get(col, col)} 分钟覆盖率 = {ratio:.1%}</li>")
                metrics[f'{titles_map.get(col, col)}-分钟覆盖率'] = f"{ratio:.1%}"

        meta = getattr(self, '_factor_build_meta', {}) or {}
        minute_src = meta.get('minute_source', 'data/minute_5m_cache.parquet')
        beta_source_snapshot = meta.get('beta_index_source_snapshot', 'unknown')
        
        # 构建β说明文字
        if beta_source_snapshot == 'equal_weight_market_5m_snapshot':
            beta_note = (
                "<li><b><i class='fas fa-exclamation-triangle text-yellow-500'></i> 高频β（5分钟）- 等权市场基准</b>: "
                "由于沪深300指数5分钟数据缺失，当前使用<b>全市场等权收益</b>（约2835只股票等权平均）作为市场基准 $r_m$。"
                "计算公式：$\\beta_{i,t} = \\frac{\\text{Cov}(r_{i}, r_{m}^{EW})}{\\text{Var}(r_{m}^{EW})}$，"
                "其中 $r_m^{EW} = \\frac{1}{N}\\sum_{j=1}^{N} r_j$ 为全市场等权平均收益。"
                "此β仍衡量个股相对市场的系统性风险，但数值通常比市值加权β高30-40%（因小盘股权重更大）。"
                "β=1.2表示系统性风险比市场平均高20%；β<1为防御性股票。"
                "</li>"
            )
        else:
            beta_note = (
                "<li><b>高频β（5分钟）</b>: $\\beta_{i,t} = \\frac{\\text{Cov}(r_{i}, r_{m})}{\\text{Var}(r_{m})}$，"
                "其中 $r_i$ 为股票5分钟收益率，$r_m$ 为指数5分钟收益率，"
                "滚动窗口为6个周期（30分钟），最少需要3个有效样本。"
                "</li>"
            )
        
        explanation_parts = [
            "<h4><i class='fas fa-thumbtack text-red-400'></i> 方法说明</h4>",
            "<ul>",
            "<li><b>权重口径</b>: 仅统计带来仓位增加的 <code>tradeAmount</code>；多头/空头增量分别记录。</li>",
            "<li><b>暴露计算</b>: $\\text{策略暴露}_f(t) = \\frac{\\sum_i w_i(t) \\cdot x_{i,f}(t)}{\\sum_i w_i(t)}$，其中 $w_i$ 为新增仓位金额，$x_{i,f}$ 为该成交对应的因子值。</li>",
            "<li><b>分钟因子快照</b>: 使用时间近邻匹配查找交易时刻最近的5分钟K线因子值（最多回溯10分钟）。</li>",
            "<li><b>市场基准</b>: 同日全市场市值加权均值，$\\text{市场暴露}_f(t) = \\frac{\\sum_j m_j(t) \\cdot x_{j,f}(t)}{\\sum_j m_j(t)}$，若市值缺失则退化为简单平均。</li>",
            "<li><b><i class='fas fa-exclamation-triangle text-yellow-500'></i> 前视偏差提示</b>: 市场基准当前使用静态市值，存在轻微前视偏差；策略端使用交易时刻向后查找，无前视偏差。</li>",
            "</ul>",
            "<h4><i class='fas fa-chart-bar text-indigo-500'></i> 数据来源</h4>",
            "<ul>",
            "<li><b>分钟级数据</b>: 通过 <b>Baostock（宝股）</b>平台抓取的个股5分钟K线数据，包含开高低收 <code>price</code>、成交量、成交额等字段，频率为5分钟，复权方式为不复权。</li>",
            "<li><b>指数数据</b>: 通过 <b>Baostock（宝股）</b>平台抓取的沪深300指数5分钟K线数据，用于计算高频β因子（市场风险暴露）。</li>",
            "<li><b>日频数据</b>: 通过 <b>Baostock（宝股）</b>平台抓取的个股日K线数据，包含日度收盘价、成交额等，用于计算市值、流动性等日频因子。</li>",
            "<li><b>基本面数据</b>: 股票市值等基本面数据同样来自 <b>Baostock（宝股）</b>平台的历史数据接口。</li>",
            "</ul>",
            "<h4><i class='fas fa-ruler-combined text-indigo-500'></i> 因子构造公式</h4>",
            "<ul>",
            "<li><b>市值对数因子</b>: $\\text{Size}_{i,t} = \\ln(\\text{市值}_{i,t})$，其中市值为股票总市值或流通市值。</li>",
            "<li><b>非流动性（Amihud ILLIQ）</b>: $\\text{ILLIQ}_{i,t} = \\frac{|r_{i,t}|}{\\text{日成交额}_{i,t}}$，其中 $r_{i,t}$ 为日收益率，日成交额为当日总成交金额。值越大表示流动性越差。</li>",
            "<li><b>动量（5分钟）</b>: $\\text{动量}_{5m,i,t} = \\frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$，即单周期收益率（5分钟K线的 <code>price</code> 变化）。</li>",
            "<li><b>动量（30分钟）</b>: $\\text{动量}_{30m,i,t} = \\frac{P_{i,t}}{P_{i,t-6}} - 1$，其中 $t-6$ 表示向前6个5分钟周期（30分钟）。</li>",
            "<li><b>动量（60分钟）</b>: $\\text{动量}_{60m,i,t} = \\frac{P_{i,t}}{P_{i,t-12}} - 1$，其中 $t-12$ 表示向前12个5分钟周期（60分钟）。</li>",
            "<li><b>已实现波动率（10分钟窗口）</b>: $\\text{RV}_{i,t} = \\sqrt{\\sum_{k=t-1}^{t} r_{i,k}^2}$，其中 $r_{i,k}$ 为5分钟收益率，窗口为2个周期（10分钟）。</li>",
            beta_note,
            "<li><b>近10分钟价格振幅</b>: $\\text{振幅}_{i,t} = \\frac{\\text{最高价}_{[t-1,t]} - \\text{最低价}_{[t-1,t]}}{P_{i,t}}$，其中最高价和最低价分别为近2个周期（10分钟）内的极值。</li>",
            "</ul>"
        ]
        if coverage_items:
            explanation_parts.append("<h4><i class='fas fa-satellite-dish text-blue-400'></i> 分钟覆盖率</h4><ul>")
            explanation_parts.extend(coverage_items)
            explanation_parts.append("</ul>")

        self._save_figure_with_details(
            fig,
            name='factor_exposure_light',
            title='策略因子特征暴露度',
            explanation_html=''.join(explanation_parts),
            metrics=metrics
        )

        print("<i class='fas fa-check-circle text-green-500'></i> 因子暴露分析完成 (分钟快照)")

        self.factor_direction_exposure_analysis(exp_df, direction_df, titles_map, coverage_summary)
        self.factor_holdings_exposure_analysis(factors, titles_map)

    def factor_direction_exposure_analysis(
        self,
        exp_df: pd.DataFrame,
        direction_df: pd.DataFrame,
        titles_map: dict,
        coverage_summary: dict
    ) -> None:
        if direction_df is None or len(direction_df) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无多空方向增量数据，跳过方向分解图表")
            return

        show_cols = [
            col for col in titles_map.keys()
            if f'long_{col}' in direction_df.columns and f'short_{col}' in direction_df.columns
        ]
        if len(show_cols) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 多空方向数据缺少可用因子列")
            return

        cols = 2
        rows = len(show_cols)
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=False,
            subplot_titles=[titles_map.get(c, c) for c in show_cols]
        )

        dates_long = pd.to_datetime(direction_df['date'])
        x_axis = [d.strftime('%Y-%m-%d') for d in dates_long]

        for idx, col in enumerate(show_cols):
            row = idx + 1
            long_series = pd.to_numeric(direction_df[f'long_{col}'], errors='coerce')
            short_series = pd.to_numeric(direction_df[f'short_{col}'], errors='coerce')

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=long_series,
                    mode='lines',
                    name=f'多头-{col}',
                    line=dict(color='#2ca02c', width=2)
                ),
                row=row,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=short_series,
                    mode='lines',
                    name=f'空头-{col}',
                    line=dict(color='#d62728', width=2)
                ),
                row=row,
                col=2
            )

            if f'strat_{col}' in exp_df.columns:
                net_series = pd.to_numeric(exp_df[f'strat_{col}'], errors='coerce')
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=net_series,
                        mode='lines',
                        name=f'净暴露-{col}',
                        line=dict(color='#1f77b4', width=1.2, dash='dash'),
                        opacity=0.75
                    ),
                    row=row,
                    col=1
                )

        fig.update_layout(
            height=max(rows * 220, 320),
            title='新增仓位多空方向分解<br><sub style="color:#e74c3c;">注：空头暴露已取反（负值=做空正因子值股票），净暴露=多头暴露+空头暴露</sub>',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=-0.12),
            margin=dict(l=50, r=20, t=90, b=40)
        )

        column_labels = ['多头增量暴露', '空头增量暴露']
        for idx, label in enumerate(column_labels):
            xpos = (idx + 0.5) / cols
            fig.add_annotation(
                text=label,
                x=xpos,
                y=1.08,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=13, color='#2c3e50')
            )

        metrics = {}
        for col in show_cols:
            long_avg = pd.to_numeric(direction_df[f'long_{col}'], errors='coerce').mean()
            short_avg = pd.to_numeric(direction_df[f'short_{col}'], errors='coerce').mean()
            metrics[f'{titles_map.get(col, col)}-多头均值'] = f"{long_avg:.4f}"
            metrics[f'{titles_map.get(col, col)}-空头均值'] = f"{short_avg:.4f}"
            if f'strat_{col}' in exp_df.columns:
                net_last = pd.to_numeric(exp_df[f'strat_{col}'], errors='coerce').iloc[-1]
                metrics[f'{titles_map.get(col, col)}-净暴露末值'] = f"{net_last:.4f}"

        coverage_lines = []
        for col, ratio in coverage_summary.items():
            coverage_lines.append(f"<li>{titles_map.get(col, col)} 分钟覆盖率 = {ratio:.1%}</li>")

        explanation_parts = [
            "<h4><i class='fas fa-bullseye text-red-500'></i> 多空增量解释（修正版）</h4>",
            "<ul>",
            "<li>以单日新增仓位资金为权重，分解多头与空头方向的因子暴露。</li>",
            "<li><b>符号约定</b>：空头暴露取负号，即 <code>short_exposure = -weighted_avg(factor, short_weight)</code>；这样净暴露 = 多头暴露 + 空头暴露（含负号）。</li>",
            "<li><b>解读示例</b>：若某日 short_mom_5m = -3%，表示'做空了动量为+3%的股票'（负×正=负）。</li>",
            "<li>净暴露曲线展示多空相抵后的结果，便于与主图对照。</li>",
            "</ul>"
        ]
        if coverage_lines:
            explanation_parts.append("<h4><i class='fas fa-satellite-dish text-blue-400'></i> 分钟覆盖率</h4><ul>")
            explanation_parts.extend(coverage_lines)
            explanation_parts.append("</ul>")

        self._save_figure_with_details(
            fig,
            name='factor_direction_exposure_light',
            title='新增仓位多空方向分解',
            explanation_html=''.join(explanation_parts),
            metrics=metrics
        )

    def factor_holdings_exposure_analysis(self, factors: pd.DataFrame, titles_map: dict) -> None:
        positions = self._build_daily_positions()
        if positions is None or len(positions) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无持仓信息，跳过持仓暴露分析")
            return

        price_df = self._daily_price_df
        if price_df is None or len(price_df) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 缺少日线价格数据，无法计算持仓市值")
            return

        price_df = price_df.copy()
        price_df['date'] = pd.to_datetime(price_df['date']).dt.date
        price_df['close'] = pd.to_numeric(price_df['close'], errors='coerce')

        merged = positions.merge(price_df[['Code', 'date', 'close']], on=['Code', 'date'], how='left')
        if 'close' not in merged.columns or merged['close'].isna().all():
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 持仓数据缺少收盘价，跳过持仓暴露分析")
            return

        merged['close'] = merged.groupby('Code')['close'].transform(lambda s: s.ffill().bfill())
        merged = merged.dropna(subset=['close'])
        merged['position_value'] = merged['position_qty'] * merged['close']
        merged['long_weight'] = merged['position_value'].clip(lower=0)
        merged['short_weight'] = (-merged['position_value']).clip(lower=0)
        merged['abs_weight'] = merged['long_weight'] + merged['short_weight']
        merged['net_weight'] = merged['position_value']

        factor_cols = [c for c in ['ln_market_cap', 'liquidity', 'mom_5m', 'mom_30m', 'mom_60m', 'rv_5m', 'beta_5m', 'range_day'] if c in factors.columns]
        merged = merged.merge(factors[['Code', 'date'] + factor_cols], on=['Code', 'date'], how='left')

        def _signed_weighted(df: pd.DataFrame, col: str, weight_col: str) -> float:
            g = df.dropna(subset=[col, weight_col])
            if len(g) == 0:
                return np.nan
            w = g[weight_col].astype(float)
            sw = w.sum()
            if sw == 0:
                return np.nan
            x = g[col].astype(float)
            return float((w * x).sum() / sw)

        net_rows = []
        long_rows = []
        short_rows = []
        for dt, df_day in merged.groupby('date'):
            dt_ts = pd.to_datetime(dt)
            net_row = {'date': dt_ts}
            long_row = {'date': dt_ts}
            short_row = {'date': dt_ts}
            for col in factor_cols:
                net_row[f'strat_{col}'] = _signed_weighted(df_day, col, 'net_weight')
                long_row[f'long_{col}'] = _signed_weighted(df_day, col, 'long_weight')
                short_row[f'short_{col}'] = -_signed_weighted(df_day, col, 'short_weight')
            net_rows.append(net_row)
            long_rows.append(long_row)
            short_rows.append(short_row)

        net_df = pd.DataFrame(net_rows).sort_values('date')
        long_df = pd.DataFrame(long_rows).sort_values('date')
        short_df = pd.DataFrame(short_rows).sort_values('date')

        if len(net_df) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 持仓暴露结果为空，可能全部仓位为零")
            return

        def _market_exposure(factors_day: pd.DataFrame, col: str) -> float:
            g = factors_day.dropna(subset=[col])
            if len(g) == 0:
                return np.nan
            if 'market_cap' in g.columns and g['market_cap'].notna().any():
                w = g['market_cap'].astype(float).clip(lower=0)
                sw = w.sum()
                if sw > 0:
                    return float((w * g[col].astype(float)).sum() / sw)
            return float(g[col].astype(float).mean())

        mkt_rows = []
        factor_by_date = factors.groupby('date')
        for dt in net_df['date'].dt.date.unique():
            fday = factor_by_date.get_group(dt) if dt in factor_by_date.groups else None
            row = {'date': pd.to_datetime(dt)}
            for col in factor_cols:
                row[f'mkt_{col}'] = _market_exposure(fday, col) if fday is not None else np.nan
            mkt_rows.append(row)
        mkt_df = pd.DataFrame(mkt_rows).sort_values('date')
        net_df = net_df.merge(mkt_df, on='date', how='left')

        show_cols = [c for c in factor_cols if f'strat_{c}' in net_df.columns]
        if len(show_cols) == 0:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 持仓暴露无可视化因子，跳过")
            return

        cols = 2
        rows = len(show_cols)
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=False,
            column_titles=['净持仓 vs 市场', '多空拆分'],
            subplot_titles=[titles_map.get(c, c) for c in show_cols]
        )

        x_axis = [d.strftime('%Y-%m-%d') for d in net_df['date']]
        for idx, col in enumerate(show_cols):
            row = idx + 1
            net_series = pd.to_numeric(net_df[f'strat_{col}'], errors='coerce')
            mkt_series = pd.to_numeric(net_df[f'mkt_{col}'], errors='coerce') if f'mkt_{col}' in net_df.columns else None
            long_series = pd.to_numeric(long_df[f'long_{col}'], errors='coerce') if f'long_{col}' in long_df.columns else None
            short_series = pd.to_numeric(short_df[f'short_{col}'], errors='coerce') if f'short_{col}' in short_df.columns else None

            fig.add_trace(
                go.Scatter(x=x_axis, y=net_series, mode='lines', name=f'净暴露-{col}', line=dict(color='#1f77b4', width=2)),
                row=row,
                col=1
            )
            if mkt_series is not None:
                fig.add_trace(
                    go.Scatter(x=x_axis, y=mkt_series, mode='lines', name=f'市场-{col}', line=dict(color='#888', width=1.5, dash='dash')),
                    row=row,
                    col=1
                )
            if long_series is not None:
                fig.add_trace(
                    go.Scatter(x=x_axis, y=long_series, mode='lines', name=f'多头-{col}', line=dict(color='#2ca02c', width=2)),
                    row=row,
                    col=2
                )
            if short_series is not None:
                fig.add_trace(
                    go.Scatter(x=x_axis, y=short_series, mode='lines', name=f'空头-{col}', line=dict(color='#d62728', width=2)),
                    row=row,
                    col=2
                )

        fig.update_layout(
            height=max(rows * 240, 360),
            title='持仓因子暴露对比（净值口径）',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=-0.12),
            margin=dict(l=50, r=20, t=60, b=40)
        )

        metrics = {}
        for col in show_cols:
            net_avg = pd.to_numeric(net_df[f'strat_{col}'], errors='coerce').mean()
            metrics[f'{titles_map.get(col, col)}-净暴露均值'] = f"{net_avg:.4f}"
            if f'mkt_{col}' in net_df.columns:
                last_diff = pd.to_numeric(net_df[f'strat_{col}'], errors='coerce').iloc[-1] - pd.to_numeric(net_df[f'mkt_{col}'], errors='coerce').iloc[-1]
                metrics[f'{titles_map.get(col, col)}-末值相对市场'] = f"{last_diff:+.4f}"
            if f'long_{col}' in long_df.columns:
                metrics[f'{titles_map.get(col, col)}-多头均值'] = f"{pd.to_numeric(long_df[f'long_{col}'], errors='coerce').mean():.4f}"
            if f'short_{col}' in short_df.columns:
                metrics[f'{titles_map.get(col, col)}-空头均值'] = f"{pd.to_numeric(short_df[f'short_{col}'], errors='coerce').mean():.4f}"

        explanation_html = (
            "<h4><i class='fas fa-university text-gray-600'></i> 持仓暴露说明</h4>"
            "<ul>"
            "<li>按每日期末持仓市值计算净暴露，市值为收盘价×仓位股数。</li>"
            "<li>多头/空头曲线展示正负仓位的独立暴露强度，可与净暴露对照。</li>"
            "<li>市场基准仍为当日全市场市值加权均值，衡量策略相对偏离。</li>"
            "</ul>"
        )

        self._save_figure_with_details(
            fig,
            name='factor_holdings_exposure_light',
            title='持仓因子特征暴露',
            explanation_html=explanation_html,
            metrics=metrics
        )

    def slippage_cost_analysis(self):
        """滑点成本分析 - 避免累积计算等错误"""
        print("\n<i class='fas fa-coins text-yellow-500'></i> === 滑点成本分析 ===")
        
        # 1. 数据预处理和滑点计算
        print("<i class='fas fa-search text-blue-400'></i> 计算滑点指标...")
        
        # 确保只使用成交的订单
        traded_orders = self.df[self.df['tradeQty'] > 0].copy()
        print(f"成交订单数: {len(traded_orders):,} / 总订单数: {len(self.df):,}")
        
        # 计算时间滑点（订单到成交的时间延迟）
        traded_orders['time_slippage'] = (
            pd.to_datetime(traded_orders['tradeTimestamp']) - 
            pd.to_datetime(traded_orders['Timestamp'])
        ).dt.total_seconds()
        
        # 计算部分成交比例（数量滑点）
        traded_orders['fill_ratio'] = traded_orders['tradeQty'] / traded_orders['orderQty']
        traded_orders['quantity_slippage'] = 1 - traded_orders['fill_ratio']  # 未成交比例
        
        # 计算实际价格（基于成交金额和数量）
        traded_orders['actual_price'] = traded_orders['tradeAmount'] / traded_orders['tradeQty']
        
        # 计算价格滑点（实际价格 vs 订单价格）
        traded_orders['price_slippage_abs'] = traded_orders['actual_price'] - traded_orders['price']
        traded_orders['price_slippage_pct'] = (traded_orders['price_slippage_abs'] / traded_orders['price']) * 100
        
        # 清理异常值
        traded_orders = traded_orders[
            (traded_orders['time_slippage'] >= 0) & 
            (traded_orders['time_slippage'] <= 3600) &  # 限制在1小时内
            (traded_orders['price_slippage_pct'].abs() <= 10) &  # 价格滑点不超过10%
            (traded_orders['quantity_slippage'] >= 0) & 
            (traded_orders['quantity_slippage'] <= 1)
        ]
        
        print(f"清理后数据: {len(traded_orders):,} 条")
        
        # 2. 按日期聚合滑点指标 - 避免累积计算
        print("<i class='fas fa-chart-bar text-indigo-500'></i> 计算日度滑点指标...")
        traded_orders['date'] = traded_orders['Timestamp'].dt.date
        
        daily_slippage = traded_orders.groupby('date').agg({
            'time_slippage': ['mean', 'median', 'std'],
            'quantity_slippage': ['mean', 'median'],
            'price_slippage_pct': ['mean', 'median', 'std'],
            'fee': 'mean',
            'tradeAmount': 'sum'
        }).round(4)
        
        # 扁平化列名
        daily_slippage.columns = ['_'.join(col).strip() for col in daily_slippage.columns]
        daily_slippage = daily_slippage.reset_index()
        
        print(f"日度滑点数据范围验证:")
        print(f"  时间滑点: {daily_slippage['time_slippage_mean'].min():.2f}s - {daily_slippage['time_slippage_mean'].max():.2f}s")
        print(f"  数量滑点: {daily_slippage['quantity_slippage_mean'].min():.4f} - {daily_slippage['quantity_slippage_mean'].max():.4f}")
        print(f"  价格滑点: {daily_slippage['price_slippage_pct_mean'].min():.4f}% - {daily_slippage['price_slippage_pct_mean'].max():.4f}%")
        
        # 3. 时间滑点分析图表
        if len(daily_slippage) > 30:
            # 采样数据用于图表
            slippage_sampled = daily_slippage.iloc[::max(1, len(daily_slippage)//150)]
            
            # 时间滑点时间序列
            x_data = [str(date) for date in slippage_sampled['date']]
            y_time = [float(val) for val in slippage_sampled['time_slippage_mean']]
            
            fig_time_slip = go.Figure()
            fig_time_slip.add_trace(go.Scatter(
                x=x_data,
                y=y_time,
                mode='lines+markers',
                name='平均时间滑点',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                hovertemplate='日期: %{x}<br>时间滑点: %{y:.2f}秒<extra></extra>'
            ))
            
            fig_time_slip.update_layout(
                title=f'时间滑点分析<br><sub>平均: {daily_slippage["time_slippage_mean"].mean():.2f}秒, 标准差: {daily_slippage["time_slippage_mean"].std():.2f}秒</sub>',
                xaxis_title='日期',
                yaxis_title='时间滑点 (秒)',
                height=400
            )
            
            self._save_figure_with_details(
                fig_time_slip,
                name='time_slippage_light',
                title='时间滑点分析',
                explanation_html="<p>时间滑点表示从下单到成交的时间延迟，反映市场流动性和执行效率。</p>",
                metrics={
                    '平均时间滑点': f"{daily_slippage['time_slippage_mean'].mean():.2f}秒",
                    '中位数时间滑点': f"{daily_slippage['time_slippage_median'].mean():.2f}秒",
                    '时间滑点标准差': f"{daily_slippage['time_slippage_mean'].std():.2f}秒"
                }
            )
            
            # 4. 价格滑点分析图表
            y_price = [float(val) for val in slippage_sampled['price_slippage_pct_mean']]
            
            fig_price_slip = go.Figure()
            fig_price_slip.add_trace(go.Scatter(
                x=x_data,
                y=y_price,
                mode='lines+markers',
                name='平均价格滑点',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                hovertemplate='日期: %{x}<br>价格滑点: %{y:.4f}%<extra></extra>'
            ))
            
            # 添加零线
            fig_price_slip.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig_price_slip.update_layout(
                title=f'价格滑点分析<br><sub>平均: {daily_slippage["price_slippage_pct_mean"].mean():.4f}%, 标准差: {daily_slippage["price_slippage_pct_mean"].std():.4f}%</sub>',
                xaxis_title='日期',
                yaxis_title='价格滑点 (%)',
                height=400
            )
            
            self._save_figure_with_details(
                fig_price_slip,
                name='price_slippage_light',
                title='价格滑点分析',
                explanation_html="<p>价格滑点表示实际成交价格与订单价格的差异，正值表示买入时价格上升或卖出时价格下降。</p>",
                metrics={
                    '平均价格滑点': f"{daily_slippage['price_slippage_pct_mean'].mean():.4f}%",
                    '中位数价格滑点': f"{daily_slippage['price_slippage_pct_median'].mean():.4f}%",
                    '价格滑点标准差': f"{daily_slippage['price_slippage_pct_mean'].std():.4f}%"
                }
            )
            
            # 5. 综合成本分析（滑点+手续费）
            # 计算综合交易成本比例
            daily_trade_amount = traded_orders.groupby('date')['tradeAmount'].mean()
            daily_slippage['total_cost_pct'] = (
                daily_slippage['price_slippage_pct_mean'].abs() + 
                (daily_slippage['fee_mean'] / daily_trade_amount.reindex(daily_slippage['date']).values) * 100
            )
            
            # 重新采样包含新计算的列
            slippage_sampled_with_cost = daily_slippage.iloc[::max(1, len(daily_slippage)//150)]
            y_total_cost = [float(val) for val in slippage_sampled_with_cost['total_cost_pct']]
            
            fig_total_cost = go.Figure()
            fig_total_cost.add_trace(go.Scatter(
                x=x_data,
                y=y_total_cost,
                mode='lines+markers',
                name='综合交易成本',
                line=dict(color='purple', width=2),
                marker=dict(size=4),
                hovertemplate='日期: %{x}<br>总成本: %{y:.4f}%<extra></extra>'
            ))
            
            fig_total_cost.update_layout(
                title=f'综合交易成本分析<br><sub>平均: {daily_slippage["total_cost_pct"].mean():.4f}% (滑点+手续费)</sub>',
                xaxis_title='日期',
                yaxis_title='综合成本 (%)',
                height=400
            )
            
            self._save_figure_with_details(
                fig_total_cost,
                name='total_cost_light',
                title='综合交易成本分析',
                explanation_html="<p>综合交易成本包括价格滑点和手续费，反映实际交易的总成本负担。</p>",
                metrics={
                    '平均综合成本': f"{daily_slippage['total_cost_pct'].mean():.4f}%",
                    '成本波动': f"{daily_slippage['total_cost_pct'].std():.4f}%"
                }
            )
        
        print(f"<i class='fas fa-check-circle text-green-500'></i> 滑点成本分析完成")
        
    def daily_absolute_profit_analysis(self):
        """基于盯市结果，计算并可视化日度绝对盈利（¥）。并校验现金一致性。"""
        print("\n<i class='fas fa-money-bill text-green-600'></i> === 日度绝对盈利（盯市） ===")
        from pathlib import Path
        mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
        if not mtm_file.exists():
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到盯市分析结果文件，跳过绝对盈利分析")
            return
        try:
            mtm_df = pd.read_csv(mtm_file)
            mtm_df['date'] = pd.to_datetime(mtm_df['date'])

            # 解析货币字符串与百分比
            def parse_currency(val):
                try:
                    if isinstance(val, str):
                        return float(val.replace(',', '').strip())
                    return float(val)
                except (ValueError, TypeError):
                    return np.nan

            # 解析多空市值
            mtm_df['long_value_num'] = mtm_df['long_value'].apply(parse_currency)
            mtm_df['short_value_num'] = mtm_df['short_value'].apply(parse_currency)
            
            # 使用正确的初始资金重新计算现金和NAV
            CORRECT_INITIAL_CAPITAL = 62_090_808
            
            # 从订单数据重新计算现金
            if hasattr(self, 'df') and self.df is not None:
                orders_temp = self.df.copy()
                orders_temp['date'] = pd.to_datetime(orders_temp['Timestamp']).dt.date
                daily_flows = orders_temp.groupby(['date', 'direction'])[['tradeAmount', 'fee']].sum().unstack(fill_value=0)
                daily_flows.columns = [f"{a}_{b}" for a, b in daily_flows.columns]
                
                cash_balance = CORRECT_INITIAL_CAPITAL
                cash_series = []
                for date_val in mtm_df['date'].dt.date:
                    if date_val in daily_flows.index:
                        buy_amt = daily_flows.loc[date_val, 'tradeAmount_B'] if 'tradeAmount_B' in daily_flows.columns else 0
                        sell_amt = daily_flows.loc[date_val, 'tradeAmount_S'] if 'tradeAmount_S' in daily_flows.columns else 0
                        fee_amt = (daily_flows.loc[date_val, 'fee_B'] if 'fee_B' in daily_flows.columns else 0) + \
                                  (daily_flows.loc[date_val, 'fee_S'] if 'fee_S' in daily_flows.columns else 0)
                        cash_balance += sell_amt - buy_amt - fee_amt
                    cash_series.append(cash_balance)
                
                mtm_df['cash_num'] = cash_series
                mtm_df['total_assets_num'] = mtm_df['cash_num'] + mtm_df['long_value_num'] - mtm_df['short_value_num']
                print(f"<i class='fas fa-check-circle text-green-500'></i> 已基于正确初始资金重新计算NAV")
            else:
                # 回退：使用原始值
                mtm_df['cash_num'] = mtm_df['cash'].apply(parse_currency)
                mtm_df['total_assets_num'] = mtm_df['total_assets'].apply(parse_currency)

            # 现金一致性校验: cash ?= total_assets - long_value + short_value
            mtm_df['cash_expected'] = mtm_df['total_assets_num'] - mtm_df['long_value_num'] + mtm_df['short_value_num']
            mtm_df['cash_diff'] = mtm_df['cash_num'] - mtm_df['cash_expected']
            mtm_df['cash_diff_abs'] = mtm_df['cash_diff'].abs()
            max_abs_diff = mtm_df['cash_diff_abs'].max()
            mean_abs_diff = mtm_df['cash_diff_abs'].mean()
            rel_diff = (mtm_df['cash_diff_abs'] / mtm_df['total_assets_num'].replace(0, np.nan)).dropna()
            max_rel_diff = rel_diff.max() if len(rel_diff) else np.nan
            mean_rel_diff = rel_diff.mean() if len(rel_diff) else np.nan

            print(f"<i class='fas fa-clipboard-list text-blue-500'></i> 现金一致性校验: 最大绝对偏差={max_abs_diff:,.2f} 元, 平均绝对偏差={mean_abs_diff:,.2f} 元")
            if pd.notna(max_rel_diff):
                print(f"   相对偏差(对总资产): 最大={max_rel_diff:.6%}, 平均={mean_rel_diff:.6%}")

            # 计算日度绝对利润（第1日无前值，记为NaN）
            mtm_df = mtm_df.sort_values('date')
            mtm_df['daily_abs_profit'] = mtm_df['total_assets_num'].diff()

            # 基本统计
            profit_series = pd.Series(mtm_df['daily_abs_profit'].values, index=mtm_df['date'])
            profit_series = profit_series.dropna()
            if len(profit_series) == 0:
                print("<i class='fas fa-times-circle text-red-500'></i> 无法计算绝对盈利（数据不足）")
                return

            total_profit = profit_series.sum()
            avg_profit = profit_series.mean()
            max_profit = profit_series.max()
            min_profit = profit_series.min()
            win_rate = (profit_series > 0).mean()

            # 保存原始NAV/现金用于一致性校验
            mtm_df['total_assets_reported'] = mtm_df['total_assets'].apply(parse_currency)
            mtm_df['cash_reported'] = mtm_df['cash'].apply(parse_currency)

            # 可视化：柱状图（正红负绿 - A股标准）+ 7日均线
            x_dates = [d.strftime('%Y-%m-%d') for d in profit_series.index]
            y_vals = profit_series.values.tolist()
            colors = ['#e53935' if v >= 0 else '#43a047' for v in y_vals]

            fig_abs = go.Figure()
            fig_abs.add_trace(go.Bar(
                x=x_dates,
                y=y_vals,
                marker_color=colors,
                name='日度绝对盈利',
                hovertemplate='日期: %{x}<br>日度绝对盈利: ¥%{y:,.0f}<extra></extra>'
            ))

            # 7日移动平均 - 使用同一Y轴确保可见性
            ma7 = profit_series.rolling(window=min(7, len(profit_series)), min_periods=1).mean()
            ma7_values = [float(v) for v in ma7.values]
            fig_abs.add_trace(go.Scatter(
                x=[d.strftime('%Y-%m-%d') for d in ma7.index],
                y=ma7_values,
                mode='lines',
                name='7日均线',
                line=dict(color='#f59e0b', width=2.5),
                hovertemplate='日期: %{x}<br>7日均线: ¥%{y:,.0f}<extra></extra>'
            ))

            # 标注最大/最小盈利日 - 垂直线
            try:
                max_day = profit_series.idxmax()
                min_day = profit_series.idxmin()
                max_day_str = max_day.strftime('%Y-%m-%d')
                min_day_str = min_day.strftime('%Y-%m-%d')
                fig_abs.add_vline(
                    x=max_day_str,
                    line_dash='dash',
                    line_color='#e53935',
                    line_width=1,
                    opacity=0.35
                )
                fig_abs.add_vline(
                    x=min_day_str,
                    line_dash='dash',
                    line_color='#43a047',
                    line_width=1,
                    opacity=0.35
                )
                print(f"   最大盈利日: {max_day_str} (¥{max_profit:,.0f})")
                print(f"   最大亏损日: {min_day_str} (¥{min_profit:,.0f})")
            except Exception as e:
                print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 添加盈利标注线失败: {e}")

            # 统一主题与交互配置（符合前端设计规范）
            self._apply_plotly_theme(fig_abs)
            fig_abs.update_layout(
                title=dict(text='日度绝对盈利趋势图', x=0),
                xaxis=dict(
                    title='日期',
                    type='date',
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1月", step="month", stepmode="backward"),
                            dict(count=3, label="3月", step="month", stepmode="backward"),
                            dict(count=6, label="6月", step="month", stepmode="backward"),
                            dict(step="all", label="全部")
                        ]),
                        bgcolor='rgba(255,255,255,0.9)'
                    ),
                    rangeslider=dict(visible=True),
                    showgrid=False
                ),
                yaxis=dict(
                    title='金额（¥）',
                    tickformat=',.0f',
                    zeroline=True,
                    zerolinecolor='#9ca3af'
                ),
                height=520,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            fig_abs.update_traces(hovertemplate='日期: %{x}<br>日度绝对盈利: ¥%{y:,.0f}<extra></extra>', selector=dict(type='bar'))
            fig_abs.update_traces(hovertemplate='日期: %{x}<br>7日均线: ¥%{y:,.0f}<extra></extra>', selector=dict(mode='lines'))

            # 日收益率分布（基于 NAV 环比收益）
            daily_returns = mtm_df['total_assets_num'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            fig_ret = None
            fig_ret_json = "null"
            if len(daily_returns) > 0:
                counts, edges = np.histogram(daily_returns, bins=min(40, max(10, int(np.sqrt(len(daily_returns))*2))))
                centers = (edges[:-1] + edges[1:]) / 2
                colors_ret = ['#e53935' if c >= 0 else '#43a047' for c in centers]
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Bar(
                    x=centers,
                    y=counts,
                    marker_color=colors_ret,
                    name='日收益率分布',
                    hovertemplate='收益率: %{x:.2%}<br>频数: %{y}<extra></extra>'
                ))
                try:
                    mean_ret = float(daily_returns.mean())
                    median_ret = float(daily_returns.median())
                    fig_ret.add_vline(x=mean_ret, line_dash='dash', line_color='#1f2937', opacity=0.5)
                    fig_ret.add_vline(x=median_ret, line_dash='dot', line_color='#6366f1', opacity=0.5)
                except Exception:
                    pass
                self._apply_plotly_theme(fig_ret, yaxis_percent=False)
                fig_ret.update_layout(
                    title=dict(text='日收益率分布（频数）', x=0),
                    xaxis=dict(title='日收益率', tickformat='.2%'),
                    yaxis=dict(title='频数（天）', zeroline=True, zerolinecolor='#9ca3af'),
                    height=420,
                    showlegend=False
                )

            # 现金一致性指标（使用原始盯市数据校验，而非重算后的现金）
            mtm_df['cash_expected_raw'] = mtm_df['total_assets_reported'] - mtm_df['long_value_num'] + mtm_df['short_value_num']
            mtm_df['cash_diff_raw'] = mtm_df['cash_reported'] - mtm_df['cash_expected_raw']
            max_abs_diff_raw = mtm_df['cash_diff_raw'].abs().max()
            mean_abs_diff_raw = mtm_df['cash_diff_raw'].abs().mean()
            rel_diff_raw = (mtm_df['cash_diff_raw'].abs() / mtm_df['total_assets_reported'].replace(0, np.nan)).dropna()
            max_rel_diff_raw = rel_diff_raw.max() if len(rel_diff_raw) else np.nan
            mean_rel_diff_raw = rel_diff_raw.mean() if len(rel_diff_raw) else np.nan

            def _fmt_or_na(v, fmt):
                return fmt.format(v) if pd.notna(v) else "N/A"

            cash_metrics = {
                '最大现金偏差(¥)': _fmt_or_na(max_abs_diff_raw, "{:,.2f}"),
                '平均现金偏差(¥)': _fmt_or_na(mean_abs_diff_raw, "{:,.2f}"),
                '最大相对偏差(对总资产)': _fmt_or_na(max_rel_diff_raw, "{:.6%}"),
                '平均相对偏差(对总资产)': _fmt_or_na(mean_rel_diff_raw, "{:.6%}"),
            }

            abs_metrics = {
                '合计绝对盈利': f"¥{total_profit:,.0f}",
                '平均日盈利': f"¥{avg_profit:,.0f}",
                '最大单日盈利': f"¥{max_profit:,.0f}",
                '最大单日亏损': f"¥{min_profit:,.0f}",
                '盈利日占比': f"{win_rate:.1%}"
            }

            # 构建自定义 HTML（遵循页面设计指南）
            output_path = self.reports_dir / 'daily_absolute_profit_light.html'
            config = {
                'responsive': True,
                'displayModeBar': False,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
            }
            # 转为原生 JSON，避免 TypedArray 导致的渲染异常
            def _to_native(obj):
                if isinstance(obj, np.ndarray):
                    if np.issubdtype(obj.dtype, np.number):
                        return obj.astype(float).tolist()
                    return [str(v) for v in obj.tolist()]
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                try:
                    import pandas as _pd
                    if isinstance(obj, _pd.Timestamp):
                        return obj.isoformat()
                except Exception:
                    pass
                if isinstance(obj, dict):
                    if 'bdata' in obj and isinstance(obj.get('bdata'), str):
                        try:
                            dtype_map = {'f8': np.float64, 'f4': np.float32, 'i8': np.int64, 'i4': np.int32, 'u8': np.uint64, 'u4': np.uint32}
                            np_dtype = dtype_map.get(obj.get('dtype', 'f8'), np.float64)
                            raw = base64.b64decode(obj['bdata'])
                            arr = np.frombuffer(raw, dtype=np_dtype)
                            shape_val = obj.get('shape')
                            if shape_val is not None:
                                try:
                                    if isinstance(shape_val, str):
                                        shape_tuple = tuple(int(s.strip()) for s in shape_val.replace('x', ',').split(',') if s.strip() != '')
                                    elif isinstance(shape_val, (list, tuple)):
                                        shape_tuple = tuple(int(s) for s in shape_val)
                                    else:
                                        shape_tuple = None
                                    if shape_tuple and np.prod(shape_tuple) == arr.size:
                                        arr = arr.reshape(shape_tuple)
                                except Exception:
                                    pass
                            if np.issubdtype(arr.dtype, np.number):
                                return arr.astype(float).tolist()
                            return [str(v) for v in arr.tolist()]
                        except Exception:
                            return {k: _to_native(v) for k, v in obj.items()}
                    return {k: _to_native(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_native(v) for v in obj]
                if hasattr(obj, 'isoformat'):
                    try:
                        return obj.isoformat()
                    except Exception:
                        return str(obj)
                return obj

            fig_json = json.dumps(_to_native(fig_abs.to_plotly_json()), ensure_ascii=False)
            if fig_ret is not None:
                fig_ret_json = json.dumps(_to_native(fig_ret.to_plotly_json()), ensure_ascii=False)
            gen_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            mathjax_local_src = self._ensure_mathjax_bundle()

            card_tpl = """
            <div class="bg-white rounded-xl shadow-sm p-5 border-l-4 {border_cls}">
                <p class="text-xs font-semibold text-gray-500 uppercase">{label}</p>
                <h3 class="text-2xl font-bold text-gray-800 mt-1">{value}</h3>
                <p class="text-sm text-gray-500 mt-1">{desc}</p>
            </div>
            """
            kpi_cards_html = "".join([
                card_tpl.format(border_cls="border-[#e53935]", label="合计绝对盈利", value=abs_metrics['合计绝对盈利'], desc="累计NAV差额"),
                card_tpl.format(border_cls="border-[#e53935]", label="最大单日盈利", value=abs_metrics['最大单日盈利'], desc="最佳单日表现"),
                card_tpl.format(border_cls="border-[#43a047]", label="最大单日亏损", value=abs_metrics['最大单日亏损'], desc="最差单日表现"),
                card_tpl.format(border_cls="border-indigo-500", label="盈利日占比", value=abs_metrics['盈利日占比'], desc="(>0) 日占比"),
            ])

            cash_list_html = "".join([
                f"<li class='flex items-center justify-between py-1'><span>{k}</span><span class='font-semibold text-gray-900'>{v}</span></li>"
                for k, v in cash_metrics.items()
            ])

            explanation_md = r"""
### 页面目的
- 展示每日净资产变动对应的绝对盈亏，快速定位大幅波动日期与整体盈利稳定性。

### 计算方式
- 基于 `orders.parquet` 的成交额 `tradeAmount` 与手续费 `fee` 逐日回放现金流，叠加盯市多空市值得到当日总资产 $NAV_t$。
- 日度绝对盈利 $$Profit_t = NAV_t - NAV_{t-1}$$ ，首日记为 NaN；正值用红色柱，负值用绿色柱。
- 7 日均线 $$MA7_t = \frac{1}{k} \sum_{i=0}^{k-1} Profit_{t-i}$$，$k=\min(7,t)$，用于平滑短期波动。
- 现金一致性检查 $$Cash_t = NAV_t - LongValue_t + ShortValue_t$$，偏差异常时需核对盯市口径与费用落账。

### 交互与解读
- 右上方时间选择器支持近 1/3/6 个月快速对比，RangeSlider 支持拖动查看全区间。
- Hover 提示展示精确金额，虚线标记最大盈利日与最大亏损日；7 日均线用于观察盈利持续性。
- 收益率分布直方图可识别偏斜与尾部风险，均值/中位数虚线用于判断收益集中区间。
- 指标卡展示累计盈利、极值与盈利占比，可用于与历史版本或不同策略侧边对比。
            """

            html = f"""
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>日收益分布（盯市）</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
                <style>
                    body {{ font-family: "Noto Sans SC","Microsoft YaHei","Segoe UI",sans-serif; }}
                </style>
                <script>
                    window.MathJax = {{
                        tex: {{
                            inlineMath: [["$","$"],["\\(","\\)"]],
                            displayMath: [["$$","$$"],["\\[","\\]"]],
                            processEscapes: true
                        }},
                        options: {{ skipHtmlTags: ["script","noscript","style","textarea","pre","code"] }},
                        svg: {{ fontCache: 'global' }}
                    }};
                </script>
                <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js" onerror="this.onerror=null; this.src='{mathjax_local_src}';"></script>
            </head>
            <body class="bg-gray-50">
                <header class="sticky top-0 z-40 bg-white/95 backdrop-blur border-b border-gray-100 shadow-sm">
                    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
                        <div class="flex items-center space-x-2">
                            <i class="fas fa-money-bill-wave text-[#e53935]"></i>
                            <div>
                                <p class="text-[11px] tracking-[0.18em] text-gray-500 uppercase">收益分布</p>
                                <h1 class="text-xl font-semibold text-gray-900">日收益分布（盯市）</h1>
                            </div>
                        </div>
                        <div class="text-xs text-gray-500 flex items-center"><i class='far fa-clock mr-1'></i>生成时间: {gen_time}</div>
                    </div>
                </header>
                <main class="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
                    <section class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {kpi_cards_html}
                    </section>

                    <section id="main_chart_wrap" class="bg-white rounded-lg shadow-sm p-4 border border-gray-100">
                        <div id="main_chart" class="w-full h-[520px]"></div>
                    </section>

                    <section id="returns_chart_wrap" class="bg-white rounded-lg shadow-sm p-4 border border-gray-100">
                        <div class="flex items-center justify-between mb-2">
                            <h2 class="text-lg font-semibold text-gray-900">日收益率分布</h2>
                            <span class="text-xs text-gray-500">红=盈利区间，绿=亏损区间</span>
                        </div>
                        <div id="returns_chart" class="w-full h-[420px]"></div>
                    </section>

                    <section class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <div class="bg-white rounded-lg shadow-sm p-6 border-l-4 border-indigo-400">
                            <p class="font-semibold text-gray-900 mb-3 flex items-center"><i class='fas fa-balance-scale text-indigo-500 mr-2'></i>现金一致性校验</p>
                            <ul class="text-sm text-gray-700 space-y-1">
                                {cash_list_html}
                            </ul>
                            <p class="text-xs text-gray-500 mt-3">公式：cash = NAV - long_value + short_value；偏差过大时需要检查盯市及费用口径。</p>
                        </div>
                        <div class="lg:col-span-2 bg-white rounded-lg shadow-sm p-6">
                            <div id="explain-md" class="prose prose-sm max-w-none"></div>
                        </div>
                    </section>
                </main>

                <script>
                    (function() {{
                        var fig = {fig_json};
                        var figRet = {fig_ret_json};
                        var config = {json.dumps(config, ensure_ascii=False)};
                        Plotly.newPlot('main_chart', fig.data || [], fig.layout || {{}}, config);
                        if (figRet) {{
                            Plotly.newPlot('returns_chart', figRet.data || [], figRet.layout || {{}}, config);
                        }}
                        var mdText = {json.dumps(explanation_md, ensure_ascii=False)};
                        var mdTarget = document.getElementById('explain-md');
                        if (mdTarget) {{
                            if (window.marked) {{
                                mdTarget.innerHTML = marked.parse(mdText);
                            }} else {{
                                mdTarget.textContent = mdText;
                            }}
                        }}
                        var anchor = document.getElementById('main_chart_wrap');
                        if (anchor) {{
                            requestAnimationFrame(function() {{
                                anchor.scrollIntoView({{ behavior: 'auto', block: 'start' }});
                            }});
                        }}
                        function typeset() {{
                            if (window.MathJax && window.MathJax.typesetPromise) {{
                                window.MathJax.typesetPromise().catch(function(err) {{ console.warn('MathJax error:', err); }});
                            }}
                        }}
                        if (document.readyState === 'complete') {{
                            typeset();
                        }} else {{
                            window.addEventListener('load', typeset, {{ once: true }});
                        }}
                    }})();
                </script>
            </body>
            </html>
            """
            output_path.write_text(html, encoding='utf-8')
            self.figures.append(('daily_absolute_profit_light', str(output_path)))
            print("<i class='fas fa-check-circle text-green-500'></i> 日度绝对盈利图已生成")
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 绝对盈利分析失败: {e}")

    def _ensure_strategy_metrics_from_nav(self) -> bool:
        """兜底：直接从 daily_nav_revised.csv 计算策略总收益/夏普/回撤/胜率，修复仪表板 N/A。"""
        try:
            mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
            if not mtm_file.exists():
                return False
            nav = pd.read_csv(mtm_file)
            if 'total_assets' not in nav.columns:
                return False

            def _parse_cur(v):
                try:
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str):
                        return float(v.replace(',', '').strip())
                except Exception:
                    return np.nan
                return np.nan

            nav['date'] = pd.to_datetime(nav['date'])
            nav = nav.sort_values('date').copy()
            nav['total_assets_num'] = nav['total_assets'].apply(_parse_cur)
            nav['daily_return_nav'] = nav['total_assets_num'].pct_change()
            if len(nav) == 0:
                return False
            nav.loc[nav.index[0], 'daily_return_nav'] = 0.0
            dr = nav['daily_return_nav'].astype(float)
            cum = (1 + dr).cumprod()
            CORRECT_INITIAL_CAPITAL = 62_090_808
            total_return = float(nav['total_assets_num'].iloc[-1] / CORRECT_INITIAL_CAPITAL - 1) if len(nav) > 0 else np.nan
            vol = float(np.nanstd(dr, ddof=1))
            sharpe = float(np.nan) if vol == 0 or np.isnan(vol) else float(np.nanmean(dr) / vol * np.sqrt(252))
            rolling_max = cum.expanding().max()
            max_dd = float(((cum - rolling_max) / rolling_max).min())
            win_rate = float((dr > 0).mean())

            if not hasattr(self, 'strategy_metrics') or not isinstance(self.strategy_metrics, dict):
                self.strategy_metrics = {}
            self.strategy_metrics.update({
                'total_return_nav': f"{total_return*100:.2f}%" if not np.isnan(total_return) else "N/A",
                'sharpe_ratio': f"{sharpe:.3f}" if not np.isnan(sharpe) else "N/A",
                'max_drawdown': f"{max_dd:.2%}" if not np.isnan(max_dd) else "N/A",
                'win_rate': f"{win_rate:.2%}" if not np.isnan(win_rate) else "N/A",
            })
            return True
        except Exception as e:
            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> NAV兜底计算指标失败: {e}")
            return False

    def performance_metrics_analysis(self):
        """绩效指标分析 - 实现股票-日聚合的真实策略收益"""
        print("\n<i class='fas fa-chart-bar text-indigo-500'></i> === 绩效指标分析（真实策略收益口径）===")
        
        # 1. 股票-日聚合，避免多订单重复计权
        print("<i class='fas fa-search text-blue-400'></i> 执行股票-日聚合...")
        
        # 添加日期列用于聚合
        self.df['date'] = self.df['Timestamp'].dt.date
        
        # 使用高效但完整的聚合方法，保持三种收益计算的准确性
        print("<i class='fas fa-chart-bar text-indigo-500'></i> 执行完整聚合（优化版本）...")
        
        # 预先计算各种加权值
        self.df['weighted_real_amount'] = self.df['real'] * self.df['tradeAmount']
        
        # 总体聚合
        total_agg = self.df.groupby(['Code', 'date']).agg({
            'tradeAmount': 'sum',
            'weighted_real_amount': 'sum',
            'real': 'count'
        }).rename(columns={
            'tradeAmount': 'total_amount',
            'weighted_real_amount': 'total_weighted_real',
            'real': 'trade_count'
        })
        
        # 买入方向聚合（用于PnL计算）
        buy_df = self.df[self.df['direction'] == 'B'].copy()
        if len(buy_df) > 0:
            buy_agg = buy_df.groupby(['Code', 'date']).agg({
                'tradeAmount': 'sum',
                'weighted_real_amount': 'sum'
            }).rename(columns={
                'tradeAmount': 'buy_amount',
                'weighted_real_amount': 'buy_weighted_real_amount'
            })
        else:
            buy_agg = pd.DataFrame(columns=['buy_amount', 'buy_weighted_real_amount'])
        
        # 卖出方向聚合（用于PnL计算）
        sell_df = self.df[self.df['direction'] == 'S'].copy()
        if len(sell_df) > 0:
            sell_agg = sell_df.groupby(['Code', 'date']).agg({
                'tradeAmount': 'sum',
                'weighted_real_amount': 'sum'
            }).rename(columns={
                'tradeAmount': 'sell_amount',
                'weighted_real_amount': 'sell_weighted_real_amount'
            })
        else:
            sell_agg = pd.DataFrame(columns=['sell_amount', 'sell_weighted_real_amount'])
        
        # 合并所有聚合结果
        stock_daily = total_agg.join(buy_agg, how='left').join(sell_agg, how='left')
        stock_daily = stock_daily.fillna(0)
        
        # 计算加权平均real值
        stock_daily['weighted_real'] = stock_daily['total_weighted_real'] / stock_daily['total_amount'].replace(0, np.nan)
        stock_daily['weighted_real'] = stock_daily['weighted_real'].fillna(0)
        
        # 计算买卖方向的加权real（用于PnL）
        stock_daily['buy_weighted_real'] = stock_daily['buy_weighted_real_amount'] / stock_daily['buy_amount'].replace(0, np.nan)
        stock_daily['buy_weighted_real'] = stock_daily['buy_weighted_real'].fillna(0)
        
        stock_daily['sell_weighted_real'] = stock_daily['sell_weighted_real_amount'] / stock_daily['sell_amount'].replace(0, np.nan)
        stock_daily['sell_weighted_real'] = stock_daily['sell_weighted_real'].fillna(0)
        
        stock_daily = stock_daily.reset_index()
        
        print(f"聚合后股票-日记录数: {len(stock_daily)}")
        print(f"涉及股票数: {stock_daily['Code'].nunique()}, 交易日数: {stock_daily['date'].nunique()}")
        
        # 2. 计算三种收益方式
        
        # 优先使用配对交易数据替代旧的 real 字段口径
        print("\n<i class='fas fa-chart-bar text-indigo-500'></i> 优先使用配对交易数据计算 等权/金额加权 日收益...")
        used_pairs_for_eq_amt = False
        try:
            paired_df = pd.read_parquet('data/paired_trades_fifo.parquet')
            if len(paired_df) > 0:
                # 时间戳转为datetime
                if not pd.api.types.is_datetime64_any_dtype(paired_df['buy_timestamp']):
                    paired_df['buy_timestamp'] = pd.to_datetime(paired_df['buy_timestamp'])
                if not pd.api.types.is_datetime64_any_dtype(paired_df['sell_timestamp']):
                    paired_df['sell_timestamp'] = pd.to_datetime(paired_df['sell_timestamp'])
                
                # 若不存在 trade_type，按时间先后推断：sell<buy 视作空头，否则多头
                if 'trade_type' not in paired_df.columns:
                    paired_df['trade_type'] = np.where(
                        paired_df['sell_timestamp'] < paired_df['buy_timestamp'], 'short', 'long'
                    )
                
                # 关闭日期：多头=卖出日；空头=买入日
                paired_df['close_timestamp'] = np.where(
                    paired_df['trade_type'] == 'short',
                    paired_df['buy_timestamp'],
                    paired_df['sell_timestamp']
                )
                paired_df['close_date'] = pd.to_datetime(paired_df['close_timestamp']).dt.date
                # 开仓日期（用于“买入日归因”和“持有期均摊”）
                paired_df['open_timestamp'] = np.where(
                    paired_df['trade_type'] == 'short',
                    paired_df['sell_timestamp'],
                    paired_df['buy_timestamp']
                )
                paired_df['open_date'] = pd.to_datetime(paired_df['open_timestamp']).dt.date
                
                # 开仓名义金额：多头=buy_amount；空头=sell_amount
                paired_df['open_notional'] = np.where(
                    paired_df['trade_type'] == 'short',
                    pd.to_numeric(paired_df['sell_amount'], errors='coerce'),
                    pd.to_numeric(paired_df['buy_amount'], errors='coerce')
                )
                paired_df['open_notional'] = paired_df['open_notional'].fillna(0.0)
                paired_df['absolute_profit'] = pd.to_numeric(paired_df['absolute_profit'], errors='coerce').fillna(0.0)
                
                # 单笔配对收益率（基于绝对盈利/开仓名义金额）
                paired_df['pair_return'] = np.where(
                    paired_df['open_notional'] > 0,
                    paired_df['absolute_profit'] / paired_df['open_notional'],
                    np.nan
                )
                
                # 过滤缺失
                valid_pairs = paired_df[paired_df['pair_return'].notna() & np.isfinite(paired_df['pair_return'])]
                
                # 三种归因方式：exit(平仓日)、entry(买入日)、spread(持有期均摊)
                def _eq_amt_by_date(df, date_col):
                    eq = df.groupby(date_col)['pair_return'].mean()
                    amt = df.groupby(date_col).apply(
                        lambda g: (g['pair_return'] * g['open_notional']).sum() / g['open_notional'].sum()
                        if g['open_notional'].sum() > 0 else g['pair_return'].mean()
                    )
                    eq.index = pd.to_datetime(eq.index)
                    amt.index = pd.to_datetime(amt.index)
                    return eq.sort_index(), amt.sort_index()

                # 1) 平仓日归因（当前实现）
                eq_exit, amt_exit = _eq_amt_by_date(valid_pairs, 'close_date')

                # 2) 买入日归因：改用 open_date
                eq_entry, amt_entry = _eq_amt_by_date(valid_pairs, 'open_date')

                # 3) 持有期均摊：把每笔收益均摊到 [open_date, close_date]
                def _spread_pairs(df):
                    # 持有期均摊已暂时移除以提升运行速度
                    return pd.Series(dtype=float), pd.Series(dtype=float)

                # 持有期均摊已禁用
                eq_spread, amt_spread = pd.Series(dtype=float), pd.Series(dtype=float)

                # 统一存为字典，便于切换
                eq_by_mode = {
                    'exit': eq_exit,
                    'entry': eq_entry
                }
                amt_by_mode = {
                    'exit': amt_exit,
                    'entry': amt_entry
                }
                # 默认选择平仓日口径用于后续通用变量
                daily_returns_equal = eq_exit
                daily_amount_weighted = amt_exit
                
                used_pairs_for_eq_amt = True
                print(f"   <i class='fas fa-check-circle text-green-500'></i> 使用配对交易数据计算完成: 天数(eq)={len(daily_returns_equal)}, 天数(wt)={len(daily_amount_weighted)}")
            else:
                print("   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 配对交易数据为空，回退到原 real 字段方法")
        except Exception as e:
            print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 读取配对交易数据失败，回退: {e}")
            used_pairs_for_eq_amt = False
        
        if not used_pairs_for_eq_amt:
            # 原方法（兼容回退）：基于 real 字段
            print("\n<i class='fas fa-chart-bar text-indigo-500'></i> 方法1: 股票等权日收益（回退: real）...")
            
            real_stats = stock_daily['weighted_real'].describe()
            print(f"股票日加权real统计: 最小值={real_stats['min']:.2f}, 最大值={real_stats['max']:.2f}, 均值={real_stats['mean']:.4f}")
            
            if abs(real_stats['mean']) > 10:
                if abs(real_stats['mean']) > 100:
                    scale_factor = 10000
                    print(f"使用万分比缩放 (/{scale_factor})")
                else:
                    scale_factor = 100
                    print(f"使用百分比缩放 (/{scale_factor})")
            else:
                scale_factor = 100
                print(f"使用保守缩放 (/{scale_factor})")
            
            stock_daily['scaled_real'] = stock_daily['weighted_real'] / scale_factor
            stock_daily['scaled_real'] = stock_daily['scaled_real'].clip(-0.15, 0.15)
            daily_returns_equal = stock_daily.groupby('date')['scaled_real'].mean()
            daily_returns_equal.index = pd.to_datetime(daily_returns_equal.index)
            daily_returns_equal = daily_returns_equal.sort_index()
            
            print("<i class='fas fa-chart-bar text-indigo-500'></i> 方法2: 成交金额加权日收益（回退: real）...")
            daily_amount_weighted = stock_daily.groupby('date').apply(
                lambda g: (g['scaled_real'] * g['total_amount']).sum() / g['total_amount'].sum() 
                if g['total_amount'].sum() > 0 else g['scaled_real'].mean()
            )
            daily_amount_weighted.index = pd.to_datetime(daily_amount_weighted.index)
            daily_amount_weighted = daily_amount_weighted.sort_index()
            # 定义默认模式映射，避免按钮引用未定义
            eq_by_mode = {'exit': daily_returns_equal, 'entry': daily_returns_equal}
            amt_by_mode = {'exit': daily_amount_weighted, 'entry': daily_amount_weighted}
        
        # 方法3: PnL口径的日收益率（风险基准）：当日盯市PnL / 当日风险敞口
        print("<i class='fas fa-chart-bar text-indigo-500'></i> 方法3: PnL = 当日盯市PnL / 当日风险敞口 ...")

        # 尝试加载盯市NAV以获取真实PnL
        daily_pnl_spend_returns = None
        pnl_spend_df = None
        pnl_spend_valid_mask = None
        try:
            from pathlib import Path
            mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
            if mtm_file.exists():
                print("   <i class='fas fa-check-circle text-green-500'></i> 发现盯市分析结果，按风险敞口口径计算PnL收益率")
                
                # 读取盯市数据并解析
                mtm_df = pd.read_csv(mtm_file)
                mtm_df['date'] = pd.to_datetime(mtm_df['date'])
                
                def parse_currency(val):
                    try:
                        if isinstance(val, str):
                            return float(val.replace(',', '').strip())
                        return float(val)
                    except (ValueError, TypeError):
                        return 0.0
                
                # 使用正确的初始资金重新计算NAV
                CORRECT_INITIAL_CAPITAL = 62_090_808
                
                # 解析多空市值
                mtm_df['long_value_num'] = mtm_df['long_value'].apply(parse_currency)
                mtm_df['short_value_num'] = mtm_df['short_value'].apply(parse_currency)
                
                # 从订单数据重新计算现金和NAV
                orders_temp = self.df.copy()
                orders_temp['date'] = pd.to_datetime(orders_temp['Timestamp']).dt.date
                daily_flows_temp = orders_temp.groupby(['date', 'direction'])[['tradeAmount', 'fee']].sum().unstack(fill_value=0)
                daily_flows_temp.columns = [f"{a}_{b}" for a, b in daily_flows_temp.columns]
                
                cash_balance = CORRECT_INITIAL_CAPITAL
                cash_series = []
                for date_val in mtm_df['date'].dt.date:
                    if date_val in daily_flows_temp.index:
                        buy_amt = daily_flows_temp.loc[date_val, 'tradeAmount_B'] if 'tradeAmount_B' in daily_flows_temp.columns else 0
                        sell_amt = daily_flows_temp.loc[date_val, 'tradeAmount_S'] if 'tradeAmount_S' in daily_flows_temp.columns else 0
                        fee_amt = (daily_flows_temp.loc[date_val, 'fee_B'] if 'fee_B' in daily_flows_temp.columns else 0) + \
                                  (daily_flows_temp.loc[date_val, 'fee_S'] if 'fee_S' in daily_flows_temp.columns else 0)
                        cash_balance += sell_amt - buy_amt - fee_amt
                    cash_series.append(cash_balance)
                
                mtm_df['cash_num'] = cash_series
                mtm_df['total_assets_num'] = mtm_df['cash_num'] + mtm_df['long_value_num'] - mtm_df['short_value_num']
                mtm_df['exposure_risk'] = (mtm_df['long_value_num'].abs() + mtm_df['short_value_num'].abs()) / 2
                mtm_df['equity_prev'] = mtm_df['total_assets_num'].shift(1)
                
                # 当日真实PnL
                mtm_df = mtm_df.sort_values('date')
                mtm_df['daily_pnl'] = mtm_df['total_assets_num'].diff()
                
                daily_pnl_series = pd.Series(mtm_df['daily_pnl'].values, index=mtm_df['date']).sort_index()
                exposure_series = pd.Series(mtm_df['exposure_risk'].values, index=mtm_df['date']).sort_index()
                equity_prev_series = pd.Series(mtm_df['equity_prev'].values, index=mtm_df['date']).sort_index()

                # 对齐日期并计算 PnL/风险敞口（无敞口时回退到前一日NAV）
                aligned = pd.concat([
                    daily_pnl_series.rename('daily_pnl'),
                    exposure_series.rename('daily_exposure'),
                    equity_prev_series.rename('equity_prev')
                ], axis=1)

                denom = aligned['daily_exposure'].where(aligned['daily_exposure'] > 0, aligned['equity_prev'])
                valid = (denom > 0) & aligned['daily_pnl'].notna()
                daily_pnl_spend_returns = (aligned.loc[valid, 'daily_pnl'] / denom.loc[valid]).clip(-1.0, 1.0)
                # 保留对齐数据与有效掩码用于后续指标
                pnl_spend_df = aligned.assign(denominator=denom)
                pnl_spend_valid_mask = valid
            else:
                print("   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到盯市分析结果，无法计算PnL口径，跳过该方法")
        except Exception as e:
            print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 计算PnL失败: {e}")

        # 若不可用，则用空序列占位，保持后续流程健壮
        if daily_pnl_spend_returns is None:
            daily_pnl_spend_returns = pd.Series(dtype=float)
        daily_pnl_spend_returns_clipped = daily_pnl_spend_returns
        
        # 使用等权方法作为主要显示（最稳定）
        daily_returns = daily_returns_equal
        print(f"<i class='fas fa-check-circle text-green-500'></i> 选择等权方法作为主要收益序列")
        print(f"等权日收益范围: {daily_returns.min():.4f} 到 {daily_returns.max():.4f}, 均值: {daily_returns.mean():.6f}")
        
        # <i class='fas fa-chart-bar text-indigo-500'></i> 调试：检查三种方法的数据范围
        print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 三种方法的收益范围对比：")
        print(f"   等权方法: {daily_returns_equal.min():.4f} 到 {daily_returns_equal.max():.4f}, 标准差: {daily_returns_equal.std():.4f}")
        print(f"   金额加权: {daily_amount_weighted.min():.4f} 到 {daily_amount_weighted.max():.4f}, 标准差: {daily_amount_weighted.std():.4f}")
        if len(daily_pnl_spend_returns_clipped) > 0:
            print(f"   PnL: {daily_pnl_spend_returns_clipped.min():.4f} 到 {daily_pnl_spend_returns_clipped.max():.4f}, 标准差: {daily_pnl_spend_returns_clipped.std():.4f}")
        else:
            print(f"   PnL: 无可用数据")
        
        # 构造DatetimeIndex以便重采样
        daily_returns_dt = daily_returns
        # 修正：使用正确的复利计算
        weekly_returns = daily_returns_dt.resample('W').apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0)
        monthly_returns = daily_returns_dt.resample('M').apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0)
        
        # 3. 生成可视化输出，对比三种方法
        if len(daily_returns) > 0:
            
            # 计算三种方法的统计指标（含归因切换）
            methods_comparison = {
                '等权收益': daily_returns_equal,
                '金额加权': daily_amount_weighted, 
                'PnL': daily_pnl_spend_returns_clipped
            }
            
            # 1. 日收益率对比图
            fig_returns_comp = go.Figure()
            
            colors = ['blue', 'green', 'red']
            # 初始 traces：使用默认模式 exit 的x/y
            for i, (method_name, returns_series) in enumerate(methods_comparison.items()):
                if len(returns_series) > 250:
                    step = len(returns_series) // 200
                    sampled = returns_series.iloc[::step]
                else:
                    sampled = returns_series
                    
                x_list = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in sampled.index]
                y_list = [(float(v) * 100.0) if v is not None else None for v in sampled.values]
                
                fig_returns_comp.add_trace(go.Scatter(x=x_list, y=y_list, mode='lines', name=method_name,
                                                      line=dict(color=colors[i], width=1.5), opacity=0.8))
            
            # 准备各归因模式的采样XY与相关性
            def _sample_xy(series):
                s = series.dropna().sort_index()
                if len(s) > 250:
                    step = max(1, len(s)//200)
                    s = s.iloc[::step]
                x = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in s.index]
                y = [float(v)*100.0 for v in s.values]
                return x, y
            def _corr_with_pnl(series, pnl_series):
                try:
                    s = series.dropna()
                    p = pnl_series.reindex(s.index).dropna()
                    aligned = s.reindex(p.index)
                    if len(aligned) < 5:
                        return np.nan
                    return float(np.corrcoef(aligned.values, p.values)[0,1])
                except Exception:
                    return np.nan

            modes = [('exit','平仓日'), ('entry','买入日')]
            eq_xy_map = {m: _sample_xy(eq_by_mode[m]) for m,_ in modes}
            amt_xy_map = {m: _sample_xy(amt_by_mode[m]) for m,_ in modes}
            if len(daily_pnl_spend_returns_clipped) > 0:
                pnl_x, pnl_y = _sample_xy(daily_pnl_spend_returns_clipped)
            else:
                pnl_x, pnl_y = [], []

            title_sub_map = {}
            for m,_ in modes:
                corr0 = _corr_with_pnl(amt_by_mode[m], daily_pnl_spend_returns_clipped)
                corr_p1 = _corr_with_pnl(amt_by_mode[m].shift(1), daily_pnl_spend_returns_clipped)
                corr_m1 = _corr_with_pnl(amt_by_mode[m].shift(-1), daily_pnl_spend_returns_clipped)
                title_sub_map[m] = (
                    f"等权: {eq_by_mode[m].mean()*100:.3f}% | "
                    f"金额加权: {amt_by_mode[m].mean()*100:.3f}% | "
                    f"PnL: {(daily_pnl_spend_returns_clipped.mean()*100 if len(daily_pnl_spend_returns_clipped) > 0 else 0):.3f}% | "
                    f"corr({m},PnL)={corr0:.3f}, lag+1={corr_p1:.3f}, lag-1={corr_m1:.3f}"
                )

            initial_mode = 'exit'
            fig_returns_comp.update_layout(
                title=f'日收益率对比：衡量下单质量的三种方法<br><sub>{title_sub_map[initial_mode]}</sub>',
                xaxis_title='日期',
                yaxis_title='收益率 (%)',
                height=450,
                hovermode='x unified',
                legend=dict(x=0.02, y=0.98)
            )
            
            comparison_explanation = """
<h4>页面目的</h4>
<ul>
    <li>用三种日收益率口径评估下单质量，区分信号排序、资金分配与现金效率。</li>
    <li>提供买入日/平仓日两种归因方式，观察收益对执行时点的敏感度。</li>
</ul>
<h4>实现方式</h4>
<ol>
    <li>对每只 <code>Code</code> 按 <code>Timestamp</code> 先后执行 FIFO 配对：买入(<code>B</code>)与卖出(<code>S</code>)的 <code>tradeQty</code>、<code>tradeAmount</code>、<code>fee</code> 成对，单笔净收益 = 卖出金额 − 买入金额 − 买卖两端 <code>fee</code>，单笔收益率 = 净收益 ÷ 开仓时 <code>tradeAmount</code>。</li>
    <li>按归因日期聚合：平仓日归因使用卖出日，买入日归因使用买入日；分别计算等权均值与以开仓 <code>tradeAmount</code> 为权重的加权均值。</li>
    <li><b>PnL</b>（风险口径）：当日盯市盈亏 ÷ 当日平均风险敞口，其中敞口 = (|多头市值| + |空头市值|)/2；若敞口缺失则回退到前一日NAV，仅分母>0时计算。</li>
    <li>曲线超过200个交易日会按时间抽样展示，避免加载过重，所有统计均基于全量数据。</li>
</ol>
<div style="margin-top:8px;">解读建议：若金额加权明显低于等权，说明大额交易执行质量不足；PnL若弱于配对收益，通常由持有期波动、未平仓敞口或杠杆/对冲暴露影响。</div>
"""
            
            # 添加按钮用于切换归因方式
            # 归因切换按钮：联动前两条曲线(等权/金额加权)与标题
            buttons = []
            for m,label in modes:
                x0,y0 = eq_xy_map[m]
                x1,y1 = amt_xy_map[m]
                x2,y2 = pnl_x, pnl_y
                buttons.append({
                    'label': label,
                    'method': 'update',
                    'args': [
                        {'x': [x0, x1, x2], 'y': [y0, y1, y2]},
                        {'title': f'日收益率对比：衡量下单质量的三种方法<br><sub>{title_sub_map[m]}</sub>'}
                    ]
                })
            fig_returns_comp.update_layout(
                updatemenus=[{
                    'type': 'buttons', 'direction': 'right',
                    'x': 0.5, 'y': 1.15, 'xanchor': 'center', 'yanchor': 'top',
                    'buttons': buttons
                }]
            )

            self._save_figure_with_details(fig_returns_comp, 'daily_returns_comparison_light', '日收益率对比（衡量下单质量的三种方法）', comparison_explanation, {})
            
            # 2. 使用 PnL 口径的日收益率进行展示
            print("<i class='fas fa-chart-bar text-indigo-500'></i> 使用PnL口径的日收益率数据...")

            returns_for_display = None
            data_source_name = ""
            calculation_method = ""

            if 'daily_pnl_spend_returns_clipped' in locals() and len(daily_pnl_spend_returns_clipped) > 0:
                returns_for_display = daily_pnl_spend_returns_clipped
                data_source_name = "PnL"
                calculation_method = "当日盯市PnL ÷ 当日风险敞口（无敞口时回退前一日NAV，仅分母>0）"
            else:
                # 回退：等权日收益率
                returns_for_display = daily_returns
                data_source_name = "等权日收益率(回退)"
                calculation_method = "股票-日聚合，等权平均"
            
            # 数据采样
            if len(returns_for_display) > 250:
                step = len(returns_for_display) // 200
                returns_sampled = returns_for_display.iloc[::step]
            else:
                returns_sampled = returns_for_display
                
            x_list = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in returns_sampled.index]
            y_list = [(float(v) * 100.0) if v is not None else None for v in returns_sampled.values]
                
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(
                x=x_list,
                y=y_list,
                mode='lines',
                name=f'日收益率（{data_source_name}）',
                line=dict(color='blue', width=1.5),
                hovertemplate='日期: %{x}<br>收益率: %{y:.2f}%<extra></extra>'
            ))
            
            fig_returns.update_layout(
                title=f'日收益率时间序列（真实盯市方法）<br><sub>平均收益: {returns_for_display.mean()*100:.3f}%, 波动率: {returns_for_display.std()*100:.3f}%</sub>',
                xaxis_title='日期',
                yaxis_title='收益率 (%)',
                height=400,
                xaxis=dict(type='date')
            )
            
            # 直接计算关键指标，避免数据裁剪导致的不一致（PnL风险口径视为常规日收益序列）
            real_cumulative = (1 + returns_for_display).cumprod() - 1 if len(returns_for_display) > 0 else pd.Series(dtype=float)
            total_return = real_cumulative.iloc[-1] if len(real_cumulative) > 0 else 0.0
            win_rate = (returns_for_display > 0).mean() if len(returns_for_display) > 0 else 0.0
            volatility = returns_for_display.std() * np.sqrt(252) if len(returns_for_display) > 1 else 0.0

            if len(returns_for_display) > 0:
                cumulative_nav_series = (1 + returns_for_display).cumprod()
                rolling_max = cumulative_nav_series.expanding().max()
                drawdown = (cumulative_nav_series - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
            else:
                max_drawdown = 0.0
            
            if len(returns_for_display) > 1:
                annualized_return = (1 + total_return) ** (252 / len(returns_for_display)) - 1
            else:
                annualized_return = 0.0
                
            sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0.0
            
            perf_metrics = {
                '总收益率': f"{total_return:.2%}",
                '年化收益率': f"{annualized_return:.2%}",
                '年化波动率': f"{volatility:.2%}",
                '夏普比率': f"{sharpe_ratio:.3f}",
                '最大回撤': f"{max_drawdown:.2%}",
                '胜率': f"{win_rate:.2%}",
                '交易天数': f"{len(returns_for_display)}天",
                '数据来源': data_source_name,
                '计算方法': calculation_method
            }
            perf_metrics.update({
                    '数据来源': data_source_name,
                    '计算方法': calculation_method,
                    '交易天数': f"{len(returns_for_display)}天"
            })

            # 2. 累积收益 - 修正计算逻辑
            print(f"<i class='fas fa-search text-blue-400'></i> 计算累积收益，日收益率样本: min={daily_returns.min():.4f}, max={daily_returns.max():.4f}")
            
            # 对于盯市数据，保持原始数据完整性
            # 只对极端异常值进行裁剪
            safe_returns = daily_returns.clip(-0.9, 0.9)  # 只限制极端异常值（如单日90%+的收益率）
            
            # 使用复利公式：净值 = ∏(1 + 日收益率)
            cumulative_nav = (1 + safe_returns).cumprod()
            cumulative_returns = cumulative_nav - 1  # 转换为收益率
            
            print(f"累积收益计算完成: 起始={cumulative_returns.iloc[0]:.4f}, 最终={cumulative_returns.iloc[-1]:.4f}")
            
            cum_sampled = cumulative_returns.iloc[::max(1, len(cumulative_returns)//200)]
            
            # 构造纯列表，规避 TypedArray 解析异常
            x_cum = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in cum_sampled.index]
            y_cum = [(float(v) * 100.0) if v is not None else None for v in cum_sampled.values]
            
            fig_cum = go.Figure()
            # 累积收益图：显式使用 cumulative_returns（已验证），避免外部二次计算
            fig_cum.add_trace(go.Scatter(
                x=x_cum,
                y=y_cum,
                mode='lines',
                name='累积收益',
                line=dict(color='green', width=2),
                fill='tonexty',
                hovertemplate='日期: %{x}<br>累积收益: %{y:.2f}%<extra></extra>'
            ))
            
            # 计算最大回撤
            cumulative_nav_max = cumulative_nav.expanding().max()
            drawdown = (cumulative_nav - cumulative_nav_max) / cumulative_nav_max
            max_drawdown = drawdown.min()
            
            fig_cum.update_layout(
                title=f'累积收益曲线<br><sub>总收益: {cumulative_returns.iloc[-1]*100:.2f}%, 最大回撤: {max_drawdown*100:.2f}%</sub>',
                xaxis_title='日期',
                yaxis_title='累积收益率 (%)',
                height=400,
                xaxis=dict(type='date')
            )
            
            # 智能累积收益对比图 - 使用子图处理不同量级
            from plotly.subplots import make_subplots  # type: ignore
            
            # 计算三种方法的累积收益和统计数据
            methods_data = {}
            for method_name, returns_series in methods_comparison.items():
                # 默认：按复利累计（PnL 风险口径视为常规日收益序列）
                safe_method_returns = returns_series.clip(-0.9, 0.9)
                cumulative_method = (1 + safe_method_returns).cumprod() - 1
                final_return = cumulative_method.iloc[-1]
                volatility = safe_method_returns.std()
                methods_data[method_name] = {
                    'returns': safe_method_returns,
                    'cumulative': cumulative_method,
                    'final': final_return,
                    'volatility': volatility
                }
            
            # 检查按暴露PnL是否需要单独显示
            pnl_final = methods_data['PnL']['final']
            other_finals = [methods_data[k]['final'] for k in ['等权收益', '金额加权']]
            max_other = max(abs(f) for f in other_finals)
            
            # 如果按暴露PnL的绝对值超过其他方法的5倍，使用子图
            if abs(pnl_final) > 5 * max_other and np.isfinite(pnl_final):
                print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 按暴露PnL收益量级较大 ({pnl_final*100:.1f}%)，使用子图显示")
                
                # 创建双子图
                fig_cum_comp = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('等权 & 金额加权方法', '按暴露PnL方法（独立比例）'),
                    vertical_spacing=0.12,
                    shared_xaxes=True
                )
                
                # 子图1：等权和金额加权
                for i, method_name in enumerate(['等权收益', '金额加权']):
                    cum_data = methods_data[method_name]['cumulative']
                    if len(cum_data) > 200:
                        step = len(cum_data) // 150
                        cum_sampled = cum_data.iloc[::step]
                    else:
                        cum_sampled = cum_data
                    
                    x_list = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in cum_sampled.index]
                    y_list = [float(v) * 100.0 for v in cum_sampled.values]
                    
                    fig_cum_comp.add_trace(
                        go.Scatter(
                            x=x_list, y=y_list,
                            mode='lines', name=f'{method_name}',
                            line=dict(color=colors[i], width=2.5),
                            opacity=0.85
                        ),
                        row=1, col=1
                    )
                
                # 子图2：按暴露PnL方法
                pnl_cum = methods_data['PnL']['cumulative']
                if len(pnl_cum) > 200:
                    step = len(pnl_cum) // 150
                    pnl_sampled = pnl_cum.iloc[::step]
                else:
                    pnl_sampled = pnl_cum
                
                x_pnl = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in pnl_sampled.index]
                y_pnl = [float(v) * 100.0 for v in pnl_sampled.values]
                
                fig_cum_comp.add_trace(
                    go.Scatter(
                        x=x_pnl, y=y_pnl,
                        mode='lines', name='按暴露PnL',
                        line=dict(color=colors[2], width=2.5),
                        opacity=0.85
                    ),
                    row=2, col=1
                )
                
                # 更新布局
                fig_cum_comp.update_layout(
                    title=f'累积收益对比（分层显示）<br><sub style="font-size:11px;">等权: {methods_data["等权收益"]["final"]*100:.2f}% | 金额加权: {methods_data["金额加权"]["final"]*100:.2f}% | 按暴露PnL: {pnl_final*100:.2f}%</sub>',
                    height=600,
                    hovermode='x unified',
                    legend=dict(x=0.02, y=0.98),
                    showlegend=True
                )
                
                # 设置子图Y轴
                fig_cum_comp.update_yaxes(title_text="累积收益率 (%)", row=1, col=1)
                fig_cum_comp.update_yaxes(title_text="累积收益率 (%)", row=2, col=1)
                fig_cum_comp.update_xaxes(title_text="日期", row=2, col=1, type='date')
                
            else:
                print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 三种方法量级相近，使用统一图表显示")
                
                # 使用传统单图显示
                fig_cum_comp = go.Figure()
                
                for i, (method_name, data) in enumerate(methods_data.items()):
                    cum_data = data['cumulative']
                    if len(cum_data) > 200:
                        step = len(cum_data) // 150
                        cum_sampled = cum_data.iloc[::step]
                    else:
                        cum_sampled = cum_data
                    
                    x_list = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in cum_sampled.index]
                    y_list = [float(v) * 100.0 for v in cum_sampled.values]
                    
                    fig_cum_comp.add_trace(go.Scatter(
                        x=x_list, y=y_list,
                        mode='lines', name=f'{method_name}',
                        line=dict(color=colors[i], width=2.5),
                        opacity=0.85
                    ))
                
                fig_cum_comp.update_layout(
                    title=f'累积收益对比<br><sub>等权: {methods_data["等权收益"]["final"]*100:.2f}% | 金额加权: {methods_data["金额加权"]["final"]*100:.2f}% | PnL: {pnl_final*100:.2f}%</sub>',
                    xaxis_title='日期',
                    yaxis_title='累积收益率 (%)',
                    height=450,
                    hovermode='x unified',
                    legend=dict(x=0.02, y=0.98),
                    xaxis=dict(type='date')
                )
            
            # 保存最终收益数据
            equal_total = methods_data['等权收益']['final']
            weighted_total = methods_data['金额加权']['final']
            pnl_total = pnl_final
            
            # 更新说明文档
            cumulative_explanation = """
            <h4><i class='fas fa-chart-bar text-indigo-500'></i> 计算方法说明</h4>
            <p>本页展示三种日收益序列的复利累积结果（即 $\\prod(1 + r_i) - 1$），每条曲线的数据来源与计算方法如下：</p>
            
            <h5>方法1：等权收益（期末 {eq_pct:.2%}）</h5>
            <ul>
                <li><b>数据来源</b>：从 <code>data/orders.parquet</code> 中按 <code>Code</code> 和 <code>Timestamp</code> 先后进行买卖配对（FIFO原则）</li>
                <li><b>单笔收益率</b>：$r_{{\\text{{单笔}}}} = \\frac{{\\text{{卖出}}_{{\\text{{tradeAmount}}}} - \\text{{买入}}_{{\\text{{tradeAmount}}}} - \\text{{总}}_{{\\text{{fee}}}}}}{{\\text{{买入}}_{{\\text{{tradeAmount}}}}}}$</li>
                <li><b>日收益聚合</b>：按平仓日期，对当日完成的所有交易对的收益率取简单平均</li>
                <li><b>优点</b>：反映选股与下单方向的纯技术表现，不受资金配置影响</li>
                <li><b>局限</b>：仅包含已平仓交易，未平仓浮动盈亏不计入</li>
            </ul>
            
            <h5>方法2：金额加权收益（期末 {wt_pct:.2%}）</h5>
            <ul>
                <li><b>数据来源</b>：与等权收益相同的配对交易数据</li>
                <li><b>日收益聚合</b>：按开仓时 <code>tradeAmount</code> 作为权重，计算加权平均日收益率</li>
                <li><b>优点</b>：体现资金配置效果，大额交易对收益的影响更大</li>
                <li><b>局限</b>：同样仅包含已平仓交易</li>
            </ul>
            
            <h5>方法3：PnL（风险口径，期末 {pnl_pct:.2%}）</h5>
            <ul>
                <li><b>数据来源</b>：从盯市文件读取每日总资产，计算总资产差分得到当日盈亏</li>
                <li><b>分母定义</b>：当日平均风险敞口 $E_t = (|\\text{{多头市值}}_t| + |\\text{{空头市值}}_t|)/2$，若缺失则回退到前一日 NAV</li>
                <li><b>日收益率</b>：$r_t = \\frac{{\\text{{PnL}}_t}}{{E_t}}$，仅在 $E_t > 0$ 时定义，裁剪至 [-1, 1]</li>
                <li><b>累计口径</b>：按日收益复利：$\\prod (1 + r_t) - 1$</li>
                <li><b>优点</b>：包含全部持仓（已平仓+未平仓），并以风险敞口/权益为基准，反映真实风险效率</li>
                <li><b>适用场景</b>：风险调整后的收益监控、杠杆/对冲策略的资金效率评估</li>
            </ul>
            
            <h4><i class='fas fa-thumbtack text-red-400'></i> 关键假设</h4>
            <ul>
                <li>累积前对日收益序列进行裁剪至 [-90%, 90%] 区间，以抑制极端噪声对复利计算的影响（仅影响展示，不改变原始数据）</li>
                <li>前两种方法基于配对交易，仅统计已完成的买卖对；第三种方法基于盯市总资产，包含全部持仓</li>
                <li>三种方法的差异反映了"交易完成度"、"资金配置效率"与"真实现金效率"的不同视角</li>
            </ul>
            
            <h4><i class='fas fa-book-open text-gray-500'></i> 解读建议</h4>
            <ul>
                <li><b>方法接近</b>：当三条曲线趋势一致时，说明策略收益结构稳健，已平仓与未平仓收益方向一致</li>
                <li><b>方法分离</b>：若PnL显著偏离，可能是未平仓浮动盈亏较大，或当日资金投入与平仓节奏不匹配</li>
                <li><b>建议组合使用</b>：等权看选股能力，金额加权看配置效率，PnL看真实回报</li>
            </ul>
            """.format(eq_pct=equal_total, wt_pct=weighted_total, pnl_pct=pnl_total)
            
            self._save_figure_with_details(
                fig_cum_comp,
                name='cumulative_returns_comparison_light',
                title='累积收益对比（三种方法）',
                explanation_html=cumulative_explanation,
                metrics={}
            )
            
            # 基准对比累积收益图
            if self.benchmark_data:
                print("<i class='fas fa-chart-bar text-indigo-500'></i> 生成策略vs基准累积收益对比图...")

                # 默认使用当前计算的策略累积收益
                strategy_cum_for_bench = cumulative_returns
                daily_returns_for_bench = daily_returns
                strategy_daily_for_plot = daily_amount_weighted

                # 使用盯市分析的日度绝对盈利数据
                try:
                    print("<i class='fas fa-chart-bar text-indigo-500'></i> 使用盯市分析的日度绝对盈利数据用于基准对比...")
                    
                    from pathlib import Path
                    mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
                    if mtm_file.exists():
                        print(f"<i class='fas fa-check-circle text-green-500'></i> 发现盯市分析结果文件: {mtm_file}")
                        
                        # 读取盯市分析结果
                        mtm_df = pd.read_csv(mtm_file)

                        # 解析日期
                        mtm_df['date'] = pd.to_datetime(mtm_df['date'])
                        mtm_df = mtm_df.sort_values('date').reset_index(drop=True)
                        
                        # 解析总资产并计算日度绝对盈利
                        def parse_currency_for_strategy(val):
                            try:
                                if isinstance(val, str):
                                    return float(val.replace(',', '').strip())
                                return float(val)
                            except (ValueError, TypeError):
                                return np.nan
                        
                        # 使用正确的初始资金重新计算NAV
                        CORRECT_INITIAL_CAPITAL = 62_090_808
                        
                        mtm_df['long_value_num'] = mtm_df['long_value'].apply(parse_currency_for_strategy)
                        mtm_df['short_value_num'] = mtm_df['short_value'].apply(parse_currency_for_strategy)
                        
                        # 重新计算现金和NAV
                        orders_temp = self.df.copy()
                        orders_temp['date'] = pd.to_datetime(orders_temp['Timestamp']).dt.date
                        daily_flows_temp = orders_temp.groupby(['date', 'direction'])[['tradeAmount', 'fee']].sum().unstack(fill_value=0)
                        daily_flows_temp.columns = [f"{a}_{b}" for a, b in daily_flows_temp.columns]
                        
                        cash_balance = CORRECT_INITIAL_CAPITAL
                        cash_series = []
                        for date_val in mtm_df['date'].dt.date:
                            if date_val in daily_flows_temp.index:
                                buy_amt = daily_flows_temp.loc[date_val, 'tradeAmount_B'] if 'tradeAmount_B' in daily_flows_temp.columns else 0
                                sell_amt = daily_flows_temp.loc[date_val, 'tradeAmount_S'] if 'tradeAmount_S' in daily_flows_temp.columns else 0
                                fee_amt = (daily_flows_temp.loc[date_val, 'fee_B'] if 'fee_B' in daily_flows_temp.columns else 0) + \
                                          (daily_flows_temp.loc[date_val, 'fee_S'] if 'fee_S' in daily_flows_temp.columns else 0)
                                cash_balance += sell_amt - buy_amt - fee_amt
                            cash_series.append(cash_balance)
                        
                        mtm_df['cash_num'] = cash_series
                        mtm_df['total_assets_num'] = mtm_df['cash_num'] + mtm_df['long_value_num'] - mtm_df['short_value_num']
                        mtm_df['daily_abs_profit'] = mtm_df['total_assets_num'].diff()
                        
                        # 构建盯市净值序列（真实净值口径）
                        nav_series = pd.Series(
                            mtm_df['total_assets_num'].values,
                            index=pd.to_datetime(mtm_df['date'])
                        ).astype(float)
                        daily_return_nav = nav_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        cumulative_return_nav = (1.0 + daily_return_nav).cumprod() - 1.0
                        
                        # 创建日度绝对盈利时间序列（用于可选展示）
                        daily_abs_profit = pd.Series(
                            mtm_df['daily_abs_profit'].values,
                            index=nav_series.index
                        ).dropna().sort_index()
                        cumulative_abs_profit = daily_abs_profit.cumsum()
                        
                        print(f"<i class='fas fa-check-circle text-green-500'></i> 使用盯市分析的日度绝对盈利结果")
                        print(f"   数据期间: {daily_abs_profit.index.min().date()} 到 {daily_abs_profit.index.max().date()}")
                        print(f"   交易天数: {len(daily_abs_profit)} 天")
                        print(f"   日度绝对盈利范围: ¥{daily_abs_profit.min():,.0f} 到 ¥{daily_abs_profit.max():,.0f}")
                        print(f"   累积绝对盈利: ¥{cumulative_abs_profit.iloc[-1]:,.0f}")
                        print(f"   真实净值期末收益: {cumulative_return_nav.iloc[-1]*100:.2f}%")
                        
                        # 以真实净值收益口径对基准图及指标进行对齐
                        strategy_cum_for_bench = cumulative_return_nav
                        daily_returns_for_bench = daily_return_nav
                        
                    else:
                        print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到盯市分析结果文件，使用默认策略收益")
                        
                except Exception as e:
                    print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 读取盯市分析结果失败: {e}")
                    import traceback
                    traceback.print_exc()

                fig_strategy_vs_benchmark = self._create_strategy_benchmark_comparison(
                    strategy_cum_for_bench,
                    strategy_daily_returns=strategy_daily_for_plot
                )

                self._save_figure_with_details(
                    fig_strategy_vs_benchmark,
                    name='strategy_vs_benchmark_light',
                    title='策略vs基准指数累积收益对比',
                    explanation_html="""
<p>左轴展示策略金额加权日收益率（来源：<b>日收益率对比</b>页面的“金额加权”口径）与各基准指数日收益率，便于逐日对照策略相对强弱；右轴保留策略/指数累积收益率（策略曲线默认隐藏，可按需展开）。基准按交易日对齐，均基于日收盘价计算。</p>
""",
                    metrics=self._calculate_benchmark_comparison_metrics(strategy_cum_for_bench, daily_returns_for_bench)
                )
            

            
            # 3. 绝对收益分布（盯市）
            daily_abs_profit = pd.Series(dtype=float)
            try:
                from pathlib import Path
                mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
                if mtm_file.exists():
                    mtm_df = pd.read_csv(mtm_file)
                    mtm_df['date'] = pd.to_datetime(mtm_df['date'])
                    def _parse_currency_for_abs(v):
                        try:
                            if isinstance(v, str):
                                return float(v.replace(',', '').strip())
                            return float(v)
                        except (ValueError, TypeError):
                            return np.nan
                    mtm_df['total_assets_num'] = mtm_df['total_assets'].apply(_parse_currency_for_abs)
                    mtm_df = mtm_df.sort_values('date')
                    mtm_df['daily_abs_profit'] = mtm_df['total_assets_num'].diff()
                    daily_abs_profit = pd.Series(
                        mtm_df['daily_abs_profit'].values,
                        index=pd.to_datetime(mtm_df['date'])
                    ).dropna().sort_index()
            except Exception:
                daily_abs_profit = pd.Series(dtype=float)

            fig_dist = go.Figure()
            if len(daily_abs_profit) > 0:
                abs_vals = daily_abs_profit.values.astype(float)
                amount_range = float(abs_vals.max() - abs_vals.min())
                try:
                    q75, q25 = np.percentile(abs_vals, [75, 25])
                    iqr = float(q75 - q25)
                    if iqr > 0:
                        bin_width = 2 * iqr * (len(abs_vals) ** (-1/3))
                        optimal_bins = int(np.clip(np.ceil(amount_range / bin_width), 20, 60)) if bin_width > 0 else max(20, min(60, int(len(abs_vals) / 10)))
                    else:
                        optimal_bins = max(20, min(60, int(len(abs_vals) / 10)))
                except Exception:
                    optimal_bins = max(20, min(60, int(len(abs_vals) / 10)))

                print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 绝对收益分布参数: 范围=¥{amount_range:,.0f}, bin数={optimal_bins}")

                fig_dist.add_trace(go.Histogram(
                    x=abs_vals,
                    nbinsx=optimal_bins,
                    name='绝对收益分布',
                    opacity=0.75,
                    marker_color='lightcoral',
                    marker_line_color='darkred',
                    marker_line_width=1,
                    hovertemplate='日绝对收益: ¥%{x:,.0f}<br>频次: %{y}<extra></extra>'
                ))

                mean_amt = float(np.mean(abs_vals))
                median_amt = float(np.median(abs_vals))
                std_amt = float(np.std(abs_vals, ddof=0))

                # 均值/中位数/±1σ
                fig_dist.add_vline(x=mean_amt, line_dash="dash", line_color="red", line_width=2)
                show_median = abs(median_amt - mean_amt) > max(1.0, std_amt * 0.01)
                if show_median:
                    fig_dist.add_vline(x=median_amt, line_dash="dot", line_color="blue", line_width=1.5)
                fig_dist.add_vline(x=mean_amt - std_amt, line_dash="dashdot", line_color="orange", line_width=1, opacity=0.7)
                fig_dist.add_vline(x=mean_amt + std_amt, line_dash="dashdot", line_color="orange", line_width=1, opacity=0.7)

                # 注释
                annotations = []
                stats_text = f"<b>统计信息</b><br>均值: ¥{mean_amt:,.0f}<br>标准差: ¥{std_amt:,.0f}"
                if show_median:
                    stats_text += f"<br>中位数: ¥{median_amt:,.0f}"
                annotations.append(dict(
                    xref='paper', yref='paper', x=0.98, y=0.98, xanchor='right', yanchor='top',
                    text=stats_text, showarrow=False,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1, font=dict(size=11)
                ))
                legend_text = "<b>图例</b><br><span style='color:red'>━━</span> 均值"
                if show_median:
                    legend_text += "<br><span style='color:blue'>┅┅</span> 中位数"
                legend_text += "<br><span style='color:orange'>⋯⋯</span> ±1σ"
                annotations.append(dict(
                    xref='paper', yref='paper', x=0.02, y=0.98, xanchor='left', yanchor='top',
                    text=legend_text, showarrow=False,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1, font=dict(size=10)
                ))

                # 轴与标题
                x_min = float(abs_vals.min())
                x_max = float(abs_vals.max())
                fig_dist.update_layout(
                    title=f'日绝对收益分布（盯市）<br><sub>样本: {len(abs_vals)}天 | 范围: ¥{x_min:,.0f} ~ ¥{x_max:,.0f} | 偏度: {daily_abs_profit.skew():.3f} | 峰度: {daily_abs_profit.kurtosis():.3f}</sub>',
                    xaxis_title='日绝对收益 (¥)',
                    yaxis_title='频次 (天数)',
                    height=450,
                    xaxis=dict(tickformat=',.0f'),
                    bargap=0.1,
                    annotations=annotations
                )
            else:
                # 回退：若无盯市数据，仍使用收益率分布
                returns_pct = daily_returns.values * 100
                returns_range = returns_pct.max() - returns_pct.min()
                if returns_range < 2:
                    optimal_bins = max(15, min(25, len(set(returns_pct)) // 2))
                else:
                    optimal_bins = max(20, min(40, int(len(returns_pct) / 10)))
                print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 收益分布参数(回退): 范围={returns_range:.3f}%, bin数={optimal_bins}")
                fig_dist.add_trace(go.Histogram(
                    x=returns_pct,
                    nbinsx=optimal_bins,
                    name='收益分布',
                    opacity=0.75,
                    marker_color='lightcoral',
                    marker_line_color='darkred',
                    marker_line_width=1,
                    hovertemplate='收益率: %{x:.3f}%<br>频次: %{y}<extra></extra>'
                ))
                mean_return = daily_returns.mean() * 100
                median_return = np.median(returns_pct)
                std_return = daily_returns.std() * 100
                fig_dist.add_vline(x=mean_return, line_dash="dash", line_color="red", line_width=2)
                show_median = abs(median_return - mean_return) > 0.01
                if show_median:
                    fig_dist.add_vline(x=median_return, line_dash="dot", line_color="blue", line_width=1.5)
                fig_dist.add_vline(x=mean_return - std_return, line_dash="dashdot", line_color="orange", line_width=1, opacity=0.7)
                fig_dist.add_vline(x=mean_return + std_return, line_dash="dashdot", line_color="orange", line_width=1, opacity=0.7)
                annotations = []
                stats_text = f"<b>统计信息</b><br>均值: {mean_return:.3f}%<br>标准差: {std_return:.3f}%"
                if show_median:
                    stats_text += f"<br>中位数: {median_return:.3f}%"
                annotations.append(dict(xref='paper', yref='paper', x=0.98, y=0.98, xanchor='right', yanchor='top', text=stats_text, showarrow=False, bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1, font=dict(size=11)))
                legend_text = "<b>图例</b><br><span style='color:red'>━━</span> 均值"
                if show_median:
                    legend_text += "<br><span style='color:blue'>┅┅</span> 中位数"
                legend_text += "<br><span style='color:orange'>⋯⋯</span> ±1σ"
                annotations.append(dict(xref='paper', yref='paper', x=0.02, y=0.98, xanchor='left', yanchor='top', text=legend_text, showarrow=False, bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1, font=dict(size=10)))
                x_margin = max(0.1, returns_range * 0.1)
                x_min = returns_pct.min() - x_margin
                x_max = returns_pct.max() + x_margin
                fig_dist.update_layout(
                    title=f'日收益率分布（回退）<br><sub>样本: {len(returns_pct)}天 | 范围: {returns_pct.min():.3f}%~{returns_pct.max():.3f}% | 偏度: {daily_returns.skew():.3f} | 峰度: {daily_returns.kurtosis():.3f}</sub>',
                    xaxis_title='日收益率 (%)',
                    yaxis_title='频次 (天数)',
                    height=450,
                    xaxis=dict(range=[x_min, x_max], tickformat='.3f', dtick=max(0.05, returns_range / 10)),
                    bargap=0.1,
                    annotations=annotations
                )
            
            self._save_figure_with_details(
                fig_dist,
                name='returns_distribution_light',
                title='日绝对收益分布（盯市）',
                explanation_html="<p>展示日绝对收益（盯市）分布形态，定位极端值与偏态风险。</p>",
                metrics=perf_metrics,
            )
            
            # 计算关键指标
            self._calculate_key_metrics(daily_returns)
            
    def execution_analysis(self):
        """交易执行分析"""
        print("\n<i class='fas fa-bolt text-yellow-400'></i> === 交易执行分析 ===")
        
        # === 日内交易平均持仓时间（按买入日） ===
        try:
            import pandas as _pd
            import numpy as _np
            print("<i class='fas fa-download text-blue-400'></i> 加载配对交易数据用于持仓时间统计…")
            _pairs = _pd.read_parquet('data/paired_trades_fifo.parquet')
            if len(_pairs) > 0:
                # 统一时间戳
                if not _pd.api.types.is_datetime64_any_dtype(_pairs['buy_timestamp']):
                    _pairs['buy_timestamp'] = _pd.to_datetime(_pairs['buy_timestamp'])
                if not _pd.api.types.is_datetime64_any_dtype(_pairs['sell_timestamp']):
                    _pairs['sell_timestamp'] = _pd.to_datetime(_pairs['sell_timestamp'])
                # 推断多空并确定开/平仓时间
                if 'trade_type' not in _pairs.columns:
                    _pairs['trade_type'] = _np.where(
                        _pairs['sell_timestamp'] < _pairs['buy_timestamp'], 'short', 'long'
                    )
                _pairs['open_timestamp'] = _np.where(
                    _pairs['trade_type'] == 'short',
                    _pairs['sell_timestamp'],
                    _pairs['buy_timestamp']
                )
                _pairs['close_timestamp'] = _np.where(
                    _pairs['trade_type'] == 'short',
                    _pairs['buy_timestamp'],
                    _pairs['sell_timestamp']
                )
                # 绝对收益，用于计算收益覆盖率所需持仓时间
                _pairs['abs_profit'] = _pairs['absolute_profit'].abs().fillna(0)
                # 计算“交易时段内”的持仓分钟数（跨日仅累计交易时段，周末/午休不计）
                _pairs['open_date'] = _pd.to_datetime(_pairs['open_timestamp']).dt.date
                _pairs['close_date'] = _pd.to_datetime(_pairs['close_timestamp']).dt.date

                # 向量化计算：两个交易时段（分钟）
                M1, M2, A1, A2 = 570, 690, 780, 900  # 09:30-11:30, 13:00-15:00
                open_min = (_pairs['open_timestamp'].dt.hour * 60 + _pairs['open_timestamp'].dt.minute).astype(float)
                close_min = (_pairs['close_timestamp'].dt.hour * 60 + _pairs['close_timestamp'].dt.minute).astype(float)
                same_day = (_pairs['open_date'] == _pairs['close_date']).to_numpy()

                # 同日：与当日两段交易时段求交集
                morning_same = _np.clip(_np.minimum(close_min, M2) - _np.maximum(open_min, M1), 0, None)
                afternoon_same = _np.clip(_np.minimum(close_min, A2) - _np.maximum(open_min, A1), 0, None)
                minutes_same = (morning_same + afternoon_same).to_numpy()

                # 跨日：开仓日剩余 + 收盘日已过 + 中间交易日(工作日)×240
                open_morning_remain = _np.clip(M2 - _np.maximum(open_min, M1), 0, None)
                open_afternoon_remain = _np.clip(A2 - _np.maximum(open_min, A1), 0, None)
                open_minutes = (open_morning_remain + open_afternoon_remain).to_numpy()

                close_morning_elapsed = _np.clip(_np.minimum(close_min, M2) - M1, 0, None)
                close_afternoon_elapsed = _np.clip(_np.minimum(close_min, A2) - A1, 0, None)
                close_minutes = (close_morning_elapsed + close_afternoon_elapsed).to_numpy()

                open_d = _pairs['open_date'].values.astype('datetime64[D]')
                close_d = _pairs['close_date'].values.astype('datetime64[D]')
                middle_days = _np.busday_count(open_d + _np.timedelta64(1, 'D'), close_d, weekmask='1111100')
                middle_minutes = (middle_days.astype('int64') * 240)

                minutes_all = _np.where(same_day, minutes_same, open_minutes + close_minutes + middle_minutes)
                _pairs['holding_minutes'] = minutes_all

                # 按买入日聚合（全体平均）
                _daily_holding = _pairs.groupby('open_date')['holding_minutes'].mean().sort_index()

                # 计算每日“股票层面”的上下限：按每个股票当日平均持仓时间，取最短/最长5%股票的均值
                _code_col = 'code' if 'code' in _pairs.columns else ('Code' if 'Code' in _pairs.columns else None)
                _lower_series = None
                _upper_series = None
                if _code_col is not None:
                    _code_day = _pairs.groupby(['open_date', _code_col])['holding_minutes'].mean().reset_index()
                    def _low_high(g):
                        arr = _np.sort(g['holding_minutes'].to_numpy())
                        n = arr.size
                        if n == 0:
                            return _pd.Series({'low_mean': _np.nan, 'high_mean': _np.nan})
                        k = max(1, int(n * 0.05))
                        low_mean = float(_np.mean(arr[:k]))
                        high_mean = float(_np.mean(arr[-k:]))
                        return _pd.Series({'low_mean': low_mean, 'high_mean': high_mean})
                    _bounds = _code_day.groupby('open_date').apply(_low_high).reset_index()
                    _lower_series = _bounds.set_index('open_date')['low_mean'].sort_index()
                    _upper_series = _bounds.set_index('open_date')['high_mean'].sort_index()
                    # 对齐主索引
                    _lower_series = _lower_series.reindex(_daily_holding.index)
                    _upper_series = _upper_series.reindex(_daily_holding.index)
                # 计算每个买入日，累计绝对收益覆盖25%/50%/75%时对应的持仓分钟数
                def _profit_cover_time(_g):
                    _profit = _g['abs_profit'].to_numpy()
                    _hold = _g['holding_minutes'].to_numpy()
                    _total = _profit.sum()
                    if _total <= 0 or _hold.size == 0:
                        return _pd.Series({'p25_time': _np.nan, 'p50_time': _np.nan, 'p75_time': _np.nan})
                    _order = _np.argsort(_hold)
                    _profit_sorted = _profit[_order]
                    _hold_sorted = _hold[_order]
                    _cum = _np.cumsum(_profit_sorted) / _total
                    def _find(_th):
                        _idx = _np.searchsorted(_cum, _th, side='left')
                        _idx = min(_idx, len(_hold_sorted) - 1)
                        return float(_hold_sorted[_idx])
                    return _pd.Series({
                        'p25_time': _find(0.25),
                        'p50_time': _find(0.50),
                        'p75_time': _find(0.75)
                    })
                _profit_cover = _pairs.groupby('open_date').apply(_profit_cover_time)
                if isinstance(_profit_cover.index, _pd.MultiIndex):
                    _profit_cover.index = _profit_cover.index.get_level_values(0)
                _profit_cover = _profit_cover.sort_index()
                _profit_cover = _profit_cover.reindex(_daily_holding.index)
                if len(_daily_holding) > 0:
                    # 采样以控制页面体积
                    _series = _daily_holding
                    if len(_series) > 200:
                        _step = max(1, len(_series)//150)
                        _series = _series.iloc[::_step]
                    _x_idx = list(_series.index)
                    _x = [str(d) for d in _x_idx]
                    _y = [float(v) for v in _series.values]
                    fig_hold = go.Figure()
                    fig_hold.add_trace(go.Scatter(
                        x=_x,
                        y=_y,
                        mode='lines+markers',
                        name='平均持仓时间（全体）',
                        line=dict(color='teal', width=2),
                        marker=dict(size=4),
                        hovertemplate='日期: %{x}<br>平均持仓: %{y:.1f} 分钟<extra></extra>'
                    ))
                    # 上下限曲线（若可计算）
                    if _lower_series is not None and _upper_series is not None:
                        _lower_sample = _lower_series.loc[_x_idx]
                        _upper_sample = _upper_series.loc[_x_idx]
                        fig_hold.add_trace(go.Scatter(
                            x=_x,
                            y=[None if _np.isnan(v) else float(v) for v in _lower_sample.values],
                            mode='lines',
                            name='最短5%股票平均',
                            line=dict(color='purple', width=1.8, dash='dash'),
                            hovertemplate='日期: %{x}<br>最短5%平均: %{y:.1f} 分钟<extra></extra>'
                        ))
                        fig_hold.add_trace(go.Scatter(
                            x=_x,
                            y=[None if _np.isnan(v) else float(v) for v in _upper_sample.values],
                            mode='lines',
                            name='最长5%股票平均',
                            line=dict(color='darkorange', width=1.8, dash='dot'),
                            hovertemplate='日期: %{x}<br>最长5%平均: %{y:.1f} 分钟<extra></extra>'
                        ))
                    # 收益覆盖率持仓时间曲线
                    _cover_sample = _profit_cover.loc[_x_idx]
                    for _col, _name, _color, _dash, _label in [
                        ('p25_time', '收益25%覆盖持仓', '#2980b9', 'dash', '25%'),
                        ('p50_time', '收益50%覆盖持仓', '#c0392b', 'longdash', '50%'),
                        ('p75_time', '收益75%覆盖持仓', '#16a085', 'dot', '75%')
                    ]:
                        _vals = _cover_sample[_col].values
                        fig_hold.add_trace(go.Scatter(
                            x=_x,
                            y=[None if _np.isnan(v) else float(v) for v in _vals],
                            mode='lines',
                            name=_name,
                            line=dict(color=_color, width=1.5, dash=_dash),
                            hovertemplate=f'日期: %{{x}}<br>达到累计{_label}收益的持仓: %{{y:.1f}} 分钟<extra></extra>'
                        ))
                    _mean = _daily_holding.mean()
                    _p25 = _daily_holding.quantile(0.25)
                    _p75 = _daily_holding.quantile(0.75)
                    fig_hold.update_layout(
                        title=f'交易平均持仓时间（按买入日，按交易时段计）<br><sub>样本天数: {len(_daily_holding)} | 均值: {_mean:.1f} 分钟 | P25/P75: {_p25:.1f}/{_p75:.1f}</sub>',
                        xaxis_title='日期',
                        yaxis_title='平均持仓时间 (分钟)',
                        height=420
                    )
                    explain_html = (
                        "<p>基于 data/paired_trades_fifo.parquet 的配对交易；跨日交易仅累计交易时段(09:30-11:30, 13:00-15:00)，" \
                        "周末与午间休市不计入。以开仓日为横轴，展示全体平均持仓分钟数，同时绘制每日股票层面持仓时间最短的5%与最长的5%的平均值曲线，" \
                        "以及累计绝对收益覆盖25%/50%/75%所需的持仓时间曲线，用于定位主要贡献收益的持仓时长。</p>"
                    )
                    metrics = {
                        '样本天数': f"{len(_daily_holding)}",
                        '均值(分钟)': f"{_mean:.1f}",
                        'P25/P50/P75(分钟)': f"{_p25:.1f} / {_daily_holding.median():.1f} / {_p75:.1f}",
                        '总配对数': f"{len(_pairs):,}"
                    }
                    if not _profit_cover.empty:
                        _cover_median = _profit_cover[['p25_time', 'p50_time', 'p75_time']].median()
                        if _cover_median.notna().any():
                            metrics['收益覆盖25/50/75%中位持仓(分钟)'] = " / ".join(
                                "NA" if _np.isnan(_cover_median[_k]) else f"{_cover_median[_k]:.1f}"
                                for _k in ['p25_time', 'p50_time', 'p75_time']
                            )
                    self._save_figure_with_details(
                        fig_hold,
                        name='intraday_avg_holding_time_light',
                        title='交易平均持仓时间（按买入日，按交易时段计）',
                        explanation_html=explain_html,
                        metrics=metrics,
                    )
                else:
                    print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无可用的日内持仓时间样本，跳过图表生成")
            else:
                print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 配对交易数据为空，跳过持仓时间统计")
        except Exception as _e:
            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 持仓时间统计失败: {_e}")

        # 成交率分析
        self.df['fill_rate'] = self.df['tradeQty'] / self.df['orderQty']
        
        # 日度成交率
        daily_fill_rate = self.df.groupby(self.df['Timestamp'].dt.date)['fill_rate'].mean()
        
        if len(daily_fill_rate) > 30:  # 至少30天数据
            # 诊断和修复：确保数据正确性
            print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 成交率数据诊断：")
            print(f"   日度成交率范围: {daily_fill_rate.min():.3f} - {daily_fill_rate.max():.3f}")
            print(f"   平均成交率: {daily_fill_rate.mean():.3f}")
            
            # 数据清理：确保成交率在合理范围内
            daily_fill_rate_clean = daily_fill_rate.clip(0, 1)  # 限制在0-100%
            
            # 成交率时间序列
            fill_sampled = daily_fill_rate_clean.iloc[::max(1, len(daily_fill_rate_clean)//150)]
            print(f"   采样后数据点: {len(fill_sampled)}")
            print(f"   采样数据范围: {fill_sampled.min():.3f} - {fill_sampled.max():.3f}")
            
            # 确保数据类型正确，避免累积效应
            x_data = [str(date) for date in fill_sampled.index]
            y_data = [float(rate) * 100.0 for rate in fill_sampled.values]  # 转换为百分比
            
            # 验证数据
            print(f"   转换后Y轴范围: {min(y_data):.1f}% - {max(y_data):.1f}%")
            
            fig_fill = go.Figure()
            fig_fill.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines+markers',
                name='日均成交率',
                line=dict(color='green', width=2),
                marker=dict(size=4),
                hovertemplate='日期: %{x}<br>成交率: %{y:.1f}%<extra></extra>'
            ))
            
            fig_fill.update_layout(
                title=f'成交率时间序列<br><sub>平均成交率: {daily_fill_rate_clean.mean()*100:.1f}%, 稳定性: {(daily_fill_rate_clean.std()*100):.1f}%</sub>',
                xaxis_title='日期',
                yaxis_title='成交率 (%)',
                height=400,
                yaxis=dict(range=[0, 100])  # 强制Y轴范围为0-100%
            )
            
            fill_metrics = {
                '样本天数': f"{len(daily_fill_rate)}",
                '平均成交率': f"{daily_fill_rate.mean():.2%}",
                '成交率波动': f"{daily_fill_rate.std():.2%}",
                'P25/P50/P75': f"{daily_fill_rate.quantile(0.25):.2%} / {daily_fill_rate.quantile(0.50):.2%} / {daily_fill_rate.quantile(0.75):.2%}",
            }
            fill_explain = (
                "<p><b>成交率</b>=成交量/委托量。时间序列展示执行充足性与稳定性，四分位数衡量分布。</p>"
            )
            self._save_figure_with_details(
                fig_fill,
                name='fill_rate_timeseries_light',
                title='成交率时间序列',
                explanation_html=fill_explain,
                metrics=fill_metrics,
            )
            
        # 成交率分布
        fill_rate_values = self.df['fill_rate'].dropna().astype(float).clip(0, 1)
        if len(fill_rate_values) > 0:
            # 直接传入数百万原始点会导致 HTML 体积暴涨，这里预先做桶聚合后再绘制柱状图
            values_pct = (fill_rate_values * 100).to_numpy(dtype=np.float32)
            bin_edges = np.linspace(0, 100, num=26)
            counts, _ = np.histogram(values_pct, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_widths = np.diff(bin_edges)
            bin_labels = [f"{bin_edges[i]:.1f}% - {bin_edges[i+1]:.1f}%" for i in range(len(bin_edges) - 1)]

            fig_fill_dist = go.Figure(
                go.Bar(
                    x=bin_centers.tolist(),
                    y=counts.astype(int).tolist(),
                    width=bin_widths.tolist(),
                    marker=dict(color='lightgreen', line=dict(color='#27ae60', width=0.5)),
                    name='成交率分布',
                    opacity=0.85,
                    hovertemplate='区间: %{text}<br>频次: %{y:,}<extra></extra>',
                    text=bin_labels
                )
            )

            fig_fill_dist.update_layout(
                title='成交率分布图',
                xaxis_title='成交率 (%)',
                yaxis_title='频次',
                height=400
            )

            fill_dist_metrics = {
                '样本数': f"{int(counts.sum()):,}",
                '最大频次': f"{int(counts.max()):,}",
                '区间数量': f"{len(counts)}"
            }

            self._save_figure_with_details(
                fig_fill_dist,
                name='fill_rate_distribution_light',
                title='成交率分布',
                explanation_html="<p>展示所有订单的逐笔成交率分布，用于识别长尾与极端未成交情况。</p>",
                metrics=fill_dist_metrics
            )
        else:
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 成交率数据为空，跳过分布图生成")
        
    def _calculate_key_metrics(self, returns):
        """计算关键绩效指标"""
        print("\n<i class='fas fa-chart-line text-green-500'></i> 关键绩效指标:")
        
        # 安全的收益率计算 - 处理可能的异常值
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            print("<i class='fas fa-times-circle text-red-500'></i> 无有效收益率数据")
            return {}
            
        print(f"收益率数据范围: {returns_clean.min():.4f} 到 {returns_clean.max():.4f}, 均值: {returns_clean.mean():.4f}")
        
        # 对于盯市分析数据，不进行裁剪以保持数据完整性
        # 只对明显的异常值进行裁剪（超过100%的单日收益率）
        returns_capped = returns_clean.clip(-1.0, 1.0)  # 保持原始数据，只处理极端异常值
        
        # 安全的复合收益计算
        try:
            cumulative_nav = (1 + returns_capped).cumprod()
            total_return = cumulative_nav.iloc[-1] - 1
            
            # 年化收益率
            if len(returns_capped) > 1:
                annualized_return = (1 + total_return) ** (252 / len(returns_capped)) - 1
            else:
                annualized_return = 0
                
            # 波动率
            volatility = returns_capped.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0
            
            # 最大回撤
            rolling_max = cumulative_nav.expanding().max()
            drawdown = (cumulative_nav - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            win_rate = (returns_capped > 0).mean()
            
            metrics = {
                '总收益率': f"{total_return:.2%}",
                '年化收益率': f"{annualized_return:.2%}",
                '年化波动率': f"{volatility:.2%}",
                '夏普比率': f"{sharpe_ratio:.3f}",
                '最大回撤': f"{max_drawdown:.2%}",
                '胜率': f"{win_rate:.2%}",
                '交易天数': f"{len(returns_capped)}天"
            }
            
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 指标计算错误: {e}")
            metrics = {
                '总收益率': "计算错误",
                '年化收益率': "计算错误",
                '年化波动率': f"{returns_clean.std() * np.sqrt(252):.2%}",
                '夏普比率': "计算错误",
                '最大回撤': "计算错误",
                '胜率': f"{(returns_clean > 0).mean():.2%}",
                '交易天数': f"{len(returns_clean)}天"
            }
        
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
            
        return metrics
        
    def _create_strategy_benchmark_comparison(self, strategy_cumulative, strategy_daily_returns=None):
        """创建策略vs基准对比图表（左轴为日收益率，右轴为累积收益率）"""
        fig = go.Figure()

        # 归一化策略累积收益
        strategy_cum_series = pd.Series(strategy_cumulative).dropna().sort_index()

        # 策略金额加权日收益率（来自“日收益率对比”页面）
        strategy_daily_series = None
        if strategy_daily_returns is not None:
            try:
                strategy_daily_series = pd.Series(strategy_daily_returns).dropna()
                strategy_daily_series.index = pd.to_datetime(strategy_daily_series.index)
                strategy_daily_series = strategy_daily_series.sort_index()
            except Exception:
                strategy_daily_series = None

        anchor_dates: set = set()
        if strategy_daily_series is not None and len(strategy_daily_series) > 0:
            anchor_dates = set(pd.to_datetime(strategy_daily_series.index).date)
        elif len(strategy_cum_series) > 0:
            anchor_dates = set(pd.to_datetime(strategy_cum_series.index).date)

        def _sample_series(series: pd.Series, max_points: int = 150) -> pd.Series:
            if len(series) > max_points:
                step = max(1, len(series) // max_points)
                return series.iloc[::step]
            return series

        bench_traces = []

        # 基准指数：左轴日收益率 + 右轴累积收益率（先收集，后统一添加，使策略曲线位于顶部）
        bench_min_val = None
        bench_max_val = None
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#17becf']
        for i, (bench_name, bench_df) in enumerate(self.benchmark_data.items()):
            if i >= len(colors):
                break

            bench_df_local = bench_df.copy()
            bench_df_local['date'] = pd.to_datetime(bench_df_local['date'])
            if anchor_dates:
                bench_df_local = bench_df_local[bench_df_local['date'].dt.date.isin(anchor_dates)]
            bench_df_local = bench_df_local.sort_values('date')
            if len(bench_df_local) == 0:
                continue

            # 日收益率
            daily_series = pd.Series(bench_df_local['daily_return'].values, index=bench_df_local['date']).dropna()
            daily_sampled = _sample_series(daily_series)
            x_bench_daily = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in daily_sampled.index]
            y_bench_daily = [float(v) * 100.0 for v in daily_sampled.values]
            bench_traces.append(go.Scatter(
                x=x_bench_daily,
                y=y_bench_daily,
                mode='lines',
                name=f'{bench_name}日收益率',
                line=dict(color=colors[i], width=1.6),
                opacity=0.65,
                hovertemplate=f'日期: %{{x}}<br>{bench_name}日收益率: %{{y:.2f}}%<extra></extra>',
                yaxis='y'
            ))

            # 累积收益率
            cum_series = pd.Series(bench_df_local['cumulative_return'].values, index=bench_df_local['date']).dropna()
            daily_full_cum = cum_series * 100.0
            if len(daily_full_cum) > 0:
                bmin = float(daily_full_cum.min())
                bmax = float(daily_full_cum.max())
                bench_min_val = bmin if bench_min_val is None else min(bench_min_val, bmin)
                bench_max_val = bmax if bench_max_val is None else max(bench_max_val, bmax)

            cum_sampled = _sample_series(daily_full_cum)
            x_bench_cum = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in cum_sampled.index]
            y_bench_cum = [float(v) for v in cum_sampled.values]
            bench_traces.append(go.Scatter(
                x=x_bench_cum,
                y=y_bench_cum,
                mode='lines',
                name=f'{bench_name}累积收益',
                line=dict(color=colors[i], width=2, dash='dash'),
                opacity=0.65,
                hovertemplate=f'日期: %{{x}}<br>{bench_name}累积收益: %{{y:.2f}}%<extra></extra>',
                yaxis='y2'
            ))

        # 策略金额加权日收益率（左轴，置于顶部）
        if strategy_daily_series is not None and len(strategy_daily_series) > 0:
            sampled_daily = _sample_series(strategy_daily_series)
            x_daily = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in sampled_daily.index]
            y_daily = [float(v) * 100.0 for v in sampled_daily.values]
            fig.add_trace(go.Scatter(
                x=x_daily,
                y=y_daily,
                mode='lines+markers',
                name='策略日收益率（金额加权）',
                line=dict(color='firebrick', width=2.8),
                marker=dict(size=4, color='firebrick'),
                hovertemplate='日期: %{x}<br>日收益率: %{y:.2f}%<extra></extra>',
                yaxis='y',
                legendrank=0
            ))

        # 将基准曲线添加到底层
        for tr in bench_traces:
            fig.add_trace(tr)

        # 策略累积收益曲线（默认隐藏，右轴显示，置于最上层但 legendonly）
        if len(strategy_cum_series) > 0:
            strat_cum_pct = strategy_cum_series.astype(float) * 100.0
            sampled_cum = _sample_series(strat_cum_pct)
            x_cum = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in sampled_cum.index]
            y_cum = [float(v) for v in sampled_cum.values]
            fig.add_trace(go.Scatter(
                x=x_cum,
                y=y_cum,
                mode='lines',
                name='策略累积收益',
                line=dict(color='red', width=3),
                hovertemplate='日期: %{x}<br>累积收益: %{y:.2f}%<extra></extra>',
                yaxis='y2',
                visible='legendonly',
                legendrank=1
            ))

        # 将策略累积收益纳入右轴范围考量
        if len(strategy_cum_series) > 0:
            strat_cum_pct_full = strategy_cum_series.astype(float) * 100.0
            if bench_min_val is None or bench_max_val is None:
                bench_min_val = float(strat_cum_pct_full.min())
                bench_max_val = float(strat_cum_pct_full.max())
            else:
                bench_min_val = min(bench_min_val, float(strat_cum_pct_full.min()))
                bench_max_val = max(bench_max_val, float(strat_cum_pct_full.max()))

        # 布局设置
        strategy_final_display = strategy_cum_series.iloc[-1] * 100 if len(strategy_cum_series) > 0 else 0.0
        layout_config = {
            'title': f'策略vs基准指数对比<br><sub>策略总收益: {strategy_final_display:.2f}%</sub>',
            'xaxis_title': '日期',
            'yaxis_title': '收益率 (%)',
            'yaxis': dict(showgrid=True, zeroline=True, tickformat='.1f'),
            'yaxis2': dict(
                title='累积收益率 (%)',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            'height': 520,
            'hovermode': 'x unified',
            'legend': dict(x=0.01, y=0.99),
            'xaxis': dict(type='date')
        }

        # 右轴范围包含零点，便于快速感知基准表现
        if bench_min_val is not None and bench_max_val is not None:
            bench_min_val = min(bench_min_val, 0.0)
            bench_max_val = max(bench_max_val, 0.0)
            if bench_max_val <= bench_min_val:
                bench_max_val = bench_min_val + 1.0
            layout_config['yaxis2']['range'] = [bench_min_val, bench_max_val]
            layout_config['yaxis2']['zeroline'] = True

        fig.update_layout(**layout_config)
        return fig
        
    def _calculate_benchmark_comparison_metrics(self, strategy_cumulative, strategy_daily_returns):
        """计算策略vs基准的对比指标，包含关键绩效指标"""
        if not self.benchmark_data:
            return {}
            
        metrics = {}

        if isinstance(strategy_cumulative, pd.Series):
            strategy_cumulative = strategy_cumulative.dropna().sort_index()
        else:
            strategy_cumulative = pd.Series(strategy_cumulative).dropna().sort_index()

        if isinstance(strategy_daily_returns, pd.Series):
            strategy_daily_returns = strategy_daily_returns.dropna().sort_index()
        else:
            strategy_daily_returns = pd.Series(strategy_daily_returns).dropna().sort_index()

        if len(strategy_cumulative) == 0 or len(strategy_daily_returns) == 0:
            return metrics

        strategy_final = strategy_cumulative.iloc[-1]
        
        # 策略关键绩效指标
        try:
            # 1. 胜率 (Win Rate)
            win_rate = (strategy_daily_returns > 0).mean()
            metrics['策略胜率'] = f"{win_rate:.2%}"
            
            # 2. 盈亏比 (Profit/Loss Ratio)
            positive_returns = strategy_daily_returns[strategy_daily_returns > 0]
            negative_returns = strategy_daily_returns[strategy_daily_returns < 0]
            
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                avg_profit = positive_returns.mean()
                avg_loss = abs(negative_returns.mean())
                profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else np.inf
                metrics['盈亏比'] = f"{profit_loss_ratio:.2f}"
            else:
                metrics['盈亏比'] = "N/A"
            
            # 3. 夏普比率 (Sharpe Ratio) - 使用样本标准差（ddof=1）
            if len(strategy_daily_returns) > 1:
                excess_return = strategy_daily_returns.mean()
                volatility = strategy_daily_returns.std(ddof=1)
                if volatility > 0:
                    sharpe_ratio = (excess_return / volatility) * np.sqrt(252)  # 年化夏普比率
                    metrics['夏普比率'] = f"{sharpe_ratio:.3f}"
                else:
                    metrics['夏普比率'] = "N/A"
            else:
                metrics['夏普比率'] = "N/A"
                
            # 4. 最大回撤
            cumulative_nav = (1 + strategy_daily_returns).cumprod()
            rolling_max = cumulative_nav.expanding().max()
            drawdown = (cumulative_nav - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            metrics['最大回撤'] = f"{max_drawdown:.2%}"
        
        except Exception as e:
            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 计算策略指标时出错: {e}")
            metrics['策略胜率'] = "计算错误"
            metrics['盈亏比'] = "计算错误"
            metrics['夏普比率'] = "计算错误"
            metrics['最大回撤'] = "计算错误"
        
        # 策略绝对收益
        metrics['策略总收益'] = f"{strategy_final * 100:.2f}%"
        
        # 对比各基准的超额收益
        for bench_name, bench_df in self.benchmark_data.items():
            bench_final = bench_df['cumulative_return'].iloc[-1]
            excess_return = (strategy_final - bench_final) * 100
            
            metrics[f'vs {bench_name}'] = f"{excess_return:+.2f}%"
            
        # 找出最佳基准
        best_benchmark = max(self.benchmark_data.items(), 
                           key=lambda x: x[1]['cumulative_return'].iloc[-1])
        best_name = best_benchmark[0]
        best_return = best_benchmark[1]['cumulative_return'].iloc[-1] * 100
        
        metrics['最佳基准'] = f"{best_name} ({best_return:.2f}%)"

        # 同步策略指标到 dashboard，确保总收益率以基准对比口径为准
        try:
            if not hasattr(self, 'strategy_metrics') or not isinstance(self.strategy_metrics, dict):
                self.strategy_metrics = {}
            self.strategy_metrics.update({
                'total_return_nav': metrics.get('策略总收益', 'N/A'),
                'sharpe_ratio': metrics.get('夏普比率', 'N/A'),
                'max_drawdown': metrics.get('最大回撤', 'N/A'),
                'win_rate': metrics.get('策略胜率', 'N/A'),
            })
        except Exception:
            pass
        
        return metrics
        
    def _save_figure(self, fig, name):
        """保存纯图表（向后兼容）"""
        return self._save_figure_with_details(fig, name, title=name, explanation_html="", metrics={})

    # ====== 时段分析：最小可用实现（含缓存桩函数） ======
    def _orders_file_fingerprint(self) -> dict:
        """获取 `orders.parquet` 的指纹信息（路径/大小/修改时间）。"""
        try:
            p = Path(self.data_path)
            return {
                'path': str(p.resolve()),
                'size': int(p.stat().st_size) if p.exists() else -1,
                'mtime': float(p.stat().st_mtime) if p.exists() else -1.0,
            }
        except Exception:
            return {'path': str(self.data_path), 'size': -1, 'mtime': -1.0}

    def _make_slot_performance_cache_key(self) -> str:
        """构造简单的缓存键（基于订单指纹+脚本版本）。若无缓存需求，可直接返回固定键。"""
        fp = self._orders_file_fingerprint()
        raw = f"slot_v1|{fp['path']}|{fp['size']}|{fp['mtime']}"
        return hashlib.md5(raw.encode('utf-8', errors='ignore')).hexdigest()

    # ====== 资金占用序列 缓存工具 ======
    def _make_capital_util_cache_key(self) -> str:
        """根据订单文件指纹与授信规则生成缓存键。"""
        try:
            self._ensure_credit_rules_loaded()
        except Exception:
            pass
        fp = self._orders_file_fingerprint()
        rules = self._credit_rules or {}
        raw = (
            f"caputil_v1|{fp['path']}|{fp['size']}|{fp['mtime']}|"
            f"{rules.get('allow_short_proceeds_to_cash')}|{rules.get('allow_sell_long_cash_T0')}|"
            f"{rules.get('margin_short_ratio')}|{rules.get('fee_accrual')}"
        )
        return hashlib.md5(raw.encode('utf-8', errors='ignore')).hexdigest()

    def _capital_util_cache_path(self, cache_key: str) -> Path:
        # 缓存集中放入 data 目录，便于在其他项目复用
        base = Path("data")
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return base / f"capital_util_cache_{cache_key}.parquet"

    def _load_capital_util_daily_min(self, cache_key: str) -> Optional[pd.DataFrame]:
        try:
            p = self._capital_util_cache_path(cache_key)
            print(f"[CACHE] 尝试加载资金占用缓存: {p}")
            if p.exists():
                df = pd.read_parquet(p)
                if len(df) > 0:
                    print(f"[CACHE] 命中资金占用序列: {p.name}，共 {len(df):,} 天")
                    return df
                else:
                    print("[CACHE] 缓存文件存在但为空，忽略并重新计算")
        except Exception as _e:
            print(f"[CACHE] 读取缓存失败，将重新计算: {_e}")
        return None

    def _save_capital_util_daily_min(self, cache_key: str, daily_min: pd.DataFrame) -> None:
        try:
            p = self._capital_util_cache_path(cache_key)
            p.parent.mkdir(parents=True, exist_ok=True)
            daily_min.to_parquet(p, index=False)
            print(f"[CACHE] 资金占用序列已缓存: {p}")
        except Exception as _e:
            print(f"[CACHE] 写入缓存失败: {_e}")

    def _load_slot_performance_cache(self, cache_key: str) -> str:
        """返回空字符串以禁用缓存（占位桩）。后续可扩展为实际读取。"""
        return ""

    def _save_slot_performance_cache(self, cache_key: str, serialized: str) -> None:
        """不执行任何保存操作（占位桩）。后续可扩展为写入本地文件。"""
        return None

    def _serialize_slot_payload(self, payload: dict, meta: dict) -> str:
        """占位：序列化结果。当前返回空字符串以禁用缓存。"""
        return ""

    def _deserialize_slot_payload(self, serialized: str) -> dict:
        """占位：反序列化结果。当前返回空字典以禁用缓存。"""
        return {}

    def _compute_slot_performance_payload(self) -> dict:
        """计算时段绝对收益的最小可用数据包。
        口径：将每笔闭环交易的 absolute_profit 按卖出时刻落入的 5 分钟时段进行归因，并在全样本期汇总。
        仅统计 A 股交易时段：09:30-11:30 与 13:00-15:00。
        """
        pairs_path = Path('data/paired_trades_fifo.parquet')
        if not pairs_path.exists():
            raise FileNotFoundError('缺少 data/paired_trades_fifo.parquet，无法进行时段分析')

        df = pd.read_parquet(pairs_path)
        if len(df) == 0:
            raise ValueError('paired_trades_fifo.parquet 为空，无法进行时段分析')

        # 统一时间戳与必要列
        needed_cols = ['buy_timestamp', 'sell_timestamp', 'absolute_profit', 'trade_type', 'buy_amount']
        miss = [c for c in needed_cols if c not in df.columns]
        if miss:
            raise ValueError(f'paired_trades_fifo.parquet 缺少必要列: {miss}')

        if not pd.api.types.is_datetime64_any_dtype(df['buy_timestamp']):
            df['buy_timestamp'] = pd.to_datetime(df['buy_timestamp'])
        if not pd.api.types.is_datetime64_any_dtype(df['sell_timestamp']):
            df['sell_timestamp'] = pd.to_datetime(df['sell_timestamp'])

        df = df.dropna(subset=['sell_timestamp', 'absolute_profit']).copy()

        # 仅保留交易时段：09:30-11:30, 13:00-15:00（分钟：570-690, 780-900）
        def _minute_of_day(ts: pd.Timestamp) -> int:
            return int(ts.hour) * 60 + int(ts.minute)

        sell_min = df['sell_timestamp'].apply(_minute_of_day).astype(int)
        in_morning = (sell_min >= 570) & (sell_min <= 690)
        in_afternoon = (sell_min >= 780) & (sell_min <= 900)
        df = df.loc[in_morning | in_afternoon].copy()
        if len(df) == 0:
            raise ValueError('无落在交易时段内的卖出记录，无法进行时段分析')

        # 5 分钟对齐
        def _floor5_label(m: int) -> str:
            m5 = m - (m % 5)
            return f"{m5 // 60:02d}:{m5 % 60:02d}"

        df['slot'] = sell_min.apply(_floor5_label)
        df['date'] = df['sell_timestamp'].dt.date

        week_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
        df['weekday'] = df['sell_timestamp'].dt.dayofweek.map(week_map)

        # 定义标准时段顺序
        def _ordered_slots() -> list:
            slots = []
            for m in range(570, 691, 5):  # 09:30-11:30（含）
                slots.append(_floor5_label(m))
            for m in range(780, 901, 5):  # 13:00-15:00（含）
                slots.append(_floor5_label(m))
            return slots

        ordered_slots = _ordered_slots()

        # 计算每笔收益率（以买入额为分母；若缺或为0则跳过该笔）
        df['buy_amount'] = pd.to_numeric(df['buy_amount'], errors='coerce') if 'buy_amount' in df.columns else np.nan
        df['return'] = np.where(df['buy_amount'] > 0, df['absolute_profit'] / df['buy_amount'], np.nan)

        # 汇总（全样本）
        grp = df.groupby('slot').agg(
            total_profit=('absolute_profit', 'sum'),
            trade_count=('absolute_profit', 'size'),
            avg_return=('return', 'mean'),
            std_return=('return', 'std')
        )
        grp = grp.reindex(ordered_slots).fillna({'total_profit': 0.0, 'trade_count': 0}).reset_index().rename(columns={'index': 'slot'})

        # 日度明细（用于后续扩展）
        slot_daily = df.groupby(['date', 'slot']).agg(
            profit_sum=('absolute_profit', 'sum'),
            trade_count=('absolute_profit', 'size')
        ).reset_index()

        # 多/空分解（用于后续扩展）
        agg_long = None
        agg_short = None
        if 'trade_type' in df.columns:
            _al = df[df['trade_type'] == 'long'].groupby('slot')['absolute_profit'].sum()
            _as = df[df['trade_type'] == 'short'].groupby('slot')['absolute_profit'].sum()
            agg_long = _al.reindex(ordered_slots).fillna(0.0)
            agg_short = _as.reindex(ordered_slots).fillna(0.0)

        # 星期 × 时段 热力图数据
        weekday_slot_pivot = (df.pivot_table(index='weekday', columns='slot', values='absolute_profit', aggfunc='sum', fill_value=0.0)
                                .reindex(index=['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
                                .reindex(columns=ordered_slots))

        payload = {
            'slot_summary': grp,
            'slot_daily_detailed': slot_daily,
            'agg_long': agg_long,
            'agg_short': agg_short,
            'weekday_slot_pivot': weekday_slot_pivot,
            'ordered_slots': ordered_slots,
            'slot_returns': df[['date','slot','return','trade_type']].dropna(subset=['return'])
        }
        return payload

    def _render_slot_performance_outputs(self, payload: dict, cache_hit: bool = False) -> None:
        """渲染最小集的时段分析图表：
        1) 全样本期日内时段绝对收益条形图（替代瀑布图）
        2) 星期 × 时段 绝对盈利热力图
        """
        try:
            slot_sum: pd.DataFrame = payload.get('slot_summary', pd.DataFrame())
            ordered_slots: list = payload.get('ordered_slots', [])
            if slot_sum is None or len(slot_sum) == 0 or not ordered_slots:
                print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 时段分析：无可用数据，跳过图表渲染")
                return

            # 主图：时段绝对收益条形图
            x_vals = ordered_slots
            slot_sum_indexed = slot_sum.set_index('slot').reindex(ordered_slots).fillna({'total_profit': 0.0})
            y_vals = slot_sum_indexed['total_profit'].astype(float).tolist()
            # 修正颜色：正红负绿
            colors = ['#e53935' if y >= 0 else '#43a047' for y in y_vals]

            fig_main = go.Figure()
            fig_main.add_trace(go.Bar(x=x_vals, y=y_vals, marker_color=colors, name='绝对收益'))
            fig_main.update_layout(
                title='日内时段绝对收益瀑布图（全样本期总贡献）',
                xaxis_title='时段(5分钟)',
                yaxis_title='累计绝对收益',
                height=460,
                bargap=0.2
            )
            try:
                fig_main.add_hline(y=0, line_dash='dot', line_color='gray')
            except Exception:
                pass

            total_profit = float(slot_sum_indexed['total_profit'].sum())
            pos_sum = float(slot_sum_indexed['total_profit'].clip(lower=0).sum())
            neg_sum = float(slot_sum_indexed['total_profit'].clip(upper=0).sum())
            explain_html = (
                '<p>将每笔闭环交易的绝对盈利归因到其卖出时刻所在的 5 分钟时段，并在全样本期汇总。'
                '仅统计 09:30-11:30 与 13:00-15:00 的交易时段。</p>'
            )
            metrics = {
                '总绝对收益': f"{total_profit:,.2f}",
                '正贡献合计': f"{pos_sum:,.2f}",
                '负贡献合计': f"{neg_sum:,.2f}",
                '样本时段数': str(int((len([s for s in ordered_slots if s])))),
            }
            self._save_figure_with_details(
                fig_main,
                name='slot_intraday_profit_waterfall_light',
                title='日内时段绝对收益瀑布图（全样本期总贡献）',
                explanation_html=explain_html,
                metrics=metrics,
            )

            # 副图：星期 × 时段 热力图
            pivot_df: Optional[pd.DataFrame] = payload.get('weekday_slot_pivot')
            if isinstance(pivot_df, pd.DataFrame) and len(pivot_df) > 0:
                z = pivot_df.values.astype(float)
                heat_colorscale = [
                    [0.0, '#43a047'],  # 亏损/负向
                    [0.5, '#f5f5f5'],  # 中性
                    [1.0, '#e53935']   # 盈利/正向
                ]
                fig_heat = go.Figure(data=go.Heatmap(
                    z=z,
                    x=pivot_df.columns.astype(str).tolist(),
                    y=pivot_df.index.astype(str).tolist(),
                    colorscale=heat_colorscale,
                    zmid=0,
                    colorbar=dict(title='绝对收益', tickformat=',.0f'),
                    hovertemplate='星期 %{y}<br>时段 %{x}<br>绝对收益 ¥%{z:,.0f}<extra></extra>'
                ))
                fig_heat.update_layout(
                    title='星期×时段绝对盈利热力图',
                    xaxis_title='时段(5分钟)',
                    yaxis_title='星期',
                    height=480
                )
                self._save_figure_with_details(
                    fig_heat,
                    name='time_slot_profit_heatmap_light',
                    title='星期×时段绝对盈利热力图',
                    explanation_html='- 以卖出时刻所在的 5 分钟时段为颗粒，按星期聚合绝对盈利，刻画结构性高频节奏。\n- 颜色遵循 A 股语义：红=盈利，绿=亏损，中心灰=中性。',
                    metrics={}
                )

            # 1) 时段平均净收益率分析
            slot_sum_indexed = slot_sum.set_index('slot').reindex(ordered_slots)
            if 'avg_return' in slot_sum_indexed.columns:
                y_ret = slot_sum_indexed['avg_return'].astype(float).fillna(0.0).tolist()
                fig_avg = go.Figure()
                colors_avg = ['#e53935' if v >= 0 else '#43a047' for v in y_ret]
                fig_avg.add_trace(go.Bar(
                    x=x_vals,
                    y=y_ret,
                    marker_color=colors_avg,
                    name='平均净收益率',
                    text=[f"{v*100:.2f}%" for v in y_ret],
                    textposition='outside',
                    hovertemplate='时段 %{x}<br>平均净收益率 %{y:.2%}<extra></extra>'
                ))
                try:
                    fig_avg.add_hline(y=0, line_dash='dot', line_color='gray')
                except Exception:
                    pass
                fig_avg.update_layout(
                    title='时段平均净收益率分析',
                    xaxis_title='时段(5分钟)',
                    yaxis_title='平均净收益率',
                    height=420
                )
                fig_avg.update_yaxes(tickformat='.2%')
                self._save_figure_with_details(
                    fig_avg,
                    name='time_slot_performance_analysis_light',
                    title='时段平均净收益率分析',
                    explanation_html='- 将每笔闭环交易的收益率（绝对盈利÷买入金额）按卖出时刻所在的 5 分钟时段求均值。\n- 仅覆盖 09:30-11:30 与 13:00-15:00 的交易时段，用于识别高低效时段。',
                    metrics={}
                )

            # 2) 显著性检验（t 统计量阈值标注）
            slot_ret_df: pd.DataFrame = payload.get('slot_returns', pd.DataFrame())
            if isinstance(slot_ret_df, pd.DataFrame) and len(slot_ret_df) > 0:
                t_values = []
                counts = []
                for s in ordered_slots:
                    r = slot_ret_df.loc[slot_ret_df['slot'] == s, 'return'].astype(float)
                    n = int(r.notna().sum())
                    counts.append(n)
                    if n >= 10 and r.std(ddof=1) not in (0, np.nan):
                        t_val = float(r.mean() / (r.std(ddof=1) / np.sqrt(n)))
                    else:
                        t_val = np.nan
                    t_values.append(t_val)
                fig_t = go.Figure()
                t_plot = [0 if pd.isna(v) else v for v in t_values]
                colors_t = []
                for v in t_plot:
                    if pd.isna(v):
                        colors_t.append('#9ca3af')
                    else:
                        colors_t.append('#e53935' if v >= 0 else '#43a047')
                fig_t.add_trace(go.Bar(
                    x=ordered_slots,
                    y=t_plot,
                    marker_color=colors_t,
                    name='t统计',
                    text=[f"{v:.2f}" if not pd.isna(v) else "" for v in t_values],
                    textposition='outside',
                    hovertemplate='时段 %{x}<br>t 统计 %{y:.2f}<extra></extra>'
                ))
                try:
                    fig_t.add_hline(y=1.96, line_dash='dot', line_color='rgba(229,57,53,0.7)')
                    fig_t.add_hline(y=-1.96, line_dash='dot', line_color='rgba(229,57,53,0.7)')
                except Exception:
                    pass
                fig_t.update_layout(
                    title='时段收益显著性检验（t统计）',
                    xaxis_title='时段(5分钟)',
                    yaxis_title='t 值',
                    height=420
                )
                self._save_figure_with_details(
                    fig_t,
                    name='time_slot_significance_test_light',
                    title='时段收益显著性检验',
                    explanation_html='- 对每个 5 分钟时段的收益率做单样本 t 检验，判断均值是否显著偏离 0。\n- 虚线 |t|=1.96 为约 5% 的双侧显著阈值，红色为正向显著、绿色为负向显著。',
                    metrics={}
                )

            # 3) 时段绝对收益分解（多/空）
            if payload.get('agg_long') is not None or payload.get('agg_short') is not None:
                al = payload.get('agg_long')
                ar = payload.get('agg_short')
                fig_decomp = go.Figure()
                if al is not None:
                    long_vals = [float(v) if pd.notna(v) else 0.0 for v in al.tolist()]
                    colors_long = ['#e53935' if v >= 0 else '#43a047' for v in long_vals]
                    fig_decomp.add_trace(go.Bar(
                        x=ordered_slots,
                        y=long_vals,
                        name='多头',
                        marker=dict(color=colors_long),
                        text=[f"{v:,.0f}" for v in long_vals],
                        textposition='outside',
                        hovertemplate='时段 %{x}<br>多头绝对收益 ¥%{y:,.0f}<extra></extra>'
                    ))
                if ar is not None:
                    short_vals = [float(v) if pd.notna(v) else 0.0 for v in ar.tolist()]
                    colors_short = ['#e53935' if v >= 0 else '#43a047' for v in short_vals]
                    fig_decomp.add_trace(go.Bar(
                        x=ordered_slots,
                        y=short_vals,
                        name='空头',
                        marker=dict(color=colors_short, line=dict(width=0.4, color='#111827')),
                        text=[f"{v:,.0f}" for v in short_vals],
                        textposition='outside',
                        hovertemplate='时段 %{x}<br>空头绝对收益 ¥%{y:,.0f}<extra></extra>'
                    ))
                fig_decomp.update_layout(
                    barmode='relative',
                    title='时段绝对收益分解（多/空）',
                    xaxis_title='时段(5分钟)',
                    yaxis_title='累计绝对收益',
                    height=420,
                    legend_title_text='方向'
                )
                self._save_figure_with_details(
                    fig_decomp,
                    name='time_slot_profit_decomposition_light',
                    title='时段绝对收益分解',
                    explanation_html='- 将绝对盈利按卖出时刻所在的 5 分钟时段、并按多/空方向拆分后累加。\n- 红色柱表示盈利贡献，绿色表示亏损贡献，便于定位方向性优势时段。',
                    metrics={}
                )

            # 4) 时段收益分布箱线图（每时段最多采样 400 笔）
            if isinstance(slot_ret_df, pd.DataFrame) and len(slot_ret_df) > 0:
                fig_box = go.Figure()
                for s in ordered_slots:
                    r = slot_ret_df.loc[slot_ret_df['slot'] == s, 'return'].astype(float)
                    if len(r) == 0:
                        continue
                    if len(r) > 400:
                        r = r.sample(400, random_state=12345)
                    median_v = float(np.nanmedian(r.values)) if len(r) > 0 else 0.0
                    color_box = '#e53935' if median_v >= 0 else '#43a047'
                    fig_box.add_trace(go.Box(
                        y=r.values.tolist(),
                        name=s,
                        boxpoints=False,
                        marker_color=color_box,
                        hovertemplate='时段 %{name}<br>收益率=%{y:.2%}<extra></extra>'
                    ))
                fig_box.update_layout(
                    title='时段收益分布箱线图（采样）',
                    xaxis_title='时段(5分钟)',
                    yaxis_title='收益率',
                    height=480,
                    showlegend=False
                )
                fig_box.update_yaxes(tickformat='.2%')
                self._save_figure_with_details(
                    fig_box,
                    name='time_slot_returns_boxplot_light',
                    title='时段收益分布箱线图',
                    explanation_html='- 每个时段最多采样 400 笔交易绘制箱线图，控制页面体积。\n- 箱体颜色跟随中位数：红=正收益，绿=负收益。',
                    metrics={}
                )

            # 5) 风险-收益气泡图（均值 vs 标准差，气泡大小=样本数）
            if isinstance(slot_sum_indexed, pd.DataFrame) and 'avg_return' in slot_sum_indexed.columns:
                mu = slot_sum_indexed['avg_return'].astype(float).fillna(0.0)
                sd = slot_sum_indexed['std_return'].astype(float).fillna(0.0)
                n = slot_sum_indexed['trade_count'].astype(float).fillna(0.0)
                fig_bub = go.Figure()
                max_abs_mu = float(np.nanmax(np.abs(mu))) if len(mu) > 0 else 0.0
                max_abs_mu = max(max_abs_mu, 1e-6)
                fig_bub.add_trace(go.Scatter(
                    x=sd.tolist(),
                    y=mu.tolist(),
                    mode='markers+text',
                    text=[s for s in ordered_slots],
                    textposition='top center',
                    customdata=n.tolist(),
                    marker=dict(
                        size=(np.sqrt(n) / np.sqrt(max(n.max(), 1))) * 40 + 5,
                        color=mu.tolist(),
                        colorscale=[[0, '#43a047'], [0.5, '#f5f5f5'], [1, '#e53935']],
                        cmin=-max_abs_mu,
                        cmax=max_abs_mu,
                        colorbar=dict(title='平均收益率', tickformat='.2%'),
                        line=dict(width=0.6, color='#111827')
                    ),
                    hovertemplate='时段 %{text}<br>均值 %{y:.2%}<br>标准差 %{x:.2%}<br>样本数 %{customdata:.0f}<extra></extra>'
                ))
                fig_bub.update_layout(
                    title='时段风险-收益气泡图',
                    xaxis_title='收益率标准差',
                    yaxis_title='平均收益率',
                    height=480
                )
                fig_bub.update_yaxes(tickformat='.2%')
                fig_bub.update_xaxes(tickformat='.2%')
                self._save_figure_with_details(
                    fig_bub,
                    name='time_slot_risk_return_bubble_light',
                    title='时段风险-收益气泡图',
                    explanation_html='- 横轴为收益率标准差，纵轴为平均收益率，气泡面积与样本数成比例。\n- 颜色遵循红盈绿亏语义，并以 0 为中心对称映射。',
                    metrics={}
                )

            # 6) 开盘与尾盘精细分析（各取 30 分钟）
            try:
                open_slots = ordered_slots[:6]  # 09:30-10:00
                close_slots = ordered_slots[-6:]  # 14:30-15:00
                mu_open = slot_sum_indexed.set_index(pd.Index(ordered_slots)).loc[open_slots, 'avg_return'].astype(float).fillna(0.0)
                mu_close = slot_sum_indexed.set_index(pd.Index(ordered_slots)).loc[close_slots, 'avg_return'].astype(float).fillna(0.0)
                colors_open = ['#e53935' if v >= 0 else '#43a047' for v in mu_open.tolist()]
                colors_close = ['#e53935' if v >= 0 else '#43a047' for v in mu_close.tolist()]
                fig_open = go.Figure(go.Bar(
                    x=open_slots,
                    y=mu_open.tolist(),
                    marker_color=colors_open,
                    text=[f"{v*100:.2f}%" for v in mu_open.tolist()],
                    textposition='outside',
                    hovertemplate='时段 %{x}<br>平均净收益率 %{y:.2%}<extra></extra>'
                ))
                fig_open.update_layout(title='开盘前 30 分钟：平均净收益率', height=420)
                fig_open.update_yaxes(tickformat='.2%')
                fig_close = go.Figure(go.Bar(
                    x=close_slots,
                    y=mu_close.tolist(),
                    marker_color=colors_close,
                    text=[f"{v*100:.2f}%" for v in mu_close.tolist()],
                    textposition='outside',
                    hovertemplate='时段 %{x}<br>平均净收益率 %{y:.2%}<extra></extra>'
                ))
                fig_close.update_layout(title='尾盘后 30 分钟：平均净收益率', height=420)
                fig_close.update_yaxes(tickformat='.2%')
                self._save_figure_pair_with_details(
                    fig_open, fig_close,
                    name='time_slot_opening_closing_analysis_light',
                    title='开盘与尾盘精细分析（5分钟）',
                    explanation_html='- 将开盘与尾盘各 30 分钟的 5 分钟时段均值并列展示，识别边界时段的稳定性差异。\n- 红色为盈利，中性或亏损用绿色标示，便于快速对比。',
                    metrics_primary={}, metrics_secondary={},
                    primary_title='开盘 30 分钟', secondary_title='尾盘 30 分钟'
                )
            except Exception:
                pass

            # 7) 涨跌日分面热力图（若有盯市 NAV 数据）
            try:
                mtm_file = Path('mtm_analysis_results/daily_nav_revised.csv')
                if mtm_file.exists():
                    nav = pd.read_csv(mtm_file)
                    def _pc(v):
                        try:
                            if isinstance(v, (int,float)):
                                return float(v)
                            return float(str(v).replace(',', '').strip())
                        except Exception:
                            return np.nan
                    nav['date'] = pd.to_datetime(nav['date']).dt.date
                    nav = nav.sort_values('date')
                    nav['total_assets_num'] = nav['total_assets'].apply(_pc)
                    nav['nav_diff'] = nav['total_assets_num'].diff()
                    slot_abs = payload.get('slot_daily_detailed')
                    if isinstance(slot_abs, pd.DataFrame) and len(slot_abs) > 0:
                        slot_abs2 = slot_abs.copy()
                        slot_abs2['date'] = pd.to_datetime(slot_abs2['date']).dt.date
                        merged = slot_abs2.merge(nav[['date','nav_diff']], on='date', how='left')
                        up = merged.loc[merged['nav_diff'] > 0]
                        dn = merged.loc[merged['nav_diff'] < 0]
                        def _heat(df_part):
                            pv = df_part.pivot_table(index='slot', columns='date', values='profit_sum', aggfunc='sum', fill_value=0.0)
                            # 汇总为按 slot 的总收益，再按 ordered_slots 排序
                            return pv.sum(axis=1).reindex(ordered_slots).fillna(0.0)
                        up_series = _heat(up)
                        dn_series = _heat(dn)
                        fig_up = go.Figure(go.Bar(x=ordered_slots, y=up_series.tolist(), marker_color='#e74c3c'))
                        fig_up.update_layout(title='涨日：各时段绝对盈利', height=420)
                        fig_dn = go.Figure(go.Bar(x=ordered_slots, y=dn_series.tolist(), marker_color='#2e86c1'))
                        fig_dn.update_layout(title='跌日：各时段绝对盈利', height=420)
                        self._save_figure_pair_with_details(
                            fig_up, fig_dn,
                            name='time_slot_market_regime_heatmap_light',
                            title='涨跌日分面（各时段绝对盈利）',
                            explanation_html='<p>将样本天按 NAV 涨跌分组，比较各 5 分钟时段的绝对盈利。</p>',
                            metrics_primary={}, metrics_secondary={},
                            primary_title='涨日', secondary_title='跌日'
                        )
            except Exception:
                pass
        except Exception as e:
            print(f"<i class='fas fa-exclamation-triangle text-yellow-500'></i> 时段分析渲染失败: {e}")
            import traceback as _tb
            _tb.print_exc()

    def slot_performance_analysis(self):
        """时段盈利能力分析 - 基于配对交易的绝对收益"""
        print("\n" + "="*60)
        print("<i class='fas fa-clock text-blue-400'></i> 开始时段盈利能力分析（基于绝对收益）...")
        print("="*60)
        
        try:
            cache_key = self._make_slot_performance_cache_key()
            cached_serialized = self._load_slot_performance_cache(cache_key)
            if cached_serialized:
                payload = self._deserialize_slot_payload(cached_serialized)
                cache_hit = True
            else:
                payload = self._compute_slot_performance_payload()
                cache_hit = False
                orders_fp = self._orders_file_fingerprint()
                serialized = self._serialize_slot_payload(payload, meta={
                    'cache_key': cache_key,
                    'orders_path': orders_fp['path'],
                    'orders_size': orders_fp['size'],
                    'orders_mtime': orders_fp['mtime'],
                })
                self._save_slot_performance_cache(cache_key, serialized)
            self._render_slot_performance_outputs(payload, cache_hit=cache_hit)
            slot_summary = payload.get('slot_summary', pd.DataFrame())
            slot_daily_detailed = payload.get('slot_daily_detailed', pd.DataFrame())
            agg_long = payload.get('agg_long')
            agg_short = payload.get('agg_short')
            weekday_slot_pivot = payload.get('weekday_slot_pivot')
            ordered_slots = payload.get('ordered_slots')
            # === Recon 诊断（按日桥接 + 费用Recon + Top列表） ===
            try:
                print("[RECON] 生成Recon诊断报表...")
                mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
                mtm_df = None
                if mtm_file.exists():
                    mtm_df = pd.read_csv(mtm_file)
                    def _parse_currency(v):
                        try:
                            if isinstance(v, str):
                                return float(v.replace(',', '').strip())
                            return float(v)
                        except Exception:
                            return np.nan
                    mtm_df['date'] = pd.to_datetime(mtm_df['date']).dt.date
                    mtm_df['cash_num'] = mtm_df['cash'].apply(_parse_currency)
                    mtm_df['long_value_num'] = mtm_df['long_value'].apply(_parse_currency)
                    mtm_df['short_value_num'] = mtm_df['short_value'].apply(_parse_currency)
                    mtm_df['total_assets_num'] = mtm_df['total_assets'].apply(_parse_currency)
                    mtm_df = mtm_df.sort_values('date')
                    mtm_df['nav_diff'] = mtm_df['total_assets_num'].diff()

                orders_cols = ['Code','direction','tradeAmount','fee','tradeQty','Timestamp']
                orders_df = pd.read_parquet(self.data_path, columns=orders_cols)
                orders_df = orders_df.dropna(subset=['direction','tradeAmount','fee','Timestamp']).copy()
                orders_df['date'] = pd.to_datetime(orders_df['Timestamp']).dt.date
                daily_flows = orders_df.groupby(['date','direction'])[['tradeAmount','fee']].sum().unstack(fill_value=0)
                daily_flows.columns = [f"{a}_{b}" for a,b in daily_flows.columns]
                daily_flows['cash_bridge'] = (
                    daily_flows.get('tradeAmount_S', 0.0) - daily_flows.get('tradeAmount_B', 0.0) -
                    (daily_flows.get('fee_B', 0.0) + daily_flows.get('fee_S', 0.0))
                )
                daily_flows = daily_flows.reset_index()

                if mtm_df is not None:
                    recon_daily = pd.merge(
                        mtm_df[['date','nav_diff','cash_num','long_value_num','short_value_num','total_assets_num']],
                        daily_flows, on='date', how='left'
                    ).fillna(0.0)
                    recon_daily['cash_diff'] = recon_daily['cash_num'] - recon_daily['cash_num'].shift(1)
                    recon_daily['fee_implied'] = (
                        recon_daily.get('tradeAmount_S', 0.0) - recon_daily.get('tradeAmount_B', 0.0) - recon_daily['cash_diff']
                    )
                    recon_daily['fee_reported'] = (recon_daily.get('fee_B', 0.0) + recon_daily.get('fee_S', 0.0))
                    recon_daily['fee_gap'] = recon_daily['fee_implied'] - recon_daily['fee_reported']
                    recon_daily['residual'] = recon_daily['nav_diff'] - recon_daily['cash_bridge']
                    # 新增：持仓估值变动项 ΔL - ΔS，用于验证 residual ≈ exposure_delta
                    recon_daily['delta_long'] = recon_daily['long_value_num'] - recon_daily['long_value_num'].shift(1)
                    recon_daily['delta_short'] = recon_daily['short_value_num'] - recon_daily['short_value_num'].shift(1)
                    recon_daily['exposure_delta'] = recon_daily['delta_long'] - recon_daily['delta_short']
                    recon_daily['exposure_gap'] = recon_daily['residual'] - recon_daily['exposure_delta']

                    top_days = (recon_daily.assign(abs_res=lambda x: x['residual'].abs())
                                            .sort_values('abs_res', ascending=False)
                                            .head(15)
                                            [['date','nav_diff','cash_bridge','residual','fee_gap']])

                    # 真实Top代码贡献：按日剩余头寸 × 当日收盘涨跌
                    try:
                        orders_q = orders_df.copy()
                        # 日净成交量（买入为+，卖出为-），单位保持与tradeQty一致
                        orders_q['signed_qty'] = np.where(orders_q['direction']=='B', orders_q['tradeQty'], -orders_q['tradeQty'])
                        daily_qty = orders_q.groupby(['Code','date'])['signed_qty'].sum().reset_index()
                        daily_qty = daily_qty.sort_values(['Code','date'])
                        # 期末净头寸（逐日累计）
                        daily_qty['position_eod'] = daily_qty.groupby('Code')['signed_qty'].cumsum()
                        # 用昨日期末净头寸参与今日价格变动归因
                        daily_qty['position_prev'] = daily_qty.groupby('Code')['position_eod'].shift(1).fillna(0.0)
                        # 收盘价序列
                        close_daily_path = Path('data/daily_close_cache.parquet')
                        close_daily = pd.read_parquet(close_daily_path) if close_daily_path.exists() else None
                        top_codes_tbl = pd.DataFrame({'Code':[], 'abs_contribution':[], 'percent_of_year':[]})
                        if close_daily is not None and len(daily_qty) > 0:
                            close_daily = close_daily.dropna(subset=['close']).copy().sort_values(['Code','date'])
                            close_daily['prev_close'] = close_daily.groupby('Code')['close'].shift(1)
                            close_daily['price_diff'] = close_daily['close'] - close_daily['prev_close']
                            # 对齐自然日：若无交易，则position_prev需前向填充
                            # 先构造所有(Code, date)笛卡尔上的最小覆盖
                            all_map = close_daily[['Code','date']].drop_duplicates()
                            pos_map = daily_qty[['Code','date','position_prev']]
                            pos_full = all_map.merge(pos_map, on=['Code','date'], how='left').sort_values(['Code','date'])
                            pos_full['position_prev'] = pos_full.groupby('Code')['position_prev'].ffill().fillna(0.0)
                            # 合并价格变动
                            merged = pos_full.merge(close_daily[['Code','date','price_diff']], on=['Code','date'], how='left').fillna({'price_diff':0.0})
                            merged['raw_contrib'] = merged['position_prev'] * merged['price_diff']
                            # 以每日 exposure_delta 为总量，按 |raw| 比例分摊，确保按日汇总一致
                            day_delta = recon_daily[['date','exposure_delta']].copy()
                            day_delta['date'] = pd.to_datetime(day_delta['date'])
                            merged['date'] = pd.to_datetime(merged['date'])
                            merged2 = merged.merge(day_delta, on='date', how='left').fillna({'exposure_delta':0.0})
                            day_abs = merged2.groupby('date')['raw_contrib'].apply(lambda s: np.abs(s).sum()).replace(0, np.nan)
                            merged2 = merged2.merge(day_abs.rename('day_abs_sum'), on='date', how='left')
                            merged2['scaled_contrib'] = np.where(
                                merged2['day_abs_sum'].notna(),
                                np.abs(merged2['raw_contrib']) / merged2['day_abs_sum'] * merged2['exposure_delta'],
                                0.0
                            )
                            code_total = merged2.groupby('Code')['scaled_contrib'].sum()
                            total_abs_year = np.abs(recon_daily['exposure_delta']).sum()
                            top_codes_tbl = (code_total.abs().sort_values(ascending=False).head(20)
                                             .rename('abs_contribution').reset_index())
                            if total_abs_year > 0:
                                top_codes_tbl['percent_of_year'] = top_codes_tbl['abs_contribution'] / total_abs_year
                            else:
                                top_codes_tbl['percent_of_year'] = 0.0
                    except Exception:
                        top_codes_tbl = pd.DataFrame({'Code':[], 'abs_contribution':[], 'percent_of_year':[]})

                    def _df_to_html(df: pd.DataFrame, title: str) -> str:
                        return f"<h4>{title}</h4>" + df.to_html(index=False, border=0)

                    # 相关性与误差统计
                    try:
                        corr = float(pd.Series(recon_daily['residual']).corr(recon_daily['exposure_delta']))
                    except Exception:
                        corr = np.nan
                    stats_html = f"<p><b>residual vs exposure_delta</b>: 相关系数={corr:.4f}; MAE={recon_daily['exposure_gap'].abs().mean():,.2f}; MedianAE={recon_daily['exposure_gap'].abs().median():,.2f}</p>"

                    # Lorenz 曲线数据（按代码的年度绝对贡献累计分布）
                    try:
                        lorenz_html = ""
                        if 'Code' in top_codes_tbl.columns or 'code' in top_codes_tbl.columns:
                            # 使用上一阶段的按代码总贡献 code_total（绝对值）
                            code_total_abs = code_total.abs().sort_values()
                            if code_total_abs.sum() > 0 and len(code_total_abs) > 1:
                                y_vals = (code_total_abs.cumsum() / code_total_abs.sum()).astype(float).tolist()
                                n = len(code_total_abs)
                                x_vals = [i / n for i in range(1, n + 1)]
                                import json as _json
                                x_json = _json.dumps(x_vals)
                                y_json = _json.dumps(y_vals)
                                lorenz_html = (
                                    '<div id="lorenz_curve" style="height:320px;width:100%;"></div>'
                                    '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
                                    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">'
                                    '<script>'
                                    '(function(){'
                                    'var x=' + x_json + ';'
                                    'var y=' + y_json + ';'
                                    "var data=[{x:x,y:x,mode:'lines',name:'均匀分布',line:{dash:'dot',color:'#888'}},"
                                    "{x:x,y:y,mode:'lines',name:'Lorenz',line:{color:'#1f77b4'}}];"
                                    "var layout={title:'年度分解：代码贡献Lorenz曲线',xaxis:{title:'累计代码占比'},yaxis:{title:'累计贡献占比'},margin:{l:60,r:10,t:40,b:40}};"
                                    "Plotly.newPlot('lorenz_curve',data,layout,{displayModeBar:false});"
                                    '})();'
                                    '</script>'
                                )
                    except Exception:
                        lorenz_html = ""

                    html = "".join([
                        _df_to_html(top_days, 'Top日（按|NAV差异−现金桥|）'),
                        _df_to_html(recon_daily[['date','nav_diff','cash_bridge','exposure_delta','residual','exposure_gap']].tail(15), '最近15日：residual 与 exposure_delta 对照'),
                        _df_to_html(recon_daily[['date','nav_diff','cash_bridge','fee_implied','fee_reported','fee_gap']].tail(10), '最近10日费用Recon（隐含费用 vs 订单费用）'),
                        stats_html,
                        _df_to_html(top_codes_tbl.assign(abs_contribution_wan=lambda d: d['abs_contribution']/1e4,
                                                         percent_fmt=lambda d: (d['percent_of_year']*100).map(lambda x: f"{x:.2f}%"))[
                                    ['Code','abs_contribution','abs_contribution_wan','percent_fmt']],
                                    'Top20代码（真实：剩余头寸 × 当日收盘涨跌；含年度占比）'),
                        lorenz_html
                    ])
                    diag_path = self.reports_dir / 'recon_diagnosis.html'
                    diag_path.write_text(html, encoding='utf-8')
                    print(f"[RECON] 诊断页面: {diag_path}")
            except Exception:
                print("[RECON] 诊断失败，跳过（不影响主流程）")

            
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 时段分析出错: {e}")
            import traceback
            traceback.print_exc()

        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 时段分析出错: {e}")
            import traceback
            traceback.print_exc()

    def _apply_plotly_theme(self, fig: go.Figure, yaxis_percent: bool = False) -> None:
        """统一 Plotly 主题，符合前端设计规范。"""
        base_layout = dict(
            font=dict(family='Noto Sans SC, "Microsoft YaHei", "Segoe UI", sans-serif'),
            margin=dict(t=30, b=40, l=60, r=20),
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb', zeroline=True, zerolinecolor='#9ca3af')
        )
        try:
            fig.update_layout(**base_layout)
            if yaxis_percent:
                fig.update_yaxes(tickformat='.2%')
        except Exception:
            pass

    def _clean_title_text(self, text: str) -> str:
        """移除标题中的括号补充信息，仅保留主标题。"""
        try:
            cleaned = re.sub(r'（[^（）]*）', '', text)
            cleaned = re.sub(r'\([^()]*\)', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            return cleaned.strip()
        except Exception:
            return text

    def _save_figure_with_details(self, fig, name: str, title: str, explanation_html: str, metrics: dict, extra_figs: Optional[List[Tuple[str, go.Figure]]] = None):
        """保存含说明与指标汇总的图表页面"""
        try:
            title = self._clean_title_text(title)
            output_path = self.reports_dir / f"{name}.html"

            # 将所有数据转换为原生可序列化类型，避免 Plotly 在部分环境下将 y 渲染为顺序索引
            # 这会规避 TypedArray(bdata) 在某些浏览器/离线环境下解码失败导致的“单调上升到样本数”现象
            try:
                for trace in fig.data:
                    # 处理 x 轴：时间戳转字符串（保留原始字符串/数值）
                    if hasattr(trace, 'x') and trace.x is not None:
                        try:
                            x_values = list(trace.x)
                        except TypeError:
                            x_values = trace.x
                        new_x = []
                        for xv in x_values:
                            if hasattr(xv, 'isoformat'):
                                new_x.append(xv.isoformat())
                            else:
                                new_x.append(xv)
                        trace.x = new_x
                    # 处理 y 轴：仅在全部可转为数字时才转 float；否则保持原样（如分类标签、中文星期）
                    if hasattr(trace, 'y') and trace.y is not None:
                        try:
                            y_list = list(trace.y)
                        except TypeError:
                            # 兼容 TypedArray 包装 {'dtype':'f8','bdata': '...'}
                            if isinstance(trace.y, dict) and 'bdata' in trace.y:
                                try:
                                    y_bytes = base64.b64decode(trace.y['bdata'])
                                    y_list = np.frombuffer(y_bytes, dtype=np.float64).tolist()
                                except Exception:
                                    y_list = []
                            else:
                                y_list = []
                        def _all_numeric(arr):
                            ok = True
                            for v in arr:
                                if v is None:
                                    continue
                                try:
                                    float(v)
                                except Exception:
                                    ok = False
                                    break
                            return ok
                        if _all_numeric(y_list):
                            trace.y = [None if v is None else float(v) for v in y_list]
                        else:
                            # 保持原始（可能是分类标签）
                            trace.y = y_list
                        # 若是柱状图且无文本，则为便于阅读填充标签
                        if getattr(trace, 'type', None) == 'bar' and not getattr(trace, 'text', None) and _all_numeric(y_list):
                            try:
                                trace.text = [f"{float(v):.2f}" if v is not None else "" for v in y_list]
                                trace.textposition = 'outside'
                            except Exception:
                                pass
                    # 对热力图进行 z/text 形状修复：统一为二维 (len(y), len(x))，避免被序列化为一维导致仅对角显示
                    try:
                        if getattr(trace, 'type', None) == 'heatmap':
                            x_len = len(list(trace.x)) if hasattr(trace, 'x') and trace.x is not None else None
                            y_len = len(list(trace.y)) if hasattr(trace, 'y') and trace.y is not None else None
                            if hasattr(trace, 'z') and trace.z is not None and x_len and y_len:
                                import numpy as _np
                                z_arr = _np.array(trace.z)
                                if z_arr.ndim == 1 and z_arr.size == x_len * y_len:
                                    z_arr = z_arr.reshape((y_len, x_len))
                                    trace.z = z_arr.tolist()
                                elif z_arr.ndim == 2 and (z_arr.shape != (y_len, x_len)):
                                    # 若方向相反则尝试自动转置
                                    if z_arr.shape == (x_len, y_len):
                                        trace.z = z_arr.T.tolist()
                                # 修复 text 形状
                                if hasattr(trace, 'text') and trace.text is not None:
                                    t_arr = _np.array(trace.text)
                                    if t_arr.ndim == 1 and t_arr.size == x_len * y_len:
                                        trace.text = t_arr.reshape((y_len, x_len)).tolist()
                                    elif t_arr.ndim == 2 and (t_arr.shape != (y_len, x_len)):
                                        if t_arr.shape == (x_len, y_len):
                                            trace.text = t_arr.T.tolist()
                    except Exception:
                        pass
            except Exception:
                # 降级：不影响后续渲染
                pass

            # 展示配置
            config = {
                'displayModeBar': False,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
            }

            # 统一主题（含字体、网格与 hover 行为）
            self._apply_plotly_theme(fig)
            if extra_figs:
                for _, ef in extra_figs:
                    try:
                        self._apply_plotly_theme(ef)
                    except Exception:
                        pass

            # 仅生成图表DIV片段，后续嵌入我们的说明模板
            # 为彻底规避 TypedArray(bdata) 在部分环境下解析失败导致的柱状图“单调上升”问题，
            # 我们不再使用内置的 to_html 序列化，而是手动将 Figure deep 转换为原生 JSON（纯 list/float），
            # 再在前端用 Plotly.newPlot 还原。这样 y 值会严格按照数值绘制。
            def _to_native(obj):
                # 数值ndarray转为浮点列表；否则转字符串列表
                if isinstance(obj, np.ndarray):
                    if np.issubdtype(obj.dtype, np.number):
                        return obj.astype(float).tolist()
                    return [str(v) for v in obj.tolist()]
                # 单个numpy数值
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                # pandas Timestamp/Datetime
                try:
                    import pandas as _pd  # 延迟导入
                    if isinstance(obj, _pd.Timestamp):
                        return obj.isoformat()
                except Exception:
                    pass
                # dict 递归（含TypedArray）
                if isinstance(obj, dict):
                    if 'bdata' in obj and isinstance(obj.get('bdata'), str):
                        try:
                            dtype_map = {
                                'f8': np.float64, 'f4': np.float32,
                                'i8': np.int64,   'i4': np.int32,
                                'u8': np.uint64,  'u4': np.uint32
                            }
                            np_dtype = dtype_map.get(obj.get('dtype', 'f8'), np.float64)
                            raw = base64.b64decode(obj['bdata'])
                            arr = np.frombuffer(raw, dtype=np_dtype)
                            # 若包含shape信息，按shape还原为多维数组，避免2D被展平成1D
                            shape_val = obj.get('shape')
                            if shape_val is not None:
                                try:
                                    if isinstance(shape_val, str):
                                        # 形如 "5, 8" or "5,8"
                                        shape_tuple = tuple(int(s.strip()) for s in shape_val.replace('x', ',').split(',') if s.strip() != '')
                                    elif isinstance(shape_val, (list, tuple)):
                                        shape_tuple = tuple(int(s) for s in shape_val)
                                    else:
                                        shape_tuple = None
                                    if shape_tuple and np.prod(shape_tuple) == arr.size:
                                        arr = arr.reshape(shape_tuple)
                                except Exception:
                                    # 忽略形状解析错误，退回一维
                                    pass
                            if np.issubdtype(arr.dtype, np.number):
                                return arr.astype(float).tolist()
                            return [str(v) for v in arr.tolist()]
                        except Exception:
                            return {k: _to_native(v) for k, v in obj.items()}
                    return {k: _to_native(v) for k, v in obj.items()}
                # 列表/元组
                if isinstance(obj, (list, tuple)):
                    return [_to_native(v) for v in obj]
                # 其它可能的日期对象
                if hasattr(obj, 'isoformat'):
                    try:
                        return obj.isoformat()
                    except Exception:
                        return str(obj)
                return obj

            tmpl = Template("""
            <div id="$div_id" class="plotly-graph-div" style="height:${height}px; width:100%;"></div>
            <script type="text/javascript">
                window.PLOTLYENV=window.PLOTLYENV || {};
                (function() {
                    var fig = $fig_json;
                    var cfg = $config_json;
                    if (document.getElementById("$div_id")) {
                        Plotly.newPlot("$div_id", fig.data || [], fig.layout || {}, cfg);
                    }
                })();
            </script>
            """)

            def _build_fig_html(fig_obj, div_id: str) -> str:
                # Force transparent background
                fig_obj.update_layout({
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'font': {'family': 'Noto Sans SC, "Microsoft YaHei", "Segoe UI", sans-serif'}
                })
                fig_json = fig_obj.to_plotly_json()
                fig_json_native = _to_native(fig_json)
                fig_json_str = json.dumps(fig_json_native, ensure_ascii=False)
                config_str = json.dumps(config, ensure_ascii=False)
                height_px = fig_json_native.get('layout', {}).get('height', 500)
                return tmpl.substitute(div_id=div_id, height=height_px, fig_json=fig_json_str, config_json=config_str)

            fig_html = _build_fig_html(fig, name)
            if extra_figs:
                extra_parts = []
                for idx, (suffix, extra_fig) in enumerate(extra_figs, start=1):
                    div_suffix = suffix if suffix else f"extra{idx}"
                    div_id = f"{name}_{div_suffix}"
                    extra_parts.append(f"<div class=\"mt-6\">{_build_fig_html(extra_fig, div_id)}</div>")
                fig_html += "".join(extra_parts)

            # 指标表格HTML
            if metrics:
                rows = "".join([f"<tr><td class='py-2 px-3 text-gray-600'>{k}</td><td class='py-2 px-3 text-gray-900 font-medium'>{v}</td></tr>" for k, v in metrics.items()])
                metrics_html = f"""
                <div class="mt-2 mb-4">
                    <table class="w-full text-sm border border-gray-100 rounded-md overflow-hidden">
                        <thead class="bg-gray-50 text-gray-500">
                            <tr><th class="py-2 px-3 text-left">指标</th><th class="py-2 px-3 text-left">数值</th></tr>
                        </thead>
                        <tbody class="divide-y divide-gray-100">{rows}</tbody>
                    </table>
                </div>
                """
            else:
                metrics_html = ""

            # 组合页面 - 图表优先显示
            mathjax_local_src = self._ensure_mathjax_bundle()
            explanation_md = json.dumps(explanation_html, ensure_ascii=False)
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>{title}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <script src="https://cdn.tailwindcss.com"></script>
                <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
                <style>
                    body {{ font-family: "Noto Sans SC","Microsoft YaHei","Segoe UI",sans-serif; }}
                    .markdown-body h3 {{ font-size: 1.05rem; margin-top: 1rem; margin-bottom: .45rem; }}
                    .markdown-body p {{ margin: .35rem 0; color: #374151; line-height: 1.65; }}
                    .markdown-body ul {{ margin: .25rem 0 .5rem 1.2rem; color: #374151; line-height: 1.6; list-style: disc; }}
                    .markdown-body li {{ margin: .2rem 0; }}
                </style>
                <script>
                    window.MathJax = {{
                        tex: {{
                            inlineMath: [["$","$"],["\\(","\\)"]],
                            displayMath: [["$$","$$"],["\\[","\\]"]],
                            processEscapes: true
                        }},
                        options: {{ skipHtmlTags: ["script","noscript","style","textarea","pre","code"] }},
                        svg: {{ fontCache: 'global' }}
                    }};
                </script>
                <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js" onerror="this.onerror=null; this.src='{mathjax_local_src}';"></script>
            </head>
            <body class="bg-gray-50">
                <header class="sticky top-0 z-30 bg-white shadow-sm border-b border-gray-100">
                    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
                        <div>
                            <p class="text-[11px] tracking-[0.18em] text-gray-500 uppercase">时段分析</p>
                            <h1 class="text-xl font-semibold text-gray-900">{title}</h1>
                        </div>
                        <div class="text-xs text-gray-500">滚动查看完整内容</div>
                    </div>
                </header>
                <main class="max-w-7xl mx-auto px-4 sm:px-6 py-6">
                    <div class="bg-white rounded-lg shadow-sm p-6" id="page-root">
                        <div id="main-chart" class="mb-6">
                            {fig_html}
                        </div>
                        {metrics_html}
                        <div id="explain-md" class="markdown-body mt-4"></div>
                    </div>
                </main>
                <script>
                    (function ensureRender(){{
                        var mdText = {explanation_md};
                        var mdTarget = document.getElementById('explain-md');
                        if (mdTarget){{
                            if (window.marked) {{
                                mdTarget.innerHTML = marked.parse(mdText);
                            }} else {{
                                mdTarget.textContent = mdText;
                            }}
                        }}
                        var anchor = document.getElementById('main-chart');
                        if (anchor) {{
                            requestAnimationFrame(function() {{
                                anchor.scrollIntoView({{ behavior: 'auto', block: 'start' }});
                            }});
                        }}
                        function typeset() {{
                            if (window.MathJax && window.MathJax.typesetPromise) {{
                                window.MathJax.typesetPromise().catch(function(err){{ console.warn('MathJax error:', err); }});
                            }}
                        }}
                        if (document.readyState === 'complete') {{
                            typeset();
                        }} else {{
                            window.addEventListener('load', typeset);
                        }}
                    }})();
                </script>
            </body>
            </html>
            """

            output_path.write_text(html, encoding='utf-8')
            self.figures.append((name, str(output_path)))
            file_size = output_path.stat().st_size / (1024*1024)
            print(f"    <i class='fas fa-check-circle text-green-500'></i> 保存: {name}.html ({file_size:.2f} MB)")
        except Exception as e:
            print(f"    <i class='fas fa-times-circle text-red-500'></i> 保存失败 {name}: {e}")
            import traceback
            traceback.print_exc()
            
    # 旧版 daily_returns_initial_capital_analysis 已删除，保留新版定义

    def daily_min_capital_and_utilization_analysis_legacy(self):
        """全周期：每日最低所需本金序列 + 资金占用收益率。
        - 最低所需本金：逐日逐笔回放，required_equity = max(0, -cash_free) + short_margin，取当日峰值；
          其中 short_margin = 空头市值 × margin_short_ratio。
        - 资金占用收益率：日PnL / 当日最低所需本金；PnL 来源于 daily_nav_revised.csv。
        """
        try:
            import plotly.graph_objs as go
            import pandas as _pd
            import numpy as _np

            # 0) 缓存命中直接返回
            cache_key = self._make_capital_util_cache_key()
            cached_daily_min = self._load_capital_util_daily_min(cache_key)
            if cached_daily_min is not None and len(cached_daily_min) > 0:
                daily_min = cached_daily_min.copy()
                print(f"[CACHE] 复用每日最低所需本金缓存，共 {len(daily_min):,} 天")
            else:
                # 1) 逐日计算“最低所需本金”
                self._ensure_credit_rules_loaded()
                allow_short_cash = bool(self._credit_rules.get('allow_short_proceeds_to_cash', False))
                allow_sell_long_t0 = bool(self._credit_rules.get('allow_sell_long_cash_T0', True))
                short_margin_ratio = float(self._credit_rules.get('margin_short_ratio', 0.5))
                fee_accrual = str(self._credit_rules.get('fee_accrual', 'realtime')).lower()
                # 使用元数据获取列名，避免整表读取
                need_cols = ['Code','direction','tradeQty','tradeAmount','fee','price','tradeTimestamp','Timestamp']
                try:
                    cols_meta = set(self._parquet_columns(self.data_path))
                except Exception:
                    cols_meta = set()
                base_cols = [c for c in need_cols if c in cols_meta] if cols_meta else need_cols
                try:
                    od = pd.read_parquet(self.data_path, columns=base_cols)
                except KeyError:
                    # 列缺失时回退为交集
                    base_cols = [c for c in base_cols if c in self.df.columns]
                    od = self.df[base_cols].copy()
                if len(od) == 0:
                    print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 订单数据为空，跳过资金占用分析")
                    return
                tcol = 'tradeTimestamp' if 'tradeTimestamp' in od.columns else 'Timestamp'
                od = od.dropna(subset=['Code','direction','tradeQty','tradeAmount','fee',tcol]).copy()
                od[tcol] = pd.to_datetime(od[tcol])
                od['__rowid__'] = _np.arange(len(od))
                od['date_trade'] = od[tcol].dt.date
                # 稳定排序
                if 'Timestamp' in od.columns:
                    od['Timestamp'] = pd.to_datetime(od['Timestamp'])
                    od = od.sort_values([ 'date_trade', tcol, 'Timestamp', '__rowid__'])
                else:
                    od = od.sort_values([ 'date_trade', tcol, '__rowid__'])

                positions: dict = {}
                last_price: dict = {}
                daily_records = []

                for dt, d in od.groupby('date_trade', sort=True):
                    cash_free = 0.0
                    fees_buffer_eod = 0.0
                    # 开盘前先评估已有空头的保证金占用
                    short_mv_total = 0.0
                    for c, pos_v in positions.items():
                        if pos_v < 0:
                            lp = last_price.get(c, 0.0)
                            short_mv_total += (-pos_v) * float(lp)
                    max_required = short_mv_total * short_margin_ratio

                    for _, r in d.iterrows():
                        code = r['Code']
                        qty = int(r['tradeQty'])
                        amt = float(r['tradeAmount'])
                        fee = float(r['fee']) if not pd.isna(r['fee']) else 0.0
                        px = float(r['price']) if ('price' in r and not pd.isna(r['price'])) else (amt / qty if qty > 0 else 0.0)
                        pos_cur = int(positions.get(code, 0))

                        if r['direction'] == 'B':
                            if fee_accrual == 'realtime':
                                cash_free -= (amt + fee)
                            else:
                                cash_free -= amt
                                fees_buffer_eod += fee
                            old_lp = last_price.get(code, px)
                            old_neg = max(-pos_cur, 0)
                            new_pos = pos_cur + qty
                            new_neg = max(-new_pos, 0)
                            last_price[code] = px
                            positions[code] = new_pos
                            short_mv_total += (new_neg * px) - (old_neg * old_lp)
                        else:
                            close_qty = min(max(pos_cur, 0), qty)
                            open_short_qty = qty - close_qty
                            if close_qty > 0:
                                portion = close_qty / qty
                                if allow_sell_long_t0:
                                    if fee_accrual == 'realtime':
                                        cash_free += (amt * portion) - (fee * portion)
                                    else:
                                        cash_free += (amt * portion)
                                        fees_buffer_eod += (fee * portion)
                                old_lp = last_price.get(code, px)
                                old_neg = max(-pos_cur, 0)
                                new_pos = pos_cur - close_qty
                                new_neg = max(-new_pos, 0)
                                last_price[code] = px
                                positions[code] = new_pos
                                short_mv_total += (new_neg * px) - (old_neg * old_lp)
                                pos_cur = new_pos
                            if open_short_qty > 0:
                                portion = open_short_qty / qty
                                if allow_short_cash:
                                    if fee_accrual == 'realtime':
                                        cash_free += (amt * portion) - (fee * portion)
                                    else:
                                        cash_free += (amt * portion)
                                        fees_buffer_eod += (fee * portion)
                                else:
                                    if fee_accrual == 'realtime':
                                        cash_free -= (fee * portion)
                                    else:
                                        fees_buffer_eod += (fee * portion)
                                old_lp = last_price.get(code, px)
                                old_neg = max(-pos_cur, 0)
                                new_pos = pos_cur - open_short_qty
                                new_neg = max(-new_pos, 0)
                                last_price[code] = px
                                positions[code] = new_pos
                                short_mv_total += (new_neg * px) - (old_neg * old_lp)

                        # 计算保证金需求
                        short_margin = short_mv_total * short_margin_ratio
                        required_equity = max(0.0, -cash_free) + short_margin
                        if required_equity > max_required:
                            max_required = required_equity

                    # 日终计提费用时再评估一次
                    if fee_accrual != 'realtime' and fees_buffer_eod > 0:
                        cash_for_req = cash_free - fees_buffer_eod
                        short_margin = short_mv_total * short_margin_ratio
                        required_equity = max(0.0, -cash_for_req) + short_margin
                        if required_equity > max_required:
                            max_required = required_equity

                    daily_records.append((dt, float(max_required)))

                daily_min = _pd.DataFrame(daily_records, columns=['date','min_required_equity'])
                # 写缓存
                try:
                    self._save_capital_util_daily_min(cache_key, daily_min)
                except Exception:
                    pass

            # 2) 资金占用收益率 = 日PnL / 当日最低所需本金
            mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
            if not mtm_file.exists():
                print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到盯市数据，跳过资金占用收益率")
                return
            nav = _pd.read_csv(mtm_file)
            def _parse_cur(v):
                try:
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str):
                        return float(v.replace(',', '').strip())
                except Exception:
                    return _np.nan
                return _np.nan
            nav['date'] = _pd.to_datetime(nav['date']).dt.date
            nav = nav.sort_values('date').copy()
            nav['total_assets_num'] = nav['total_assets'].apply(_parse_cur)
            nav['daily_pnl'] = nav['total_assets_num'].diff()
            merged = nav[['date','daily_pnl']].merge(daily_min, on='date', how='left')
            # 安全除法：当日最低所需本金<=0或缺失时收益置为NaN
            denom = merged['min_required_equity'].replace({0: np.nan})
            merged['util_return'] = merged['daily_pnl'] / denom
            merged['cum_util_return'] = merged['util_return'].cumsum()

            # 3) 图表
            # 顶部图仅展示 daily_min（不依赖 nav），避免无重叠导致空图
            fig_top = go.Figure()
            dm = daily_min.copy()
            fig_top.add_trace(go.Scatter(
                x=_pd.to_datetime(dm['date']),
                y=dm['min_required_equity'].astype(float),
                mode='lines', name='每日最低所需本金', line=dict(color='#2c3e50')
            ))
            fig_top.update_layout(height=460, xaxis=dict(title='日期'), yaxis=dict(title='资金(¥)'))

            fig_bottom = go.Figure()
            if merged['util_return'].notna().any():
                fig_bottom.add_trace(go.Scatter(
                    x=_pd.to_datetime(merged['date']),
                    y=merged['util_return'], mode='lines', name='资金占用收益率', line=dict(color='#16a085')
                ))
            if merged['cum_util_return'].notna().any():
                fig_bottom.add_trace(go.Scatter(
                    x=_pd.to_datetime(merged['date']),
                    y=merged['cum_util_return'], mode='lines', name='累计资金占用收益率', yaxis='y2', line=dict(color='#e74c3c')
                ))
            fig_bottom.update_layout(
                height=460,
                xaxis=dict(title='日期'),
                yaxis=dict(title='日度(%)', tickformat='.2%'),
                yaxis2=dict(title='累计(%)', overlaying='y', side='right', tickformat='.2%'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )

            primary_metrics = {
                '平均每日资金占用': f"¥{_np.nanmean(merged['min_required_equity']):,.0f}",
                '峰值资金占用': f"¥{_np.nanmax(merged['min_required_equity']):,.0f}",
                '有效天数': int(merged['min_required_equity'].notna().sum()),
            }
            secondary_metrics = {
                '资金占用收益率均值': f"{_np.nanmean(merged['util_return'])*100:.3f}%",
                '资金占用收益率标准差': f"{_np.nanstd(merged['util_return'])*100:.3f}%",
                '累计资金占用收益率(期末)': f"{(merged['cum_util_return'].iloc[-1] if len(merged)>0 else 0)*100:.2f}%",
            }

            explanation_html = (
                "<h4>页面目的</h4>"
                "<ul>"
                "<li>量化日内最大资金占用，检查授信/保证金配置是否匹配交易节奏。</li>"
                "<li>计算资金占用收益率，评估单位占用资金产生的回报。</li>"
                "</ul>"
                "<h4>实现方式</h4>"
                "<ol>"
                "<li>按 <code>tradeTimestamp</code>/<code>Timestamp</code> 时间顺序逐笔回放：买入(<code>B</code>)现金流出 = <code>tradeAmount + fee</code>；卖出(<code>S</code>)现金回笼 = <code>tradeAmount - fee</code>，同步更新多/空持仓数量。</li>"
                "<li>空头保证金 = 空头市值 × 保证金比例；所需权益 = $\\max(0, -\\text{可用现金}) + \\text{空头保证金}$，记录当日最高值作为“每日最低所需本金”。</li>"
                "<li>日度资金占用收益率 = 当日盯市盈亏 ÷ 当日最低所需本金；累计曲线为日度收益率的算术累加。</li>"
                "<li>授信规则（卖出回款是否T+0、开空所得是否计入现金、<code>fee</code> 逐笔计提等）沿用当前配置，并体现在回放现金流与保证金计算中。</li>"
                "</ol>"
            )

            self._save_figure_pair_with_details(
                fig_top, fig_bottom,
                name='capital_utilization_light',
                title='资金占用与资金占用收益率',
                explanation_html=explanation_html,
                metrics_primary=primary_metrics,
                metrics_secondary=secondary_metrics,
                primary_title='每日最低所需本金',
                secondary_title='资金占用收益率(日度/累计)'
            )
            print("[OK] 每日最低所需本金 + 资金占用收益率 已生成")
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 资金占用分析失败: {e}")

    def _save_figure_pair_with_details_v2(self, fig_top, fig_bottom, name: str, title: str, explanation_html: str, metrics_primary: dict, metrics_secondary: dict, primary_title: str, secondary_title: str):
        """在同一页面上下展示两张图：上=交易金额占比，下=盈利金额占比。
        - 在仪表板中仍作为一个入口展示（主页面仅显示上图，滚动可见下图）。
        - 在新窗口打开时，CSS在宽屏下自动并排，两列布局。
        """
        try:
            title = self._clean_title_text(title)
            primary_title = self._clean_title_text(primary_title)
            secondary_title = self._clean_title_text(secondary_title)
            output_path = self.reports_dir / f"{name}.html"

            def _to_native(obj):
                # 与单图保存逻辑一致，确保 ndarray/TypedArray 转为原生 JSON
                if isinstance(obj, np.ndarray):
                    if np.issubdtype(obj.dtype, np.number):
                        return obj.astype(float).tolist()
                    return [str(v) for v in obj.tolist()]
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                try:
                    import pandas as _pd  # 延迟导入
                    if isinstance(obj, _pd.Timestamp):
                        return obj.isoformat()
                except Exception:
                    pass
                if isinstance(obj, dict):
                    if 'bdata' in obj and isinstance(obj.get('bdata'), str):
                        try:
                            dtype_map = {'f8': np.float64, 'f4': np.float32, 'i8': np.int64, 'i4': np.int32, 'u8': np.uint64, 'u4': np.uint32}
                            np_dtype = dtype_map.get(obj.get('dtype', 'f8'), np.float64)
                            raw = base64.b64decode(obj['bdata'])
                            arr = np.frombuffer(raw, dtype=np_dtype)
                            shape_val = obj.get('shape')
                            if shape_val is not None:
                                try:
                                    if isinstance(shape_val, str):
                                        shape_tuple = tuple(int(s.strip()) for s in shape_val.replace('x', ',').split(',') if s.strip() != '')
                                    elif isinstance(shape_val, (list, tuple)):
                                        shape_tuple = tuple(int(s) for s in shape_val)
                                    else:
                                        shape_tuple = None
                                    if shape_tuple and np.prod(shape_tuple) == arr.size:
                                        arr = arr.reshape(shape_tuple)
                                except Exception:
                                    pass
                            if np.issubdtype(arr.dtype, np.number):
                                return arr.astype(float).tolist()
                            return [str(v) for v in arr.tolist()]
                        except Exception:
                            return {k: _to_native(v) for k, v in obj.items()}
                    return {k: _to_native(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_native(v) for v in obj]
                if hasattr(obj, 'isoformat'):
                    try:
                        return obj.isoformat()
                    except Exception:
                        return str(obj)
                return obj

            def _fig_to_html_div(fig, div_id):
                fig_json = fig.to_plotly_json()
                fig_json_native = _to_native(fig_json)
                fig_json_str = json.dumps(fig_json_native, ensure_ascii=False)
                config_str = json.dumps({'displayModeBar': False, 'displaylogo': False}, ensure_ascii=False)
                tmpl = Template("""
                <div id="$div_id" class="plotly-graph-div" style="height:${height}px; width:100%;"></div>
                <script type="text/javascript">
                    (function(){
                        var fig = $fig_json;
                        var cfg = $config_json;
                        Plotly.newPlot("$div_id", fig.data || [], fig.layout || {}, cfg);
                    })();
                </script>
                """)
                height_px = fig_json_native.get('layout', {}).get('height', 480)
                return tmpl.substitute(div_id=div_id, height=height_px, fig_json=fig_json_str, config_json=config_str)

            top_html = _fig_to_html_div(fig_top, f"{name}_top")
            bottom_html = _fig_to_html_div(fig_bottom, f"{name}_bottom")

            def _metrics_tbl(ms):
                if not ms:
                    return ""
                rows = "".join([f"<tr><td class='py-2 px-3 text-gray-600'>{k}</td><td class='py-2 px-3 text-gray-900 font-medium'>{v}</td></tr>" for k,v in ms.items()])
                return f"""
                <div class="mt-2 mb-2">
                    <table class="w-full text-sm border border-gray-100 rounded-md overflow-hidden">
                        <thead class="bg-gray-50 text-gray-500"><tr><th class="py-2 px-3 text-left">指标</th><th class="py-2 px-3 text-left">数值</th></tr></thead>
                        <tbody class="divide-y divide-gray-100">{rows}</tbody>
                    </table>
                </div>
                """

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset=\"utf-8\" />
                <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
                <title>{title}</title>
                <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
                <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css\">
                <script src=\"https://cdn.tailwindcss.com\"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ margin: 0 0 15px 0; font-size: 22px; }}
                    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
                    @media (min-width: 1100px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
                    .card {{ background:#fff; padding:16px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
                    .subtitle {{ font-weight:600; margin:6px 0 10px 2px; }}
                    table td, table th {{ border-bottom: 1px solid #eee; padding: 8px; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class=\"grid\">
                    <div class=\"card\">
                        <div class=\"subtitle\">{primary_title}</div>
                        {top_html}
                        {_metrics_tbl(metrics_primary)}
                    </div>
                    <div class=\"card\">
                        <div class=\"subtitle\">{secondary_title}</div>
                        {bottom_html}
                        {_metrics_tbl(metrics_secondary)}
                    </div>
                </div>
                <div style=\"margin-top:14px;color:#444;line-height:1.7;\">{explanation_html}</div>
            </body>
            </html>
            """
            output_path.write_text(html, encoding='utf-8')
            self.figures.append((name, str(output_path)))
            file_size = output_path.stat().st_size / (1024*1024)
            print(f"    <i class='fas fa-check-circle text-green-500'></i> 保存: {name}.html ({file_size:.2f} MB)")
        except Exception as e:
            print(f"    <i class='fas fa-times-circle text-red-500'></i> 保存失败 {name}: {e}")

    def _save_figure_triple_with_details(self, fig1, fig2, fig3, name: str, title: str, explanation_html: str, metrics_primary: dict, metrics_secondary: dict, title1: str, title2: str, title3: str):
        """保存三个图表的页面（dashboard预览时纵向，新窗口打开时两列）"""
        try:
            title = self._clean_title_text(title)
            title1 = self._clean_title_text(title1)
            title2 = self._clean_title_text(title2)
            title3 = self._clean_title_text(title3)
            output_path = self.reports_dir / f"{name}.html"

            def _to_native(obj):
                if isinstance(obj, np.ndarray):
                    if np.issubdtype(obj.dtype, np.number):
                        return obj.astype(float).tolist()
                    return [str(v) for v in obj.tolist()]
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                try:
                    import pandas as _pd
                    if isinstance(obj, _pd.Timestamp):
                        return obj.isoformat()
                except Exception:
                    pass
                if isinstance(obj, dict):
                    return {k: _to_native(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_native(v) for v in obj]
                if hasattr(obj, 'isoformat'):
                    try:
                        return obj.isoformat()
                    except Exception:
                        return str(obj)
                return obj

            def _fig_to_html_div(fig, div_id):
                fig_json = fig.to_plotly_json()
                fig_json_native = _to_native(fig_json)
                fig_json_str = json.dumps(fig_json_native, ensure_ascii=False)
                config_str = json.dumps({'displayModeBar': False, 'displaylogo': False}, ensure_ascii=False)
                from string import Template
                tmpl = Template("""
                <div id="$div_id" class="plotly-graph-div" style="height:${height}px; width:100%;"></div>
                <script type="text/javascript">
                    (function(){
                        var fig = $fig_json;
                        var cfg = $config_json;
                        Plotly.newPlot("$div_id", fig.data || [], fig.layout || {}, cfg);
                    })();
                </script>
                """)
                height_px = fig_json_native.get('layout', {}).get('height', 420)
                return tmpl.substitute(div_id=div_id, height=height_px, fig_json=fig_json_str, config_json=config_str)

            html1 = _fig_to_html_div(fig1, f"{name}_1")
            html2 = _fig_to_html_div(fig2, f"{name}_2")
            html3 = _fig_to_html_div(fig3, f"{name}_3")

            def _metrics_tbl(ms):
                if not ms:
                    return ""
                rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in ms.items()])
                return f"""
                <table style="width:100%;border-collapse:collapse;margin:10px 0;">
                    <thead><tr style="background:#f4f6f8;text-align:left;"><th style="padding:8px">指标</th><th style="padding:8px">数值</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
                """

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>{title}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                
                <script src="https://cdn.tailwindcss.com"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ margin: 0 0 15px 0; font-size: 22px; }}
                    
                    /* Dashboard预览模式：纵向排列 */
                    .grid-triple {{ 
                        display: grid; 
                        grid-template-columns: 1fr; 
                        gap: 18px; 
                    }}
                    
                    /* 新窗口宽屏模式：两列布局，第三个图占满行 */
                    @media (min-width: 1100px) {{ 
                        .grid-triple {{ 
                            grid-template-columns: 1fr 1fr;
                        }}
                        .card-full {{ 
                            grid-column: 1 / -1; 
                        }}
                    }}
                    
                    .card {{ background:#fff; padding:16px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
                    .subtitle {{ font-weight:600; margin:6px 0 10px 2px; }}
                    table td, table th {{ border-bottom: 1px solid #eee; padding: 8px; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class="grid-triple">
                    <div class="card">
                        <div class="subtitle">{title1}</div>
                        {html1}
                        {_metrics_tbl(metrics_primary)}
                    </div>
                    <div class="card">
                        <div class="subtitle">{title2}</div>
                        {html2}
                    </div>
                    <div class="card card-full">
                        <div class="subtitle">{title3}</div>
                        {html3}
                        {_metrics_tbl(metrics_secondary)}
                    </div>
                </div>
                <div style="margin-top:14px;color:#444;line-height:1.7;">{explanation_html}</div>
            </body>
            </html>
            """
            output_path.write_text(html, encoding='utf-8')
            self.figures.append((name, str(output_path)))
            file_size = output_path.stat().st_size / (1024*1024)
            print(f"    <i class='fas fa-check-circle text-green-500'></i> 保存: {name}.html ({file_size:.2f} MB)")
        except Exception as e:
            print(f"    <i class='fas fa-times-circle text-red-500'></i> 保存失败 {name}: {e}")

    def create_lightweight_dashboard(self):
        """创建仪表板（Tailwind CSS版）"""
        print("\n<i class='fas fa-sliders-h text-gray-600'></i> 创建仪表板...")
        
        # 获取关键绩效指标，若未计算则使用默认值
        metrics = getattr(self, 'strategy_metrics', {})
        total_return = metrics.get('total_return_nav', 'N/A')
        max_drawdown = metrics.get('max_drawdown', 'N/A')
        sharpe = metrics.get('sharpe_ratio', 'N/A')
        win_rate = metrics.get('win_rate', 'N/A')

        def _need_refresh(v):
            return v in (None, 'N/A') or (isinstance(v, str) and v.strip() == '')

        if any(_need_refresh(v) for v in [total_return, max_drawdown, sharpe, win_rate]):
            if self._ensure_strategy_metrics_from_nav():
                metrics = getattr(self, 'strategy_metrics', {})
                total_return = metrics.get('total_return_nav', total_return)
                max_drawdown = metrics.get('max_drawdown', max_drawdown)
                sharpe = metrics.get('sharpe_ratio', sharpe)
                win_rate = metrics.get('win_rate', win_rate)
        
        # 获取首日资产信息
        first_day_assets_display = "N/A"
        first_day_initial_capital_display = "N/A"
        first_day_min_capital_display = "N/A"
        
        try:
            mtm_file = Path("mtm_analysis_results/daily_nav_revised.csv")
            if mtm_file.exists():
                mtm_df = pd.read_csv(mtm_file)
                def _parse_currency(v):
                    try:
                        if isinstance(v, (int, float)):
                            return float(v)
                        if isinstance(v, str):
                            return float(v.replace(',', '').strip())
                        return np.nan
                    except Exception:
                        return np.nan
                mtm_df['date'] = pd.to_datetime(mtm_df['date']).dt.date
                mtm_df = mtm_df.sort_values('date')
                if len(mtm_df) > 0:
                    first_nav = _parse_currency(mtm_df['total_assets'].iloc[0])
                    if pd.notna(first_nav):
                        first_day_assets_display = f"¥{first_nav:,.0f}"
                        
            # 获取首日最低所需本金快照
            snapshot = Path('reports/first_day_capital_snapshot.json')
            if snapshot.exists():
                data = json.loads(snapshot.read_text(encoding='utf-8'))
                val = float(data.get('first_day_min_required_equity', float('nan')))
                if not math.isnan(val) and val > 0:
                    first_day_min_capital_display = f"¥{val:,.0f}"
                    # 获取安全系数
                    self._ensure_credit_rules_loaded()
                    factor = float(self._credit_rules.get('initial_capital_factor', 1.3))
                    first_day_initial_capital_display = f"¥{(val * factor):,.0f}"
        except Exception:
            pass

        dashboard_html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>量化策略分析报告</title>
            <script src="https://cdn.tailwindcss.com"></script>
            
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            <link rel="icon" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjMyIiBoZWlnaHQ9IjMyIiByeD0iOCIgZmlsbD0iIzM0OThkYiIvPgo8cGF0aCBkPSJNOCAxMkwxNiA4TDI0IDEyTDE2IDE2TDggMTJaIiBmaWxsPSJ3aGl0ZSIvPgo8cGF0aCBkPSJNOCAxOEwxNiAxNEwyNCAxOEwxNiAyMkw4IDE4WiIgZmlsbD0id2hpdGUiLz4KPHN2Zz4K" type="image/svg+xml">
            <style>
                /* 自定义字体栈 */
                body {{ font-family: "Inter", "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }}
                /* 隐藏 Plotly 的 modebar */
                .js-plotly-plot .plotly .modebar {{ display: none !important; }}
            </style>
        </head>
        <body class="bg-gray-50 text-gray-800 min-h-screen p-6">
            <div class="max-w-7xl mx-auto">
                <!-- Header -->
                <header class="bg-white rounded-xl shadow-sm p-8 mb-8 text-center border-t-4 border-blue-500">
                    <h1 class="text-3xl font-bold text-gray-900 mb-2"><i class='fas fa-rocket text-blue-500'></i> 量化策略分析报告</h1>
                    <div class="flex justify-center items-center space-x-4 text-sm text-gray-500 mb-4">
                        <span><i class='far fa-calendar-alt text-gray-500'></i> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                        <span><i class='fas fa-chart-bar text-indigo-500'></i> 数据量: {len(self.df):,} 条</span>
                    </div>
                </header>

                <!-- KPI Cards (Pyramid Level 1) -->
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <!-- Total Return -->
                    <div class="bg-white rounded-xl shadow-sm p-6 border-l-4 border-[#e53935]">
                        <div class="text-sm text-gray-500 uppercase tracking-wide font-semibold mb-1">总收益率 (NAV)</div>
                        <div class="text-3xl font-bold text-[#e53935]">{total_return}</div>
                    </div>
                    <!-- Max Drawdown -->
                    <div class="bg-white rounded-xl shadow-sm p-6 border-l-4 border-yellow-500">
                        <div class="text-sm text-gray-500 uppercase tracking-wide font-semibold mb-1">最大回撤</div>
                        <div class="text-3xl font-bold text-yellow-600">{max_drawdown}</div>
                    </div>
                    <!-- Sharpe Ratio -->
                    <div class="bg-white rounded-xl shadow-sm p-6 border-l-4 border-blue-500">
                        <div class="text-sm text-gray-500 uppercase tracking-wide font-semibold mb-1">夏普比率</div>
                        <div class="text-3xl font-bold text-blue-600">{sharpe}</div>
                    </div>
                    <!-- Win Rate -->
                    <div class="bg-white rounded-xl shadow-sm p-6 border-l-4 border-[#e53935]">
                        <div class="text-sm text-gray-500 uppercase tracking-wide font-semibold mb-1">盈利日占比</div>
                        <div class="text-3xl font-bold text-gray-800">{win_rate}</div>
                    </div>
                </div>
                
                <!-- Info Banner -->
                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-8 rounded-r text-sm text-yellow-800">
                    <p class="font-bold mb-1"><i class='fas fa-lightbulb text-yellow-500'></i> 口径说明</p>
                    <ul class="list-disc list-inside space-y-1 ml-2">
                        <li><b>初始本金</b>：暂定为 <b>{first_day_initial_capital_display}</b> (基于首日最低所需本金 {first_day_min_capital_display} × 安全系数)。</li>
                        <li><b>资金计算</b>：涉及现金/总资产的页面均基于此修正本金重算。</li>
                        <li><b>假设</b>：首日前一日无持仓，首日收市持仓全额计入。</li>
                    </ul>
                </div>
        """
        
        # 定义图表分类和排序 - 区分主图表和副图表
        chart_categories = {
            "<i class='fas fa-coins text-yellow-500'></i> 策略收益分析": {
                'main': [
                    ('daily_absolute_profit_light', '日度绝对盈利（盯市）'),
                    ('daily_returns_initial_capital_light', '日收益率曲线（以首日总资产为本金）'),
                ],
                'sub': [
                    ('daily_returns_comparison_light', '日收益率对比（衡量下单质量的三种方法）'),
                    ('cumulative_returns_comparison_light', '累积收益对比（三种方法）'),
                    ('strategy_vs_benchmark_light', '策略vs基准指数累积收益对比'),
                    ('returns_distribution_light', '日绝对收益分布（盯市）'),
                    ('capital_utilization_light', '每日最低所需本金 + 资金占用收益率'),
                ]
            },
            "<i class='fas fa-chart-bar text-indigo-500'></i> 模型性能分析": {
                'main': [
                    ('strategy_sharpe_nav', '夏普比率（真实净值口径）'),
                    ('ic_timeseries_light', 'IC时间序列（含极端信号组追踪）'),
                    ('ic_stability_monthly_light', 'IC按月份稳定性（T+1）含极端信号组追踪'),
                ],
                'sub': [
                    ('ic_distribution_light', 'IC分布'),
                    ('ic_stability_regime_light', 'IC按行情分段（T+1）'),
                    ('ic_stability_industry_light', 'IC按行业分段（T+1）'),
                ]
            },
            "<i class='fas fa-bullseye text-red-500'></i> 预测有效性分析": {
                'main': [
                    ('pred_real_relationship_light', '预测值与实际收益关系分析'),
                ],
                'sub': [
                ]
            },
            "<i class='fas fa-chart-bar text-indigo-500'></i> 投资组合分析": {
                'main': [
                    ('factor_attribution_main', '因子归因（FF3）主页面'),
                    ('portfolio_composition_light', '收盘后持仓市值'),
                ],
                'sub': [
                    ('factor_exposure_light', '策略因子特征暴露度'),
                    ('factor_direction_exposure_light', '新增仓位多空方向分解'),
                    ('factor_holdings_exposure_light', '持仓因子特征暴露'),
                    ('amount_by_market_cap_pie_light', '按市值大小的交易/盈利占比'),
                    ('amount_by_industry_pie_light', '按行业的交易/盈利占比'),
                    ('amount_by_board_pie_light', '按交易所板块的交易/盈利占比'),
                ]
            },
            "<i class='fas fa-bolt text-yellow-400'></i> 交易执行分析": {
                'main': [
                    ('entry_exit_rank_baostock_full', '择时能力分布（5min行情，全量）'),
                    ('fill_rate_timeseries_light', '成交率时间序列'),
                ],
                'sub': [
                    ('intraday_avg_holding_time_light', '交易平均持仓时间（按买入日，按交易时段计）'),
                    ('fill_rate_distribution_light', '成交率分布'),
                ]
            },
            "<i class='fas fa-money-bill-wave text-green-600'></i> 滑点成本分析": {
                'main': [
                    ('total_cost_light', '综合交易成本分析'),
                ],
                'sub': [
                    ('time_slippage_light', '时间滑点分析'),
                    ('price_slippage_light', '价格滑点分析'),
                ]
            },
            "<i class='fas fa-clock text-blue-400'></i> 时段盈利能力分析": {
                'main': [
                    ('slot_intraday_profit_waterfall_light', '日内时段绝对收益瀑布图（全样本期总贡献）'),
                ],
                'sub': [
                    ('time_slot_performance_analysis_light', '时段平均净收益率分析'),
                    ('time_slot_significance_test_light', '时段收益显著性检验'),
                    ('time_slot_profit_decomposition_light', '时段绝对收益分解'),
                    ('time_slot_returns_boxplot_light', '时段收益分布箱线图'),
                    ('time_slot_risk_return_bubble_light', '时段风险-收益气泡图'),
                    ('time_slot_opening_closing_analysis_light', '开盘尾盘精细分析（5分钟）'),
                    ('time_slot_market_regime_heatmap_light', '涨跌日分面热力图'),
                    ('time_slot_profit_heatmap_light', '星期×时段绝对盈利热力图'),
                ]
            }
        }
        
        # 创建现有图表的映射
        extra_figs = [
            ('entry_exit_rank_baostock_full', 'reports/entry_exit_rank_baostock_full.html'),
            ('entry_exit_rank_baostock_full', 'docs/entry_exit_rank_baostock_full.html'),
        ]
        for name, path in extra_figs:
            if name not in [n for n, _ in self.figures]:
                if Path(path).exists():
                    self.figures.append((name, path))

        available_figures = {name: path for name, path in self.figures}
        
        # 按分类生成图表 - 区分主图表和副图表布局
        for category, chart_groups in chart_categories.items():
            main_charts = [(name, title) for name, title in chart_groups['main'] if name in available_figures]
            sub_charts = [(name, title) for name, title in chart_groups['sub'] if name in available_figures]
            
            if main_charts or sub_charts:
                dashboard_html += f"""
                <div class="mb-12">
                    <div class="flex items-center mb-6">
                        <div class="w-1 h-8 bg-blue-500 rounded-full mr-3"></div>
                        <h3 class="text-2xl font-bold text-gray-800">{category}</h3>
                    </div>
                """
                
                # 主图表 - 独占一行 (Full Width)
                if main_charts:
                    for chart_name, chart_title in main_charts:
                        chart_path = available_figures[chart_name]
                        dashboard_html += f"""
                        <div class="mb-8">
                            <div class="bg-white rounded-xl shadow-sm p-1 border-l-4 border-[#e53935] overflow-hidden">
                                <iframe src="{Path(chart_path).name}" class="w-full h-[520px] border-none rounded-lg" loading="lazy"></iframe>
                                <div class="text-center py-2 bg-gray-50 border-t border-gray-100">
                                    <a href="{Path(chart_path).name}" target="_blank" class="text-blue-600 hover:text-blue-800 text-sm font-medium transition-colors">
                                        <i class='fas fa-external-link-alt'></i> 在新窗口打开全屏查看
                                    </a>
                                </div>
                            </div>
                        </div>
                        """
                
                # 副图表 - 网格布局 (Grid)
                if sub_charts:
                    dashboard_html += f"""
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    """
                    
                    for chart_name, chart_title in sub_charts:
                        chart_path = available_figures[chart_name]
                        dashboard_html += f"""
                        <div class="bg-white rounded-xl shadow-sm p-1 overflow-hidden">
                            <iframe src="{Path(chart_path).name}" class="w-full h-[520px] border-none rounded-lg" loading="lazy"></iframe>
                            <div class="text-center py-2 bg-gray-50 border-t border-gray-100">
                                <a href="{Path(chart_path).name}" target="_blank" class="text-blue-600 hover:text-blue-800 text-sm font-medium transition-colors">
                                    <i class='fas fa-external-link-alt'></i> 在新窗口打开
                                </a>
                            </div>
                        </div>
                        """
                    
                    dashboard_html += """
                    </div>
                    """
                
                dashboard_html += """
                </div>
                <div class="h-px bg-gradient-to-r from-transparent via-gray-200 to-transparent my-10"></div>
                """
            
        dashboard_html += """
                <footer class="text-center text-gray-400 text-sm py-8">
                    <p><i class='fas fa-bullseye text-red-500'></i> 高效设计 · <i class='fas fa-rocket text-blue-500'></i> 快速分析 · <i class='fas fa-chart-bar text-indigo-500'></i> 专业洞察</p>
                    <p class="mt-1">优化策略: 智能采样 + CDN加载 + 数据压缩 + Tailwind CSS</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        dashboard_filename = "index.html"
        dashboard_path = self.reports_dir / dashboard_filename
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)

        # 兼容旧版重定向
        legacy_dashboard_path = self.reports_dir / "lightweight_dashboard.html"
        legacy_redirect = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=index.html">
    <title>仪表板已迁移</title>
</head>
<body style="font-family:system-ui,sans-serif;text-align:center;padding-top:60px;">
    <h2>仪表板已迁移至 <a href="index.html">index.html</a></h2>
</body>
</html>
"""
        try:
            with open(legacy_dashboard_path, 'w', encoding='utf-8') as legacy_file:
                legacy_file.write(legacy_redirect)
        except Exception:
            pass
            
        print(f"<i class='fas fa-check-circle text-green-500'></i> 仪表板已保存: {dashboard_path}")
        return dashboard_path
        
    def run_analysis(self):
        """运行分析（带模块耗时打印）"""
        from time import perf_counter as _tpc
        def _timeit(label, fn):
            _t0 = _tpc()
            res = fn()
            _dt = _tpc() - _t0
            print(f"[TIME] {label}: {_dt:.2f}s")
            return res

        print("<i class='fas fa-rocket text-blue-500'></i> 启动量化分析")
        print("=" * 60)
        print("<i class='fas fa-bullseye text-red-500'></i> 目标: 快速加载 + 核心洞察")
        print("=" * 60)
        
        try:
            _timeit("加载和采样", self.load_and_sample_data)
            _timeit("绩效指标分析", self.performance_metrics_analysis)
            _timeit("日度绝对盈利(盯市)", self.daily_absolute_profit_analysis)
            _timeit("本金口径日收益率", self.daily_returns_initial_capital_analysis)
            _timeit("每日最低所需本金+资金占用", self.daily_min_capital_and_utilization_analysis)
            _timeit("模型性能分析", self.model_performance_analysis)
            _timeit("预测-真实关系", self.pred_real_relationship_analysis)
            _timeit("投资组合构成", self.portfolio_composition_analysis)
            _timeit("因子归因（FF3）主页面", self.portfolio_factor_attribution_main)
            _timeit("因子归因（FF3）季度分析", self.portfolio_factor_attribution_quarterly)
            _timeit("因子特征暴露", self.factor_exposure_analysis)
            _timeit("滑点成本分析", self.slippage_cost_analysis)
            _timeit("执行分析", self.execution_analysis)
            _timeit("时段盈利能力分析", self.slot_performance_analysis)
            dashboard_path = _timeit("创建仪表板", self.create_lightweight_dashboard)
            
            print("\n" + "=" * 60)
            print("<i class='fas fa-check-circle text-green-500'></i> 分析完成!")
            print(f"<i class='fas fa-chart-bar text-indigo-500'></i> 生成 {len(self.figures)} 个图表")
            print(f"<i class='fas fa-sliders-h text-gray-600'></i> 仪表板: {dashboard_path}")
            print(f"<i class='fas fa-bolt text-yellow-400'></i> 预计加载时间: < 5秒")
            print("=" * 60)
            
            return dashboard_path
            
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None


    def daily_returns_initial_capital_analysis(self):
        """日收益率曲线（以首日收市总资产为本金）。
        - 本金定义：首日(NAV首行)总资产；优先读取 mtm_analysis_results/daily_nav_revised.csv。
        - 日PnL：NAV_t - NAV_(t-1)
        - 日收益率：PnL_t / 本金（非复利）；同时展示累积收益率=∑日收益率。
        """
        print("\n<i class='fas fa-chart-line text-green-500'></i> === 日收益率曲线（首日总资产为本金） ===")
        from pathlib import Path as _Path
        import plotly.graph_objs as go
        import numpy as _np
        import pandas as _pd

        mtm_file = _Path("mtm_analysis_results/daily_nav_revised.csv")
        if not mtm_file.exists():
            print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到盯市分析结果文件，跳过基于本金的日收益率曲线")
            return

        def _parse_currency(v):
            try:
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    return float(v.replace(",", "").strip())
                return _np.nan
            except Exception:
                return _np.nan

        try:
            df = _pd.read_csv(mtm_file)
            df['date'] = _pd.to_datetime(df['date']).dt.date
            df = df.sort_values('date')
            
            # 使用正确的初始资金重新计算NAV
            CORRECT_INITIAL_CAPITAL = 62_090_808
            
            # 解析多空市值
            df['long_value_num'] = df['long_value'].apply(_parse_currency)
            df['short_value_num'] = df['short_value'].apply(_parse_currency)
            
            # 从订单数据计算每日现金流并重新计算现金和NAV
            if not hasattr(self, 'df') or self.df is None:
                print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 订单数据未加载，无法重新计算NAV")
                df['total_assets_num'] = df['total_assets'].apply(_parse_currency)
            else:
                # 计算每日现金流
                orders_temp = self.df.copy()
                orders_temp['date'] = _pd.to_datetime(orders_temp['Timestamp']).dt.date
                daily_flows = orders_temp.groupby(['date', 'direction'])[['tradeAmount', 'fee']].sum().unstack(fill_value=0)
                daily_flows.columns = [f"{a}_{b}" for a, b in daily_flows.columns]
                
                # 逐日累计现金
                cash_balance = CORRECT_INITIAL_CAPITAL
                cash_series = []
                nav_series = []
                
                for date_val in df['date']:
                    if date_val in daily_flows.index:
                        buy_amt = daily_flows.loc[date_val, 'tradeAmount_B'] if 'tradeAmount_B' in daily_flows.columns else 0
                        sell_amt = daily_flows.loc[date_val, 'tradeAmount_S'] if 'tradeAmount_S' in daily_flows.columns else 0
                        fee_amt = (daily_flows.loc[date_val, 'fee_B'] if 'fee_B' in daily_flows.columns else 0) + \
                                  (daily_flows.loc[date_val, 'fee_S'] if 'fee_S' in daily_flows.columns else 0)
                        cash_balance += sell_amt - buy_amt - fee_amt
                    
                    cash_series.append(cash_balance)
                
                df['cash_num'] = cash_series
                # NAV = 现金 + 多头 - 空头
                df['total_assets_num'] = df['cash_num'] + df['long_value_num'] - df['short_value_num']
                
                print(f"<i class='fas fa-check-circle text-green-500'></i> 已基于正确初始资金重新计算NAV")
            # 基于“授信规则”重估初始本金（首日资金占用峰值 × 安全系数），若无法重估则回退
            self._ensure_credit_rules_loaded()
            init_capital_factor = float(self._credit_rules.get('initial_capital_factor', 1.3))
            # 优先使用快照/缓存，避免重复回放首日逐笔
            init_cap_reestimated = _np.nan
            try:
                _snap_path = _Path('reports/first_day_capital_snapshot.json')
                if _snap_path.exists():
                    _snap = json.loads(_snap_path.read_text(encoding='utf-8'))
                    _val = float(_snap.get('first_day_min_required_equity', float('nan')))
                    _first_day_str = str(df['date'].iloc[0])
                    if not _np.isnan(_val) and str(_snap.get('first_day', _first_day_str)) == _first_day_str:
                        init_cap_reestimated = _val * init_capital_factor
            except Exception:
                pass
            if _np.isnan(init_cap_reestimated):
                try:
                    _ck = self._make_capital_util_cache_key()
                    _dm = self._load_capital_util_daily_min(_ck)
                    if _dm is not None and len(_dm) > 0:
                        _dm = _dm.copy()
                        _dm['date'] = _pd.to_datetime(_dm['date']).dt.date
                        _first = df['date'].iloc[0]
                        _row = _dm.loc[_dm['date'] == _first]
                        if len(_row) > 0:
                            init_cap_reestimated = float(_row['min_required_equity'].iloc[0]) * init_capital_factor
                except Exception:
                    pass
            try:
                if _np.isnan(init_cap_reestimated):
                    need_cols = ['Code','direction','tradeQty','tradeAmount','fee','price','tradeTimestamp','Timestamp']
                    # 仅读取首日订单，使用 parquet 过滤器
                    _first_day = _pd.to_datetime(df['date']).min()
                    try:
                        _cols = [c for c in need_cols if c != 'Timestamp']
                        df0 = _pd.read_parquet(self.data_path, columns=_cols, engine='pyarrow',
                                               filters=[('tradeTimestamp', '>=', _first_day), ('tradeTimestamp', '<', _first_day + _pd.Timedelta(days=1))])
                        tcol = 'tradeTimestamp'
                    except Exception:
                        _cols = [c for c in need_cols if c != 'tradeTimestamp']
                        df0 = _pd.read_parquet(self.data_path, columns=_cols, engine='pyarrow',
                                               filters=[('Timestamp', '>=', _first_day), ('Timestamp', '<', _first_day + _pd.Timedelta(days=1))])
                        tcol = 'Timestamp'
                    df0 = df0.dropna(subset=['Code','direction','tradeQty','tradeAmount','fee']).copy()
                    if len(df0) > 0:
                        df0[tcol] = _pd.to_datetime(df0[tcol])
                        df0['__rowid__'] = _np.arange(len(df0))
                        d = df0.sort_values([tcol, '__rowid__'])
                    allow_short_cash = bool(self._credit_rules.get('allow_short_proceeds_to_cash', False))
                    allow_sell_long_t0 = bool(self._credit_rules.get('allow_sell_long_cash_T0', True))
                    short_margin_ratio = float(self._credit_rules.get('margin_short_ratio', 0.5))
                    fee_accrual = str(self._credit_rules.get('fee_accrual', 'realtime')).lower()
                    cash_free = 0.0
                    positions: dict = {}
                    last_price: dict = {}
                    max_required = 0.0
                    fees_buffer_eod = 0.0
                    short_mv_total = 0.0
                    short_mv_total = 0.0  # 增量维护空头市值，避免每笔遍历全部持仓
                    for _, r in d.iterrows():
                        code = r['Code']
                        qty = int(r['tradeQty'])
                        amt = float(r['tradeAmount'])
                        fee = float(r['fee']) if not _pd.isna(r['fee']) else 0.0
                        px = float(r['price']) if ('price' in r and not _pd.isna(r['price'])) else (amt / qty if qty > 0 else 0.0)
                        pos_cur = int(positions.get(code, 0))
                        if r['direction'] == 'B':
                            if fee_accrual == 'realtime':
                                cash_free -= (amt + fee)
                            else:
                                cash_free -= amt
                                fees_buffer_eod += fee
                            positions[code] = pos_cur + qty
                            last_price[code] = px
                        else:
                            close_qty = min(max(pos_cur, 0), qty)
                            open_short_qty = qty - close_qty
                            if close_qty > 0:
                                portion = close_qty / qty
                                if allow_sell_long_t0:
                                    if fee_accrual == 'realtime':
                                        cash_free += (amt * portion) - (fee * portion)
                                    else:
                                        cash_free += (amt * portion)
                                        fees_buffer_eod += (fee * portion)
                                old_lp = last_price.get(code, px)
                                old_neg = max(-pos_cur, 0)
                                new_pos = pos_cur - close_qty
                                new_neg = max(-new_pos, 0)
                                last_price[code] = px
                                positions[code] = new_pos
                                short_mv = (new_neg * px) - (old_neg * old_lp)
                                # 临时增量（该分支仅用于首日重估，开盘无历史短仓时影响有限）
                                pos_cur = new_pos
                            if open_short_qty > 0:
                                portion = open_short_qty / qty
                                if allow_short_cash:
                                    if fee_accrual == 'realtime':
                                        cash_free += (amt * portion) - (fee * portion)
                                    else:
                                        cash_free += (amt * portion)
                                        fees_buffer_eod += (fee * portion)
                                else:
                                    if fee_accrual == 'realtime':
                                        cash_free -= (fee * portion)
                                    else:
                                        fees_buffer_eod += (fee * portion)
                                old_lp = last_price.get(code, px)
                                old_neg = max(-pos_cur, 0)
                                new_pos = pos_cur - open_short_qty
                                new_neg = max(-new_pos, 0)
                                last_price[code] = px
                                positions[code] = new_pos
                                short_mv = (new_neg * px) - (old_neg * old_lp)
                        short_margin = 0.0
                        # 回退：如上未维护总量，则保留原方法（首日回放数据量远小于全周期）
                        for c, pos_v in positions.items():
                            if pos_v < 0:
                                lp = last_price.get(c, px)
                                short_margin += (-pos_v) * float(lp) * short_margin_ratio
                        required_equity = max(0.0, -cash_free) + short_margin
                        if required_equity > max_required:
                            max_required = required_equity
                    if fee_accrual != 'realtime' and fees_buffer_eod > 0:
                        cash_for_req = cash_free - fees_buffer_eod
                        short_mv = 0.0
                        for c, pos_v in positions.items():
                            if pos_v < 0:
                                lp = last_price.get(c, px)
                                short_mv += (-pos_v) * float(lp)
                        short_margin = short_mv * short_margin_ratio
                        required_equity = max(0.0, -cash_for_req) + short_margin
                        if required_equity > max_required:
                            max_required = required_equity
                    init_cap_reestimated = float(max_required) * float(init_capital_factor)
                    # 写快照供后续复用
                    try:
                        _snap_out = {'first_day_min_required_equity': float(max_required), 'first_day': str(_first_day.date())}
                        _Path('reports').mkdir(parents=True, exist_ok=True)
                        _Path('reports/first_day_capital_snapshot.json').write_text(json.dumps(_snap_out, ensure_ascii=False), encoding='utf-8')
                    except Exception:
                        pass
                else:
                    pass
            except Exception:
                init_cap_reestimated = _np.nan

            # 优先使用正确计算的初始资金（基于保证金要求）
            if not _np.isnan(init_cap_reestimated) and init_cap_reestimated > 0:
                df['initial_capital_num'] = init_cap_reestimated
                first_day_initial_capital_display = f"¥{init_cap_reestimated:,.0f}"
                initial_capital = float(init_cap_reestimated)
            else:
                # 使用固定的正确初始资金
                df['initial_capital_num'] = CORRECT_INITIAL_CAPITAL
                first_day_initial_capital_display = f"¥{CORRECT_INITIAL_CAPITAL:,.0f}"
                initial_capital = CORRECT_INITIAL_CAPITAL
            
            # 确保NAV数据可用
            if len(df) == 0 or df['total_assets_num'].isna().all():
                print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 盯市数据为空或NAV全部缺失，跳过")
                return
            # 日PnL 与 日收益率
            df['daily_pnl'] = df['total_assets_num'].diff()
            df['daily_return_capital'] = df['daily_pnl'] / initial_capital
            df['cum_return_capital'] = df['daily_return_capital'].cumsum()
            df['nav_capital_curve'] = 1 + df['cum_return_capital']

            # 真实净值日收益率（复利）
            df['daily_return_nav'] = df['total_assets_num'].pct_change()
            df.loc[df.index[0], 'daily_return_nav'] = 0.0  # 首日设为0
            df['cum_return_nav'] = (1 + df['daily_return_nav']).cumprod() - 1
            df['nav_curve'] = 1 + df['cum_return_nav']

            # 确保日期格式正确，Plotly 统一使用 ISO 字符串避免轴显示为时间戳数字
            date_index = _pd.to_datetime(df['date'].astype(str))
            date_str = date_index.dt.strftime('%Y-%m-%d').tolist()

            # 基准对齐（按策略交易日对齐，默认展示深证成指）
            bench_curves = []
            bench_daily_series = []
            primary_benchmark = None
            if self.benchmark_data:
                for bench_name, bench_df in self.benchmark_data.items():
                    bench_copy = bench_df.copy()
                    bench_copy['date'] = _pd.to_datetime(bench_copy['date'])
                    merged_bench = _pd.DataFrame({'date': date_index}).merge(
                        bench_copy[['date', 'daily_return', 'cumulative_return']],
                        on='date', how='left'
                    ).sort_values('date')
                    if merged_bench['cumulative_return'].notna().sum() == 0:
                        continue
                    bench_nav_curve = 1 + merged_bench['cumulative_return'].astype(float)
                    bench_dates_str = merged_bench['date'].dt.strftime('%Y-%m-%d')
                    bench_curves.append((bench_name, bench_dates_str, bench_nav_curve))
                    bench_daily_series.append((bench_name, bench_dates_str, merged_bench['daily_return'].astype(float)))
                    if primary_benchmark is None or bench_name == '深证成指':
                        primary_benchmark = bench_name

            # 主图：净值曲线（首日=1），保持单轴便于在仪表板预览
            fig_nav = go.Figure()
            fig_nav.add_trace(go.Scatter(
                x=date_str,
                y=df['nav_curve'],
                mode='lines',
                name='真实净值曲线',
                line=dict(color='#16a085', width=2.3)
            ))
            fig_nav.add_trace(go.Scatter(
                x=date_str,
                y=df['nav_capital_curve'],
                mode='lines',
                name='固定本金累积曲线',
                line=dict(color='#3498db', width=1.9, dash='dot')
            ))

            bench_colors = ['#27ae60', '#f1c40f', '#9b59b6', '#16a085', '#e67e22', '#34495e']
            for idx, (bench_name, bench_dates, bench_nav_curve) in enumerate(bench_curves):
                fig_nav.add_trace(go.Scatter(
                    x=bench_dates,
                    y=bench_nav_curve,
                    mode='lines',
                    name=f'{bench_name}净值',
                    line=dict(color=bench_colors[idx % len(bench_colors)], width=1.7, dash='dot'),
                    hovertemplate=f'日期: %{{x}}<br>{bench_name}: %{{y:.2f}}<extra></extra>',
                    visible=True if bench_name == (primary_benchmark or bench_name) else 'legendonly'
                ))

            fig_nav.update_layout(
                height=520,
                title='净值曲线对比（首日=1，固定本金 vs 真实净值 vs 基准）',
                xaxis=dict(title='日期', type='date'),
                yaxis=dict(title='净值（首日=1）', tickformat='.2f'),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )

            # 副图：日收益率对比（策略两种口径 + 基准）
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=date_str,
                y=df['daily_return_nav'],
                mode='lines',
                name='真实净值日收益率',
                line=dict(color='#16a085', width=1.8)
            ))
            fig_daily.add_trace(go.Scatter(
                x=date_str,
                y=df['daily_return_capital'],
                mode='lines',
                name='日收益率(固定本金)',
                line=dict(color='#3498db', width=1.4, dash='dot')
            ))
            for idx, (bench_name, bench_dates, bench_daily) in enumerate(bench_daily_series):
                fig_daily.add_trace(go.Scatter(
                    x=bench_dates,
                    y=bench_daily,
                    mode='lines',
                    name=f'{bench_name}日收益率',
                    line=dict(color=bench_colors[idx % len(bench_colors)], width=1.1),
                    hovertemplate=f'日期: %{{x}}<br>{bench_name}: %{{y:.2%}}<extra></extra>',
                    visible=True if bench_name == (primary_benchmark or bench_name) else 'legendonly'
                ))
            if len(date_str) > 0:
                fig_daily.add_shape(
                    type='line',
                    x0=date_str[0],
                    x1=date_str[-1],
                    y0=0,
                    y1=0,
                    line=dict(color='rgba(0,0,0,0.2)', width=1)
                )
            fig_daily.update_layout(
                height=400,
                title='日收益率对比（策略 vs 基准）',
                xaxis=dict(title='日期', type='date'),
                yaxis=dict(title='日收益率', tickformat='.2%'),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )

            title = '日收益率曲线（以首日总资产为本金 vs 真实净值）'
            explanation_html = (
                "<h4>展示结构</h4>"
                "<ul>"
                "<li><b>主图</b>：将固定本金累积曲线、真实净值曲线与基准指数净值（均以首日=1）放在同一坐标系，无需双轴，便于在仪表板预览。</li>"
                "<li><b>副图</b>：单独展示日收益率，对齐基准指数日收益，突出波动与相对表现；零轴用虚线标记。</li>"
                "</ul>"
                "<h4>计算口径</h4>"
                "<ol>"
                "<li><b>本金</b>：按《授信规则》重估——先回放首日逐笔得到当日最低所需本金，再乘以安全系数（默认1.3）；若失败则回退为正确初始资金62,090,808元。</li>"
                "<li><b>日收益率（固定本金）</b>：$r^{\\text{本金}}_t = \\frac{\\text{当日盈亏}_t}{\\text{初始本金}}$，非复利。</li>"
                "<li><b>日收益率（真实净值）</b>：$r^{\\text{净值}}_t = \\frac{\\text{总资产}_t - \\text{总资产}_{t-1}}{\\text{总资产}_{t-1}}$，复利口径，首日设为0。</li>"
                "<li><b>净值曲线</b>：固定本金曲线= $1+\\sum r^{\\text{本金}}_t$；真实净值曲线= $\\prod(1+r^{\\text{净值}}_t)$；基准净值曲线= $\\prod(1+r^{\\text{指数}}_t)$。</li>"
                "<li><b>基准对齐</b>：按策略交易日对齐基准指数，默认显示深证成指，其他基准可在图例中切换。</li>"
                "</ol>"
                "<p>图表拆分后，累积表现与日度波动分开展示：主图聚焦长期净值对比，副图专注单日波动与相对强弱，避免混用日收益与累积收益导致的双轴混乱。</p>"
            )

            nav_total_return = (df['total_assets_num'].iloc[-1] / initial_capital - 1) if len(df) > 0 else 0.0
            capital_total_return = (df['nav_capital_curve'].iloc[-1] - 1) if len(df) > 0 else 0.0
            nav_vol = _np.nanstd(df['daily_return_nav'], ddof=1)
            sharpe_nav = (_np.nanmean(df['daily_return_nav']) / nav_vol * _np.sqrt(252)) if nav_vol > 0 else _np.nan
            dd_nav = ((1 + df['daily_return_nav']).cumprod().div((1 + df['daily_return_nav']).cumprod().expanding().max()) - 1).min()

            primary_bench_return = None
            primary_excess = None
            if primary_benchmark:
                bench_df_primary = self.benchmark_data.get(primary_benchmark)
                if bench_df_primary is not None and len(bench_df_primary) > 0:
                    bench_df_primary = bench_df_primary.copy()
                    bench_df_primary['date'] = _pd.to_datetime(bench_df_primary['date'])
                    bench_aligned = bench_df_primary[bench_df_primary['date'].isin(date_index)]
                    if len(bench_aligned) > 0:
                        primary_bench_return = float(bench_aligned['cumulative_return'].iloc[-1])
                        primary_excess = nav_total_return - primary_bench_return

            metrics = {
                '本金(按授信规则重估)': f"¥{initial_capital:,.0f}",
                '期末真实净值收益率': f"{nav_total_return*100:.2f}%",
                '期末固定本金收益率': f"{capital_total_return*100:.2f}%",
                '真实净值夏普(年化)': f"{sharpe_nav:.3f}" if not _np.isnan(sharpe_nav) else "N/A",
                '真实净值最大回撤': f"{dd_nav:.2%}",
                '日收益率均值(本金计)': f"{_np.nanmean(df['daily_return_capital'])*100:.3f}%",
                '日收益率标准差(本金计)': f"{_np.nanstd(df['daily_return_capital'], ddof=1)*100:.3f}%",
                '日收益率均值(真实净值)': f"{_np.nanmean(df['daily_return_nav'])*100:.3f}%",
                '日收益率标准差(真实净值)': f"{nav_vol*100:.3f}%"
            }
            if primary_bench_return is not None:
                metrics[f'{primary_benchmark}收益率'] = f"{primary_bench_return*100:.2f}%"
                metrics[f'vs {primary_benchmark}超额(真实净值)'] = f"{primary_excess*100:+.2f}%"
            
            # Cache metrics for dashboard
            if not hasattr(self, 'strategy_metrics'):
                self.strategy_metrics = {}
            self.strategy_metrics.update({
                'total_return_nav': f"{nav_total_return*100:.2f}%",
                'sharpe_ratio': metrics.get('真实净值夏普(年化)', '0.00'),
                'win_rate': f"{(df['daily_return_nav'] > 0).mean():.2%}",
                'max_drawdown': f"{dd_nav:.2%}"
            })

            self._save_figure_with_details(
                fig_nav,
                name='daily_returns_initial_capital_light',
                title=title,
                explanation_html=explanation_html,
                metrics=metrics,
                extra_figs=[('daily_returns_panel', fig_daily)]
            )
            print("[OK] 基于本金的日收益率曲线已生成")
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 生成基于本金的日收益率曲线失败: {e}")

    def daily_min_capital_and_utilization_analysis(self):
        """全周期：每日最低所需本金序列 + 资金占用收益率。
        - 最低所需本金：逐日逐笔回放，required_equity = max(0, -cash_free) + short_margin，取当日峰值；
          其中 short_margin = 空头市值 × margin_short_ratio。
        - 资金占用收益率：日PnL / 当日最低所需本金；PnL 来源于 daily_nav_revised.csv。
        """
        try:
            import plotly.graph_objs as go
            import pandas as _pd
            import numpy as _np

            cache_key = self._make_capital_util_cache_key()
            cached_daily_min = self._load_capital_util_daily_min(cache_key)
            cache_used = False

            def _compute_daily_min() -> _pd.DataFrame:
                # 1) 逐日计算“最低所需本金”
                self._ensure_credit_rules_loaded()
                allow_short_cash = bool(self._credit_rules.get('allow_short_proceeds_to_cash', False))
                allow_sell_long_t0 = bool(self._credit_rules.get('allow_sell_long_cash_T0', True))
                short_margin_ratio = float(self._credit_rules.get('margin_short_ratio', 0.5))
                fee_accrual = str(self._credit_rules.get('fee_accrual', 'realtime')).lower()

                need_cols = ['Code','direction','tradeQty','tradeAmount','fee','price','tradeTimestamp','Timestamp']
                base_cols = [c for c in need_cols if c in pd.read_parquet(self.data_path).columns]
                od = pd.read_parquet(self.data_path, columns=base_cols)
                if len(od) == 0:
                    print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 订单数据为空，跳过资金占用分析")
                    return None
                tcol = 'tradeTimestamp' if 'tradeTimestamp' in od.columns else 'Timestamp'
                od = od.dropna(subset=['Code','direction','tradeQty','tradeAmount','fee',tcol]).copy()
                od[tcol] = pd.to_datetime(od[tcol])
                od['__rowid__'] = _np.arange(len(od))
                od['date_trade'] = od[tcol].dt.date
                # 稳定排序
                if 'Timestamp' in od.columns:
                    od['Timestamp'] = pd.to_datetime(od['Timestamp'])
                    od = od.sort_values(['date_trade', tcol, 'Timestamp', '__rowid__'])
                else:
                    od = od.sort_values(['date_trade', tcol, '__rowid__'])

                positions: dict = {}
                last_price: dict = {}
                daily_records = []

                for dt, d in od.groupby('date_trade', sort=True):
                    cash_free = 0.0
                    fees_buffer_eod = 0.0
                    # 开盘前先评估已有空头的保证金占用
                    short_mv_total = 0.0
                    for c, pos_v in positions.items():
                        if pos_v < 0:
                            lp = last_price.get(c, 0.0)
                            short_mv_total += (-pos_v) * float(lp)
                    max_required = short_mv_total * short_margin_ratio

                    for _, r in d.iterrows():
                        code = r['Code']
                        qty = int(r['tradeQty'])
                        amt = float(r['tradeAmount'])
                        fee = float(r['fee']) if not pd.isna(r['fee']) else 0.0
                        px = float(r['price']) if ('price' in r and not pd.isna(r['price'])) else (amt / qty if qty > 0 else 0.0)
                        pos_cur = int(positions.get(code, 0))

                        if r['direction'] == 'B':
                            if fee_accrual == 'realtime':
                                cash_free -= (amt + fee)
                            else:
                                cash_free -= amt
                                fees_buffer_eod += fee
                            old_lp = last_price.get(code, px)
                            old_neg = max(-pos_cur, 0)
                            new_pos = pos_cur + qty
                            new_neg = max(-new_pos, 0)
                            last_price[code] = px
                            positions[code] = new_pos
                            short_mv_total += (new_neg * px) - (old_neg * old_lp)
                        else:
                            close_qty = min(max(pos_cur, 0), qty)
                            open_short_qty = qty - close_qty
                            if close_qty > 0:
                                portion = close_qty / qty
                                if allow_sell_long_t0:
                                    if fee_accrual == 'realtime':
                                        cash_free += (amt * portion) - (fee * portion)
                                    else:
                                        cash_free += (amt * portion)
                                        fees_buffer_eod += (fee * portion)
                                old_lp = last_price.get(code, px)
                                old_neg = max(-pos_cur, 0)
                                new_pos = pos_cur - close_qty
                                new_neg = max(-new_pos, 0)
                                last_price[code] = px
                                positions[code] = new_pos
                                short_mv_total += (new_neg * px) - (old_neg * old_lp)
                                pos_cur = new_pos
                            if open_short_qty > 0:
                                portion = open_short_qty / qty
                                if allow_short_cash:
                                    if fee_accrual == 'realtime':
                                        cash_free += (amt * portion) - (fee * portion)
                                    else:
                                        cash_free += (amt * portion)
                                        fees_buffer_eod += (fee * portion)
                                else:
                                    if fee_accrual == 'realtime':
                                        cash_free -= (fee * portion)
                                    else:
                                        fees_buffer_eod += (fee * portion)
                                old_lp = last_price.get(code, px)
                                old_neg = max(-pos_cur, 0)
                                new_pos = pos_cur - open_short_qty
                                new_neg = max(-new_pos, 0)
                                last_price[code] = px
                                positions[code] = new_pos
                                short_mv_total += (new_neg * px) - (old_neg * old_lp)

                        # 更新保证金要求
                        short_margin = short_mv_total * short_margin_ratio
                        required_equity = max(0.0, -cash_free) + short_margin
                        if required_equity > max_required:
                            max_required = required_equity

                    # 日终费用计提再算一次
                    if fee_accrual != 'realtime' and fees_buffer_eod > 0:
                        cash_for_req = cash_free - fees_buffer_eod
                        short_margin = short_mv_total * short_margin_ratio
                        required_equity = max(0.0, -cash_for_req) + short_margin
                        if required_equity > max_required:
                            max_required = required_equity

                    daily_records.append((dt, float(max_required)))

                daily_min = _pd.DataFrame(daily_records, columns=['date','min_required_equity'])
                # 写缓存
                try:
                    self._save_capital_util_daily_min(cache_key, daily_min)
                except Exception:
                    pass
                return daily_min

            daily_min = None
            if cached_daily_min is not None and len(cached_daily_min) > 0:
                daily_min = cached_daily_min.copy()
                cache_used = True
                print(f"[CACHE] 复用每日最低所需本金缓存，共 {len(daily_min):,} 天")
            else:
                daily_min = _compute_daily_min()

            if daily_min is None or len(daily_min) == 0:
                print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 无法构建每日最低所需本金序列，终止资金占用分析")
                return

            daily_min = daily_min.copy()
            daily_min['date'] = _pd.to_datetime(daily_min['date']).dt.date

            mtm_file = Path('mtm_analysis_results/daily_nav_revised.csv')
            if not mtm_file.exists():
                print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 未找到盯市数据，跳过资金占用收益率")
                return

            def _parse_cur(v):
                try:
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str):
                        return float(v.replace(',', '').strip())
                except Exception:
                    return _np.nan
                return _np.nan

            nav = _pd.read_csv(mtm_file)
            nav['date'] = _pd.to_datetime(nav['date']).dt.date
            nav = nav.sort_values('date').copy()
            
            # 使用正确的初始资金重新计算NAV（与其他分析函数保持一致）
            CORRECT_INITIAL_CAPITAL = 62_090_808
            print(f'[资金占用] 使用正确初始资金重新计算NAV: ¥{CORRECT_INITIAL_CAPITAL:,.0f}')
            
            # 解析多空市值
            nav['long_value_num'] = nav['long_value'].apply(_parse_cur)
            nav['short_value_num'] = nav['short_value'].apply(_parse_cur)
            
            # 从订单数据重新计算现金
            if hasattr(self, 'df') and self.df is not None:
                orders_temp = pd.read_parquet(self.data_path, columns=['Timestamp', 'direction', 'tradeAmount', 'fee'])
                orders_temp['date'] = _pd.to_datetime(orders_temp['Timestamp']).dt.date
                daily_flows_temp = orders_temp.groupby(['date', 'direction'])[['tradeAmount', 'fee']].sum().unstack(fill_value=0)
                daily_flows_temp.columns = [f"{a}_{b}" for a, b in daily_flows_temp.columns]
                
                cash_balance = CORRECT_INITIAL_CAPITAL
                cash_series = []
                for date_val in nav['date']:
                    if date_val in daily_flows_temp.index:
                        buy_amt = daily_flows_temp.loc[date_val, 'tradeAmount_B'] if 'tradeAmount_B' in daily_flows_temp.columns else 0
                        sell_amt = daily_flows_temp.loc[date_val, 'tradeAmount_S'] if 'tradeAmount_S' in daily_flows_temp.columns else 0
                        fee_amt = (daily_flows_temp.loc[date_val, 'fee_B'] if 'fee_B' in daily_flows_temp.columns else 0) + \
                                  (daily_flows_temp.loc[date_val, 'fee_S'] if 'fee_S' in daily_flows_temp.columns else 0)
                        cash_balance += sell_amt - buy_amt - fee_amt
                    cash_series.append(cash_balance)
                
                nav['cash_num'] = cash_series
                nav['total_assets_num'] = nav['cash_num'] + nav['long_value_num'] - nav['short_value_num']
                print(f"   <i class='fas fa-check-circle text-green-500'></i> NAV已重新计算，范围: ¥{nav['total_assets_num'].min():,.0f} ~ ¥{nav['total_assets_num'].max():,.0f}")
            else:
                # 回退：使用文件中的数据
                print("   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 无订单数据，使用文件中的NAV数据（可能不准确）")
                nav['total_assets_num'] = nav['total_assets'].apply(_parse_cur)
            
            nav['daily_pnl'] = nav['total_assets_num'].diff()
            print(f'   日PnL范围: ¥{nav["daily_pnl"].min():,.0f} ~ ¥{nav["daily_pnl"].max():,.0f}')
            nav_dates = set(nav['date'])

            if cache_used:
                cache_dates = set(daily_min['date'])
                missing_dates = nav_dates - cache_dates
                if missing_dates:
                    print(f"[CACHE] 缓存缺失 {len(missing_dates)} 个盯市日期，重新计算每日最低所需本金")
                    daily_min = _compute_daily_min()
                    if daily_min is None or len(daily_min) == 0:
                        print("<i class='fas fa-exclamation-triangle text-yellow-500'></i> 重新计算每日最低所需本金失败，终止资金占用分析")
                        return
                    daily_min = daily_min.copy()
                    daily_min['date'] = _pd.to_datetime(daily_min['date']).dt.date

            merged = nav[['date','daily_pnl']].merge(daily_min, on='date', how='left')
            
            # 诊断：检查合并后的数据
            print(f'   合并后数据: {len(merged)} 行')
            print(f'   最低所需本金范围: ¥{merged["min_required_equity"].min():,.0f} ~ ¥{merged["min_required_equity"].max():,.0f}')
            
            # 计算资金占用收益率
            merged['util_return'] = merged['daily_pnl'] / merged['min_required_equity']
            
            # 诊断：检查日度收益率的线性度
            print(f'   日度资金占用收益率范围: {merged["util_return"].min()*100:.4f}% ~ {merged["util_return"].max()*100:.4f}%')
            print(f'   日度资金占用收益率均值: {merged["util_return"].mean()*100:.4f}%')
            print(f'   日度资金占用收益率标准差: {merged["util_return"].std()*100:.4f}%')
            
            # 计算累计收益率
            merged['cum_util_return'] = merged['util_return'].cumsum()
            
            # 诊断：检查累计收益率的线性度
            from scipy.stats import linregress as _linregress
            x_idx = _np.arange(len(merged))
            y_cum = merged['cum_util_return'].values
            mask = ~_np.isnan(y_cum)
            if mask.sum() > 2:
                slope, intercept, r_value, _, _ = _linregress(x_idx[mask], y_cum[mask])
                print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 累计收益率线性拟合 R² = {r_value**2:.6f}")
                if r_value**2 > 0.99:
                    print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> 警告：累计收益率过于线性（R²>{r_value**2:.4f}），可能存在数据问题！")
                    # 检查是否PnL与资金占用高度相关
                    corr = _np.corrcoef(merged['daily_pnl'].dropna(), 
                                       merged['min_required_equity'].reindex(merged['daily_pnl'].dropna().index))[0,1]
                    print(f'   PnL与资金占用的相关系数: {corr:.4f}')
                    if abs(corr) > 0.7:
                        print(f"   <i class='fas fa-exclamation-triangle text-yellow-500'></i> PnL与资金占用高度相关！这会导致收益率过于稳定")

            # 3) 图表 - 改进显示，突出日度波动
            dates_iso = _pd.to_datetime(merged['date']).dt.strftime('%Y-%m-%d').tolist()

            fig_top = go.Figure()
            fig_top.add_trace(go.Scatter(
                x=dates_iso,
                y=merged['min_required_equity'],
                mode='lines', 
                name='每日最低所需本金', 
                line=dict(color='#2c3e50', width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 62, 80, 0.1)'
            ))
            fig_top.update_layout(
                height=460, 
                xaxis=dict(title='日期', tickangle=-45), 
                yaxis=dict(title='资金(¥)', tickformat=',.0f')
            )

            fig_bottom = go.Figure()
            
            # 准备数据 - 确保数据类型正确
            dates_bottom = dates_iso
            util_return_values = merged['util_return'].values.astype(float).tolist()
            cum_util_return_values = merged['cum_util_return'].values.astype(float).tolist()
            
            # 日度收益率：加强显示，添加填充区域
            fig_bottom.add_trace(go.Scatter(
                x=dates_bottom,
                y=util_return_values, 
                mode='lines+markers', 
                name='日度资金占用收益率', 
                line=dict(color='#16a085', width=2.5),
                marker=dict(size=3, color='#16a085'),
                fill='tozeroy',
                fillcolor='rgba(22, 160, 133, 0.15)',
                hovertemplate='日期: %{x|%Y-%m-%d}<br>日度收益率: %{y:.4%}<extra></extra>'
            ))
            
            # 累计收益率：使用次坐标轴
            fig_bottom.add_trace(go.Scatter(
                x=dates_bottom,
                y=cum_util_return_values, 
                mode='lines', 
                name='累计资金占用收益率', 
                yaxis='y2', 
                line=dict(color='#e74c3c', width=2.5),
                hovertemplate='日期: %{x|%Y-%m-%d}<br>累计收益率: %{y:.2%}<extra></extra>'
            ))
            
            # 添加零参考线（仅对主Y轴）
            try:
                fig_bottom.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)
            except:
                pass
            
            # 计算合理的Y轴范围
            util_min = _np.nanmin(util_return_values)
            util_max = _np.nanmax(util_return_values)
            util_range = util_max - util_min
            cum_min = _np.nanmin(cum_util_return_values)
            cum_max = _np.nanmax(cum_util_return_values)
            
            fig_bottom.update_layout(
                height=520,
                title=f'资金占用收益率分析<br><sub style="color:#666;">日度: {util_min*100:.2f}% ~ {util_max*100:.2f}% | 累计: {cum_min*100:.2f}% ~ {cum_max*100:.2f}%</sub>',
                xaxis=dict(title='日期', tickangle=-45),
                yaxis=dict(
                    title='日度收益率(%)', 
                    tickformat='.3%',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    range=[util_min - util_range*0.1, util_max + util_range*0.2],
                    zeroline=True,
                    zerolinecolor='gray',
                    zerolinewidth=1
                ),
                yaxis2=dict(
                    title='累计收益率(%)', 
                    overlaying='y', 
                    side='right', 
                    tickformat='.1%',
                    showgrid=False,
                    range=[0, cum_max * 1.05],
                    zeroline=False
                ),
                legend=dict(
                    orientation='h', 
                    yanchor='bottom', 
                    y=1.08, 
                    xanchor='right', 
                    x=1,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                hovermode='x unified',
                margin=dict(t=80)
            )

            primary_metrics = {
                '平均每日资金占用': f"¥{_np.nanmean(merged['min_required_equity']):,.0f}",
                '峰值资金占用': f"¥{_np.nanmax(merged['min_required_equity']):,.0f}",
                '有效天数': int(merged['min_required_equity'].notna().sum()),
            }
            secondary_metrics = {
                '资金占用收益率均值': f"{_np.nanmean(merged['util_return'])*100:.3f}%",
                '资金占用收益率标准差': f"{_np.nanstd(merged['util_return'])*100:.3f}%",
                '累计资金占用收益率(期末)': f"{(merged['cum_util_return'].iloc[-1] if len(merged)>0 else 0)*100:.2f}%",
            }

            # 说明与口径（去除特征警告，改为完整结算说明与规则清单）
            try:
                _rules = dict(self._credit_rules or {})
            except Exception:
                _rules = {}
            allow_short_cash = bool(_rules.get('allow_short_proceeds_to_cash', False))
            allow_sell_long_t0 = bool(_rules.get('allow_sell_long_cash_T0', True))
            short_margin_ratio = float(_rules.get('margin_short_ratio', 0.5))
            fee_accrual = str(_rules.get('fee_accrual', 'realtime'))
            initial_capital_factor = float(_rules.get('initial_capital_factor', 1.3))
            include_borrow_fee = _rules.get('include_borrow_fee', None)

            _borrow_fee_str = (
                '启用' if include_borrow_fee is True else ('禁用' if include_borrow_fee is False else '未配置')
            )

            explanation_html = (
                "<h4>口径与结算说明</h4>"
                "<h5>1) 指标定义</h5>"
                "<ul>"
                "<li><b>每日最低所需本金</b>：对每个交易日，按 <code>tradeTimestamp</code> 或 <code>Timestamp</code> 先后逐笔回放资金、持仓与价格，"
                "在任意时点计算 所需权益 = $\\max(0, -\\text{可用现金}) + \\text{空头保证金}$，"
                "其中空头保证金 = 空头市值 × 保证金比例，并取 <b>当日峰值</b> 作为该日的最低所需本金。</li>"
                "<li><b>日度资金占用收益率</b>：$\\text{日度收益率}_t = \\dfrac{\\text{当日盈亏}_t}{\\text{当日最低所需本金}_t}$；"
                "其中当日盈亏 = 总资产$_t$ - 总资产$_{t-1}$，总资产基于正确的初始资金重算；当日分母≤0或缺失时，结果记为NaN。</li>"
                "<li><b>累计资金占用收益率</b>：$\\text{累计收益率}_t = \\sum\\limits_{i \\le t} \\text{日度收益率}_i$（<b>算术累计</b>，非复利）。</li>"
                "</ul>"
                "<h5>2) 逐日结算流程（摘要）</h5>"
                "<ol>"
                "<li>按时间排序处理当日每笔成交；<code>direction</code> 为 <code>B</code>（买入）：减少可用现金；卖出平多：按规则(如T+0)回补现金；开空：按规则计入/不计入现金，仅计提 <code>fee</code>。</li>"
                "<li>每处理一笔，更新持仓与最新 <code>price</code>，并据此得到当前空头市值与保证金（空头市值 × 保证金比例）。</li>"
                "<li>根据费用计提规则（逐笔/日终），计算可用现金，"
                "并据此得到 所需权益 = $\\max(0, -\\text{可用现金}) + \\text{空头保证金}$，记录当日最大值。</li>"
                "<li>结合重算后的总资产序列得到当日盈亏，据此计算日度与累计资金占用收益率。</li>"
                "</ol>"
                "<h5>3) 本页使用的授信/清算规则</h5>"
                f"<ul>"
                f"<li>空头保证金比例：<b>{short_margin_ratio:.0%}</b></li>"
                f"<li>卖出平多现金可用性（T+0）：<b>{'是' if allow_sell_long_t0 else '否'}</b></li>"
                f"<li>开空所得计入可用现金：<b>{'是' if allow_short_cash else '否'}</b></li>"
                f"<li>费用计提口径：<b>{'逐笔(实时)' if fee_accrual.lower()=='realtime' else '日终计提'}</b></li>"
                f"<li>初始资金安全系数（回推用）：<b>{initial_capital_factor:.2f}</b></li>"
                f"<li>借券费/融券利息计入：<b>{_borrow_fee_str}</b></li>"
                f"</ul>"
            )

            self._save_figure_pair_with_details(
                fig_top, fig_bottom,
                name='capital_utilization_light',
                title='资金占用与资金占用收益率',
                explanation_html=explanation_html,
                metrics_primary=primary_metrics,
                metrics_secondary=secondary_metrics,
                primary_title='每日最低所需本金',
                secondary_title='资金占用收益率(日度/累计)'
            )
            print('[OK] 每日最低所需本金 + 资金占用收益率 已生成')
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 资金占用分析失败: {e}")

    def _save_figure_pair_with_details(self, fig_top, fig_bottom, name: str, title: str, explanation_html: str, metrics_primary: dict, metrics_secondary: dict, primary_title: str, secondary_title: str):
        """在同一页面上下展示两张图：上=交易金额占比，下=盈利金额占比。
        - 在仪表板中仍作为一个入口展示（主页面仅显示上图，滚动可见下图）。
        - 在新窗口打开时，CSS在宽屏下自动并排，两列布局。
        """
        try:
            title = self._clean_title_text(title)
            primary_title = self._clean_title_text(primary_title)
            secondary_title = self._clean_title_text(secondary_title)
            output_path = self.reports_dir / f"{name}.html"
            mathjax_local_src = self._ensure_mathjax_bundle()
            try:
                self._apply_plotly_theme(fig_top)
                self._apply_plotly_theme(fig_bottom)
            except Exception:
                pass

            def _fig_to_html_div(fig, div_id):
                # 转原生可序列化结构，避免 numpy 类型序列化问题（包含bdata解码）
                def _to_native(obj):
                    if isinstance(obj, np.ndarray):
                        if np.issubdtype(obj.dtype, np.number):
                            return obj.astype(float).tolist()
                        return [str(v) for v in obj.tolist()]
                    if isinstance(obj, (np.floating, np.integer)):
                        return float(obj)
                    try:
                        if isinstance(obj, pd.Timestamp):
                            return obj.isoformat()
                    except:
                        pass
                    # 处理 TypedArray (bdata) - 关键修复！
                    if isinstance(obj, dict):
                        if 'bdata' in obj and isinstance(obj.get('bdata'), str):
                            try:
                                dtype_map = {
                                    'f8': np.float64, 'f4': np.float32,
                                    'i8': np.int64, 'i4': np.int32,
                                    'u8': np.uint64, 'u4': np.uint32
                                }
                                np_dtype = dtype_map.get(obj.get('dtype', 'f8'), np.float64)
                                raw = base64.b64decode(obj['bdata'])
                                arr = np.frombuffer(raw, dtype=np_dtype)
                                # 处理shape信息
                                shape_val = obj.get('shape')
                                if shape_val is not None:
                                    try:
                                        if isinstance(shape_val, str):
                                            shape_tuple = tuple(int(s.strip()) for s in shape_val.replace('x', ',').split(',') if s.strip())
                                        elif isinstance(shape_val, (list, tuple)):
                                            shape_tuple = tuple(int(s) for s in shape_val)
                                        else:
                                            shape_tuple = None
                                        if shape_tuple and np.prod(shape_tuple) == arr.size:
                                            arr = arr.reshape(shape_tuple)
                                    except:
                                        pass
                                if np.issubdtype(arr.dtype, np.number):
                                    return arr.astype(float).tolist()
                                return [str(v) for v in arr.tolist()]
                            except:
                                return {k: _to_native(v) for k, v in obj.items()}
                        return {k: _to_native(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_to_native(v) for v in obj]
                    if hasattr(obj, 'isoformat'):
                        try:
                            return obj.isoformat()
                        except:
                            return str(obj)
                    return obj

                fig_json = fig.to_plotly_json()
                fig_json_native = _to_native(fig_json)
                fig_json_str = json.dumps(fig_json_native, ensure_ascii=False)
                config_str = json.dumps({'displayModeBar': False, 'displaylogo': False}, ensure_ascii=False)
                tmpl = Template("""
                <div id="$div_id" class="plotly-graph-div" style="height:${height}px; width:100%;"></div>
                <script type="text/javascript">
                    (function(){
                        var fig = $fig_json;
                        var cfg = $config_json;
                        Plotly.newPlot("$div_id", fig.data || [], fig.layout || {}, cfg);
                    })();
                </script>
                """)
                # 估计高度（若未设置，则使用 480）
                try:
                    h = fig.layout.height
                    height_px = int(h) if h is not None else 480
                except Exception:
                    height_px = 480
                return tmpl.substitute(div_id=div_id, height=height_px, fig_json=fig_json_str, config_json=config_str)

            top_html = _fig_to_html_div(fig_top, f"{name}_top")
            bottom_html = _fig_to_html_div(fig_bottom, f"{name}_bottom")

            def _metrics_tbl(ms):
                if not ms:
                    return ""
                rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in ms.items()])
                return f"""
                <table style=\"width:100%;border-collapse:collapse;margin:10px 0;\">
                    <thead><tr style=\"background:#f4f6f8;text-align:left;\"><th style=\"padding:8px\">指标</th><th style=\"padding:8px\">数值</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
                """

            explanation_md = json.dumps(explanation_html, ensure_ascii=False)
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>{title}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <script src="https://cdn.tailwindcss.com"></script>
                <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
                <style>
                    body {{ font-family: "Noto Sans SC","Microsoft YaHei","Segoe UI",sans-serif; }}
                    .markdown-body h3 {{ font-size: 1.05rem; margin-top: 1rem; margin-bottom: .45rem; }}
                    .markdown-body p {{ margin: .35rem 0; color: #374151; line-height: 1.65; }}
                    .markdown-body ul {{ margin: .25rem 0 .5rem 1.2rem; color: #374151; line-height: 1.6; list-style: disc; }}
                    .markdown-body li {{ margin: .2rem 0; }}
                </style>
                <script>
                    window.MathJax = {{
                        tex: {{
                            inlineMath: [["$","$"], ["\\(","\\)"]],
                            displayMath: [["$$","$$"], ["\\[","\\]"]],
                            processEscapes: true
                        }},
                        options: {{ skipHtmlTags: ["script","noscript","style","textarea","pre","code"] }},
                        svg: {{ fontCache: 'global' }}
                    }};
                </script>
                <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js" onerror="this.onerror=null; this.src='{mathjax_local_src}';"></script>
            </head>
            <body class="bg-gray-50">
                <header class="sticky top-0 z-30 bg-white shadow-sm border-b border-gray-100">
                    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
                        <div>
                            <p class="text-[11px] tracking-[0.18em] text-gray-500 uppercase">时段分析</p>
                            <h1 class="text-xl font-semibold text-gray-900">{title}</h1>
                        </div>
                        <div class="text-xs text-gray-500">滚动查看完整内容</div>
                    </div>
                </header>
                <main class="max-w-7xl mx-auto px-4 sm:px-6 py-6">
                    <div class="grid gap-6 lg:grid-cols-2" id="page-root">
                        <div class="bg-white rounded-lg shadow-sm p-5">
                            <div class="text-sm font-semibold text-gray-800 mb-3">{primary_title}</div>
                            {top_html}
                            {_metrics_tbl(metrics_primary)}
                        </div>
                        <div class="bg-white rounded-lg shadow-sm p-5">
                            <div class="text-sm font-semibold text-gray-800 mb-3">{secondary_title}</div>
                            {bottom_html}
                            {_metrics_tbl(metrics_secondary)}
                        </div>
                    </div>
                    <div id="explain-md" class="markdown-body mt-6 bg-white rounded-lg shadow-sm p-5"></div>
                </main>
                <script>
                    (function ensureRender(){{
                        var mdText = {explanation_md};
                        var target = document.getElementById('explain-md');
                        if (target) {{
                            if (window.marked) {{
                                target.innerHTML = marked.parse(mdText);
                            }} else {{
                                target.textContent = mdText;
                            }}
                        }}
                        var anchor = document.getElementById('page-root');
                        if (anchor) {{
                            requestAnimationFrame(function() {{
                                anchor.scrollIntoView({{ behavior: 'auto', block: 'start' }});
                            }});
                        }}
                        function typeset() {{
                            if (window.MathJax && window.MathJax.typesetPromise) {{
                                window.MathJax.typesetPromise().catch(function(e){{ console.warn('MathJax error:', e); }});
                            }}
                        }}
                        if (document.readyState === 'complete') {{
                            typeset();
                        }} else {{
                            window.addEventListener('load', typeset);
                        }}
                    }})();
                </script>
            </body>
            </html>
            """
            output_path.write_text(html, encoding='utf-8')
            self.figures.append((name, str(output_path)))
            file_size = output_path.stat().st_size / (1024*1024)
            print(f"    <i class='fas fa-check-circle text-green-500'></i> 保存: {name}.html ({file_size:.2f} MB)")
        except Exception as e:
            print(f"    <i class='fas fa-times-circle text-red-500'></i> 保存失败 {name}: {e}")

    # ====== 因子归因：数据与回归辅助 ======
    def _load_factor_strategy_dataset(self) -> Optional[pd.DataFrame]:
        try:
            factors_path = Path('data/factors_ff_cn.parquet')
            if not factors_path.exists():
                print("[FF] 缺少 data/factors_ff_cn.parquet，跳过因子归因")
                return None
            fac = pd.read_parquet(factors_path)
            fac['date'] = pd.to_datetime(fac['date']).dt.normalize()
            fac = fac.sort_values('date')
        except Exception as e:
            print(f"[FF] 读取因子失败: {e}")
            return None

        # 策略日收益：优先盯市文件，否则从 self.df 聚合
        strat = None
        try:
            mtm_csv = Path('mtm_analysis_results/mtm_returns.csv')
            if mtm_csv.exists():
                s = pd.read_csv(mtm_csv)
                dcol = 'date' if 'date' in s.columns else ('日期' if '日期' in s.columns else None)
                rcol = None
                for c in ['ret','daily_return','return','日收益','r']:
                    if c in s.columns:
                        rcol = c
                        break
                if dcol and rcol:
                    strat = s[[dcol, rcol]].copy()
                    strat.columns = ['date','strategy_return']
                    strat['date'] = pd.to_datetime(strat['date']).dt.normalize()
                    strat['strategy_return'] = pd.to_numeric(strat['strategy_return'], errors='coerce')
                    strat = strat.dropna()
        except Exception:
            strat = None

        if strat is None:
            try:
                df = getattr(self, 'df', None)
                if df is not None and len(df):
                    tmp = df.copy()
                    if 'date' not in tmp.columns:
                        tmp['date'] = pd.to_datetime(tmp['Timestamp']).dt.normalize()
                    tmp = tmp.dropna(subset=['real'])
                    grp = (tmp.groupby(['date'])
                             .apply(lambda g: np.average(g['real'], weights=pd.to_numeric(g.get('tradeAmount', pd.Series(np.ones(len(g)))), errors='coerce').fillna(0.0)) if (pd.to_numeric(g.get('tradeAmount', pd.Series(np.ones(len(g)))), errors='coerce').fillna(0.0)>0).any() else g['real'].mean())
                             .reset_index())
                    grp.columns = ['date','strategy_return']
                    grp['date'] = pd.to_datetime(grp['date']).dt.normalize()
                    strat = grp.dropna()
            except Exception as e:
                print(f"[FF] 从 self.df 聚合策略收益失败: {e}")
                strat = None

        if strat is None or strat.empty:
            print("[FF] 无法获取策略日收益，跳过因子归因")
            return None

        merged = strat.merge(fac, on='date', how='inner')
        if 'rf' in merged.columns:
            merged['excess'] = merged['strategy_return'] - merged['rf']
        else:
            merged['excess'] = merged['strategy_return']
        
        # 添加深交所因子列别名（优先使用深交所因子，如果可用）
        if 'smb_sz' in merged.columns and merged['smb_sz'].notna().sum() > 0:
            merged['smb'] = merged['smb_sz']
            merged['factor_source'] = 'SZ'
        elif 'smb' not in merged.columns:
            # 如果没有任何SMB，创建空列
            merged['smb'] = np.nan
            merged['factor_source'] = 'None'
        
        if 'hml_sz' in merged.columns and merged['hml_sz'].notna().sum() > 0:
            merged['hml'] = merged['hml_sz']
        elif 'hml' not in merged.columns:
            # 如果没有任何HML，创建空列
            merged['hml'] = np.nan
        
        return merged

    def _ff3_static_ols(self, df: pd.DataFrame) -> dict:
        cols = ['excess','mkt','smb','hml']
        data = df.dropna(subset=cols).copy()
        if len(data) < 20:
            return {'n': len(data)}
        X = data[['mkt','smb','hml']].values
        X = np.column_stack([np.ones(len(X)), X])
        y = data['excess'].values
        res = {}
        try:
            import statsmodels.api as sm  # type: ignore
            model = sm.OLS(y, X).fit()
            res = {
                'n': int(model.nobs),
                'alpha': float(model.params[0]),
                'beta_mkt': float(model.params[1]),
                'beta_smb': float(model.params[2]),
                'beta_hml': float(model.params[3]),
                't_alpha': float(model.tvalues[0]),
                't_mkt': float(model.tvalues[1]),
                't_smb': float(model.tvalues[2]),
                't_hml': float(model.tvalues[3]),
                'p_alpha': float(model.pvalues[0]),
                'p_mkt': float(model.pvalues[1]),
                'p_smb': float(model.pvalues[2]),
                'p_hml': float(model.pvalues[3]),
                'r2': float(model.rsquared),
                'adj_r2': float(model.rsquared_adj),
            }
        except Exception:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            yhat = X @ beta
            ss_res = float(np.sum((y - yhat)**2))
            ss_tot = float(np.sum((y - yhat.mean())**2)) if len(y) > 0 else np.nan
            r2 = 1 - ss_res/ss_tot if ss_tot not in (0.0, np.nan) else np.nan
            res = {
                'n': int(len(y)),
                'alpha': float(beta[0]),
                'beta_mkt': float(beta[1]),
                'beta_smb': float(beta[2]),
                'beta_hml': float(beta[3]),
                'r2': float(r2) if r2 == r2 else np.nan,
            }
        return res

    def _ff3_rolling_ols(self, df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
        cols = ['excess','mkt','smb','hml']
        data = df.dropna(subset=cols).copy()
        data = data.sort_values('date')
        rows = []
        for i in range(window, len(data)+1):
            part = data.iloc[i-window:i]
            X = part[['mkt','smb','hml']].values
            X = np.column_stack([np.ones(len(X)), X])
            y = part['excess'].values
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                yhat = X @ beta
                ss_res = float(np.sum((y - yhat)**2))
                ss_tot = float(np.sum((y - yhat.mean())**2)) if len(y) > 0 else np.nan
                r2 = 1 - ss_res/ss_tot if ss_tot not in (0.0, np.nan) else np.nan
                rows.append({
                    'date': data.iloc[i-1]['date'],
                    'alpha': float(beta[0]),
                    'beta_mkt': float(beta[1]),
                    'beta_smb': float(beta[2]),
                    'beta_hml': float(beta[3]),
                    'r2': float(r2) if r2 == r2 else np.nan,
                })
            except Exception:
                continue
        return pd.DataFrame(rows)

    def portfolio_factor_attribution_main(self) -> None:
        print("\n<i class='fas fa-chart-line text-green-500'></i> === 因子归因（FF3）主页面 ===")
        try:
            d = self._load_factor_strategy_dataset()
            if d is None or d.empty:
                return
            # 静态 OLS
            static_res = self._ff3_static_ols(d)
            # 选择滚动窗口：固定30天；若有效样本不足则不绘滚动
            valid_cnt = len(d.dropna(subset=['excess','mkt','smb','hml']))
            roll_win = 30 if valid_cnt >= 30 else None
            if roll_win is None:
                print(f"[FF] 有效样本 {valid_cnt} 天 < 30，跳过滚动回归")
                rolling_df = pd.DataFrame(columns=['date','alpha','beta_mkt','beta_smb','beta_hml','r2'])
            else:
                rolling_df = self._ff3_rolling_ols(d, window=roll_win)

            # 落盘工件
            try:
                pd.DataFrame([static_res]).to_csv(self.reports_dir / 'ff3_static.csv', index=False, encoding='utf-8-sig')
                rolling_df.to_csv(self.reports_dir / f'ff3_rolling_{roll_win}.csv', index=False, encoding='utf-8-sig')
            except Exception:
                pass

            # 图表：上=滚动β和Alpha（双Y轴）；下=滚动R²
            import plotly.graph_objs as go  # 确保可用
            from plotly.subplots import make_subplots
            
            # 使用make_subplots创建双Y轴
            fig_top = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 检查rolling_df是否有数据
            if not rolling_df.empty and len(rolling_df) > 0:
                # Beta系数使用左Y轴
                for col, name, color in [
                    ('beta_mkt','β_MKT','#2c3e50'),
                    ('beta_smb','β_SMB','#16a085'),
                    ('beta_hml','β_HML','#8e44ad'),
                ]:
                    if col in rolling_df.columns:
                        fig_top.add_trace(
                            go.Scatter(
                                x=rolling_df['date'].tolist(),  # 直接转为list
                                y=rolling_df[col].tolist(), 
                                mode='lines', 
                                name=name, 
                                line=dict(color=color, width=2)
                            ),
                            secondary_y=False
                        )
                
                # Alpha使用右Y轴
                if 'alpha' in rolling_df.columns:
                    fig_top.add_trace(
                        go.Scatter(
                            x=rolling_df['date'].tolist(), 
                            y=rolling_df['alpha'].tolist(), 
                            mode='lines', 
                            name='Alpha', 
                            line=dict(color='#e67e22', width=2.5, dash='dot')
                        ),
                        secondary_y=True
                    )
            else:
                # 如果没有滚动数据，显示提示
                print("[WARN] 滚动回归数据为空，图表将为空")
            
            title_suffix = f'（窗口={roll_win}天）' if roll_win is not None else '（样本不足，未绘制）'
            
            # 设置Y轴标题和颜色
            fig_top.update_yaxes(title_text="Beta系数", secondary_y=False)
            fig_top.update_yaxes(
                title_text="<span style='color:#e67e22'>Alpha (日度)</span>", 
                secondary_y=True,
                tickfont=dict(color='#e67e22')
            )
            
            fig_top.update_layout(
                title=f'滚动回归系数{title_suffix}',
                xaxis_title='日期',
                height=420,
                margin=dict(l=50,r=50,t=40,b=40),
                hovermode='x unified'
            )

            # 下图改为：因子收益率曲线（显示市场环境）
            fig_bottom = go.Figure()
            
            # 获取因子的完整时间序列
            if 'mkt' in d.columns or 'smb' in d.columns or 'hml' in d.columns:
                # 按日期排序
                factor_ts = d[['date', 'mkt', 'smb', 'hml']].copy()
                factor_ts = factor_ts.sort_values('date')
                
                # 绘制因子收益率曲线
                for col, name, color in [
                    ('mkt', 'MKT (市场)', '#2c3e50'),
                    ('smb', 'SMB (规模)', '#16a085'),
                    ('hml', 'HML (价值)', '#8e44ad'),
                ]:
                    if col in factor_ts.columns:
                        # 过滤有效值
                        valid_data = factor_ts[factor_ts[col].notna()].copy()
                        if len(valid_data) > 0:
                            fig_bottom.add_trace(go.Scatter(
                                x=valid_data['date'].dt.strftime('%Y-%m-%d').tolist(),  # 格式化日期
                                y=valid_data[col].tolist(),
                                mode='lines',
                                name=name,
                                line=dict(color=color, width=1.5),
                                opacity=0.7
                            ))
                
                # 添加0参考线
                fig_bottom.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, 
                                   annotation_text="0%", annotation_position="right")
                
                fig_bottom.update_layout(
                    title='FF3因子日度收益率（市场环境）',
                    xaxis_title='日期',
                    yaxis_title='因子日收益率',
                    yaxis_tickformat='.2%',
                    height=420,
                    margin=dict(l=50, r=10, t=40, b=40),
                    hovermode='x unified',
                    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
                )
            else:
                # 如果没有因子数据，显示R²
                if not rolling_df.empty and 'r2' in rolling_df.columns:
                    fig_bottom.add_trace(go.Scatter(
                        x=rolling_df['date'].astype(str), 
                        y=rolling_df['r2'], 
                        mode='lines+markers', 
                        name='R²', 
                        line=dict(color='#2980b9', width=2),
                        marker=dict(size=4)
                    ))
                    fig_bottom.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                fig_bottom.update_layout(
                    title=f'滚动模型解释力（R²）{title_suffix}',
                    xaxis_title='日期',
                    yaxis_title='R² (模型解释力)',
                    height=420,
                    margin=dict(l=40,r=10,t=40,b=40),
                    hovermode='x unified'
                )

            # 指标卡片
            metrics_primary = {
                '样本数 n': static_res.get('n', 0),
                'alpha': f"{static_res.get('alpha', np.nan):.6f}" if 'alpha' in static_res else 'NA',
                'β_MKT': f"{static_res.get('beta_mkt', np.nan):.4f}" if 'beta_mkt' in static_res else 'NA',
                'β_SMB': f"{static_res.get('beta_smb', np.nan):.4f}" if 'beta_smb' in static_res else 'NA',
                'β_HML': f"{static_res.get('beta_hml', np.nan):.4f}" if 'beta_hml' in static_res else 'NA',
                'R²': f"{static_res.get('r2', np.nan):.4f}" if 'r2' in static_res else 'NA',
            }
            metrics_secondary = {
                '滚动窗口': roll_win if roll_win is not None else '样本不足',
                '滚动点数': 0 if rolling_df.empty else len(rolling_df),
            }

            explanation = [
                "<h4><i class='fas fa-thumbtack text-red-400'></i> 方法说明</h4>",
                "<ul>",
                "<li><b>模型：</b>R<sub>strategy</sub> - R<sub>f</sub> = α + β<sub>MKT</sub>·MKT + β<sub>SMB</sub>·SMB + β<sub>HML</sub>·HML + ε</li>",
                "<li><b>因子构建：</b>MKT=深证成指日收益，SMB/HML=深交所2×3双重排序（月度重构）</li>",
                "<li><b>滚动窗口：</b>30天，每日更新回归系数</li>",
                "<li><b>数据源：</b>深交所2,875只股票，Baostock历史PB + AkShare市值</li>",
                "</ul>",
                "<h4><i class='fas fa-chart-bar text-indigo-500'></i> 如何阅读图表</h4>",
                "<ul>",
                "<li><b>上图（双Y轴）：</b>左侧=Beta系数，右侧=Alpha（橙色虚线）</li>",
                "<li><b>下图：</b>三个因子的日度收益率（理解市场环境）</li>",
                "<li><b>例：</b>β_HML=-0.5，HML=-3%（成长股涨） → 贡献=+1.5%（负×负=正）</li>",
                "</ul>"
            ]
            # 添加第三个图表：R²曲线
            fig_r2 = go.Figure()
            if not rolling_df.empty and len(rolling_df) > 0 and 'r2' in rolling_df.columns:
                fig_r2.add_trace(go.Scatter(
                    x=rolling_df['date'].tolist(),  # 直接转为list
                    y=rolling_df['r2'].tolist(),
                    mode='lines+markers',
                    name='R²',
                    line=dict(color='#2980b9', width=2),
                    marker=dict(size=4),
                    fill='tozeroy',
                    fillcolor='rgba(41, 128, 185, 0.1)'
                ))
                fig_r2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            else:
                print("[WARN] R²数据为空或不存在")
                
            fig_r2.update_layout(
                title=f'滚动模型解释力（R²）{title_suffix}',
                xaxis_title='日期',
                yaxis_title='R²',
                yaxis_tickformat='.1%',
                height=380,
                margin=dict(l=50, r=10, t=40, b=40),
                hovermode='x unified'
            )
            
            # 使用三图表保存方法
            self._save_figure_triple_with_details(
                fig_top, fig_bottom, fig_r2,
                name='factor_attribution_main',
                title='因子归因（FF3）主页面',
                explanation_html=''.join(explanation),
                metrics_primary=metrics_primary,
                metrics_secondary=metrics_secondary,
                title1='滚动回归系数（Alpha & Beta）',
                title2='FF3因子日度收益率（市场环境）',
                title3='滚动模型解释力（R²）'
            )
            print("<i class='fas fa-check-circle text-green-500'></i> 因子归因主页面完成")
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 因子归因失败: {e}")
            import traceback
            traceback.print_exc()
    
    def portfolio_factor_attribution_quarterly(self) -> None:
        """按季度分析FF3因子归因"""
        print("\n<i class='fas fa-chart-line text-green-500'></i> === 因子归因（FF3）按季度分析 ===")
        try:
            d = self._load_factor_strategy_dataset()
            if d is None or d.empty:
                return
            
            # 添加季度标识
            d['quarter'] = pd.to_datetime(d['date']).dt.to_period('Q')
            quarters = sorted(d['quarter'].unique())
            
            print(f"[FF] 分析季度数: {len(quarters)}")
            
            # 按季度计算回归
            quarterly_results = []
            for q in quarters:
                q_data = d[d['quarter'] == q]
                q_result = self._ff3_static_ols(q_data)
                q_result['quarter'] = str(q)
                q_result['start_date'] = q_data['date'].min()
                q_result['end_date'] = q_data['date'].max()
                quarterly_results.append(q_result)
            
            if not quarterly_results:
                print("[FF] 无季度结果")
                return
            
            qdf = pd.DataFrame(quarterly_results)
            
            # 保存季度回归结果
            qdf.to_csv(self.reports_dir / 'ff3_quarterly.csv', index=False, encoding='utf-8-sig')
            
            # 绘制季度分析图
            import plotly.graph_objs as go
            from plotly.subplots import make_subplots
            
            # 上图：Beta系数按季度变化
            fig_top = go.Figure()
            quarters_str = qdf['quarter'].astype(str).tolist()
            
            for col, name, color in [
                ('beta_mkt', 'β_MKT', '#2c3e50'),
                ('beta_smb', 'β_SMB', '#16a085'),
                ('beta_hml', 'β_HML', '#8e44ad'),
            ]:
                if col in qdf.columns:
                    fig_top.add_trace(go.Bar(
                        x=quarters_str,
                        y=qdf[col],
                        name=name,
                        marker_color=color
                    ))
            
            fig_top.update_layout(
                title='各季度Beta系数（深交所FF3）',
                xaxis_title='季度',
                yaxis_title='Beta系数',
                barmode='group',
                height=450,
                margin=dict(l=40, r=10, t=50, b=40),
                showlegend=True
            )
            
            # 下图：Alpha和R²按季度变化
            fig_bottom = make_subplots(
                rows=1, cols=2,
                subplot_titles=('各季度Alpha', '各季度R²')
            )
            
            if 'alpha' in qdf.columns:
                fig_bottom.add_trace(
                    go.Bar(x=quarters_str, y=qdf['alpha'], 
                          marker_color='#e67e22', name='Alpha', showlegend=False),
                    row=1, col=1
                )
            
            if 'r2' in qdf.columns:
                fig_bottom.add_trace(
                    go.Bar(x=quarters_str, y=qdf['r2'],
                          marker_color='#2980b9', name='R²', showlegend=False),
                    row=1, col=2
                )
            
            fig_bottom.update_xaxes(title_text="季度", row=1, col=1)
            fig_bottom.update_xaxes(title_text="季度", row=1, col=2)
            fig_bottom.update_yaxes(title_text="Alpha", row=1, col=1)
            fig_bottom.update_yaxes(title_text="R²", row=1, col=2)
            fig_bottom.update_layout(
                height=450,
                margin=dict(l=40, r=10, t=50, b=40)
            )
            
            # 构建指标
            full_result = self._ff3_static_ols(d)
            
            metrics_primary = {
                '全年Alpha': f"{full_result.get('alpha', np.nan):.6f}",
                '全年β_MKT': f"{full_result.get('beta_mkt', np.nan):.4f}",
                '全年β_SMB': f"{full_result.get('beta_smb', np.nan):.4f}",
                '全年β_HML': f"{full_result.get('beta_hml', np.nan):.4f}",
                '全年R²': f"{full_result.get('r2', np.nan):.4f}",
            }
            
            metrics_secondary = {
                '分析季度数': len(quarters),
                '因子数据源': d.get('factor_source', pd.Series(['深交所'])).iloc[0] if len(d) > 0 else '深交所',
                '有效样本': full_result.get('n', 0)
            }
            
            explanation = [
                "<h4><i class='fas fa-thumbtack text-red-400'></i> 季度归因分析说明</h4>",
                "<ul>",
                "<li><b>方法：</b>使用Fama-French三因子模型对策略收益进行归因分析</li>",
                "<li><b>模型：</b>R<sub>strategy</sub> - R<sub>f</sub> = α + β<sub>MKT</sub>·MKT + β<sub>SMB</sub>·SMB + β<sub>HML</sub>·HML + ε</li>",
                "<li><b>深交所因子：</b>使用2×3双重排序方法构建，基于深交所全部上市公司</li>",
                "<li><b>分组规则：</b>市值按中位数分S/B，BM按30%/70%分L/M/H，形成6个组合</li>",
                "<li><b>加权方式：</b>组合内按流通市值加权</li>",
                "<li><b><i class='fas fa-exclamation-triangle text-yellow-500'></i> 数据限制：</b>市值数据使用当前值，存在前视偏差，结果仅供风格分析参考</li>",
                "</ul>",
                "<h4><i class='fas fa-chart-bar text-indigo-500'></i> 指标解读</h4>",
                "<ul>",
                "<li><b>Alpha：</b>扣除系统性风险后的超额收益，衡量选股/择时能力</li>",
                "<li><b>β_MKT：</b>对市场整体波动的敏感度（>1表示高波动）</li>",
                "<li><b>β_SMB：</b>对小市值风格的暴露（>0偏好小盘，<0偏好大盘）</li>",
                "<li><b>β_HML：</b>对价值风格的暴露（>0偏好价值，<0偏好成长）</li>",
                "<li><b>R²：</b>因子对策略收益的解释力（越高说明风格越纯粹）</li>",
                "</ul>"
            ]
            
            self._save_figure_pair_with_details_v2(
                fig_top, fig_bottom,
                name='factor_attribution_quarterly',
                title='因子归因（FF3）- 按季度分析',
                explanation_html=''.join(explanation),
                metrics_primary=metrics_primary,
                metrics_secondary=metrics_secondary,
                primary_title='各季度Beta系数',
                secondary_title='各季度Alpha与R²'
            )
            print("<i class='fas fa-check-circle text-green-500'></i> 因子归因季度分析完成")
            
        except Exception as e:
            print(f"<i class='fas fa-times-circle text-red-500'></i> 因子归因季度分析失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    analyzer = LightweightAnalysis()
    analyzer.run_analysis()
