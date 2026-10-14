# -*- coding: utf-8 -*-
"""
AI股票趋势预测系统 - 完整整合版
功能：日线预测、分时预测、统计相似性分析、上升趋势扫描（含形态相似性预测）
特点：量价共振 + 均线确认 + 趋势过滤 + 九转序列 + 换手率辅助 + 形态相似度排序
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import requests
import baostock as bs
import xgboost as xgb
from scipy.stats import poisson
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')
import time
import random
from datetime import datetime, timedelta
import akshare as ak
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import wraps

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    ts = None

# =====================================================
# 页面配置
# =====================================================
st.set_page_config(page_title="AI股票预测系统", layout="wide")
st.title("📈 AI 股票趋势预测系统")
st.caption("量价共振 + 均线确认 + 趋势过滤 + 九转序列 + 换手率 + 形态相似度")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

DATA_CACHE_DIR = "data_cache"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


# =====================================================
# 通用工具
# =====================================================
def retry_on_failure(max_retries=3, base_delay=1.0, backoff_factor=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (backoff_factor ** attempt)
                    delay += random.uniform(0, delay * 0.5)
                    time.sleep(delay)
            if last_exc:
                raise last_exc
        return wrapper
    return decorator


def call_with_timeout(func, args=(), kwargs=None, timeout=25):
    if kwargs is None:
        kwargs = {}
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        result = future.result(timeout=timeout)
        executor.shutdown(wait=False)
        return result
    except FuturesTimeoutError:
        executor.shutdown(wait=False)
        return None
    except Exception:
        executor.shutdown(wait=False)
        return None


def is_data_ok(df, lookback_window, pred_days):
    if df is None or df.empty:
        return False
    if len(df) < lookback_window + pred_days + 10:
        return False
    required_cols = ['date', 'close']
    if not all(c in df.columns for c in required_cols):
        return False
    close = pd.to_numeric(df['close'], errors='coerce').dropna()
    if len(close) < lookback_window + pred_days + 10:
        return False
    if (close <= 0).any():
        return False
    return True


# =====================================================
# 沪深主板过滤
# =====================================================
def filter_main_board_stocks(df, exclude_st=True):
    if df is None or df.empty:
        return df
    code_col = None
    for c in ['symbol', 'code', '代码']:
        if c in df.columns:
            code_col = c
            break
    if code_col is None:
        return df

    def is_main_board(code):
        code = str(code).strip()
        if code.startswith(('sh', 'sz', 'SH', 'SZ')):
            code = code[2:]
        if len(code) != 6 or not code.isdigit():
            return False
        if code.startswith(('600', '601', '603', '605')):
            return True
        if code.startswith(('000', '001', '002', '003')):
            return True
        return False

    df = df[df[code_col].apply(is_main_board)].reset_index(drop=True)

    if exclude_st and 'name' in df.columns:
        df = df[~df['name'].astype(str).str.contains('ST', case=False, na=False)].reset_index(drop=True)

    return df


# =====================================================
# Baostock 代码格式化
# =====================================================
def format_baostock_code(code):
    code = code.strip()
    if code.startswith('6'):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


# =====================================================
# 换手率数据
# =====================================================
@st.cache_data(ttl=1800)
def get_turnover_map():
    try:
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return pd.DataFrame(columns=['symbol', '换手率'])
        rename_map = {}
        for c in df.columns:
            if c in ('代码', 'code'):
                rename_map[c] = 'symbol'
            elif c in ('换手率',):
                rename_map[c] = '换手率'
        df = df.rename(columns=rename_map)
        if 'symbol' not in df.columns or '换手率' not in df.columns:
            return pd.DataFrame(columns=['symbol', '换手率'])
        df['symbol'] = df['symbol'].astype(str).str.zfill(6)
        df['换手率'] = pd.to_numeric(df['换手率'], errors='coerce')
        df = df[['symbol', '换手率']].dropna(subset=['换手率'])
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=['symbol', '换手率'])


# =====================================================
# 模拟数据生成
# =====================================================
def get_seed_from_code(code):
    try:
        return int(code) % (2**32)
    except Exception:
        return 42


def generate_synthetic_price_data(code, start, end):
    dates = pd.date_range(start=start, end=end, freq='B')
    n = len(dates)
    seed = get_seed_from_code(code)
    np.random.seed(seed)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    open_price = close * (1 + np.random.randn(n) * 0.005)
    high = np.maximum(close, open_price) * (1 + np.random.rand(n) * 0.01)
    low = np.minimum(close, open_price) * (1 - np.random.rand(n) * 0.01)
    volume = np.random.randint(1000, 10000, n)
    return pd.DataFrame({'date': dates, 'open': open_price, 'high': high,
                         'low': low, 'close': close, 'volume': volume})


def generate_synthetic_financial_data(code, start_year, end_year):
    seed = get_seed_from_code(code)
    np.random.seed(seed)
    rows = []
    for year in range(start_year, end_year + 1):
        base_roe = 0.20 + np.random.randn() * 0.05
        base_eps = 5 + (year - start_year) * 0.5 + np.random.randn() * 0.5
        base_revenue = 100 + (year - start_year) * 10 + np.random.randn() * 5
        base_net_profit = base_revenue * 0.2 + np.random.randn() * 2
        rows.append({
            'year': year,
            'roe': max(0, base_roe),
            'eps': max(0, base_eps),
            'revenue': max(0, base_revenue),
            'net_profit': max(0, base_net_profit),
            'gross_profit_rate': 0.8 + np.random.randn() * 0.05,
            'net_profit_rate': 0.2 + np.random.randn() * 0.03,
            'debt_asset_ratio': 0.2 + np.random.randn() * 0.05,
            'current_ratio': 2.0 + np.random.randn() * 0.3,
        })
    df = pd.DataFrame(rows)
    if len(df) >= 2:
        df = df.sort_values('year')
        df['revenue_growth'] = df['revenue'].pct_change()
        df['net_profit_growth'] = df['net_profit'].pct_change()
    else:
        df['revenue_growth'] = 0
        df['net_profit_growth'] = 0
    return df.fillna(0)


# =====================================================
# 侧边栏
# =====================================================
st.sidebar.header("⚙️ 全局参数")
tushare_token = st.sidebar.text_input("🔑 Tushare Token (日线)", type="password",
                                      help="https://tushare.pro 注册获取，仅日线使用")
if tushare_token and TUSHARE_AVAILABLE:
    ts.set_token(tushare_token)
    pro = ts.pro_api()
else:
    pro = None
    if tushare_token and not TUSHARE_AVAILABLE:
        st.sidebar.warning("未安装 tushare，请执行：pip install tushare")

stock_code = st.sidebar.text_input("股票代码 (6位数字)", value="600519").strip()
prediction_mode = st.sidebar.radio("预测模式", [
    "日线预测（中长期）",
    "分时预测（日内）",
    "📊 统计与相似性分析（日线）",
    "🔍 上升趋势股票扫描"
])
st.sidebar.markdown("---")

if prediction_mode == "日线预测（中长期）":
    st.sidebar.subheader("📊 日线参数")
    start_date = st.sidebar.date_input("开始日期", value=pd.to_datetime("today") - pd.DateOffset(months=6))
    end_date = st.sidebar.date_input("结束日期", value=pd.to_datetime("today"))
    use_fundamental = st.sidebar.checkbox("启用基本面增强", True)
    lookback = st.sidebar.slider("窗口长度", 20, 120, 60, 5)
    epochs = st.sidebar.number_input("训练轮数", 10, 200, 50, 10)
    lstm_units = st.sidebar.number_input("LSTM 单元数", 16, 256, 50, 16)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
    predict_days = st.sidebar.number_input("预测天数", 1, 180, 5, 1)
    train_btn = st.sidebar.button("🚀 训练模型并预测", type="primary")
    load_btn = st.sidebar.button("📂 加载已有模型")
    search_btn = st.sidebar.button("🔍 自动搜索最佳模型")

elif prediction_mode == "分时预测（日内）":
    st.sidebar.info("数据源：Baostock，支持 5/15/30/60 分钟")
    period = st.sidebar.selectbox("K线周期 (分钟)", ["5", "15", "30", "60"], index=0)
    lookback_min = st.sidebar.slider("窗口长度 (时间步)", 10, 120, 60, 5)
    predict_full_day = st.sidebar.checkbox("预测当天剩余全部交易时间", True)
    if predict_full_day:
        predict_steps = st.sidebar.number_input("预测步数 (自动)", 1, 480, 10, 1, disabled=True)
    else:
        predict_steps = st.sidebar.number_input("预测步数 (每步周期分钟数)", 1, 480, 10, 1)
    intraday_btn = st.sidebar.button("🔮 开始分时预测", type="primary")
    refresh_btn = st.sidebar.button("🔄 强制刷新缓存", type="primary")

elif prediction_mode == "📊 统计与相似性分析（日线）":
    st.sidebar.subheader("📐 统计参数")
    stat_lookback = st.sidebar.slider("形态匹配窗口 (天)", 20, 120, 60, 5)
    top_k = st.sidebar.slider("Top K 相似片段", 1, 20, 10, 1)
    simulate_days = st.sidebar.number_input("模拟未来天数", 1, 60, 20, 1)
    stat_btn = st.sidebar.button("🔍 执行统计分析", type="primary")

else:  # 上升趋势扫描
    st.sidebar.subheader("🔍 扫描参数")
    scan_start_date = st.sidebar.date_input("数据开始日期", value=pd.to_datetime("today") - pd.DateOffset(years=1))

    only_main_board = st.sidebar.checkbox(
        "仅扫描沪深主板（新手推荐）",
        value=True,
        help="排除创业板(300/301)、科创板(688)、北交所(4/8开头)等需要额外权限的板块，同时排除 ST / *ST"
    )

    st.sidebar.markdown("### 🎯 进场信号")

    # ===== 趋势过滤 =====
    use_trend_filter = st.sidebar.checkbox(
        "启用长期趋势过滤（EMA150 > SMA200）",
        value=True,
        help="过滤掉长期下跌趋势的股票，只在长期向上的市场里选股"
    )

    # ===== 均线确认 =====
    use_ma_confirm = st.sidebar.checkbox(
        "启用均线确认（站上20日线）",
        value=True,
        help="股价放量站上20日均线，说明短期动能启动"
    )
    if use_ma_confirm:
        ma_confirm_period = st.sidebar.number_input(
            "均线周期", value=20, min_value=5, max_value=60, step=5,
            help="股价站上多少日均线，20 日是比较常用的短期趋势线"
        )
        max_deviation = st.sidebar.number_input(
            "最大偏离度 (%)", value=5.0, min_value=1.0, max_value=20.0, step=0.5,
            help="股价离均线不能超过这个百分比，超过说明已经涨过头了"
        )
    else:
        ma_confirm_period = 20
        max_deviation = 100.0

    # ===== 量价共振 =====
    use_volume_confirm = st.sidebar.checkbox(
        "启用放量确认",
        value=True,
        help="当日成交量 > 50日均量 × 倍数，说明有资金真金白银进场"
    )
    if use_volume_confirm:
        volume_ratio = st.sidebar.number_input(
            "放量倍数", value=1.5, min_value=1.0, max_value=5.0, step=0.1,
            help="1.5 倍是比较温和的放量，2 倍以上说明资金很积极"
        )
    else:
        volume_ratio = 0.0

    # ===== 九转序列 =====
    use_td9 = st.sidebar.checkbox(
        "启用九转序列（低9）",
        value=False,
        help="连续9根K线收盘价 < 之前第4根，说明下跌动能衰竭。适合波段，中长线建议关闭"
    )

    # ===== 换手率 =====
    use_turnover_filter = st.sidebar.checkbox(
        "启用换手率过滤",
        value=False,
        help="只保留日换手率在设定区间内的股票。3%~10% 通常是较活跃的健康区间"
    )
    if use_turnover_filter:
        min_turnover = st.sidebar.number_input(
            "最低换手率 (%)", value=3.0, min_value=0.0, max_value=50.0, step=0.5
        )
        max_turnover = st.sidebar.number_input(
            "最高换手率 (%)", value=15.0, min_value=0.5, max_value=100.0, step=0.5
        )
    else:
        min_turnover = 0.0
        max_turnover = 999.0

    # ===== 价格过滤 =====
    min_price = st.sidebar.number_input("最低股价过滤 (元)", value=5.0, min_value=0.0, max_value=100.0, step=0.5)

    # ===== 形态相似度（作为排序加分）=====
    lookback_window = st.sidebar.selectbox("形态匹配窗口 (天)", [30, 60, 90, 120], index=1,
                                           help="用过去多少个交易日作为形态模板")
    pred_days = st.sidebar.selectbox("预测未来天数", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], index=3)
    similarity_top_k = st.sidebar.number_input("相似片段匹配数量 (Top K)", value=10, min_value=1, max_value=50, step=1)
    min_similarity = st.sidebar.slider(
        "最低相似度阈值（标准余弦，-1~1）",
        -1.0, 1.0, 0.3, 0.05,
        help="标准余弦相似度：1=完全同向，0=不相关。作为排序加分项，不是硬过滤"
    )

    scan_all = st.sidebar.checkbox("扫描全部股票（可能耗时较长）", value=False)
    if not scan_all:
        max_stocks = st.sidebar.number_input("最大扫描数量", value=100, min_value=10, max_value=3000, step=10)
    else:
        max_stocks = 99999
        st.sidebar.info("将扫描所有股票（约5000+只），预计耗时较长")

    board_type = st.sidebar.selectbox("板块类型", ["全部", "行业板块", "概念板块"])
    if board_type == "全部":
        board_name = "全部"
    else:
        @st.cache_data(ttl=3600)
        def get_board_list(board_type):
            if board_type == "行业板块":
                df = ak.stock_board_industry_em()
                return df['板块名称'].tolist()
            else:
                df = ak.stock_board_concept_em()
                return df['板块名称'].tolist()
        try:
            board_list = get_board_list(board_type)
            board_name = st.sidebar.selectbox("选择板块", board_list)
        except Exception:
            board_name = "全部"
            st.sidebar.warning("获取板块列表失败，使用全部股票")

    cache_hours = st.sidebar.number_input("缓存有效期 (小时)", value=1.0, min_value=0.1, max_value=48.0, step=0.5)
    force_rescan = st.sidebar.checkbox("强制重新扫描（忽略缓存）", value=False)
    auto_scan_interval = st.sidebar.number_input("自动扫描间隔 (小时)", value=0.0, min_value=0.0, max_value=24.0, step=0.5)

    if 'scan_stop_requested' not in st.session_state:
        st.session_state.scan_stop_requested = False
    if 'partial_scan_result' not in st.session_state:
        st.session_state.partial_scan_result = None

    def request_stop_scan():
        st.session_state.scan_stop_requested = True

    scan_btn = st.sidebar.button("🔍 开始扫描上升趋势", type="primary")
    stop_btn = st.sidebar.button("⏹️ 停止扫描（返回已扫描结果）", on_click=request_stop_scan)

st.sidebar.markdown("---")
st.sidebar.info("日线：Tushare（需Token）｜分时/基本面：Baostock（免费）")


# =====================================================
# 日线数据获取
# =====================================================
def get_price_data_baostock(code, start, end):
    bs_code = format_baostock_code(code)
    try:
        bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code,
            fields="date,open,high,low,close,volume",
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="3"
        )
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()
        if not data_list:
            return None
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['close'] = df['close'].astype(float)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        return df
    except Exception:
        try:
            bs.logout()
        except Exception:
            pass
        return None


@st.cache_data(ttl=86400)
def get_price_data(code, start, end, _pro=None):
    cache_file = f"{code}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_price.csv"
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=['date'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception:
            pass

    if _pro is not None:
        try:
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
            df_ts = _pro.daily(ts_code=ts_code, start_date=start.strftime("%Y%m%d"),
                               end_date=end.strftime("%Y%m%d"),
                               fields="trade_date,open,high,low,close,vol")
            if df_ts is not None and not df_ts.empty:
                df_ts = df_ts.sort_values('trade_date').rename(columns={'trade_date': 'date', 'vol': 'volume'})
                df_ts['date'] = pd.to_datetime(df_ts['date'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_ts[col] = pd.to_numeric(df_ts[col], errors='coerce')
                df_ts = df_ts[(df_ts['date'] >= start) & (df_ts['date'] <= end)]
                if not df_ts.empty:
                    df_ts.to_csv(cache_file, index=False)
                    return df_ts
        except Exception:
            pass

    secid = f"0.{code}" if code.startswith(('0', '3')) else f"1.{code}"
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {"secid": secid, "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
              "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61", "klt": 101, "fqt": 1,
              "end": end.strftime("%Y%m%d"), "lmt": 1200}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('data') and data['data'].get('klines'):
                klines = data['data']['klines']
                rows = []
                for line in klines:
                    parts = line.split(',')
                    rows.append({'date': parts[0], 'open': float(parts[1]), 'close': float(parts[2]),
                                 'high': float(parts[3]), 'low': float(parts[4]), 'volume': float(parts[5])})
                df = pd.DataFrame(rows)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df = df[(df['date'] >= start) & (df['date'] <= end)]
                if not df.empty:
                    df.to_csv(cache_file, index=False)
                    return df
    except Exception:
        pass

    try:
        df_bs = get_price_data_baostock(code, start, end)
        if df_bs is not None and not df_bs.empty:
            df_bs.to_csv(cache_file, index=False)
            return df_bs
    except Exception:
        pass

    df_sim = generate_synthetic_price_data(code, start, end)
    df_sim.to_csv(cache_file, index=False)
    if 'simulated_codes' not in st.session_state:
        st.session_state.simulated_codes = set()
    st.session_state.simulated_codes.add(code)
    return df_sim


# =====================================================
# 基本面数据
# =====================================================
def get_financial_metrics_baostock(code, start_year, end_year):
    bs_code = format_baostock_code(code)
    try:
        bs.login()
        all_data = []
        for year in range(start_year, end_year + 1):
            rs_profit = bs.query_profit_data(code=bs_code, year=year, quarter=4)
            if rs_profit.error_code != '0':
                continue
            profit_row = None
            while (rs_profit.error_code == '0') and rs_profit.next():
                profit_row = rs_profit.get_row_data()
                break
            if profit_row is None:
                continue
            profit_dict = dict(zip(rs_profit.fields, profit_row))
            rs_balance = bs.query_balance_data(code=bs_code, year=year, quarter=4)
            balance_dict = {}
            if rs_balance.error_code == '0':
                while (rs_balance.error_code == '0') and rs_balance.next():
                    bal_row = rs_balance.get_row_data()
                    balance_dict = dict(zip(rs_balance.fields, bal_row))
                    break
            rs_dup = bs.query_dupont_data(code=bs_code, year=year, quarter=4)
            dupont_dict = {}
            if rs_dup.error_code == '0':
                while (rs_dup.error_code == '0') and rs_dup.next():
                    dup_row = rs_dup.get_row_data()
                    dupont_dict = dict(zip(rs_dup.fields, dup_row))
                    break
            metrics = {
                'year': year,
                'roe': float(dupont_dict.get('dupontROE', 0)) if dupont_dict.get('dupontROE') else None,
                'eps': float(profit_dict.get('profitStatementBasicEPS', 0)) if profit_dict.get('profitStatementBasicEPS') else None,
                'revenue': float(profit_dict.get('profitStatementTotalRevenue', 0)) if profit_dict.get('profitStatementTotalRevenue') else None,
                'net_profit': float(profit_dict.get('profitStatementNetProfit', 0)) if profit_dict.get('profitStatementNetProfit') else None,
                'gross_profit_rate': float(profit_dict.get('profitStatementGrossProfitRate', 0)) if profit_dict.get('profitStatementGrossProfitRate') else None,
                'net_profit_rate': float(profit_dict.get('profitStatementNetProfitRate', 0)) if profit_dict.get('profitStatementNetProfitRate') else None,
                'debt_asset_ratio': float(balance_dict.get('balanceTotalLiabilities', 0)) / float(balance_dict.get('balanceTotalAssets', 1)) if balance_dict.get('balanceTotalAssets') else None,
                'current_ratio': float(balance_dict.get('balanceCurrentAssets', 0)) / float(balance_dict.get('balanceCurrentLiabilities', 1)) if balance_dict.get('balanceCurrentLiabilities') else None,
            }
            if any(v is not None for k, v in metrics.items() if k != 'year'):
                all_data.append(metrics)
        bs.logout()
        if all_data:
            df = pd.DataFrame(all_data)
            if len(df) >= 2:
                df = df.sort_values('year')
                df['revenue_growth'] = df['revenue'].pct_change()
                df['net_profit_growth'] = df['net_profit'].pct_change()
            else:
                df['revenue_growth'] = 0
                df['net_profit_growth'] = 0
            return df.fillna(0)
        return None
    except Exception:
        try:
            bs.logout()
        except Exception:
            pass
        return None


@st.cache_data(ttl=86400)
def get_financial_metrics(code, start_year, end_year):
    cache_file = f"{code}_{start_year}_{end_year}_fin.csv"
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            return df
        except Exception:
            pass
    df_bs = get_financial_metrics_baostock(code, start_year, end_year)
    if df_bs is not None and not df_bs.empty:
        df_bs.to_csv(cache_file, index=False)
        return df_bs
    df_sim = generate_synthetic_financial_data(code, start_year, end_year)
    df_sim.to_csv(cache_file, index=False)
    return df_sim


# =====================================================
# 日线特征融合与训练
# =====================================================
def get_fundamental_features_for_date(df_price, financial_df):
    if financial_df is None or financial_df.empty:
        return None, None
    fin_cols = [c for c in financial_df.columns if c != 'year']
    for col in fin_cols:
        if col in financial_df.columns:
            financial_df[col] = pd.to_numeric(financial_df[col], errors='coerce').fillna(0)
    features = []
    for date in df_price['date']:
        year = date.year
        row = financial_df[financial_df['year'] == year]
        if len(row) == 0:
            row = financial_df[financial_df['year'] < year].tail(1)
        if len(row) == 0:
            row = financial_df[fin_cols].mean().to_frame().T
        feat = row[fin_cols].values.flatten()
        if len(feat) != len(fin_cols):
            feat = np.zeros(len(fin_cols))
        features.append(feat)
    if not features:
        return None, None
    return np.array(features, dtype=np.float32), fin_cols


def prepare_data(df_price, lookback, fundamental_features=None):
    price_data = df_price['close'].values.reshape(-1, 1)
    scaler_price = MinMaxScaler(feature_range=(0, 1))
    scaled_price = scaler_price.fit_transform(price_data)
    if fundamental_features is not None and len(fundamental_features) > 0:
        scaler_fund = MinMaxScaler(feature_range=(0, 1))
        scaled_fund = scaler_fund.fit_transform(fundamental_features)
        combined = np.concatenate([scaled_price, scaled_fund], axis=1)
    else:
        combined = scaled_price
        scaler_fund = None
    X, y = [], []
    for i in range(lookback, len(combined)):
        X.append(combined[i-lookback:i, :])
        y.append(combined[i, 0])
    X = np.array(X)
    y = np.array(y)
    split = int(len(X) * 0.8)
    if split == 0 or split == len(X):
        split = max(1, int(len(X) * 0.7))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, y_train, X_test, y_test, scaler_price, scaler_fund


def build_lstm(input_shape, units, dropout):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(model, X_train, y_train, X_test, y_test, epochs):
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[es], verbose=0)
    return history


def predict_future_generic(model, model_type, df_price, scaler_price, scaler_fund,
                           lookback, days, fundamental_features=None):
    last_price = df_price['close'].values[-lookback:].reshape(-1, 1)
    scaled_last_price = scaler_price.transform(last_price)
    if fundamental_features is not None and scaler_fund is not None and len(fundamental_features) > 0:
        last_fund = fundamental_features[-lookback:]
        scaled_last_fund = scaler_fund.transform(last_fund)
        current_window = np.concatenate([scaled_last_price, scaled_last_fund], axis=1)
    else:
        current_window = scaled_last_price
    n_features = current_window.shape[1]
    pred_scaled = []
    for _ in range(days):
        if model_type == 'deep':
            inp = current_window.reshape((1, lookback, n_features))
            pred = model.predict(inp, verbose=0)[0, 0]
        else:
            flat = current_window.flatten().reshape(1, -1)
            pred = model.predict(flat)[0]
        pred_scaled.append(pred)
        if fundamental_features is not None and scaler_fund is not None and len(fundamental_features) > 0:
            last_fund_t = current_window[-1, 1:]
            new_step = np.concatenate([[pred], last_fund_t])
        else:
            new_step = np.array([pred])
        current_window = np.append(current_window[1:], [new_step], axis=0)
    pred_price = scaler_price.inverse_transform(np.array(pred_scaled).reshape(-1, 1))
    last_date = df_price['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    return pred_price.flatten(), future_dates


def auto_search_best_model(df_price, lookback, fundamental_features=None, test_size=0.2, primary_metric='directional_accuracy'):
    X_train, y_train, X_test, y_test, scaler_price, scaler_fund = prepare_data(
        df_price, lookback, fundamental_features
    )
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("数据样本不足")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    results = []
    best_score = -np.inf if primary_metric != 'mse' else np.inf
    best_model = None
    best_name = None
    best_type = None

    def train_lstm_model(X, y):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=30, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)
        return model

    def train_xgb_model(X, y):
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, verbosity=0)
        model.fit(X, y)
        return model

    def train_rf_model(X, y):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    def directional_accuracy(y_true, y_pred):
        if len(y_true) < 2:
            return np.nan
        true_diff = np.diff(y_true.flatten())
        pred_diff = np.diff(y_pred.flatten())
        correct = np.sum((true_diff * pred_diff) > 0)
        return correct / len(true_diff)

    models = {
        'LSTM': {
            'train': lambda: train_lstm_model(X_train, y_train),
            'predict': lambda mod, X: mod.predict(X, verbose=0).flatten(),
            'type': 'deep'
        },
        'XGBoost': {
            'train': lambda: train_xgb_model(X_train_flat, y_train),
            'predict': lambda mod, X: mod.predict(X).flatten(),
            'type': 'tree'
        },
        'RandomForest': {
            'train': lambda: train_rf_model(X_train_flat, y_train),
            'predict': lambda mod, X: mod.predict(X).flatten(),
            'type': 'tree'
        }
    }
    for name, cfg in models.items():
        try:
            start_time = time.time()
            mod = cfg['train']()
            if cfg['type'] == 'deep':
                y_pred = cfg['predict'](mod, X_test)
            else:
                y_pred = cfg['predict'](mod, X_test_flat)
            elapsed = time.time() - start_time
            mse = mean_squared_error(y_test, y_pred)
            dir_acc = directional_accuracy(y_test, y_pred)
            result = {'model': name, 'mse': mse, 'directional_accuracy': dir_acc, 'train_time_sec': elapsed}
            results.append(result)
            score = result.get(primary_metric, -np.inf)
            if primary_metric == 'mse':
                if score < best_score:
                    best_score, best_model, best_name, best_type = score, mod, name, cfg['type']
            else:
                if score > best_score:
                    best_score, best_model, best_name, best_type = score, mod, name, cfg['type']
        except Exception as e:
            st.warning(f"模型 {name} 训练失败: {str(e)}")
            continue
    if not results:
        st.error("所有模型训练失败")
        return None, None, None, None, None
    results_df = pd.DataFrame(results).sort_values(by=primary_metric, ascending=(primary_metric == 'mse')).reset_index(drop=True)
    best_metrics = results_df.iloc[0].to_dict()
    return best_name, best_model, results_df, best_metrics, best_type


# =====================================================
# 进场信号检测
# =====================================================
def detect_entry_signals(df, use_trend_filter=True, use_ma_confirm=True,
                         ma_confirm_period=20, max_deviation=5.0,
                         use_volume_confirm=True, volume_ratio=1.5,
                         use_td9=False):
    """
    检测进场信号，返回 dict：
      {
        'trend_ok': bool,        # EMA150 > SMA200
        'ma_ok': bool,           # 站上均线且偏离度合理
        'volume_ok': bool,       # 放量确认
        'td9_ok': bool,          # 九转低9（可选）
        'details': dict,         # 各指标的具体数值
      }
    返回 None 表示数据不足。
    """
    if df is None or len(df) < 210:
        # 趋势过滤需要至少 200 天数据
        if use_trend_filter:
            return None
        if len(df) < max(ma_confirm_period, 50) + 5:
            return None

    df = df.copy()
    close = df['close']
    volume = pd.to_numeric(df['volume'], errors='coerce')

    details = {}

    # ===== 1. 趋势过滤：EMA150 > SMA200 =====
    if use_trend_filter:
        if len(df) < 200:
            return None
        ema150 = close.ewm(span=150, adjust=False).mean()
        sma200 = close.rolling(window=200).mean()
        trend_ok = bool(ema150.iloc[-1] > sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else False
        details['EMA150'] = round(float(ema150.iloc[-1]), 2)
        details['SMA200'] = round(float(sma200.iloc[-1]), 2) if not pd.isna(sma200.iloc[-1]) else None
    else:
        trend_ok = True
        details['EMA150'] = None
        details['SMA200'] = None

    # ===== 2. 均线确认：站上均线且偏离度合理 =====
    if use_ma_confirm:
        if len(df) < ma_confirm_period + 5:
            return None
        ma = close.rolling(window=ma_confirm_period).mean()
        latest_close = float(close.iloc[-1])
        latest_ma = float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else None
        if latest_ma is None or latest_ma <= 0:
            return None
        deviation = (latest_close - latest_ma) / latest_ma * 100  # 偏离度 %
        ma_ok = (latest_close > latest_ma) and (deviation <= max_deviation)
        details['MA周期'] = ma_confirm_period
        details['MA值'] = round(latest_ma, 2)
        details['偏离度%'] = round(deviation, 2)
    else:
        ma_ok = True
        details['MA周期'] = None
        details['MA值'] = None
        details['偏离度%'] = None

    # ===== 3. 量价共振：当日成交量 > 50日均量 × 倍数 =====
    if use_volume_confirm:
        if len(df) < 50:
            return None
        vol_ma50 = volume.rolling(window=50).mean()
        latest_vol = float(volume.iloc[-1]) if not pd.isna(volume.iloc[-1]) else 0
        latest_vol_ma = float(vol_ma50.iloc[-1]) if not pd.isna(vol_ma50.iloc[-1]) else 0
        if latest_vol_ma <= 0:
            return None
        vol_ratio_actual = latest_vol / latest_vol_ma
        volume_ok = vol_ratio_actual >= volume_ratio
        details['当日成交量'] = round(latest_vol, 0)
        details['50日均量'] = round(latest_vol_ma, 0)
        details['量比'] = round(vol_ratio_actual, 2)
    else:
        volume_ok = True
        details['当日成交量'] = None
        details['50日均量'] = None
        details['量比'] = None

    # ===== 4. 九转序列（TD9 低9）=====
    if use_td9:
        if len(df) < 20:
            return None
        td_count = 0
        for i in range(len(close) - 1, 4, -1):
            if close.iloc[i] < close.iloc[i - 4]:
                td_count += 1
            else:
                break
        td9_ok = (td_count >= 9)
        details['TD9计数'] = td_count
    else:
        td9_ok = True
        details['TD9计数'] = None

    return {
        'trend_ok': trend_ok,
        'ma_ok': ma_ok,
        'volume_ok': volume_ok,
        'td9_ok': td9_ok,
        'details': details,
    }


def format_signal_desc(signal_result):
    """把信号结果格式化为可读字符串"""
    if signal_result is None:
        return '数据不足'
    parts = []
    parts.append('趋势✓' if signal_result['trend_ok'] else '趋势✗')
    parts.append('均线✓' if signal_result['ma_ok'] else '均线✗')
    parts.append('放量✓' if signal_result['volume_ok'] else '放量✗')
    if signal_result['details'].get('TD9计数') is not None:
        parts.append(f"TD9={signal_result['details']['TD9计数']}")
    return ' '.join(parts)


# =====================================================
# 形态相似性预测
# =====================================================
def predict_stock_by_similarity(df, lookback=60, pred_days=20, top_k=10, min_sim=0.3):
    if len(df) < lookback + pred_days + 10:
        return None

    close = df['close'].values
    target = close[-lookback:]
    target_norm = (target - np.mean(target)) / (np.std(target) + 1e-8)

    matches = []
    max_start = len(close) - lookback - pred_days
    for i in range(0, max_start):
        hist = close[i:i + lookback]
        hist_norm = (hist - np.mean(hist)) / (np.std(hist) + 1e-8)
        sim = cosine(target_norm, hist_norm)
        if sim >= min_sim and sim > 0:
            future_ret = (close[i + lookback + pred_days - 1] - close[i + lookback - 1]) / close[i + lookback - 1]
            matches.append({'similarity': sim, 'future_return': future_ret})

    if not matches:
        return None

    matches_df = pd.DataFrame(matches).sort_values('similarity', ascending=False)
    top_matches = matches_df.head(top_k)
    rets = top_matches['future_return'].values * 100

    return {
        'median_ret': float(np.median(rets)),
        'mean_ret': float(np.mean(rets)),
        'up_ratio': float((rets > 0).mean()),
        'p25': float(np.percentile(rets, 25)),
        'p75': float(np.percentile(rets, 75)),
        'avg_sim': float(top_matches['similarity'].mean()),
        'match_count': len(matches),
    }


# =====================================================
# 扫描结果缓存
# =====================================================
def get_scan_cache_key(**kwargs):
    key_str = '|'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()


def load_scan_cache(cache_key, cache_hours):
    path = os.path.join(DATA_CACHE_DIR, f"scan_{cache_key}.pkl")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        if time.time() - cache.get('time', 0) > cache_hours * 3600:
            return None, None
        return cache.get('result'), cache.get('time')
    except Exception:
        return None, None


def save_scan_cache(cache_key, result_df):
    path = os.path.join(DATA_CACHE_DIR, f"scan_{cache_key}.pkl")
    try:
        with open(path, 'wb') as f:
            pickle.dump({'time': time.time(), 'result': result_df}, f)
    except Exception as e:
        print(f"缓存保存失败: {e}")


# =====================================================
# 股票列表获取
# =====================================================
def get_stock_list_fallback_baostock():
    try:
        lg = bs.login()
        if lg.error_code != '0':
            return None
        rs = bs.query_all_stock(day=datetime.now().strftime("%Y-%m-%d"))
        rows = []
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            code_full = row[0]
            name = row[1] if len(row) > 1 else code_full
            if '.' in code_full:
                code = code_full.split('.')[1]
            else:
                code = code_full
            if len(code) == 6 and code.isdigit():
                rows.append({'symbol': code, 'name': name})
        bs.logout()
        if rows:
            return pd.DataFrame(rows)
        return None
    except Exception:
        try:
            bs.logout()
        except Exception:
            pass
        return None


@retry_on_failure(max_retries=3, base_delay=1.5, backoff_factor=2.0)
def fetch_all_stocks_akshare():
    return ak.stock_info_a_code_name()


def get_all_stocks_safe():
    try:
        df = call_with_timeout(fetch_all_stocks_akshare, timeout=45)
        if df is not None and not df.empty:
            if 'code' in df.columns and 'name' in df.columns:
                df = df.rename(columns={'code': 'symbol', 'name': 'name'})
            elif '代码' in df.columns and '名称' in df.columns:
                df = df.rename(columns={'代码': 'symbol', '名称': 'name'})
            return df
    except Exception as e:
        st.warning(f"akshare 获取股票列表失败: {e}")
    st.warning("降级使用 Baostock 获取股票列表...")
    df_fallback = get_stock_list_fallback_baostock()
    if df_fallback is not None and not df_fallback.empty:
        return df_fallback
    return None


def get_stock_list_by_board(board_type, board_name):
    try:
        if board_type == "全部" or board_name == "全部":
            df = get_all_stocks_safe()
            if df is None:
                return pd.DataFrame()
            return df
        elif board_type == "行业板块":
            df = ak.stock_board_industry_cons_em(symbol=board_name)
            if '代码' in df.columns and '名称' in df.columns:
                df = df.rename(columns={'代码': 'symbol', '名称': 'name'})
            return df
        else:
            df = ak.stock_board_concept_cons_em(symbol=board_name)
            if '代码' in df.columns and '名称' in df.columns:
                df = df.rename(columns={'代码': 'symbol', '名称': 'name'})
            return df
    except Exception as e:
        st.error(f"获取板块成分股失败: {e}")
        return pd.DataFrame()


# =====================================================
# 单只股票处理
# =====================================================
def process_one_stock(symbol, name, start_date, end_date, pro,
                      lookback_window, pred_days, similarity_top_k, min_similarity,
                      min_price,
                      use_trend_filter, use_ma_confirm, ma_confirm_period, max_deviation,
                      use_volume_confirm, volume_ratio, use_td9,
                      use_turnover_filter, min_turnover, max_turnover,
                      turnover_map):
    """
    返回 ('OK', record) 或 ('SKIP', symbol, name, reason)
    """
    try:
        # ===== 换手率过滤 =====
        if use_turnover_filter and turnover_map is not None and not turnover_map.empty:
            row_t = turnover_map[turnover_map['symbol'] == symbol]
            if row_t.empty:
                return ('SKIP', symbol, name, '换手率缺失')
            turnover_val = float(row_t.iloc[0]['换手率'])
            if pd.isna(turnover_val):
                return ('SKIP', symbol, name, '换手率缺失')
            if turnover_val < min_turnover:
                return ('SKIP', symbol, name, f'换手率过低({turnover_val:.2f}%)')
            if turnover_val > max_turnover:
                return ('SKIP', symbol, name, f'换手率过高({turnover_val:.2f}%)')
        else:
            turnover_val = None

        df = get_price_data(symbol, start_date, end_date, _pro=pro)
        if df is None or df.empty:
            return ('SKIP', symbol, name, '数据为空')

        # 趋势过滤需要至少 200 天，统一要求 220 天以上更保险
        min_rows_needed = max(220, lookback_window + pred_days + 10)
        if len(df) < min_rows_needed:
            return ('SKIP', symbol, name, f'数据不足({len(df)}行<{min_rows_needed})')

        close_series = pd.to_numeric(df['close'], errors='coerce').dropna()
        if len(close_series) < min_rows_needed:
            return ('SKIP', symbol, name, f'收盘价有效值不足({len(close_series)})')
        if (close_series <= 0).any():
            return ('SKIP', symbol, name, '存在非正价格')

        latest_price = df.iloc[-1]['close']
        if pd.isna(latest_price):
            return ('SKIP', symbol, name, '最新价为空')
        if latest_price > 500:
            return ('SKIP', symbol, name, f'价格过高({latest_price:.1f})')
        if latest_price < 1:
            return ('SKIP', symbol, name, f'价格过低({latest_price:.2f})')
        if latest_price < min_price:
            return ('SKIP', symbol, name, f'价格低于{min_price}')

        # ===== 进场信号检测 =====
        signal_result = detect_entry_signals(
            df,
            use_trend_filter=use_trend_filter,
            use_ma_confirm=use_ma_confirm,
            ma_confirm_period=ma_confirm_period,
            max_deviation=max_deviation,
            use_volume_confirm=use_volume_confirm,
            volume_ratio=volume_ratio,
            use_td9=use_td9,
        )
        if signal_result is None:
            return ('SKIP', symbol, name, '信号检测数据不足')

        # 三个核心条件必须同时满足（趋势+均线+放量）
        if not signal_result['trend_ok']:
            return ('SKIP', symbol, name, '趋势向下(EMA150<SMA200)')
        if not signal_result['ma_ok']:
            return ('SKIP', symbol, name, '未站上均线或偏离过大')
        if not signal_result['volume_ok']:
            return ('SKIP', symbol, name, '未放量')
        if use_td9 and not signal_result['td9_ok']:
            return ('SKIP', symbol, name, f"TD9不足({signal_result['details']['TD9计数']})")

        # ===== 形态相似度（作为排序加分）=====
        sim_result = predict_stock_by_similarity(
            df, lookback=lookback_window, pred_days=pred_days,
            top_k=similarity_top_k, min_sim=min_similarity
        )
        if sim_result is None:
            # 没有相似片段也可以保留（只是排序分低）
            sim_median = 0.0
            sim_mean = 0.0
            sim_up_ratio = 0.5
            sim_avg_sim = 0.0
            sim_match = 0
        else:
            sim_median = sim_result['median_ret']
            sim_mean = sim_result['mean_ret']
            sim_up_ratio = sim_result['up_ratio']
            sim_avg_sim = sim_result['avg_sim']
            sim_match = sim_result['match_count']

        # ===== 综合评分 =====
        # 基础分 = 上涨概率 × 平均相似度
        # 加分项：量比越高、偏离度越小越好
        base_score = (sim_up_ratio * 100) * sim_avg_sim

        # 量比加分：1.5倍为基准，每高0.5加5分，封顶20分
        vol_ratio_val = signal_result['details'].get('量比') or 1.0
        vol_bonus = min(20, max(0, (vol_ratio_val - 1.5) * 10))

        # 偏离度加分：离均线越近越好，偏离0%加10分，偏离5%加0分
        dev_val = signal_result['details'].get('偏离度%') or 0
        dev_bonus = max(0, 10 - dev_val * 2)

        composite = base_score + vol_bonus + dev_bonus

        record = {
            '代码': symbol,
            '名称': name,
            '最新价': round(float(df.iloc[-1]['close']), 2),
            '换手率%': round(turnover_val, 2) if turnover_val is not None else None,
            '综合评分': round(composite, 2),
            '量比': round(vol_ratio_val, 2),
            '偏离度%': round(dev_val, 2),
            'EMA150': signal_result['details'].get('EMA150'),
            'SMA200': signal_result['details'].get('SMA200'),
            f'预测{pred_days}日涨跌%(中位)': round(sim_median, 2),
            f'预测{pred_days}日涨跌%(均值)': round(sim_mean, 2),
            '上涨概率': round(sim_up_ratio * 100, 1),
            '匹配片段数': sim_match,
            '平均相似度': round(sim_avg_sim, 4),
            '信号': format_signal_desc(signal_result),
        }
        return ('OK', record)

    except Exception as e:
        return ('SKIP', symbol, name, f'异常:{type(e).__name__}:{str(e)[:40]}')


# =====================================================
# 扫描核心函数
# =====================================================
def scan_uptrend_stocks_with_similarity(
    code_list, start_date, end_date,
    lookback_window=60, pred_days=20, similarity_top_k=10, min_similarity=0.3,
    max_stocks=None, progress=None, pro=None,
    min_price=5.0,
    use_trend_filter=True, use_ma_confirm=True, ma_confirm_period=20, max_deviation=5.0,
    use_volume_confirm=True, volume_ratio=1.5, use_td9=False,
    use_turnover_filter=False, min_turnover=0.0, max_turnover=999.0,
    turnover_map=None,
    stop_check=None, save_partial=None, save_every=20, per_stock_timeout=25
):
    if 'symbol' not in code_list.columns:
        if '代码' in code_list.columns:
            code_list = code_list.rename(columns={'代码': 'symbol'})
        elif 'code' in code_list.columns:
            code_list = code_list.rename(columns={'code': 'symbol'})
        else:
            st.error("股票列表缺少 'symbol' 或 '代码' 列")
            return pd.DataFrame()
    if 'name' not in code_list.columns:
        if '名称' in code_list.columns:
            code_list = code_list.rename(columns={'名称': 'name'})
        else:
            code_list['name'] = code_list['symbol']

    if max_stocks is not None and max_stocks < len(code_list):
        scan_list = code_list.head(max_stocks).reset_index(drop=True)
        total = len(scan_list)
        st.info(f"仅扫描前 {total} 只股票（共 {len(code_list)} 只）")
    else:
        scan_list = code_list.reset_index(drop=True)
        total = len(scan_list)
        st.info(f"扫描全部 {total} 只股票")

    results = []
    skipped = []
    for idx, row in scan_list.iterrows():
        if stop_check is not None and stop_check():
            if progress is not None:
                progress.empty()
            st.warning(f"⏹️ 扫描已手动停止，已处理 {idx}/{total} 只股票")
            break

        symbol = str(row['symbol'])
        name = row['name']
        if symbol.startswith(('sh', 'sz', 'SH', 'SZ')):
            symbol = symbol[2:]
        if len(symbol) != 6 or not symbol.isdigit():
            continue

        record = call_with_timeout(
            process_one_stock,
            args=(symbol, name, start_date, end_date, pro,
                  lookback_window, pred_days, similarity_top_k, min_similarity,
                  min_price,
                  use_trend_filter, use_ma_confirm, ma_confirm_period, max_deviation,
                  use_volume_confirm, volume_ratio, use_td9,
                  use_turnover_filter, min_turnover, max_turnover,
                  turnover_map),
            timeout=per_stock_timeout
        )

        if record is None:
            skipped.append({'代码': symbol, '名称': name, '原因': '超时'})
        elif record[0] == 'OK':
            results.append(record[1])
        else:
            skipped.append({'代码': record[1], '名称': record[2], '原因': record[3]})

        if save_partial is not None and len(results) > 0 and len(results) % save_every == 0:
            try:
                save_partial(pd.DataFrame(results))
            except Exception:
                pass

        if progress is not None:
            try:
                progress.progress((idx + 1) / total, text=f"已扫描 {idx+1}/{total}")
            except Exception:
                pass

    if save_partial is not None and len(results) > 0:
        try:
            save_partial(pd.DataFrame(results))
        except Exception:
            pass

    result_df = pd.DataFrame(results)
    st.session_state['scan_skipped'] = skipped
    return result_df


# =====================================================
# 分时预测（保持原样）
# =====================================================
def get_intraday_cache_path(code, period):
    return os.path.join(DATA_CACHE_DIR, f"{code}_{period}_intraday.csv")


def load_intraday_from_cache(code, period):
    path = get_intraday_cache_path(code, period)
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=['time'])
        except Exception:
            return None
    return None


def save_intraday_to_cache(df, code, period):
    path = get_intraday_cache_path(code, period)
    df.to_csv(path, index=False)


def fetch_intraday_day_baostock(code, period, date):
    bs_code = format_baostock_code(code)
    freq_map = {'5': '5', '15': '15', '30': '30', '60': '60'}
    if period not in freq_map:
        return None, f"不支持的周期: {period}"
    try:
        lg = bs.login()
        if lg.error_code != '0':
            return None, f"Baostock 登录失败: {lg.error_msg}"
        rs = bs.query_history_k_data_plus(bs_code, fields="date,time,open,high,low,close,volume",
                                          start_date=date.strftime("%Y-%m-%d"),
                                          end_date=date.strftime("%Y-%m-%d"),
                                          frequency=freq_map[period], adjustflag="3")
        if rs.error_code != '0':
            bs.logout()
            return None, f"查询失败: {rs.error_msg}"
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()
        if not data_list:
            return None, f"{date} 无数据"
        rows = []
        for row in data_list:
            date_str = row[0]
            time_str = row[1]
            time_part = time_str[:6] if len(time_str) >= 6 else time_str.zfill(6)
            dt_str = f"{date_str} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
            dt = pd.to_datetime(dt_str, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            if pd.isna(dt):
                continue
            rows.append({'time': dt, 'open': float(row[2]), 'high': float(row[3]),
                         'low': float(row[4]), 'close': float(row[5]), 'volume': float(row[6])})
        if not rows:
            return None, f"{date} 无有效数据"
        df = pd.DataFrame(rows)
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()
        df = df[((df['time'].dt.time >= morning_start) & (df['time'].dt.time <= morning_end)) |
                ((df['time'].dt.time >= afternoon_start) & (df['time'].dt.time <= afternoon_end))]
        if df.empty:
            return None, f"{date} 交易时段内无数据"
        return df, None
    except Exception as e:
        try:
            bs.logout()
        except Exception:
            pass
        return None, f"请求异常: {str(e)}"


def get_intraday_data(code, period="5", days=30, force_refresh=False, status=None):
    today = datetime.now().date()
    df_cache = load_intraday_from_cache(code, period)
    error_details = []
    if force_refresh:
        if status:
            status.update(label="🔄 强制刷新", state="running")
        start_date = today - timedelta(days=days)
        new_dfs = []
        current_date = start_date
        while current_date <= today:
            if current_date.weekday() < 5:
                df_day, err = fetch_intraday_day_baostock(code, period, current_date)
                if df_day is not None:
                    new_dfs.append(df_day)
                else:
                    error_details.append((current_date, err))
            current_date += timedelta(days=1)
        if new_dfs:
            df_combined = pd.concat(new_dfs, ignore_index=True).drop_duplicates(subset=['time']).sort_values('time')
            save_intraday_to_cache(df_combined, code, period)
            return df_combined, error_details
        return None, error_details
    if df_cache is not None and not df_cache.empty:
        last_date = df_cache['time'].max().date()
        if (today - last_date).days > 1:
            start_date = last_date + timedelta(days=1)
            new_dfs = []
            current_date = start_date
            while current_date <= today:
                if current_date.weekday() < 5:
                    df_day, err = fetch_intraday_day_baostock(code, period, current_date)
                    if df_day is not None:
                        new_dfs.append(df_day)
                    else:
                        error_details.append((current_date, err))
                current_date += timedelta(days=1)
            if new_dfs:
                df_new = pd.concat(new_dfs, ignore_index=True).drop_duplicates(subset=['time']).sort_values('time')
                df_combined = pd.concat([df_cache, df_new], ignore_index=True).drop_duplicates(subset=['time']).sort_values('time')
                save_intraday_to_cache(df_combined, code, period)
                return df_combined, error_details
            return df_cache, error_details
        return df_cache, error_details
    else:
        start_date = today - timedelta(days=days)
        new_dfs = []
        current_date = start_date
        while current_date <= today:
            if current_date.weekday() < 5:
                df_day, err = fetch_intraday_day_baostock(code, period, current_date)
                if df_day is not None:
                    new_dfs.append(df_day)
                else:
                    error_details.append((current_date, err))
            current_date += timedelta(days=1)
        if new_dfs:
            df_combined = pd.concat(new_dfs, ignore_index=True).drop_duplicates(subset=['time']).sort_values('time')
            save_intraday_to_cache(df_combined, code, period)
            return df_combined, error_details
        return None, error_details


def prepare_intraday_data(df, lookback):
    data = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8) if len(X) > 10 else max(1, int(len(X) * 0.7))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train = X_train.reshape(-1, lookback, 1)
    X_test = X_test.reshape(-1, lookback, 1)
    return X_train, y_train, X_test, y_test, scaler


def build_intraday_lstm(input_shape, units=50, dropout=0.2):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def predict_intraday_future(model, df, scaler, lookback, steps):
    last_data = df['close'].values[-lookback:].reshape(-1, 1)
    scaled_last = scaler.transform(last_data)
    current = scaled_last.copy()
    pred_scaled = []
    for _ in range(steps):
        inp = current.reshape((1, current.shape[0], 1))
        nxt = model.predict(inp, verbose=0)[0, 0]
        pred_scaled.append(nxt)
        current = np.append(current[1:], [[nxt]], axis=0)
    pred = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1))
    return pred.flatten()


def generate_trading_timestamps(start_time, period_minutes, num_steps):
    morning_start = datetime.strptime("09:30", "%H:%M").time()
    morning_end = datetime.strptime("11:30", "%H:%M").time()
    afternoon_start = datetime.strptime("13:00", "%H:%M").time()
    afternoon_end = datetime.strptime("15:00", "%H:%M").time()
    times = []
    current = start_time
    step = timedelta(minutes=period_minutes)
    while len(times) < num_steps:
        t_time = current.time()
        if morning_start <= t_time <= morning_end:
            times.append(current)
            current += step
            if current.time() > morning_end:
                current = current.replace(hour=13, minute=0, second=0, microsecond=0)
        elif afternoon_start <= t_time <= afternoon_end:
            times.append(current)
            current += step
            if current.time() > afternoon_end:
                current = current + timedelta(days=1)
                current = current.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            if t_time < morning_start:
                current = current.replace(hour=9, minute=30, second=0, microsecond=0)
            elif morning_end < t_time < afternoon_start:
                current = current.replace(hour=13, minute=0, second=0, microsecond=0)
            else:
                current = current + timedelta(days=1)
                current = current.replace(hour=9, minute=30, second=0, microsecond=0)
    return times[:num_steps]


def show_intraday_prediction(code, period, lookback, predict_steps, predict_full_day=False, refresh_cache=False):
    st.subheader(f"⏱️ 分时预测（{period}分钟周期）")
    with st.status("⏳ 启动分时预测...", expanded=True) as status:
        status.update(label="📊 获取数据", state="running")
        df, error_details = get_intraday_data(code, period, days=30, force_refresh=refresh_cache, status=status)
        if df is None:
            status.update(label="❌ 数据获取失败", state="error")
            st.error(f"无法获取 {code} 的分时数据。请检查代码或稍后重试。")
            return
        status.update(label=f"✅ 获取到 {len(df)} 条数据", state="running")
        if predict_full_day:
            now = datetime.now()
            if now.weekday() < 5:
                morning_end = now.replace(hour=11, minute=30, second=0, microsecond=0)
                afternoon_start = now.replace(hour=13, minute=0, second=0, microsecond=0)
                afternoon_end = now.replace(hour=15, minute=0, second=0, microsecond=0)
                if now < now.replace(hour=9, minute=30, second=0, microsecond=0):
                    total_minutes = 240
                elif now <= morning_end:
                    total_minutes = (morning_end - now).total_seconds() / 60 + 120
                elif now < afternoon_start:
                    total_minutes = 120
                elif now <= afternoon_end:
                    total_minutes = (afternoon_end - now).total_seconds() / 60
                else:
                    total_minutes = 240
                predict_steps = max(1, min(480, int(np.ceil(total_minutes / int(period)))))
                st.info(f"预计剩余 {predict_steps} 步")
            else:
                predict_steps = int(np.ceil(240 / int(period)))
        status.update(label="📊 准备训练数据", state="running")
        X_train, y_train, X_test, y_test, scaler = prepare_intraday_data(df, lookback)
        if len(X_train) < 10:
            status.update(label="❌ 数据不足", state="error")
            st.error("数据不足")
            return
        status.update(label="🧠 构建并训练 LSTM", state="running")
        model = build_intraday_lstm((lookback, 1), units=50, dropout=0.2)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                            validation_data=(X_test, y_test), callbacks=[es], verbose=0)
        status.update(label="📈 生成预测", state="running")
        pred_vals = predict_intraday_future(model, df, scaler, lookback, predict_steps)
        last_time = df['time'].iloc[-1]
        future_times = generate_trading_timestamps(last_time, int(period), predict_steps)
        st.subheader(f"📈 未来 {predict_steps} 步预测")
        st.dataframe(pd.DataFrame({'时间': future_times, '预测收盘价': pred_vals.round(2)}))
        fig, ax = plt.subplots(figsize=(14, 6))
        hist_df = df.iloc[-120:]
        ax.plot(hist_df['time'], hist_df['close'], label='历史', color='blue')
        ax.plot(future_times, pred_vals, 'r--o', label='预测', markersize=3)
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        status.update(label="✅ 完成", state="complete")


# =====================================================
# 主程序路由
# =====================================================
if prediction_mode == "日线预测（中长期）":
    if train_btn:
        with st.status("⏳ 训练...", expanded=True) as status:
            df_price = get_price_data(stock_code, start_date, end_date, _pro=pro)
            if df_price is None or df_price.empty:
                status.update(label="❌ 数据获取失败", state="error")
                st.stop()
            fundamental_features = None
            if use_fundamental:
                financial_df = get_financial_metrics(stock_code, start_date.year, end_date.year)
                if financial_df is not None and not financial_df.empty:
                    fundamental_features, _ = get_fundamental_features_for_date(df_price, financial_df)
            X_train, y_train, X_test, y_test, scaler_price, scaler_fund = prepare_data(
                df_price, lookback, fundamental_features
            )
            if len(X_train) == 0:
                st.error("数据不足")
                st.stop()
            model = build_lstm((lookback, X_train.shape[2]), lstm_units, dropout)
            history = train_lstm(model, X_train, y_train, X_test, y_test, epochs)
            model.save('stock_lstm_model.h5')
            pred_vals, future_dates = predict_future_generic(
                model, 'deep', df_price, scaler_price, scaler_fund,
                lookback, predict_days, fundamental_features
            )
            st.subheader(f"📈 未来 {predict_days} 天预测")
            st.dataframe(pd.DataFrame({'日期': future_dates, '预测收盘价': pred_vals.round(2)}))
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.plot(df_price['date'].iloc[-120:], df_price['close'].iloc[-120:], label='历史')
            ax3.plot(future_dates, pred_vals, 'r--o', label='预测')
            ax3.legend()
            st.pyplot(fig3)
            status.update(label="✅ 完成", state="complete")
    elif load_btn:
        if not os.path.exists('stock_lstm_model.h5'):
            st.error("未找到模型文件")
        else:
            df_price = get_price_data(stock_code, start_date, end_date, _pro=pro)
            model = load_model('stock_lstm_model.h5')
            fundamental_features = None
            if use_fundamental:
                financial_df = get_financial_metrics(stock_code, start_date.year, end_date.year)
                if financial_df is not None and not financial_df.empty:
                    fundamental_features, _ = get_fundamental_features_for_date(df_price, financial_df)
            X_train, y_train, X_test, y_test, scaler_price, scaler_fund = prepare_data(df_price, lookback, fundamental_features)
            pred_vals, future_dates = predict_future_generic(model, 'deep', df_price, scaler_price, scaler_fund,
                                                            lookback, predict_days, fundamental_features)
            st.subheader(f"📈 未来 {predict_days} 天预测")
            st.dataframe(pd.DataFrame({'日期': future_dates, '预测收盘价': pred_vals.round(2)}))
    elif search_btn:
        with st.spinner("🔍 搜索最佳模型..."):
            df_price = get_price_data(stock_code, start_date, end_date, _pro=pro)
            fundamental_features = None
            if use_fundamental:
                financial_df = get_financial_metrics(stock_code, start_date.year, end_date.year)
                if financial_df is not None and not financial_df.empty:
                    fundamental_features, _ = get_fundamental_features_for_date(df_price, financial_df)
            try:
                best_name, best_model, results_df, best_metrics, best_type = auto_search_best_model(
                    df_price, lookback, fundamental_features, primary_metric='directional_accuracy'
                )
                if best_name:
                    st.success(f"🏆 最佳模型: {best_name}")
                    st.dataframe(results_df)
                    X_train, y_train, X_test, y_test, scaler_price, scaler_fund = prepare_data(df_price, lookback, fundamental_features)
                    pred_vals, future_dates = predict_future_generic(best_model, best_type, df_price, scaler_price, scaler_fund,
                                                                    lookback, predict_days, fundamental_features)
                    st.dataframe(pd.DataFrame({'日期': future_dates, '预测收盘价': pred_vals.round(2)}))
            except Exception as e:
                st.error(f"出错: {e}")
    else:
        st.info("👈 请点击侧边栏按钮开始日线预测。")

elif prediction_mode == "分时预测（日内）":
    show_intraday_prediction(stock_code, period, lookback_min, predict_steps, predict_full_day, refresh_btn)

elif prediction_mode == "📊 统计与相似性分析（日线）":
    st.subheader("📐 统计与相似性分析")
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=3*365)
    df_price = get_price_data(stock_code, start_dt, end_dt, _pro=pro)
    if df_price is None or df_price.empty:
        st.error("无法加载数据")
    else:
        st.success(f"加载 {len(df_price)} 条日线数据")
        if stat_btn:
            # 简化统计：展示相似度分布
            close_prices = df_price['close'].values
            lookback = stat_lookback
            if len(close_prices) < lookback + simulate_days + 10:
                st.error("数据不足")
            else:
                target = close_prices[-lookback:]
                target_norm = (target - np.mean(target)) / (np.std(target) + 1e-8)
                sims = []
                for i in range(0, len(close_prices) - lookback - simulate_days):
                    hist = close_prices[i:i+lookback]
                    hist_norm = (hist - np.mean(hist)) / (np.std(hist) + 1e-8)
                    sim = cosine(target_norm, hist_norm)
                    future_ret = (close_prices[i+lookback+simulate_days-1] - close_prices[i+lookback-1]) / close_prices[i+lookback-1] * 100
                    sims.append({'similarity': sim, 'future_return': future_ret})
                sims_df = pd.DataFrame(sims).sort_values('similarity', ascending=False)
                st.dataframe(sims_df.head(top_k))
                pos = (sims_df['future_return'] > 0).sum()
                st.metric("后续上涨占比", f"{pos/len(sims_df)*100:.1f}%")
        else:
            st.info("👈 点击侧边栏 '执行统计分析'")

else:  # 上升趋势扫描
    st.subheader("🔍 上升趋势股票扫描（量价共振 + 均线确认 + 趋势过滤）")
    if 'last_scan_time' not in st.session_state:
        st.session_state.last_scan_time = None
    if 'scan_result' not in st.session_state:
        st.session_state.scan_result = None
    if 'scan_skipped' not in st.session_state:
        st.session_state.scan_skipped = None
    if 'simulated_codes' not in st.session_state:
        st.session_state.simulated_codes = set()

    auto_scan = False
    if auto_scan_interval > 0 and st.session_state.last_scan_time is not None:
        elapsed = (datetime.now() - st.session_state.last_scan_time).total_seconds() / 3600
        if elapsed >= auto_scan_interval:
            auto_scan = True
    if scan_btn:
        auto_scan = True
        st.session_state.scan_result = None
        st.session_state.partial_scan_result = None
        st.session_state.scan_stop_requested = False
        st.session_state.scan_skipped = None
        st.session_state.simulated_codes = set()

    if not auto_scan and st.session_state.scan_stop_requested and st.session_state.partial_scan_result is not None:
        st.info("⏹️ 扫描已停止，以下是已扫描到的部分结果：")
        st.session_state.scan_result = st.session_state.partial_scan_result

    if auto_scan:
        # 获取股票列表
        if board_type == "全部" or board_name == "全部":
            with st.spinner("获取股票列表..."):
                all_stocks = get_all_stocks_safe()
            if all_stocks is None or all_stocks.empty:
                st.error("获取股票列表失败")
                st.stop()
            if only_main_board:
                before = len(all_stocks)
                all_stocks = filter_main_board_stocks(all_stocks, exclude_st=True)
                st.info(f"共 {before} 只 → 沪深主板 {len(all_stocks)} 只")
        else:
            with st.spinner(f"获取板块 {board_name}..."):
                stock_df = get_stock_list_by_board(board_type, board_name)
            if stock_df is None or stock_df.empty:
                st.error("获取板块失败")
                st.stop()
            all_stocks = stock_df
            if only_main_board:
                before = len(all_stocks)
                all_stocks = filter_main_board_stocks(all_stocks, exclude_st=True)
                st.info(f"板块 {board_name}：{before} 只 → 沪深主板 {len(all_stocks)} 只")

        if all_stocks is None or all_stocks.empty:
            st.warning("过滤后没有可扫描的股票")
            st.stop()

        # 换手率
        turnover_map = pd.DataFrame(columns=['symbol', '换手率'])
        if use_turnover_filter:
            with st.spinner("获取换手率快照..."):
                turnover_map = get_turnover_map()
            if turnover_map.empty:
                st.warning("换手率数据获取失败，跳过换手率过滤")
                use_turnover_filter = False

        max_stocks_actual = len(all_stocks) if scan_all else min(max_stocks, len(all_stocks))

        cache_key = get_scan_cache_key(
            board_type=board_type, board_name=board_name, scan_start_date=str(scan_start_date),
            lookback_window=lookback_window, pred_days=pred_days,
            similarity_top_k=similarity_top_k, min_similarity=min_similarity,
            max_stocks_actual=max_stocks_actual, only_main_board=only_main_board,
            use_trend_filter=use_trend_filter, use_ma_confirm=use_ma_confirm,
            ma_confirm_period=ma_confirm_period, max_deviation=max_deviation,
            use_volume_confirm=use_volume_confirm, volume_ratio=volume_ratio,
            use_td9=use_td9, use_turnover_filter=use_turnover_filter,
            min_turnover=min_turnover, max_turnover=max_turnover, min_price=min_price
        )

        cached_result, cached_time = load_scan_cache(cache_key, cache_hours)
        if cached_result is not None and not force_rescan:
            st.session_state.scan_result = cached_result
            st.session_state.last_scan_time = datetime.fromtimestamp(cached_time)
            st.success(f"✅ 使用缓存（{datetime.fromtimestamp(cached_time).strftime('%Y-%m-%d %H:%M')}）")
        else:
            with st.spinner("🔄 扫描中..."):
                progress_bar = st.progress(0, text="开始扫描...")

                def _save_partial(df_partial):
                    st.session_state.partial_scan_result = df_partial

                result_df = scan_uptrend_stocks_with_similarity(
                    code_list=all_stocks,
                    start_date=scan_start_date,
                    end_date=datetime.now().date(),
                    lookback_window=lookback_window,
                    pred_days=pred_days,
                    similarity_top_k=similarity_top_k,
                    min_similarity=min_similarity,
                    max_stocks=max_stocks_actual,
                    progress=progress_bar,
                    pro=None,
                    min_price=min_price,
                    use_trend_filter=use_trend_filter,
                    use_ma_confirm=use_ma_confirm,
                    ma_confirm_period=ma_confirm_period,
                    max_deviation=max_deviation,
                    use_volume_confirm=use_volume_confirm,
                    volume_ratio=volume_ratio,
                    use_td9=use_td9,
                    use_turnover_filter=use_turnover_filter,
                    min_turnover=min_turnover,
                    max_turnover=max_turnover,
                    turnover_map=turnover_map,
                    stop_check=lambda: st.session_state.get('scan_stop_requested', False),
                    save_partial=_save_partial,
                    save_every=20,
                    per_stock_timeout=25
                )
                progress_bar.empty()
                st.session_state.scan_stop_requested = False

                if st.session_state.partial_scan_result is not None and (
                    result_df is None or len(result_df) <= len(st.session_state.partial_scan_result)
                ):
                    result_df = st.session_state.partial_scan_result

                st.session_state.scan_result = result_df
                st.session_state.last_scan_time = datetime.now()
                if result_df is not None and not result_df.empty:
                    save_scan_cache(cache_key, result_df)
                    st.success("✅ 扫描完成")
                else:
                    st.warning("⚠️ 未找到符合条件的股票")

    if st.session_state.scan_result is not None:
        df_res = st.session_state.scan_result
        if df_res.empty:
            st.warning("未找到符合条件的股票")
        else:
            st.success(f"✅ 共 {len(df_res)} 只股票通过所有信号")
            st.dataframe(df_res)

            if '综合评分' in df_res.columns:
                st.subheader("📊 按综合评分排序（前20只）")
                st.dataframe(df_res.sort_values('综合评分', ascending=False).head(20))

            csv = df_res.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载全部结果CSV", csv, f"uptrend_signals_{pred_days}d.csv", "text/csv")

        # 跳过原因
        if st.session_state.scan_skipped:
            skipped_df = pd.DataFrame(st.session_state.scan_skipped)
            st.subheader("⚠️ 跳过原因分析")
            col1, col2 = st.columns(2)
            col1.metric("跳过总数", len(skipped_df))
            if st.session_state.scan_result is not None:
                col2.metric("通过股票数", len(st.session_state.scan_result))

            reason_counts = skipped_df['原因'].value_counts().reset_index()
            reason_counts.columns = ['原因', '数量']
            reason_counts['占比'] = (reason_counts['数量'] / len(skipped_df) * 100).round(2)

            col_a, col_b = st.columns(2)
            with col_a:
                st.dataframe(reason_counts, use_container_width=True)
            with col_b:
                fig_r, ax_r = plt.subplots(figsize=(6, max(3, len(reason_counts) * 0.5)))
                y_pos = range(len(reason_counts))
                ax_r.barh(list(y_pos), reason_counts['数量'][::-1], color='#e57373')
                ax_r.set_yticks(list(y_pos))
                ax_r.set_yticklabels(reason_counts['原因'][::-1], fontsize=9)
                ax_r.set_xlabel('数量')
                for i, (cnt, pct) in enumerate(zip(reason_counts['数量'][::-1], reason_counts['占比'][::-1])):
                    ax_r.text(cnt, i, f' {cnt} ({pct}%)', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_r)

            st.markdown("**🔍 每类原因示例**")
            for reason in reason_counts['原因'].tolist():
                subset = skipped_df[skipped_df['原因'] == reason].head(10)
                with st.expander(f"▸ {reason}（共 {len(skipped_df[skipped_df['原因']==reason])} 只）"):
                    st.dataframe(subset[['代码', '名称']])

            csv_skip = skipped_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载跳过明细", csv_skip, "skipped.csv", "text/csv")

        if st.session_state.simulated_codes:
            st.warning(f"⚠️ {len(st.session_state.simulated_codes)} 只股票使用了模拟数据")

        if st.session_state.last_scan_time:
            st.caption(f"上次扫描：{st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("👈 在侧边栏调整参数后，点击 '开始扫描上升趋势'")