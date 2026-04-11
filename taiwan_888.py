"""
台股發發發系統 v2.0 — GitHub Actions 自動化版
每日 18:00 台灣時間自動執行
Part 1：抓資料 → taiwan_3systems_YYYYMMDD.csv
Part 2：讀 CSV → 分析 → taiwan_888_YYYYMMDD.html
最後：Telegram + Email 通知
"""

# ============================================================
# 所有 import 集中在最上面（GitHub Actions 不分 Cell）
# ============================================================
import warnings
import time
import json
import os
import re
import io
import base64
import smtplib
from collections import defaultdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import feedparser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf

warnings.filterwarnings('ignore')

# ============================================================
# 設定：從 GitHub Secrets 讀取（不 hardcode Token）
# ============================================================
FINMIND_TOKEN     = os.environ.get('FINMIND_TOKEN', '')
TELEGRAM_TOKEN    = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID  = os.environ.get('TELEGRAM_CHAT_ID', '')
GMAIL_USER        = os.environ.get('GMAIL_USER', '')
GMAIL_APP_PASS    = os.environ.get('GMAIL_APP_PASS', '')
EMAIL_TO          = os.environ.get('EMAIL_TO', '')
REPORT_URL        = os.environ.get('REPORT_URL', '')   # ← 不用 GITHUB_ 開頭

TOP500_PATH       = os.environ.get('TOP500_PATH', 'top500.csv')
FILTER_TOP_N      = 100
FETCH_MONTHLY_REV = True
FETCH_NEWS        = True

TODAY    = datetime.today().strftime('%Y-%m-%d')
DATE_1Y  = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
DATE_2Y  = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
DATE_5Y  = (datetime.today() - timedelta(days=1825)).strftime('%Y-%m-%d')
TODAY_D  = datetime.today()

FM_BASE  = 'https://api.finmindtrade.com/api/v4/data'
FM_HDR   = {'Authorization': f'Bearer {FINMIND_TOKEN}'}
_fm_calls = [0]

os.makedirs('output', exist_ok=True)

print(f'[系統] 啟動：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'[系統] 今日：{TODAY}')

# ============================================================
# ── PART 1：工具函式
# ============================================================

def fetch_fm(dataset, stock_id='', start=None, end=None, retries=3):
    params = {'dataset': dataset}
    if stock_id: params['data_id'] = str(stock_id)
    if start:    params['start_date'] = start
    if end:      params['end_date']   = end
    for attempt in range(retries):
        try:
            r = requests.get(FM_BASE, headers=FM_HDR, params=params, timeout=30)
            d = r.json()
            if d.get('status') == 200 and d.get('data'):
                _fm_calls[0] += 1
                df = pd.DataFrame(d['data'])
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                return df
            msg = str(d.get('msg', ''))
            if any(k in msg.lower() for k in ['over','exceed','limit','quota']):
                print(f'  ⛔ FinMind 額度耗盡！(已用 {_fm_calls[0]} calls)')
                raise RuntimeError('FinMind quota exceeded')
            return pd.DataFrame()
        except RuntimeError:
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5)
    return pd.DataFrame()


def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))


def calc_kd(high, low, close, n=9, m=3):
    try:
        lo_n  = low.rolling(n).min()
        hi_n  = high.rolling(n).max()
        denom = (hi_n - lo_n).replace(0, np.nan)
        rsv   = ((close - lo_n) / denom * 100).fillna(50)
        K = rsv.ewm(com=m-1, adjust=False).mean()
        D = K.ewm(com=m-1, adjust=False).mean()
        return K, D
    except:
        empty = pd.Series([np.nan]*len(close), index=close.index)
        return empty, empty


def calc_max_drawdown(series):
    if len(series) < 2: return np.nan
    peak = series.cummax()
    dd   = (series - peak) / peak.replace(0, np.nan) * 100
    val  = dd.min()
    return round(float(val), 2) if not pd.isna(val) else np.nan


def calc_beta(close_series, mkt_ret_series):
    s_ret  = close_series.pct_change().dropna()
    common = s_ret.index.intersection(mkt_ret_series.index)
    if len(common) < 20: return np.nan
    s = s_ret.loc[common].values.astype(float)
    m = mkt_ret_series.loc[common].values.astype(float)
    mask = ~np.isnan(s) & ~np.isnan(m)
    s, m = s[mask], m[mask]
    if len(s) < 20 or np.var(m) < 1e-12: return np.nan
    return round(float(np.cov(s, m)[0][1] / np.var(m)), 3)


def is_stop_fall_k(o, h, l, c):
    try:
        rng = h - l
        if rng < 1e-6: return False
        return ((min(o, c) - l) / rng > 0.28) and (c >= l + rng * 0.38)
    except:
        return False


def calc_consec_foreign(series):
    vals = [v for v in series.tolist() if not pd.isna(v)]
    if not vals: return 0
    is_buy = vals[-1] > 0
    count = 0
    for v in reversed(vals):
        if (v > 0) == is_buy: count += 1
        else: break
    return count if is_buy else -count


def q_series(df_fin, type_key, n=8):
    sub = df_fin[df_fin['type'] == type_key].copy()
    if sub.empty: return pd.DataFrame(columns=['date', 'value'])
    sub['value'] = pd.to_numeric(sub['value'], errors='coerce')
    sub = sub.dropna(subset=['value']).drop_duplicates('date')
    return sub.sort_values('date', ascending=False).head(n)[['date', 'value']].reset_index(drop=True)


def yoy_pct(q_df, idx=0):
    if q_df.empty or idx >= len(q_df): return np.nan, np.nan
    curr_date = q_df.iloc[idx]['date']
    curr_val  = float(q_df.iloc[idx]['value'])
    target    = curr_date - pd.DateOffset(months=12)
    best_diff, best_val = 9999, None
    for _, r in q_df.iterrows():
        if r['date'] == curr_date: continue
        diff = abs((r['date'] - target).days)
        if diff <= 45 and diff < best_diff:
            best_diff, best_val = diff, float(r['value'])
    if best_val is None or best_val == 0: return curr_val, np.nan
    return round(curr_val, 3), round((curr_val / abs(best_val) - 1) * 100, 2)


def process_income(df_inc):
    rec = {}
    if df_inc is None or df_inc.empty: return rec
    types  = df_inc['type'].unique().tolist()
    eps_s  = q_series(df_inc, 'EPS', 8)
    for q in range(3):
        v, y = yoy_pct(eps_s, q)
        rec[f'eps_q{q+1}'], rec[f'eps_q{q+1}_yoy'] = v, y
    valid_y = [rec.get(f'eps_q{q+1}_yoy') for q in range(3)
               if rec.get(f'eps_q{q+1}_yoy') is not None and not pd.isna(rec.get(f'eps_q{q+1}_yoy'))]
    rec['eps_3q_yoy_gt20'] = bool(len(valid_y) >= 3 and all(y > 20 for y in valid_y))
    q1, q2 = rec.get('eps_q1', np.nan), rec.get('eps_q2', np.nan)
    rec['eps_q1_qoq'] = (round((float(q1)/abs(float(q2))-1)*100, 2)
                         if not pd.isna(q1) and not pd.isna(q2) and q2 != 0 else np.nan)
    rev_s = q_series(df_inc, 'Revenue', 4)
    for q in range(3):
        rec[f'revenue_q{q+1}'] = float(rev_s.iloc[q]['value']) if len(rev_s) > q else np.nan
    gp_s = q_series(df_inc, 'GrossProfit', 4)
    for q in range(3):
        gp = float(gp_s.iloc[q]['value']) if len(gp_s) > q else np.nan
        rv = rec.get(f'revenue_q{q+1}', np.nan)
        rec[f'gross_margin_q{q+1}'] = (round(gp/rv*100, 2)
                                        if not pd.isna(gp) and not pd.isna(rv) and rv > 0 else np.nan)
    gm = [rec.get(f'gross_margin_q{q+1}') for q in range(3)]
    rec['gross_margin_improving'] = (bool(gm[0] >= gm[1] >= gm[2])
                                     if all(v is not None and not pd.isna(v) for v in gm) else None)
    oi_s = q_series(df_inc, 'OperatingIncome', 3)
    for q in range(3):
        rec[f'op_income_q{q+1}'] = float(oi_s.iloc[q]['value']) if len(oi_s) > q else np.nan
    ni_type = next((t for t in ['IncomeAfterTaxes', 'NetIncome', 'ProfitLoss'] if t in types), None)
    if ni_type:
        ni_s = q_series(df_inc, ni_type, 1)
        rec['_net_income'] = float(ni_s.iloc[0]['value']) if not ni_s.empty else np.nan
        rec['_ni_date']    = ni_s.iloc[0]['date'] if not ni_s.empty else None
    else:
        rec['_net_income'], rec['_ni_date'] = np.nan, None
    return rec


def process_balance(df_bal):
    rec = {}
    if df_bal is None or df_bal.empty: return rec
    liab_s  = q_series(df_bal, 'Liabilities', 1)
    asset_s = q_series(df_bal, 'TotalAssets', 1)
    if not liab_s.empty and not asset_s.empty:
        l, a = float(liab_s.iloc[0]['value']), float(asset_s.iloc[0]['value'])
        rec['debt_ratio'] = round(l/a*100, 2) if a > 0 else np.nan
    else:
        lp_s = q_series(df_bal, 'Liabilities_per', 1)
        rec['debt_ratio'] = round(float(lp_s.iloc[0]['value']), 2) if not lp_s.empty else np.nan
    eq_s = q_series(df_bal, 'EquityAttributableToOwnersOfParent', 2)
    rec['_equity_q1'] = float(eq_s.iloc[0]['value']) if len(eq_s) > 0 else np.nan
    rec['_equity_q2'] = float(eq_s.iloc[1]['value']) if len(eq_s) > 1 else np.nan
    return rec


def calc_roe(ni, equity):
    return (round(ni/abs(equity)*100, 2)
            if not pd.isna(ni) and not pd.isna(equity) and equity != 0 else np.nan)


def safe_merge(base, other, on='stock_id'):
    if other is None or other.empty: return base
    return base.merge(other.drop_duplicates(on), on=on, how='left')


# ============================================================
# ── PART 1 Step 1：篩選 500 → 100
# ============================================================

def to_num(x):
    try:
        return float(str(x).replace(',', '').replace(' ', '').replace('--', '').strip() or 'nan')
    except:
        return np.nan


def recent_trade_dates(n=5):
    dates, d = [], datetime.today() - timedelta(days=1)
    while len(dates) < n:
        if d.weekday() < 5: dates.append(d.strftime('%Y%m%d'))
        d -= timedelta(days=1)
    return dates


def fetch_twse_all_one_day(date_str):
    url = (f'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX'
           f'?date={date_str}&type=ALLBUT0999&response=json')
    try:
        r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        j = r.json()
        data = next((j[k] for k in ['data9', 'data8', 'data'] if k in j and j[k]), None)
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        if len(df.columns) < 9: return pd.DataFrame()
        df.columns = list(range(len(df.columns)))
        result = pd.DataFrame({
            'stock_id': df[0].astype(str).str.strip().str.zfill(4),
            'close':    df[8].apply(to_num),
            'volume':   df[2].apply(to_num),
        })
        return result[result['volume'].notna() & result['volume'].gt(0) &
                      result['stock_id'].str.match(r'^\d{4}$')].reset_index(drop=True)
    except:
        return pd.DataFrame()


def fetch_tpex_all_one_day():
    url = 'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes'
    try:
        r  = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        j  = r.json()
        df = pd.DataFrame(j)
        code_col  = next((c for c in df.columns if 'Code' in c or '代號' in c), None)
        vol_col   = next((c for c in df.columns if 'Volume' in c and 'Trade' in c), None)
        close_col = next((c for c in df.columns if c.lower() in ['close', 'closingprice']), None)
        if not code_col: return pd.DataFrame()
        result = pd.DataFrame({
            'stock_id': df[code_col].astype(str).str.strip().str.zfill(4),
            'close':    df[close_col].apply(to_num) if close_col else np.nan,
            'volume':   df[vol_col].apply(to_num) if vol_col else np.nan,
        })
        return result[result['stock_id'].str.match(r'^\d{4}$')].reset_index(drop=True)
    except:
        return pd.DataFrame()


def load_top500_and_filter():
    print('='*58)
    print('📡 Step 1: 篩選 500 → 100')
    print('='*58)

    top500 = pd.read_csv(TOP500_PATH)
    print(f'  讀取：{len(top500)} 檔')
    for old, new in [('ticker', 'stock_id'), ('code', 'stock_id'), ('symbol', 'stock_id'),
                     ('company', 'name'), ('stock_name', 'name')]:
        if old in top500.columns and new not in top500.columns:
            top500 = top500.rename(columns={old: new})
    if 'stock_id' not in top500.columns:
        top500 = top500.rename(columns={top500.columns[0]: 'stock_id'})
    if 'name' not in top500.columns:
        top500['name'] = top500['stock_id']
    top500['stock_id'] = top500['stock_id'].astype(str).str.strip().str.zfill(4)
    top500 = top500.drop_duplicates('stock_id').reset_index(drop=True)
    top500_ids = set(top500['stock_id'].tolist())
    name_map   = top500.set_index('stock_id')['name'].to_dict()
    print(f'  去重後：{len(top500_ids)} 檔')

    daily_frames = []
    for dt in recent_trade_dates(7):
        df_tw  = fetch_twse_all_one_day(dt)
        df_tpx = fetch_tpex_all_one_day()
        combined = pd.concat([df_tw, df_tpx], ignore_index=True)
        if not combined.empty:
            combined['trade_date'] = dt
            daily_frames.append(combined)
        time.sleep(0.3)
        if daily_frames:
            print(f'    {dt}: {len(daily_frames[-1])} 筆')

    if not daily_frames:
        selected_ids = top500['stock_id'].head(FILTER_TOP_N).tolist()
    else:
        df_all  = pd.concat(daily_frames, ignore_index=True)
        df_all  = df_all[df_all['stock_id'].isin(top500_ids)]
        vol_avg = (df_all.groupby('stock_id')['volume'].mean()
                   .reset_index().rename(columns={'volume': 'avg_volume'}))
        vol_avg['avg_vol_k'] = vol_avg['avg_volume'] / 1000
        close_l = (df_all.sort_values('trade_date', ascending=False)
                   .drop_duplicates('stock_id')[['stock_id', 'close']])
        vol_avg = vol_avg.merge(close_l, on='stock_id', how='left')
        vol_avg = vol_avg[vol_avg['avg_vol_k'] >= 100]
        vol_avg['activity'] = np.log1p(vol_avg['avg_vol_k'])
        selected_ids = (vol_avg.sort_values('activity', ascending=False)
                        .head(FILTER_TOP_N)['stock_id'].tolist())
        if len(selected_ids) < FILTER_TOP_N:
            extra = [s for s in top500['stock_id'] if s not in selected_ids]
            selected_ids += extra[:FILTER_TOP_N - len(selected_ids)]

    print(f'  ✅ 篩選完成：{len(selected_ids)} 檔')
    return selected_ids, name_map


# ============================================================
# ── PART 1 Step 2：日線
# ============================================================

def fetch_price_data(selected_ids):
    print('='*58)
    print('💹 Step 2: 日線（2年）')
    print('='*58)

    price_data = {}
    mkt_ret    = pd.Series(dtype=float)

    print('  抓 0050...')
    try:
        df_0050 = fetch_fm('TaiwanStockPrice', '0050', DATE_2Y, TODAY)
        if not df_0050.empty:
            df_0050['close'] = pd.to_numeric(df_0050['close'], errors='coerce')
            df_0050 = df_0050.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            mkt_ret = df_0050.set_index('date')['close'].pct_change().dropna()
            print(f'  ✅ 0050: {len(df_0050)} 筆')
        time.sleep(0.35)
    except RuntimeError:
        pass

    for i, sid in enumerate(selected_ids):
        try:
            df_p = fetch_fm('TaiwanStockPrice', sid, DATE_2Y, TODAY)
            if not df_p.empty:
                price_data[sid] = df_p
            time.sleep(0.35)
            if (i+1) % 20 == 0 or (i+1) == len(selected_ids):
                print(f'  {i+1}/{len(selected_ids)} | 有效:{len(price_data)} | calls:{_fm_calls[0]}')
        except RuntimeError:
            print(f'  ⛔ 額度耗盡（{i}完成）')
            break

    print(f'  ✅ 日線完成 | calls:{_fm_calls[0]}')
    return price_data, mkt_ret


# ============================================================
# ── PART 1 Step 3：技術指標
# ============================================================

def calc_tech_indicators(price_data, mkt_ret):
    print('='*58)
    print('📈 Step 3: 技術指標')
    print('='*58)

    tech_records = []
    for sid in price_data:
        df_p = price_data[sid].copy()
        for col in ['open', 'max', 'min', 'close']:
            df_p[col] = pd.to_numeric(df_p[col], errors='coerce')
        df_p['Trading_Volume'] = pd.to_numeric(df_p['Trading_Volume'], errors='coerce')
        df_p = df_p.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
        if len(df_p) < 20: continue

        close  = df_p['close']
        vol    = df_p['Trading_Volume']
        high   = df_p['max']
        low_   = df_p['min']
        open_  = df_p['open']
        latest = float(close.iloc[-1])
        rec    = {'stock_id': sid, 'close': round(latest, 2)}

        for period, key in [(5,'ma5'),(10,'ma10'),(20,'ma20'),(60,'ma60'),(120,'ma120')]:
            v = close.rolling(period).mean().iloc[-1]
            rec[key] = round(float(v), 2) if not pd.isna(v) else np.nan

        vol_k = vol / 1000
        rec['volume_k']      = round(float(vol_k.iloc[-1]), 0) if not pd.isna(vol_k.iloc[-1]) else np.nan
        rec['avg_vol_5d_k']  = round(float(vol_k.tail(5).mean()), 0)
        rec['avg_vol_20d_k'] = round(float(vol_k.tail(20).mean()), 0)
        avg5 = rec['avg_vol_5d_k']
        rec['volume_ratio']  = round(rec['volume_k']/avg5, 3) if avg5 > 0 else np.nan
        rec['volume_shrink'] = bool(rec['volume_k'] < avg5) if avg5 > 0 else False

        for days, key in [(5,'return_5d'),(20,'return_20d'),(60,'return_60d'),(252,'return_1y')]:
            if len(close) > days:
                base = float(close.iloc[-(days+1)])
                rec[key] = round((latest/base-1)*100, 2) if base > 0 else np.nan
            else:
                rec[key] = np.nan

        rsi_s = calc_rsi(close, 14)
        rec['rsi14'] = round(float(rsi_s.iloc[-1]), 2) if len(rsi_s) >= 15 else np.nan

        if len(df_p) >= 12:
            K_s, D_s = calc_kd(high, low_, close)
            rec['kd_k'] = round(float(K_s.iloc[-1]), 2) if not pd.isna(K_s.iloc[-1]) else np.nan
            rec['kd_d'] = round(float(D_s.iloc[-1]), 2) if not pd.isna(D_s.iloc[-1]) else np.nan
            if len(K_s) >= 2 and len(D_s) >= 2:
                rec['kd_golden_cross'] = (float(K_s.iloc[-1]) > float(D_s.iloc[-1]) and
                                          float(K_s.iloc[-2]) <= float(D_s.iloc[-2]) and
                                          float(D_s.iloc[-1]) < 35)
                rec['kd_death_cross']  = (float(K_s.iloc[-1]) < float(D_s.iloc[-1]) and
                                          float(K_s.iloc[-2]) >= float(D_s.iloc[-2]) and
                                          float(D_s.iloc[-1]) > 65)
            else:
                rec['kd_golden_cross'] = rec['kd_death_cross'] = False
            rec['kd_oversold']   = bool(rec.get('kd_k', 100) < 30)
            rec['kd_overbought'] = bool(rec.get('kd_k', 0)   > 75)
        else:
            rec['kd_k'] = rec['kd_d'] = np.nan
            rec['kd_golden_cross'] = rec['kd_death_cross'] = False
            rec['kd_oversold'] = rec['kd_overbought'] = False

        for ma_k, pct_k in [('ma20','price_vs_ma20_pct'),('ma60','price_vs_ma60_pct')]:
            mv = rec.get(ma_k)
            rec[pct_k] = round((latest/mv-1)*100, 2) if mv and mv > 0 else np.nan

        for period, bias_key, ma_key in [(5,'bias_5','ma5'),(20,'bias_20','ma20'),(60,'bias_60','ma60')]:
            mv = rec.get(ma_key)
            if mv and mv > 0:
                rec[bias_key] = round((latest/mv-1)*100, 2)
            else:
                mv2 = close.rolling(period).mean().iloc[-1]
                rec[bias_key] = (round((latest/float(mv2)-1)*100, 2)
                                 if not pd.isna(mv2) and float(mv2) > 0 else np.nan)

        rec['ma5_gt_ma10']    = bool(rec['ma5'] > rec['ma10']) if not any(pd.isna(rec.get(k, np.nan)) for k in ['ma5','ma10']) else False
        rec['ma5_gt_ma20']    = bool(rec['ma5'] > rec['ma20']) if not any(pd.isna(rec.get(k, np.nan)) for k in ['ma5','ma20']) else False
        rec['ma_bull_align']  = bool(not any(pd.isna(rec.get(k, np.nan)) for k in ['ma20','ma60','ma120']) and rec['ma20'] > rec['ma60'] > rec['ma120'])
        rec['price_above_ma20'] = bool(latest > rec['ma20']) if not pd.isna(rec.get('ma20', np.nan)) else False
        rec['price_above_ma60'] = bool(latest > rec['ma60']) if not pd.isna(rec.get('ma60', np.nan)) else False
        rec['stop_fall_k']      = bool(is_stop_fall_k(float(open_.iloc[-1]), float(high.iloc[-1]), float(low_.iloc[-1]), float(close.iloc[-1])))
        rec['not_break_prev_low'] = bool(latest > float(close.iloc[-6:-1].min())) if len(close) >= 6 else True

        win252 = close.tail(252)
        rec['high_52w']          = round(float(win252.max()), 2)
        rec['low_52w']           = round(float(win252.min()), 2)
        rec['pct_from_52w_high'] = round((latest/rec['high_52w']-1)*100, 2)
        rec['pct_from_52w_low']  = round((latest/rec['low_52w']-1)*100, 2) if rec['low_52w'] > 0 else np.nan
        rec['max_drawdown_1y']   = calc_max_drawdown(win252)
        rec['max_drawdown_3y']   = calc_max_drawdown(close)
        rec['beta_1y']           = calc_beta(df_p.set_index('date')['close'], mkt_ret) if len(mkt_ret) > 20 else np.nan
        rec['daily_money_flow']  = round(rec['volume_k'] * latest / 10000, 3) if rec['volume_k'] and latest else np.nan

        tech_records.append(rec)

    df_tech = pd.DataFrame(tech_records).drop_duplicates('stock_id').reset_index(drop=True)
    if df_tech.empty:
        df_tech['return_1y'] = pd.Series(dtype='float64')
    print(f'  ✅ 技術面：{len(df_tech)} 檔 | return_1y有效：{df_tech["return_1y"].notna().sum()}')
    return df_tech


# ============================================================
# ── PART 1 Step 4：PE/PB
# ============================================================

def fetch_per_data(price_data):
    print('='*58)
    print('💰 Step 4: PE/PB')
    print('='*58)
    per_records = []
    for i, sid in enumerate(price_data):
        try:
            df_per = fetch_fm('TaiwanStockPER', sid, DATE_1Y, TODAY)
            if not df_per.empty:
                df_per[['PER','PBR','dividend_yield']] = df_per[['PER','PBR','dividend_yield']].apply(pd.to_numeric, errors='coerce')
                lr = df_per.sort_values('date').iloc[-1]
                per_records.append({'stock_id': sid,
                    'pe': round(float(lr['PER']), 2) if not pd.isna(lr['PER']) else np.nan,
                    'pb': round(float(lr['PBR']), 2) if not pd.isna(lr['PBR']) else np.nan,
                    'dividend_yield': round(float(lr['dividend_yield']), 2) if not pd.isna(lr['dividend_yield']) else np.nan})
            else:
                per_records.append({'stock_id': sid, 'pe': np.nan, 'pb': np.nan, 'dividend_yield': np.nan})
            time.sleep(0.35)
            if (i+1) % 25 == 0 or (i+1) == len(price_data):
                print(f'  {i+1}/{len(price_data)} | calls:{_fm_calls[0]}')
        except RuntimeError:
            for s in list(price_data.keys())[i:]:
                if not any(r['stock_id'] == s for r in per_records):
                    per_records.append({'stock_id': s, 'pe': np.nan, 'pb': np.nan, 'dividend_yield': np.nan})
            break
    df_per = pd.DataFrame(per_records).drop_duplicates('stock_id').reset_index(drop=True)
    print(f'  ✅ PE/PB：{df_per["pe"].notna().sum()} 有效 | calls:{_fm_calls[0]}')
    return df_per


# ============================================================
# ── PART 1 Step 5：損益表
# ============================================================

def fetch_income_data(price_data):
    print('='*58)
    print('📋 Step 5: 損益表')
    print('='*58)
    income_records = []
    for i, sid in enumerate(price_data):
        rec = {'stock_id': sid}
        try:
            df_inc = fetch_fm('TaiwanStockFinancialStatements', sid, DATE_5Y, TODAY)
            if not df_inc.empty:
                rec.update(process_income(df_inc))
            time.sleep(0.35)
            if (i+1) % 25 == 0 or (i+1) == len(price_data):
                ok = sum(1 for r in income_records if not pd.isna(r.get('eps_q1', np.nan)))
                print(f'  {i+1}/{len(price_data)} | EPS有效:{ok} | calls:{_fm_calls[0]}')
        except RuntimeError:
            print(f'  ⛔ 耗盡({i}完成)')
            break
        income_records.append(rec)
    df_inc_data = pd.DataFrame(income_records).drop_duplicates('stock_id').reset_index(drop=True)
    print(f'  ✅ 損益表：{len(df_inc_data)} 檔 | calls:{_fm_calls[0]}')
    return df_inc_data


# ============================================================
# ── PART 1 Step 6：資產負債表
# ============================================================

def fetch_balance_data(price_data):
    print('='*58)
    print('🏦 Step 6: 資產負債')
    print('='*58)
    balance_records = []
    for i, sid in enumerate(price_data):
        rec = {'stock_id': sid}
        try:
            df_bal = fetch_fm('TaiwanStockBalanceSheet', sid, DATE_5Y, TODAY)
            if not df_bal.empty:
                rec.update(process_balance(df_bal))
            time.sleep(0.35)
            if (i+1) % 25 == 0 or (i+1) == len(price_data):
                ok = sum(1 for r in balance_records if not pd.isna(r.get('debt_ratio', np.nan)))
                print(f'  {i+1}/{len(price_data)} | 負債比:{ok} | calls:{_fm_calls[0]}')
        except RuntimeError:
            print(f'  ⛔ 耗盡({i}完成)')
            break
        balance_records.append(rec)
    df_bal_data = pd.DataFrame(balance_records).drop_duplicates('stock_id').reset_index(drop=True)
    print(f'  ✅ 資產負債：{len(df_bal_data)} 檔 | calls:{_fm_calls[0]}')
    return df_bal_data


# ============================================================
# ── PART 1 Step 7：外資持股
# ============================================================

def fetch_shareholding_data(price_data):
    print('='*58)
    print('🌏 Step 7: 外資持股')
    print('='*58)
    share_records = []
    for i, sid in enumerate(price_data):
        rec = {'stock_id': sid, 'foreign_holding_pct': np.nan}
        try:
            df_shr = fetch_fm('TaiwanStockShareholding', sid, DATE_1Y, TODAY)
            if not df_shr.empty:
                ratio_col = next((c for c in df_shr.columns if 'Foreign' in c and 'SharesRatio' in c and 'Remain' not in c), None)
                if not ratio_col:
                    remain_col = next((c for c in df_shr.columns if 'Foreign' in c and 'RemainRatio' in c), None)
                    if remain_col:
                        df_shr[remain_col] = pd.to_numeric(df_shr[remain_col], errors='coerce')
                        rec['foreign_holding_pct'] = round(100 - float(df_shr[remain_col].iloc[-1]), 2)
                else:
                    df_shr[ratio_col] = pd.to_numeric(df_shr[ratio_col], errors='coerce')
                    lv = df_shr.sort_values('date')[ratio_col].iloc[-1]
                    rec['foreign_holding_pct'] = round(float(lv), 2) if not pd.isna(lv) else np.nan
            time.sleep(0.35)
            if (i+1) % 25 == 0 or (i+1) == len(price_data):
                ok = sum(1 for r in share_records if not pd.isna(r.get('foreign_holding_pct', np.nan)))
                print(f'  {i+1}/{len(price_data)} | 有效:{ok} | calls:{_fm_calls[0]}')
        except RuntimeError:
            print(f'  ⛔ 耗盡({i}完成)')
            break
        share_records.append(rec)
    df_shr = pd.DataFrame(share_records).drop_duplicates('stock_id').reset_index(drop=True)
    print(f'  ✅ 持股：{df_shr["foreign_holding_pct"].notna().sum()} 有效 | calls:{_fm_calls[0]}')
    return df_shr


# ============================================================
# ── PART 1 Step 8：三大法人 30 天
# ============================================================

def fetch_t86_one_day(date_str):
    url = (f'https://www.twse.com.tw/rwd/zh/fund/T86'
           f'?date={date_str}&selectType=ALLBUT0999&response=json')
    try:
        r = requests.get(url, timeout=25, headers={'User-Agent': 'Mozilla/5.0'})
        j = r.json()
        data = next((j[k] for k in ['data','data9','data0'] if k in j and j[k]), None)
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        ncol = len(df.columns)
        if ncol < 13: return pd.DataFrame()
        df.columns = list(range(ncol))
        def to_i(x):
            try:
                return int(str(x).replace(',','').replace('+','').replace(' ','').replace('－','-').strip() or '0')
            except:
                return 0
        result = pd.DataFrame({
            'stock_id':   df[0].astype(str).str.strip().str.zfill(4),
            'foreign_net':df[4].apply(to_i),
            'trust_net':  df[10 if ncol >= 11 else 7].apply(to_i),
            'inst_total': df[20 if ncol >= 21 else 12].apply(to_i),
        })
        return result[result['stock_id'].str.match(r'^\d{4}$')].reset_index(drop=True)
    except:
        return pd.DataFrame()


def recent_30_trade_dates():
    dates, d = [], datetime.today() - timedelta(days=1)
    while len(dates) < 30:
        if d.weekday() < 5: dates.append(d.strftime('%Y%m%d'))
        d -= timedelta(days=1)
        if (datetime.today() - d).days > 70: break
    return sorted(dates)


def fetch_inst_data(price_data):
    print('='*58)
    print('🏦 Step 8: 三大法人 30天')
    print('='*58)
    daily_inst = {}
    for dt in recent_30_trade_dates():
        df_d = fetch_t86_one_day(dt)
        if not df_d.empty:
            daily_inst[dt] = df_d
        time.sleep(0.4)

    market_foreign_today = market_trust_today = 0
    latest_t86_date = ''
    if daily_inst:
        latest_t86_date = sorted(daily_inst.keys())[-1]
        lat = daily_inst[latest_t86_date]
        market_foreign_today = int(lat['foreign_net'].sum())
        market_trust_today   = int(lat['trust_net'].sum())
        print(f'  ✅ {len(daily_inst)} 個交易日 | 最新資料日：{latest_t86_date}')
        print(f'  大盤外資：{market_foreign_today:+,} | 投信：{market_trust_today:+,}')

    inst_records = []
    for sid in price_data:
        f_l, t_l, tot_l = [], [], []
        for dt in sorted(daily_inst.keys()):
            row = daily_inst[dt][daily_inst[dt]['stock_id'] == sid]
            f_l.append(int(row['foreign_net'].iloc[0]) if not row.empty else 0)
            t_l.append(int(row['trust_net'].iloc[0]) if not row.empty else 0)
            tot_l.append(int(row['inst_total'].iloc[0]) if not row.empty else 0)
        f_s = pd.Series(f_l); t_s = pd.Series(t_l); n = len(f_s)
        half = n // 2
        f1, f2 = f_s.iloc[:half].sum(), f_s.iloc[half:].sum()
        trend = 'up' if f2 > f1*1.1 else ('down' if f2 < f1*0.9 else 'flat')
        inst_records.append({'stock_id': sid,
            'foreign_net_today':      int(f_s.iloc[-1]) if n > 0 else 0,
            'foreign_net_5d':         int(f_s.tail(5).sum()),
            'foreign_net_20d':        int(f_s.tail(20).sum()),
            'foreign_net_30d':        int(f_s.sum()),
            'foreign_consecutive_days': int(calc_consec_foreign(f_s)),
            'foreign_3m_trend':       trend,
            'trust_net_today':        int(t_s.iloc[-1]) if n > 0 else 0,
            'trust_net_5d':           int(t_s.tail(5).sum()),
            'trust_net_20d':          int(t_s.tail(20).sum()),
            'inst_total_today':       int(pd.Series(tot_l).iloc[-1]) if n > 0 else 0,
            'market_foreign_today':   market_foreign_today,
            'market_foreign_date':    latest_t86_date,
        })

    df_inst = pd.DataFrame(inst_records).drop_duplicates('stock_id').reset_index(drop=True)
    print(f'  ✅ 法人：{len(df_inst)} 檔')
    return df_inst


# ============================================================
# ── PART 1 Step 9：月營收
# ============================================================

def fetch_monthly_revenue(price_data):
    print('='*58)
    print('📅 Step 9: 月營收')
    print('='*58)

    empty_df = pd.DataFrame(columns=['stock_id','monthly_rev_latest','monthly_rev_yoy','monthly_rev_mom','rev_month'])

    if not FETCH_MONTHLY_REV:
        print('  ℹ️ 月營收已關閉')
        return empty_df

    today_d    = datetime.today()
    cur_month  = today_d.month - 1 if today_d.month > 1 else 12
    cur_year   = today_d.year if today_d.month > 1 else today_d.year - 1
    rev_month_str = f'{cur_year}-{str(cur_month).zfill(2)}'
    rev_records = []

    for i, sid in enumerate(price_data):
        rec = {'stock_id': sid, 'monthly_rev_latest': np.nan,
               'monthly_rev_yoy': np.nan, 'monthly_rev_mom': np.nan, 'rev_month': rev_month_str}
        try:
            start_14m = (datetime.today() - timedelta(days=430)).strftime('%Y-%m-%d')
            df_rev = fetch_fm('TaiwanStockMonthRevenue', sid, start_14m, TODAY)
            if not df_rev.empty and 'revenue' in df_rev.columns:
                df_rev['revenue'] = pd.to_numeric(df_rev['revenue'], errors='coerce')
                df_rev = df_rev.dropna(subset=['revenue']).sort_values('date', ascending=False)
                if len(df_rev) >= 1:
                    latest_rev = float(df_rev.iloc[0]['revenue'])
                    rec['monthly_rev_latest'] = round(latest_rev, 0)
                if len(df_rev) >= 2:
                    prev_rev = float(df_rev.iloc[1]['revenue'])
                    if prev_rev > 0:
                        rec['monthly_rev_mom'] = round((latest_rev/prev_rev-1)*100, 2)
                if len(df_rev) >= 13:
                    yoy_rev = float(df_rev.iloc[12]['revenue'])
                    if yoy_rev > 0:
                        rec['monthly_rev_yoy'] = round((latest_rev/yoy_rev-1)*100, 2)
            time.sleep(0.35)
            if (i+1) % 25 == 0 or (i+1) == len(price_data):
                ok = sum(1 for r in rev_records if not pd.isna(r.get('monthly_rev_yoy', np.nan)))
                print(f'  {i+1}/{len(price_data)} | YoY有效:{ok} | calls:{_fm_calls[0]}')
        except RuntimeError:
            print(f'  ⛔ 額度耗盡（{i}完成）')
            break
        rev_records.append(rec)

    if rev_records:
        df_mops = pd.DataFrame(rev_records).drop_duplicates('stock_id').reset_index(drop=True)
        print(f'  ✅ 月營收：{len(df_mops)} 檔 | calls:{_fm_calls[0]}')
        return df_mops
    return empty_df


# ============================================================
# ── PART 1 Step 10：新聞情緒
# ============================================================

POSITIVE_WORDS = [
    '成長','創高','利多','擴產','接單','訂單','上修','需求強','爆發','AI','受惠','調升',
    '回溫','突破','創新高','漲價','獲利','業績','上調','看好','轉盈','超預期','強勁',
    '增加','合作','新品','拿到','贏得','股利','配息','漲','量增','大單','旺季','復甦',
    '加速','升評','目標價調高','毛利率提升','法說','上調目標','供不應求','缺貨',
]
NEGATIVE_WORDS = [
    '下滑','衰退','虧損','利空','砍單','降評','疲弱','衝擊','減產','庫存高',
    '跌','暫停','警告','下修','縮減','裁員','罰款','違約','停產','受創',
    '壓力','疑慮','不確定','下降','拖累','破底','賣出','停看聽','調降目標',
    '虧','負成長','超跌','跌停','認列損失','提列','呆帳',
]

def score_text(text):
    score = 0
    for w in POSITIVE_WORDS:
        if w in text: score += 1
    for w in NEGATIVE_WORDS:
        if w in text: score -= 1
    return max(-4, min(4, score))


def fetch_mops_news(sid):
    try:
        url = 'https://mops.twse.com.tw/mops/web/ajax_t05st01'
        payload = {
            'encodeURIComponent':'1','step':'1','firstin':'1','off':'1',
            'keyword4':'','code1':'','TYPEK2':'','checkbtn':'',
            'queryName':'co_id','inpuType':'co_id','TYPEK':'all','co_id':str(sid),
        }
        r = requests.post(url, data=payload, timeout=15,
                          headers={'User-Agent':'Mozilla/5.0',
                                   'Referer':'https://mops.twse.com.tw/mops/web/index'})
        if r.status_code != 200: return []
        titles = re.findall(r'<td[^>]*>([^<]{5,50})</td>', r.text[:8000])
        return titles[:3]
    except:
        return []


def fetch_news_data(price_data, name_map):
    print('='*58)
    print('🗞️ Step 10: 新聞情緒')
    print('='*58)

    news_score_map  = {}
    news_count_map  = {}
    news_latest_map = {}
    news_date_map   = {}

    if not FETCH_NEWS:
        print('  ℹ️ 新聞情緒已關閉')
        return news_score_map, news_count_map, news_latest_map, news_date_map

    sid_names = {sid: name_map.get(sid, sid) for sid in price_data}

    # Yahoo RSS
    rss_urls = [
        'https://tw.stock.yahoo.com/rss',
        'https://tw.finance.yahoo.com/rss/topfinstories',
    ]
    all_news = []
    for rss_url in rss_urls:
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                all_news.append({
                    'title':   entry.get('title', ''),
                    'date':    entry.get('published', ''),
                    'summary': entry.get('summary', ''),
                })
        except Exception as e:
            print(f'    Yahoo RSS 失敗：{e}')
    print(f'    Yahoo RSS：{len(all_news)} 條')

    for sid, sname in sid_names.items():
        matched = [n for n in all_news if sname in n['title'] or sid in n['title']]
        if matched:
            scores = [score_text(n['title'] + ' ' + n.get('summary', '')) for n in matched]
            news_score_map[sid]  = max(-3, min(3, sum(scores)))
            news_count_map[sid]  = len(matched)
            news_latest_map[sid] = matched[0]['title'][:60]
            news_date_map[sid]   = matched[0]['date'][:20] if matched[0].get('date') else ''

    no_news_sids = [s for s in price_data if s not in news_score_map][:30]
    if no_news_sids:
        print(f'  MOPS 補抓 {len(no_news_sids)} 檔...')
        for sid in no_news_sids:
            titles = fetch_mops_news(sid)
            if titles:
                text = ' '.join(titles)
                news_score_map[sid]  = max(-3, min(3, score_text(text)))
                news_count_map[sid]  = len(titles)
                news_latest_map[sid] = titles[0][:60]
                news_date_map[sid]   = TODAY
            time.sleep(0.3)

    total_news = sum(1 for s in price_data if s in news_score_map)
    print(f'  ✅ 新聞情緒：{total_news} 檔有訊號')
    return news_score_map, news_count_map, news_latest_map, news_date_map


# ============================================================
# ── PART 1 Step 11：週線 MA13
# ============================================================

def detect_fresh_cross_2d(sid, price_data):
    dfp = price_data.get(sid)
    if dfp is None or len(dfp) < 25: return False, ''
    try:
        cl   = pd.to_numeric(dfp['close'], errors='coerce').dropna()
        ma5  = cl.rolling(5).mean()
        ma10 = cl.rolling(10).mean()
        ma20 = cl.rolling(20).mean()
        if not (float(ma5.iloc[-1]) > float(ma20.iloc[-1]) or
                float(ma10.iloc[-1]) > float(ma20.iloc[-1])): return False, ''
        for i in range(1, 4):
            if len(cl) < i+2: break
            ci, pi = -i, -(i+1)
            if float(ma5.iloc[ci]) > float(ma20.iloc[ci]) and float(ma5.iloc[pi]) <= float(ma20.iloc[pi]):
                return True, f'MA5穿MA20({i}日前)'
            if float(ma10.iloc[ci]) > float(ma20.iloc[ci]) and float(ma10.iloc[pi]) <= float(ma20.iloc[pi]):
                return True, f'MA10穿MA20({i}日前)'
        return False, ''
    except:
        return False, ''


def calc_weekly_ma13(price_data):
    print('='*58)
    print('📊 Step 11: 週線MA13')
    print('='*58)
    wma_recs = []
    for sid, dfp in price_data.items():
        try:
            dw = dfp.copy()
            dw['close'] = pd.to_numeric(dw['close'], errors='coerce')
            dw = dw.dropna(subset=['close']).set_index('date').sort_index()
            wc   = dw['close'].resample('W').last().dropna()
            if len(wc) < 15: raise ValueError()
            wm13 = wc.rolling(13).mean()
            wup  = float(wm13.iloc[-1]) > float(wm13.iloc[-2])
            abv  = float(wc.iloc[-1])   > float(wm13.iloc[-1])
            wsig = bool(wup and abv)
            fc, cn = detect_fresh_cross_2d(sid, price_data)
            wma_recs.append({'stock_id': sid, 'weekly_ma13_signal': wsig,
                              'fresh_priority': bool(wsig and fc),
                              'fresh_cross': fc, 'cross_note': cn,
                              'weekly_ma13': round(float(wm13.iloc[-1]), 2)})
        except:
            wma_recs.append({'stock_id': sid, 'weekly_ma13_signal': False,
                              'fresh_priority': False, 'fresh_cross': False,
                              'cross_note': '', 'weekly_ma13': np.nan})
    df_wma = pd.DataFrame(wma_recs)
    print(f'  ✅ 週線MA13多頭：{int(df_wma["weekly_ma13_signal"].sum())} | 2日剛穿：{int(df_wma["fresh_priority"].sum())}')
    return df_wma


# ============================================================
# ── PART 1 Step 12：族群動能
# ============================================================

def calc_sector_momentum(df_data, price_data, finmind_token):
    print('='*58)
    print('🏭 Step 12: 族群動能')
    print('='*58)

    df_sinfo = None
    try:
        url    = 'https://api.finmind.tw/api/latest/datatables/TaiwanStockInfo'
        params = {'token': finmind_token}
        r      = requests.get(url, params=params, timeout=20)
        data   = r.json().get('data', [])
        if data:
            df_info = pd.DataFrame(data)
            cols    = [c for c in ['stock_id','industry_category'] if c in df_info.columns]
            if len(cols) == 2:
                df_sinfo = df_info[cols].dropna()
        if df_sinfo is not None:
            print(f'  ✅ 產業資料：{len(df_sinfo)} 筆')
    except:
        print('  ⚠️ 產業資料取得失敗')

    sid_sector = {}
    if df_sinfo is not None:
        for _, row in df_sinfo.iterrows():
            sid_sector[str(row['stock_id'])] = str(row['industry_category'])

    ret3d = {}
    for sid, dfp in price_data.items():
        try:
            cl = pd.to_numeric(dfp['close'], errors='coerce').dropna()
            if len(cl) >= 4:
                ret3d[str(sid)] = (float(cl.iloc[-1]) / float(cl.iloc[-4]) - 1) * 100
        except:
            pass

    sector_rets = defaultdict(list)
    for sid_s, ret in ret3d.items():
        sec = sid_sector.get(sid_s, '其他')
        sector_rets[sec].append(ret)

    sector_leader = {sec: any(r > 2.0 for r in rets) for sec, rets in sector_rets.items()}

    result = {}
    for _, row in df_data.iterrows():
        sid_s = str(row['stock_id'])
        sec   = sid_sector.get(sid_s, '其他')
        result[sid_s] = bool(sector_leader.get(sec, False))

    print(f'  ✅ 族群完成')
    return result


# ============================================================
# ── PART 1 Step 13：衍生指標 + 合併
# ============================================================

def build_derived_features(df_tech, df_per, df_inc_data, df_bal_data,
                            df_shr, df_inst, df_mops, df_wma,
                            price_data, mkt_ret, name_map,
                            news_score_map, news_count_map, news_latest_map, news_date_map):
    print('='*58)
    print('🔢 Step 13: 衍生指標 + 合併')
    print('='*58)

    # ROE
    roe_records = []
    for sid in price_data:
        inc_row = df_inc_data[df_inc_data['stock_id'] == sid]
        bal_row = df_bal_data[df_bal_data['stock_id'] == sid]
        rec = {'stock_id': sid}
        ni    = float(inc_row['_net_income'].iloc[0]) if not inc_row.empty and '_net_income' in inc_row.columns else np.nan
        eq_q1 = float(bal_row['_equity_q1'].iloc[0])  if not bal_row.empty and '_equity_q1' in bal_row.columns else np.nan
        eq_q2 = float(bal_row['_equity_q2'].iloc[0])  if not bal_row.empty and '_equity_q2' in bal_row.columns else np.nan
        rec['roe_latest']        = calc_roe(ni, eq_q1)
        rec['roe_prev_q']        = calc_roe(ni, eq_q2)
        rec['roe_improving']     = bool(not pd.isna(rec['roe_latest']) and not pd.isna(rec['roe_prev_q']) and rec['roe_latest'] >= rec['roe_prev_q'])
        rec['roe_gt5_improving'] = bool(rec['roe_improving'] and not pd.isna(rec['roe_latest']) and rec['roe_latest'] > 5)
        roe_records.append(rec)
    df_roe = pd.DataFrame(roe_records).drop_duplicates('stock_id').reset_index(drop=True)

    # 合併
    df = df_tech.copy()
    for _kd_col in ['kd_k','kd_d','kd_golden_cross','kd_death_cross','kd_oversold','kd_overbought']:
        if _kd_col not in df.columns:
            df[_kd_col] = False if any(k in _kd_col for k in ['cross','sold','bought']) else np.nan

    df = safe_merge(df, df_per[['stock_id','pe','pb','dividend_yield']])
    df = safe_merge(df, df_inc_data.drop(columns=[c for c in ['_net_income','_ni_date'] if c in df_inc_data.columns], errors='ignore'))
    df = safe_merge(df, df_bal_data.drop(columns=[c for c in ['_equity_q1','_equity_q2'] if c in df_bal_data.columns], errors='ignore'))
    df = safe_merge(df, df_shr)
    df = safe_merge(df, df_roe)
    df = safe_merge(df, df_inst)
    df = safe_merge(df, df_mops[['stock_id','monthly_rev_latest','monthly_rev_yoy','monthly_rev_mom','rev_month']])
    df = safe_merge(df, df_wma)
    df = df.drop_duplicates('stock_id').reset_index(drop=True)
    df['name'] = df['stock_id'].map(name_map).fillna(df['stock_id'])

    # 新聞
    df['news_score']        = df['stock_id'].map(news_score_map).fillna(0).astype(int)
    df['news_count']        = df['stock_id'].map(news_count_map).fillna(0).astype(int)
    df['news_latest_title'] = df['stock_id'].map(news_latest_map).fillna('')
    df['news_latest_date']  = df['stock_id'].map(news_date_map).fillna('')
    has_recent_news_map     = {sid: True for sid in news_score_map.keys()}
    df['has_recent_news']   = df['stock_id'].map(has_recent_news_map).fillna(False).astype(int)

    # PEG
    def _peg(row):
        try:
            pe, yoy = float(row.get('pe', np.nan)), float(row.get('eps_q1_yoy', np.nan))
            return round(pe/yoy, 3) if not pd.isna(pe) and not pd.isna(yoy) and yoy > 0 else np.nan
        except: return np.nan
    def _peg_note(row):
        pe, yoy = row.get('pe', np.nan), row.get('eps_q1_yoy', np.nan)
        if pd.isna(pe): return 'PE=NaN'
        if pd.isna(yoy): return 'YoY=NaN'
        try: return f'YoY負({float(yoy):.1f}%)不計' if float(yoy) <= 0 else '正常'
        except: return '異常'
    df['peg']      = df.apply(_peg, axis=1)
    df['peg_note'] = df.apply(_peg_note, axis=1)

    df['foreign_rank_today']       = df['foreign_net_today'].rank(ascending=False, method='min').astype(int)
    df['foreign_net_today_value']  = (df['foreign_net_today'].fillna(0) * df['close'].fillna(0) / 100_000_000).round(2)
    df['_fntv_rank']               = df['foreign_net_today_value'].rank(ascending=False, method='min').astype(int)

    def _chip_sync(row):
        f = int(row.get('foreign_net_today', 0) or 0)
        t = int(row.get('trust_net_today', 0) or 0)
        if f > 0 and t > 0:   return '雙買'
        elif f < 0 and t < 0: return '雙賣'
        elif f > 0: return '單買外'
        elif t > 0: return '單買投'
        elif f < 0 or t < 0: return '單賣'
        else: return '中性'
    df['chip_sync'] = df.apply(_chip_sync, axis=1)

    def _dmf_pct(row):
        f  = abs(int(row.get('foreign_net_today', 0) or 0))
        t  = abs(int(row.get('trust_net_today', 0) or 0))
        vk = float(row.get('volume_k', 0) or 0)
        return round((f+t)/(vk*1000)*100, 2) if vk > 0 else np.nan
    df['daily_money_flow_pct'] = df.apply(_dmf_pct, axis=1)

    def _fcf_proxy(row):
        qoq = row.get('eps_q1_qoq', np.nan)
        mry = row.get('monthly_rev_yoy', np.nan)
        dr  = row.get('debt_ratio', np.nan)
        qf  = float(qoq) if not pd.isna(qoq) else None
        mf  = float(mry) if not pd.isna(mry) else None
        df_ = float(dr)  if not pd.isna(dr)  else None
        return bool(qf is not None and qf > 0 and mf is not None and mf > 10 and df_ is not None and df_ < 55)
    df['fcf_proxy'] = df.apply(_fcf_proxy, axis=1)

    def _fpp(row):
        qoq = row.get('eps_q1_qoq', np.nan)
        mry = row.get('monthly_rev_yoy', np.nan)
        peg = row.get('peg', np.nan)
        qf  = float(qoq) if not pd.isna(qoq) else None
        mf  = float(mry) if not pd.isna(mry) else None
        pf  = float(peg) if not pd.isna(peg) else None
        return bool(qf is not None and qf > 15 and mf is not None and mf > 15 and pf is not None and pf < 1.3)
    df['forward_pe_proxy'] = df.apply(_fpp, axis=1)

    def _catalyst(row):
        s = 0
        rank = int(row.get('foreign_rank_today', 999) or 999)
        if rank <= 10: s += 30
        elif rank <= 20: s += 20
        elif rank <= 30: s += 10
        cd = int(row.get('foreign_consecutive_days', 0) or 0)
        if cd >= 5: s += 25
        elif cd >= 3: s += 20
        elif cd >= 1: s += 10
        elif cd <= -3: s -= 15
        sync = str(row.get('chip_sync', '中性'))
        if sync == '雙買': s += 20
        elif '單買' in sync: s += 8
        elif sync == '雙賣': s -= 10
        vr = float(row.get('volume_ratio', 1) or 1)
        if vr >= 2.5: s += 20
        elif vr >= 1.5: s += 13
        elif vr >= 1.2: s += 8
        yoy = float(row.get('eps_q1_yoy', 0) or 0) if not pd.isna(row.get('eps_q1_yoy', np.nan)) else 0
        if yoy >= 30: s += 15
        elif yoy >= 20: s += 10
        elif yoy >= 10: s += 5
        mry = float(row.get('monthly_rev_yoy', np.nan) or np.nan)
        if not pd.isna(mry):
            if mry >= 30: s += 12
            elif mry >= 15: s += 8
            elif mry >= 5: s += 4
            elif mry < -10: s -= 8
        ns = int(row.get('news_score', 0) or 0)
        if ns >= 2: s += 15
        elif ns == 1: s += 8
        elif ns <= -2: s -= 15
        elif ns == -1: s -= 5
        gm1 = row.get('gross_margin_q1', np.nan)
        gm2 = row.get('gross_margin_q2', np.nan)
        try:
            if not pd.isna(gm1) and not pd.isna(gm2) and float(gm1) > float(gm2): s += 8
        except: pass
        return max(0, min(100, s))
    df['catalyst_score'] = df.apply(_catalyst, axis=1)

    df['v7_momentum_score'] = df.apply(lambda r: round(
        float(r.get('return_5d', 0) or 0) * 0.35 +
        min(float(r.get('volume_ratio', 1) or 1), 5) * 10 * 0.30 +
        (100 - float(r.get('rsi14', 50) or 50)) * 0.35, 2), axis=1)

    df['has_news'] = (df['news_score'].abs() >= 1).astype(int)

    try:
        # 用 0050 市場相對強度（若可用）
        df_0050 = price_data.get('0050')
        if df_0050 is not None:
            cl_0050 = pd.to_numeric(df_0050['close'], errors='coerce').dropna()
            if len(cl_0050) >= 20:
                mkt_pct = (float(cl_0050.iloc[-1]) / float(cl_0050.iloc[-20]) - 1) * 100
                df['rel_strength_20d'] = (df['return_20d'].fillna(0) - mkt_pct).round(2)
            else:
                df['rel_strength_20d'] = np.nan
        else:
            df['rel_strength_20d'] = np.nan
    except:
        df['rel_strength_20d'] = np.nan

    # T1 分級
    def _t1(row):
        b20 = float(row.get('bias_20', row.get('price_vs_ma20_pct', np.nan)) or np.nan)
        if pd.isna(b20): return 'N/A'
        rsi  = float(row.get('rsi14', 50) or 50)
        r5   = float(row.get('return_5d', 0) or 0)
        vr   = float(row.get('volume_ratio', 1) or 1)
        vshr = bool(row.get('volume_shrink', False))
        stop = bool(row.get('stop_fall_k', False))
        m5   = bool(row.get('ma5_gt_ma10', False))
        ab20 = bool(row.get('price_above_ma20', False))
        cs   = str(row.get('chip_sync', '中性'))
        cd   = int(row.get('foreign_consecutive_days', 0) or 0)
        npbl = bool(row.get('not_break_prev_low', True)) and ab20
        chip_ok = cs not in ('雙賣', '單賣') and cd >= -1
        blk_k = vr > 2.5 and r5 < -3
        if r5 > 12 or rsi > 68 or b20 > 13 or blk_k: return 'C'
        ap_zone = -3 <= b20 <= 1
        ap_tech = vshr and (stop or rsi < 42) and m5 and npbl
        if ap_zone and ap_tech and rsi < 50 and chip_ok: return 'A+'
        a_zone = -4 <= b20 <= 4
        a_tech = vshr and m5 and (stop or npbl)
        if a_zone and a_tech and rsi < 57 and chip_ok: return 'A'
        am_zone = -6 <= b20 <= 6
        if am_zone and rsi < 62 and m5: return 'A-'
        if 4 < b20 <= 9 and rsi < 65 and vr < 2: return 'B+'
        if (9 < b20 <= 13) or (-8 <= b20 < -4): return 'B'
        if -12 <= b20 < -4 and not m5: return 'B-'
        return 'C'
    df['t1_grade'] = df.apply(_t1, axis=1)

    # P5 面向
    def _p5(row):
        s = []
        f1 = sum([bool(row.get('eps_3q_yoy_gt20', False)),
                  bool(row.get('roe_gt5_improving', False)),
                  bool(row.get('gross_margin_improving'))])
        s.append(1 if f1 >= 2 else 0)
        pe, peg = row.get('pe', np.nan), row.get('peg', np.nan)
        s.append(1 if (not pd.isna(pe) and float(pe) < 20) or (not pd.isna(peg) and float(peg) < 1.2) else 0)
        dr, oi1, oi2, dy = (row.get('debt_ratio', np.nan), row.get('op_income_q1', np.nan),
                             row.get('op_income_q2', np.nan), row.get('dividend_yield', np.nan))
        f3 = ((1 if not pd.isna(dr) and float(dr) < 50 else 0) +
               (1 if not pd.isna(dy) and float(dy) > 1.5 else 0) +
               (1 if not pd.isna(oi1) and not pd.isna(oi2) and float(oi1) > float(oi2) > 0 else 0))
        s.append(1 if f3 >= 2 else 0)
        fp, cd, tr = (row.get('foreign_holding_pct', np.nan),
                      row.get('foreign_consecutive_days', 0),
                      row.get('foreign_3m_trend', ''))
        s.append(1 if (not pd.isna(fp) and float(fp) > 20 and str(tr) == 'up') or
                      (not pd.isna(cd) and int(cd) >= 3) else 0)
        bull = bool(row.get('ma_bull_align', False))
        rsi  = float(row.get('rsi14', 100) or 100)
        f5   = ((1 if bull else 0) + (1 if rsi < 70 else 0) +
                (1 if bool(row.get('price_above_ma60', False)) else 0))
        s.append(1 if f5 >= 2 else 0)
        beta, mdd = row.get('beta_1y', np.nan), row.get('max_drawdown_3y', np.nan)
        s.append(1 if (not pd.isna(beta) and float(beta) < 1.2) and
                      (not pd.isna(mdd) and float(mdd) > -25) else 0)
        return sum(s)
    df['p5_face_pass'] = df.apply(_p5, axis=1)
    df['export_date']  = TODAY

    print(f'  ✅ 衍生指標完成 | 欄位：{len(df.columns)}')
    print(f'  T1：{df["t1_grade"].value_counts().to_dict()}')
    print(f'  P5≥5：{(df["p5_face_pass"]>=5).sum()} 檔')
    return df


# ============================================================
# ── PART 1 Step 14：輸出 CSV
# ============================================================

COL_ORDER = [
    'export_date','stock_id','name','close',
    'eps_q1','eps_q1_yoy','eps_q1_qoq','eps_q2','eps_q2_yoy',
    'eps_q3','eps_q3_yoy','eps_3q_yoy_gt20',
    'revenue_q1','revenue_q2','revenue_q3',
    'gross_margin_q1','gross_margin_q2','gross_margin_q3','gross_margin_improving',
    'roe_latest','roe_prev_q','roe_improving','roe_gt5_improving',
    'pe','pb','peg','peg_note','dividend_yield',
    'op_income_q1','op_income_q2','op_income_q3',
    'debt_ratio','fcf_proxy','forward_pe_proxy',
    'monthly_rev_latest','monthly_rev_yoy','monthly_rev_mom','rev_month',
    'foreign_holding_pct','foreign_net_30d','foreign_3m_trend',
    'ma20','ma60','ma120','price_vs_ma20_pct','price_vs_ma60_pct',
    'ma_bull_align','price_above_ma20','price_above_ma60',
    'beta_1y','max_drawdown_1y','max_drawdown_3y',
    'p5_face_pass',
    'return_5d','return_20d','return_60d','return_1y',
    'foreign_net_today','foreign_net_today_value','foreign_rank_today',
    'foreign_consecutive_days','foreign_net_5d','foreign_net_20d',
    'trust_net_today','trust_net_5d','trust_net_20d','inst_total_today',
    'chip_sync','market_foreign_today','market_foreign_date',
    'volume_k','avg_vol_5d_k','avg_vol_20d_k','volume_ratio','volume_shrink',
    'daily_money_flow','daily_money_flow_pct',
    'rsi14','rel_strength_20d',
    'has_news','catalyst_score','v7_momentum_score',
    'news_score','news_count','has_recent_news','news_latest_title','news_latest_date',
    't1_grade','stop_fall_k','ma5','ma10','ma5_gt_ma10','ma5_gt_ma20',
    'not_break_prev_low','pct_from_52w_high','pct_from_52w_low','high_52w','low_52w',
    'bias_5','bias_20','bias_60',
    'kd_k','kd_d','kd_golden_cross','kd_death_cross','kd_oversold','kd_overbought',
    'weekly_ma13_signal','weekly_ma13','fresh_priority','fresh_cross','cross_note',
    'sector_momentum',
]


def export_csv(df):
    print('='*58)
    print('💾 Step 14: 輸出 CSV')
    print('='*58)
    for c in COL_ORDER:
        if c not in df.columns: df[c] = np.nan
    df_out = df[COL_ORDER].drop_duplicates('stock_id').reset_index(drop=True)
    csv_fname = f'output/taiwan_3systems_{TODAY.replace("-","")}.csv'
    df_out.to_csv(csv_fname, index=False, encoding='utf-8-sig')
    print(f'  ✅ CSV 已輸出：{csv_fname}（{len(df_out)} 筆）')
    return csv_fname, df_out


# ============================================================
# ── PART 2：讀 CSV → 分析 → HTML
# ============================================================

def load_and_clean_csv(csv_fname):
    print('='*58)
    print('📂 Part2 Step 1: 讀取 CSV')
    print('='*58)
    df_raw = pd.read_csv(csv_fname)
    data_today = str(df_raw['export_date'].iloc[0])

    bool_cols = ['eps_3q_yoy_gt20','fcf_proxy','forward_pe_proxy','ma_bull_align',
                 'price_above_ma20','price_above_ma60','volume_shrink','stop_fall_k',
                 'ma5_gt_ma10','ma5_gt_ma20','not_break_prev_low','weekly_ma13_signal',
                 'fresh_priority','fresh_cross','sector_momentum','roe_improving']
    for c in bool_cols:
        if c in df_raw.columns:
            df_raw[c] = df_raw[c].astype(str).str.strip().str.lower().map({'true':True,'false':False}).fillna(False)

    num_cols = ['close','ma20','ma60','ma5','ma10','rsi14','volume_ratio','return_5d',
                'price_vs_ma20_pct','pct_from_52w_low','pct_from_52w_high',
                'monthly_rev_yoy','eps_q1_qoq','eps_q1_yoy','debt_ratio',
                'roe_latest','roe_prev_q','foreign_net_today_value','foreign_net_5d',
                'pe','pb','peg','dividend_yield','beta_1y','rel_strength_20d',
                'high_52w','low_52w','bias_5','bias_20','bias_60','weekly_ma13',
                'avg_vol_5d_k','avg_vol_20d_k','volume_k']
    for c in num_cols:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

    int_cols = ['foreign_net_5d','foreign_net_20d','foreign_net_today',
                'foreign_rank_today','foreign_consecutive_days',
                'trust_net_today','trust_net_5d','trust_net_20d',
                'news_score','catalyst_score','market_foreign_today']
    for c in int_cols:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce').fillna(0).astype(int)

    df = df_raw.copy()
    df['_fntv_rank']    = df['foreign_net_today_value'].rank(ascending=False, method='min').astype(int)
    MARKET_FOREIGN_B    = round(df['foreign_net_today_value'].fillna(0).sum(), 1)
    N_STOCKS            = len(df)
    print(f'  ✅ 讀取完成 | {N_STOCKS} 檔 | 日期：{data_today}')
    print(f'  大盤外資（樣本加總）：{MARKET_FOREIGN_B:+.1f} 億元')
    return df, N_STOCKS, MARKET_FOREIGN_B, data_today


def fetch_taiex():
    """加權指數三層 fallback，使用 FINMIND_TOKEN（從環境變數）"""
    def fetch_taiex_twse(months_back=4):
        records = []
        for m in range(months_back + 1):
            mo = (TODAY_D.month - m - 1) % 12 + 1
            yr = TODAY_D.year - ((TODAY_D.month - m - 1) // 12)
            ym = f'{yr}{mo:02d}01'
            try:
                url = f'https://www.twse.com.tw/rwd/zh/indicesReport/TAIEX?response=json&date={ym}'
                r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
                j = r.json()
                for row in j.get('data', []):
                    try:
                        parts   = str(row[0]).split('/')
                        yr_ad   = int(parts[0]) + 1911
                        date_ad = f'{yr_ad}-{parts[1]}-{parts[2]}'
                        close_v = float(str(row[4]).replace(',', ''))
                        records.append({'date': date_ad, 'close': close_v})
                    except: pass
            except: pass
        if not records: return None
        return pd.DataFrame(records).drop_duplicates('date').sort_values('date').reset_index(drop=True)

    def fetch_taiex_finmind():
        if not FINMIND_TOKEN: return None  # ← 用全域 env var，不 reset 為空字串
        try:
            end_d   = TODAY_D.strftime('%Y-%m-%d')
            start_d = (TODAY_D - timedelta(days=90)).strftime('%Y-%m-%d')
            params  = {'dataset':'TaiwanStockPrice','data_id':'TAIEX',
                       'start_date':start_d,'end_date':end_d,'token':FINMIND_TOKEN}
            r = requests.get('https://api.finmindtrade.com/api/v4/data', params=params, timeout=15)
            j = r.json()
            if j.get('status') == 200 and j.get('data'):
                records = [{'date':d['date'],'close':float(d['close'])} for d in j['data'] if d.get('close')]
                return pd.DataFrame(records).drop_duplicates('date').sort_values('date').reset_index(drop=True)
        except: pass
        return None

    def fetch_0050_proxy(months_back=4):
        records = []
        for m in range(months_back + 1):
            mo = (TODAY_D.month - m - 1) % 12 + 1
            yr = TODAY_D.year - ((TODAY_D.month - m - 1) // 12)
            ym = f'{yr}{mo:02d}01'
            try:
                url = f'https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?response=json&date={ym}&stockNo=0050'
                r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
                j = r.json()
                for row in j.get('data', []):
                    try:
                        parts   = str(row[0]).split('/')
                        yr_ad   = int(parts[0]) + 1911
                        date_ad = f'{yr_ad}-{parts[1]}-{parts[2]}'
                        close_v = float(str(row[6]).replace(',', ''))
                        records.append({'date': date_ad, 'close': close_v})
                    except: pass
            except: pass
        if not records: return None
        return pd.DataFrame(records).drop_duplicates('date').sort_values('date').reset_index(drop=True)

    taiex_df  = None
    taiex_src = '未取得'
    print('  層1：TWSE 官方...')
    try:
        taiex_df = fetch_taiex_twse()
        if taiex_df is not None and len(taiex_df) >= 20:
            taiex_src = '📡 TWSE官方'
            print(f'  ✅ 層1成功，取得 {len(taiex_df)} 筆')
        else:
            taiex_df = None; print('  ⚠️ 層1資料不足')
    except Exception as e:
        print(f'  ⚠️ 層1失敗: {e}')

    if taiex_df is None:
        print('  層2：FinMind...')
        try:
            taiex_df = fetch_taiex_finmind()
            if taiex_df is not None and len(taiex_df) >= 20:
                taiex_src = '📊 FinMind'
                print(f'  ✅ 層2成功')
            else:
                taiex_df = None; print('  ⚠️ 層2未取得')
        except Exception as e:
            print(f'  ⚠️ 層2失敗: {e}')

    if taiex_df is None:
        print('  層3：0050代理...')
        try:
            taiex_df = fetch_0050_proxy()
            if taiex_df is not None and len(taiex_df) >= 20:
                taiex_src = '🔄 0050代理'
                print(f'  ✅ 層3成功')
        except Exception as e:
            print(f'  ⚠️ 層3失敗: {e}')

    macro = {'taiex_close': None, 'taiex_ma20': None, 'taiex_above_ma20': None,
             'taiex_source': taiex_src, 'op_advice': '適合保守操作'}
    if taiex_df is not None and len(taiex_df) >= 20:
        taiex_df['ma20'] = taiex_df['close'].rolling(20).mean()
        tx_c = float(taiex_df['close'].iloc[-1])
        tx_m = float(taiex_df['ma20'].dropna().iloc[-1])
        macro['taiex_close']     = tx_c
        macro['taiex_ma20']      = tx_m
        macro['taiex_above_ma20'] = tx_c > tx_m
        print(f'  {taiex_src}｜收盤：{tx_c:,.2f} / MA20：{tx_m:,.2f}')
    return macro


def run_screening(df, N_STOCKS, MARKET_FOREIGN_B, macro):
    """執行 T1/L1/S1/Z1 篩選，與 Notebook Part2 邏輯完全一致"""

    fb = MARKET_FOREIGN_B
    if fb > 100:   macro['foreign_label'] = f'買超 {fb:+.0f} 億元 → 偏多'
    elif fb >= -100: macro['foreign_label'] = f'小幅 {fb:+.0f} 億元 → 中性'
    elif fb >= -300: macro['foreign_label'] = f'賣超 {fb:.0f} 億元 → 偏空'
    else:            macro['foreign_label'] = f'大賣超 {fb:.0f} 億元 → 極度危險'
    bullish = (macro['taiex_above_ma20'] is True and macro.get('foreign_b', fb) > -300)
    macro['op_advice'] = '適合積極操作' if bullish else '適合保守操作'
    macro['foreign_b'] = fb

    # ── T1-TOP
    def parse_cross_days(note):
        if pd.isna(note) or not note: return 999
        m = re.search(r'(\d+)日前', str(note))
        return int(m.group(1)) if m else 999

    def has_valid_cross(note):
        if pd.isna(note) or not note: return False
        s = str(note)
        return any(k in s for k in ['MA5穿MA20','MA10穿MA20', '雙穿MA20']) and parse_cross_days(s) <= 3

    def run_t1_top(df):
        results = []
        for _, row in df.iterrows():
            sid   = str(row['stock_id']).strip()
            if sid.startswith('28'): continue
            close = float(row.get('close', 0) or 0)
            if close <= 0: continue
            name  = str(row.get('name', sid))
            fresh = bool(row.get('fresh_priority', False))
            t1g   = str(row.get('t1_grade', 'C'))
            cn    = str(row.get('cross_note', ''))
            vr    = float(row.get('volume_ratio', 0) or 0)
            pct20 = float(row.get('price_vs_ma20_pct', 0) or 0)
            ma20v = float(row.get('ma20', 0) or 0)
            r5d   = float(row.get('return_5d', 0) or 0)
            if not (fresh or t1g in ['A+', 'A']): continue
            if t1g not in ['A+', 'A', 'A-']: continue
            if not has_valid_cross(cn): continue
            if vr <= 1.60: continue
            if not (-4.5 <= pct20 <= 2.0): continue
            if r5d <= 5.5: continue
            if not bool(row.get('ma5_gt_ma10', False)): continue
            if float(row.get('rsi14', 50) or 50) > 77: continue
            sl  = round(ma20v * 0.975, 0)
            tgt = round(close * 1.30, 0)
            results.append({'sid':sid,'name':name,'close':close,'t1_grade':t1g,
                            'cross_note':cn,'vr':vr,'pct20':pct20,'ma20':ma20v,
                            'rsi':float(row.get('rsi14',50) or 50),
                            'chip':str(row.get('chip_sync','')),
                            'weekly_ma13':bool(row.get('weekly_ma13_signal',False)),
                            'r5d':r5d,'sl':sl,'tgt':tgt})
        return results

    # ── L1
    def l1_gate(row):
        mry = row.get('monthly_rev_yoy', np.nan)
        qoq = row.get('eps_q1_qoq', np.nan)
        yoy = row.get('eps_q1_yoy', np.nan)
        dr  = row.get('debt_ratio', np.nan)
        pfl = row.get('pct_from_52w_low', np.nan)
        if pd.isna(mry) or float(mry) <= -20: return False
        if not ((not pd.isna(qoq) and float(qoq) >= -30) or
                (not pd.isna(yoy) and float(yoy) > 3)): return False
        if pd.isna(dr) or float(dr) >= 90: return False
        if pd.isna(pfl) or float(pfl) <= 3: return False
        return True

    l1_mask     = df.apply(l1_gate, axis=1)
    L1_full_pool = df[l1_mask].copy()
    print(f'L1 守門通過：{len(L1_full_pool)} 檔')

    def l1_score(row):
        close = float(row.get('close', 0) or 0)
        sid   = str(row['stock_id']); name = str(row.get('name', sid))
        if close <= 0: return None
        details = []
        s_grow = 0
        qoqf = float(row['eps_q1_qoq']) if not pd.isna(row.get('eps_q1_qoq')) else None
        mryf = float(row['monthly_rev_yoy']) if not pd.isna(row.get('monthly_rev_yoy')) else None
        roe_l = row.get('roe_latest', np.nan); roe_p = row.get('roe_prev_q', np.nan)
        if qoqf is not None:
            if qoqf >= 15: s_grow += 15; details.append(f'QoQ{qoqf:.0f}%≥15 +15')
            elif qoqf >= 5: s_grow += 10; details.append(f'QoQ{qoqf:.0f}%≥5 +10')
            elif qoqf >= 0: s_grow += 5; details.append(f'QoQ{qoqf:.0f}%≥0 +5')
        if bool(row.get('eps_3q_yoy_gt20', False)): s_grow += 10; details.append('EPS連3季YoY>20% +10')
        if mryf is not None:
            if mryf > 15: s_grow += 10; details.append(f'月營收YoY{mryf:.0f}%>15 +10')
            elif mryf > 0: s_grow += 5; details.append(f'月營收YoY{mryf:.0f}%>0 +5')
        if (not pd.isna(roe_l) and not pd.isna(roe_p) and float(roe_l) > float(roe_p) and float(roe_l) > 5):
            s_grow += 5; details.append(f'ROE{float(roe_l):.1f}%改善 +5')
        s_grow = min(s_grow, 40)
        s_chip = 0
        ft = str(row.get('foreign_3m_trend', ''))
        fntv_rank = int(row.get('_fntv_rank', 999))
        cs = str(row.get('chip_sync', ''))
        if ft == 'up': s_chip += 12; details.append('外資3M趨勢up +12')
        if fntv_rank <= 15: s_chip += 8; details.append(f'外資買超排名#{fntv_rank} +8')
        if cs == '雙買': s_chip += 5; details.append('雙買 +5')
        s_chip = min(s_chip, 25)
        s_fin = 0
        dr = row.get('debt_ratio', np.nan)
        if not pd.isna(dr):
            drf = float(dr)
            if drf < 40: s_fin += 8; details.append(f'負債比{drf:.0f}%<40 +8')
            elif drf < 55: s_fin += 5; details.append(f'負債比{drf:.0f}%<55 +5')
            elif drf < 70: s_fin += 3; details.append(f'負債比{drf:.0f}%<70 +3')
        if bool(row.get('fcf_proxy', False)): s_fin += 7; details.append('FCF正向 +7')
        s_fin = min(s_fin, 15)
        s_val = 0
        if bool(row.get('forward_pe_proxy', False)): s_val += 10; details.append('Forward PE上修 +10')
        s_val = min(s_val, 10)
        s_tech = 0
        pct20 = float(row.get('price_vs_ma20_pct', 0) or 0)
        bull  = bool(row.get('ma_bull_align', False))
        if pct20 > -5 and bull: s_tech += 10; details.append(f'月線{pct20:.1f}%+多頭 +10')
        elif -10 <= pct20 <= 10: s_tech += 5; details.append(f'月線{pct20:.1f}%±10% +5')
        s_tech = min(s_tech, 10)
        total = s_grow + s_chip + s_fin + s_val + s_tech
        ma20v = float(row.get('ma20', 0) or 0)
        ma60v = float(row.get('ma60', 0) or 0)
        if ma20v > 0 and pct20 > 8:   entry = f'等回踩月線≈{ma20v:.0f}'
        elif ma20v > 0 and pct20 > 3: entry = f'{ma20v*0.99:.0f}~{ma20v*1.03:.0f}'
        else:                          entry = f'{close*0.99:.0f}~{close*1.01:.0f}'
        sl_ma20 = f'≈{ma20v:.0f}' if ma20v > 0 and ma20v < close*0.999 else '月線≈現價'
        sl_ma60 = f'≈{ma60v:.0f}' if ma60v > 0 and ma60v < close*0.999 else '—'
        def _f(col, fmt='—'):
            v = row.get(col, np.nan)
            return fmt.format(float(v)) if not pd.isna(v) else '—'
        op_label = '重點追蹤' if (s_chip >= 20 and total >= 70) else '穩健型'
        return {'sid':sid,'name':name,'close':close,'total':total,
                's_grow':s_grow,'s_chip':s_chip,'s_fin':s_fin,'s_val':s_val,'s_tech':s_tech,
                'details':details,'entry':entry,'sl_ma20':sl_ma20,'sl_ma60':sl_ma60,
                'tgt':f'{round(close*1.30,0):.0f}~{round(close*1.70,0):.0f}',
                'pct20':pct20,'ma20':ma20v,'ma60':ma60v,
                't1_grade':str(row.get('t1_grade','—')),
                'rsi':float(row.get('rsi14',50) or 50),
                'op_label':op_label,
                'weekly_ma13':bool(row.get('weekly_ma13_signal',False)),
                'eps_yoy':_f('eps_q1_yoy','{:.0f}%'),'roe':_f('roe_latest','{:.1f}%'),
                'pe':_f('pe','{:.1f}'),'peg':_f('peg','{:.2f}'),'debt_ratio':_f('debt_ratio','{:.0f}%')}

    l1_scored = [r for r in (l1_score(row) for _, row in L1_full_pool.iterrows()) if r]
    l1_scored.sort(key=lambda x: -x['total'])
    L1_TOP20 = l1_scored[:20]

    # ── S1
    def run_s1(pool_df):
        results = []
        for _, row in pool_df.iterrows():
            sid   = str(row['stock_id']); name = str(row.get('name', sid))
            close = float(row.get('close', 0) or 0)
            if close <= 0: continue
            ma20v = float(row.get('ma20', 0) or 0)
            rsi   = float(row.get('rsi14', 50) or 50)
            r5    = float(row.get('return_5d', 0) or 0)
            vr    = float(row.get('volume_ratio', 0) or 0)
            m5m10 = bool(row.get('ma5_gt_ma10', False))
            cs    = str(row.get('chip_sync', ''))
            fn_r  = int(row.get('foreign_rank_today', 999) or 999)
            fntv_r = int(row.get('_fntv_rank', 999))
            if not (close > ma20v and ma20v > 0): continue
            if rsi >= 78: continue
            if r5 >= 25: continue
            conds = []
            if vr > 1.10: conds.append(f'量比{vr:.2f}x')
            if m5m10:     conds.append('MA5>MA10')
            if r5 > 1.5:  conds.append(f'5日漲{r5:.1f}%')
            if any(k in cs for k in ['雙買','單買投','單買外']): conds.append(f'籌碼({cs})')
            rank_use = min(fn_r, fntv_r)
            if rank_use <= 30: conds.append(f'外資Top{rank_use}')
            cnt = len(conds)
            if cnt >= 5:   label = '🔥主力股'
            elif cnt >= 3: label = '✅強勢股'
            else: continue
            foreign_rank_score = max(0, (30 - rank_use)) * 3 if rank_use <= 30 else 0
            weighted_score     = (vr * 12) + (r5 * 6) + (cnt * 20) + foreign_rank_score
            results.append({'sid':sid,'name':name,'close':close,'label':label,
                            'cnt':cnt,'conds':conds,'rsi':rsi,'vr':vr,'r5':r5,'chip':cs,
                            'ma20':ma20v,'sl':round(ma20v,0),'tgt':round(close*1.15,0),
                            'pct20':float(row.get('price_vs_ma20_pct',0) or 0),
                            't1_grade':str(row.get('t1_grade','—')),
                            'fresh':bool(row.get('fresh_priority',False)),
                            'cross_note':str(row.get('cross_note','')),
                            'stop_fall_k':bool(row.get('stop_fall_k',False)),
                            'volume_shrink':bool(row.get('volume_shrink',False)),
                            'ma5_gt_ma10':m5m10,
                            'weekly_ma13':bool(row.get('weekly_ma13_signal',False)),
                            'weighted_score':weighted_score})
        results.sort(key=lambda x: -x['weighted_score'])
        return results

    S1_full_pool = run_s1(L1_full_pool)
    S1_TOP12     = S1_full_pool[:12]
    S1_MAJOR     = [r for r in S1_TOP12 if '主力' in r['label']]

    # ── T1-A（從 S1 pool 挑選）
    def run_t1a_from_s1(s1_pool):
        results = []
        for r in s1_pool:
            sid   = str(r['sid']).strip()
            if sid.startswith('28'): continue
            fresh = bool(r.get('fresh', False))
            t1g   = str(r.get('t1_grade', 'C'))
            cn    = str(r.get('cross_note', ''))
            vr    = float(r.get('vr', 0) or 0)
            r5d   = float(r.get('r5', 0) or 0)
            ma20v = float(r.get('ma20', 0) or 0)
            close = float(r.get('close', 0) or 0)
            if not (fresh or t1g in ['A+', 'A']): continue
            if t1g not in ['A+', 'A', 'A-']: continue
            if not has_valid_cross(cn): continue
            if vr <= 1.45: continue
            if r5d <= 3.8: continue
            if float(r.get('rsi', 50) or 50) >= 73: continue
            sl  = round(ma20v, 0)
            tgt = round(close * 1.15, 0)
            results.append({'sid':sid,'name':r['name'],'close':close,'t1_grade':t1g,
                            'cross_note':cn,'vr':vr,
                            'pct20':float(r.get('pct20', 0) or 0),
                            'ma20':ma20v,'sl':sl,'tgt':tgt,
                            'rsi':float(r.get('rsi', 50) or 50),
                            'chip':str(r.get('chip', '')),
                            'r5d':r5d,
                            'weekly_ma13':bool(r.get('weekly_ma13', False))})
        return results

    # T1-B 已取消

    t1_top_results = run_t1_top(df)
    t1a_results    = run_t1a_from_s1(S1_full_pool)
    t1b_results    = []   # T1-B 已取消
    s1_sids        = {r['sid'] for r in S1_full_pool}
    t1_top_final   = [r for r in t1_top_results if r['sid'] in s1_sids]

    # ── Z1
    def z1_check(row):
        close = float(row.get('close', 0) or 0)
        ma20v = float(row.get('ma20', 0) or 0)
        rsi   = float(row.get('rsi14', 50) or 50)
        bull  = bool(row.get('ma_bull_align', False))
        r5    = float(row.get('return_5d', 0) or 0)
        checks = []
        if close > ma20v and ma20v > 0:
            checks.append(('✅', f'站穩月線{ma20v:.0f}'))
        else:
            checks.append(('❌', f'跌破月線{ma20v:.0f}→立即出場'))
        checks.append(('✅','均線多頭排列') if bull else ('⚠️','均線未多頭'))
        checks.append(('✅',f'RSI{rsi:.0f}未過熱') if rsi < 75 else ('⚠️',f'RSI{rsi:.0f}過熱→考慮獲利了結'))
        checks.append(('✅',f'5日{r5:.1f}%正常') if r5 > -5 else ('⚠️',f'5日{r5:.1f}%回撤偏大'))
        fail_cnt = sum(1 for c in checks if c[0] == '❌')
        warn_cnt = sum(1 for c in checks if c[0] == '⚠️')
        if fail_cnt > 0:    verdict = '🛑 立即出場'
        elif warn_cnt >= 2: verdict = '⚠️ 減碼觀察'
        else:               verdict = '✅ 續抱'
        return verdict, checks

    # Z1：只取 S1 加權分排名前5
    z1_results = []
    for r in S1_full_pool[:5]:
        sid = r['sid']
        df_row_df = df[df['stock_id'].astype(str) == sid]
        if df_row_df.empty: continue
        df_row = df_row_df.iloc[0]
        verdict, checks = z1_check(df_row)
        z1_results.append({'sid':sid,'name':r['name'],'close':r['close'],
                           'verdict':verdict,'checks':checks,'category':'💥S1主力'})

    print(f'T1-TOP:{len(t1_top_final)} L1:{len(L1_TOP20)}/{len(L1_full_pool)} S1:{len(S1_TOP12)}/{len(S1_full_pool)} T1-A:{len(t1a_results)} T1-B:{len(t1b_results)} Z1:{len(z1_results)}')

    risk_rules = [
        '單檔最高 15% 資金',
        '整體最大回撤目標控制在 30% 以內',
        'L1 長線持股止損：跌破季線（MA60）或 Z1 未過關',
        'S1/T1 持股止損：跌破 MA20 立即出場',
        '達目標一先賣 1/2，剩餘持至目標二或趨勢轉弱',
    ]

    return (t1_top_final, L1_TOP20, L1_full_pool, S1_TOP12, S1_full_pool,
            t1a_results, t1b_results, z1_results, risk_rules, N_STOCKS, macro)


# ============================================================
# ── PART 2：下跌趨勢最高勝率拉回共振計分
# ============================================================

# ── 計分參數（對應 plugin_score_test_v2.py Cell 3）────────────
_W_C1             = 4.0   # 雙買發動股
_W_C4             = 3.0   # 安全回踩
_W_C5             = 2.0   # 乖離黃金點
_W_C2             = 1.0   # 投信新認養
_SCORE_MIN        = 6.0   # 6分以下不展示
_TOP_N            = 20    # 最多顯示幾筆
_GRADE_AAA        = 8.0   # ⭐⭐⭐ 極強共振
_GRADE_AA         = 6.5   # ⭐⭐  強力推薦
# 6.0 ~ 6.4 → ⭐ 值得留意

def run_downtrend_plugin(df):
    """
    🔌 下跌趨勢最高勝率拉回共振計分
    綜合加權計分（滿分10分），6分以上列入，最多顯示前20檔。
    C1（雙買發動股4分） + C4（安全回踩3分） + C5（乖離黃金點2分） + C2（投信新認養1分）
    """
    _df = df.copy()

    # ── 四條件布林 ───────────────────────────────────────────────
    def _safe_bool(col):
        return _df[col].astype(str).str.strip().str.lower().map(
            {'true': True, 'false': False, '1': True, '0': False}
        ).fillna(False)

    chip_sync_s       = _df['chip_sync'].astype(str).str.strip()
    trust_today_s     = pd.to_numeric(_df['trust_net_today'], errors='coerce').fillna(0)
    trust_20d_s       = pd.to_numeric(_df['trust_net_20d'],   errors='coerce').fillna(0)
    ma20_pct_s        = pd.to_numeric(_df['price_vs_ma20_pct'], errors='coerce').fillna(0)
    bias_5_s          = pd.to_numeric(_df['bias_5'],  errors='coerce').fillna(0)
    bias_20_s         = pd.to_numeric(_df['bias_20'], errors='coerce').fillna(0)
    vol_shrink_s      = _safe_bool('volume_shrink')
    stop_fall_s       = _safe_bool('stop_fall_k')

    c1 = (chip_sync_s == '雙買') & (trust_today_s > 0)
    c4 = vol_shrink_s & stop_fall_s & (ma20_pct_s >= -12)
    c5 = (bias_5_s <= -3) & (bias_20_s >= -5)
    c2 = (trust_today_s > 1000) & (trust_20d_s > -200)

    score = (c1.astype(float) * _W_C1 +
             c4.astype(float) * _W_C4 +
             c5.astype(float) * _W_C5 +
             c2.astype(float) * _W_C2)

    _df['_c1']    = c1; _df['_c2'] = c2
    _df['_c4']    = c4; _df['_c5'] = c5
    _df['_score'] = score

    hit = (_df[_df['_score'] >= _SCORE_MIN]
           .sort_values('_score', ascending=False)
           .head(_TOP_N)
           .copy())

    results = []
    for _, row in hit.iterrows():
        sid   = str(row['stock_id'])
        name  = str(row.get('name', sid))
        close = float(row.get('close', 0) or 0)
        sc    = float(row['_score'])

        # 推薦等級
        grade_str = ('⭐⭐⭐' if sc >= _GRADE_AAA
                     else '⭐⭐' if sc >= _GRADE_AA
                     else '⭐')

        # 符合條件字串
        parts = []
        if row['_c1']: parts.append('雙買發動股')
        if row['_c4']: parts.append('安全回踩')
        if row['_c5']: parts.append('乖離黃金點')
        if row['_c2']: parts.append('投信新認養')
        cond_str = ' + '.join(parts)

        # 關鍵數據
        t_today = int(row.get('trust_net_today', 0) or 0)
        t_20d   = int(row.get('trust_net_20d',   0) or 0)
        ma_pct  = float(row.get('price_vs_ma20_pct', 0) or 0)
        b5      = float(row.get('bias_5',  0) or 0)
        b20     = float(row.get('bias_20', 0) or 0)
        key_data = (f'今日{t_today:+,}張 20d={t_20d:+,}張 '
                    f'MA20%={ma_pct:+.1f}% b5={b5:.1f} b20={b20:.1f}')

        # 建議觀點
        if sc >= _GRADE_AAA:
            suggest = '極強共振，可積極低接'
        elif sc >= _GRADE_AA:
            suggest = '強力共振，可考慮分批布局'
        else:
            suggest = '技術回踩中，留意量縮止穩後進場'

        # 標註
        tags = []
        if sc >= _GRADE_AAA:   tags.append('🔥 強共振')
        if row['_c1']:          tags.append('💪 雙買發動')
        if row['_c4']:          tags.append('🛡️ 安全回踩')
        tag_str = ' '.join(tags)

        results.append({
            'sid': sid, 'name': name, 'close': close,
            'score':     sc,
            'grade':     grade_str,
            'cond_str':  cond_str,
            'key_data':  key_data,
            'suggest':   suggest,
            'tags':      tag_str,
        })

    # 統計輸出
    print(f'  🔌 下跌趨勢拉回共振 | 掃描{len(_df)}檔 '
          f'| ⭐⭐⭐:{(score>=_GRADE_AAA).sum()} '
          f'⭐⭐:{((score>=_GRADE_AA)&(score<_GRADE_AAA)).sum()} '
          f'⭐:{((score>=_SCORE_MIN)&(score<_GRADE_AA)).sum()} '
          f'| 列入{len(results)}檔')
    return results


# ============================================================
# ── PART 2：K 線圖（TWSE API）
# ============================================================

def get_kline_base64(sid, months_back=3):
    records = []
    for m in range(months_back + 1):
        mo = (TODAY_D.month - m - 1) % 12 + 1
        yr = TODAY_D.year - ((TODAY_D.month - m - 1) // 12)
        ym = f'{yr}{mo:02d}01'
        for attempt in range(2):
            try:
                time.sleep(0.4)
                url = (f'https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY'
                       f'?response=json&date={ym}&stockNo={sid}')
                r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
                if r.status_code != 200: time.sleep(1); continue
                j = r.json()
                for row in j.get('data', []):
                    try:
                        parts   = str(row[0]).split('/')
                        date_ad = f'{int(parts[0])+1911}-{parts[1]}-{parts[2]}'
                        o = float(str(row[3]).replace(',', ''))
                        h = float(str(row[4]).replace(',', ''))
                        l = float(str(row[5]).replace(',', ''))
                        c = float(str(row[6]).replace(',', ''))
                        v = float(str(row[1]).replace(',', ''))
                        records.append({'Date':date_ad,'Open':o,'High':h,'Low':l,'Close':c,'Volume':v})
                    except: pass
                break
            except: time.sleep(1)
    if len(records) < 5: return None
    try:
        df_k = pd.DataFrame(records).drop_duplicates('Date')
        df_k['Date'] = pd.to_datetime(df_k['Date'])
        df_k = df_k.sort_values('Date').set_index('Date')
        mc    = mpf.make_marketcolors(up='#3fb950', down='#f85149', edge='inherit', wick='inherit',
                                       volume={'up':'#3fb950','down':'#f85149'})
        style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc,
                                    facecolor='#161b22', figcolor='#161b22',
                                    gridcolor='#21262d', gridstyle='--',
                                    rc={'font.size':7,'axes.labelsize':6,
                                        'xtick.labelsize':6,'ytick.labelsize':6})
        add_plots = []
        ma20 = df_k['Close'].rolling(20).mean()
        ma60 = df_k['Close'].rolling(60).mean()
        if ma20.notna().sum() > 2:
            add_plots.append(mpf.make_addplot(ma20, color='#f0883e', width=1.0))
        if ma60.notna().sum() > 2:
            add_plots.append(mpf.make_addplot(ma60, color='#58a6ff', width=1.0))
        kwargs = dict(type='candle', style=style, volume=True,
                      figsize=(4.5, 1.6), returnfig=True, tight_layout=True, xrotation=20)
        if add_plots: kwargs['addplot'] = add_plots
        fig, axes = mpf.plot(df_k, **kwargs)
        fig.patch.set_facecolor('#161b22')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=72, bbox_inches='tight', facecolor='#161b22')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')
    except:
        return None


def fetch_z1_klines(z1_results):
    print('  📊 抓取 Z1 K線資料...')
    cache = {}
    fail_list = []
    for r in z1_results:
        sid = r['sid']
        if sid in cache: continue
        b64 = get_kline_base64(sid)
        cache[sid] = b64
        if b64:
            print(f'    ✅ {sid} {r["name"]}')
        else:
            fail_list.append((sid, r['name']))
            print(f'    ⚠️ {sid} {r["name"]} 首次失敗')
    if fail_list:
        print(f'  🔄 重試 {len(fail_list)} 檔...')
        time.sleep(3)
        for sid, name in fail_list:
            b64 = get_kline_base64(sid)
            cache[sid] = b64
            print(f'    {"✅" if b64 else "❌"} {sid} {name}')
    ok = sum(1 for v in cache.values() if v)
    print(f'  K線完成：{ok}/{len(cache)} 檔')
    return cache


# ============================================================

# ============================================================
# ── PART 3：MAX 三系統（P5長線 · V7 v8.0鐵三角 · T1 v8.2買點）
#    共用 CSV 資料庫，共用 CSS，視覺完全統一
# ============================================================

# CSS_MAX 就是 888 的深色主題 CSS，對正下面直接讀 CSS

# ── P5 評分引擎
P5_PASS = 100

def calc_p5(row):
    close = float(row.get("close",0) or 0)
    if close <= 0: return None
    sid  = str(row.get("stock_id","")); name = str(row.get("name",""))

    # 面向1 基本面 30分
    s1=0; d1=[]
    e3q   = bool(row.get("eps_3q_yoy_gt20", False))
    qoq   = row.get("eps_q1_qoq", np.nan)
    qoqf  = float(qoq) if not pd.isna(qoq) else None
    ey1   = row.get("eps_q1_yoy", np.nan)
    roei  = bool(row.get("roe_gt5_improving", False))
    roev  = row.get("roe_latest", np.nan)
    gmi   = str(row.get("gross_margin_improving",""))
    gm1   = row.get("gross_margin_q1", np.nan)
    mry   = row.get("monthly_rev_yoy", np.nan)
    mrm   = row.get("monthly_rev_mom", np.nan)
    if e3q and qoqf is not None and qoqf >= 5:
        s1+=12; d1.append("EPS連3季YoY>20%+QoQ" + str(round(qoqf,0)) + "% +12")
    elif e3q:
        s1+=8;  d1.append("EPS連3季YoY>20% +8")
    elif not pd.isna(ey1) and float(ey1) > 20:
        s1+=4;  d1.append("EPS年增率" + str(round(float(ey1),0)) + "% +4")
    elif not pd.isna(ey1) and float(ey1) > 0:
        s1+=2;  d1.append("EPS年增率" + str(round(float(ey1),0)) + "%(弱) +2")
    if roei:
        s1+=10; d1.append("ROE" + (str(round(float(roev),1)) if not pd.isna(roev) else "") + "%>5%改善 +10")
    elif not pd.isna(roev) and float(roev) > 5:
        s1+=6;  d1.append("ROE" + str(round(float(roev),1)) + "%>5% +6")
    if gmi == "True":
        s1+=4;  d1.append("毛利率連3季改善(" + (str(round(float(gm1),1)) if not pd.isna(gm1) else "") + "%) +4")
    elif not pd.isna(gm1) and float(gm1) > 30:
        s1+=1;  d1.append("毛利率" + str(round(float(gm1),1)) + "% +1")
    if not pd.isna(mry) and float(mry) > 15 and not pd.isna(mrm) and float(mrm) > 0:
        s1+=4;  d1.append("月營收YoY" + str(round(float(mry),0)) + "%+MoM正 +4")
    elif not pd.isna(mry) and float(mry) > 10:
        s1+=2;  d1.append("月營收YoY" + str(round(float(mry),0)) + "% +2")
    s1 = min(s1, 30); f1 = s1 >= 18

    # 面向2 估值 20分（加入 forward_pe_proxy）
    s2=0; d2=[]
    pe  = row.get("pe",  np.nan)
    peg = row.get("peg", np.nan)
    dy  = row.get("dividend_yield", np.nan)
    fpp = bool(row.get("forward_pe_proxy", False))
    if not pd.isna(pe):
        pf = float(pe)
        if   pf < 15: s2+=8; d2.append("本益比" + str(round(pf,1)) + "<15 +8")
        elif pf < 20: s2+=5; d2.append("本益比" + str(round(pf,1)) + "<20 +5")
        elif pf < 25: s2+=3; d2.append("本益比" + str(round(pf,1)) + "<25 +3")
        else:              d2.append("本益比" + str(round(pf,1)) + "偏高 +0")
    else: d2.append("本益比無資料 +0")
    if not pd.isna(peg):
        pgf = float(peg)
        if   pgf < 0.8: s2+=7; d2.append("PEG" + str(round(pgf,2)) + "<0.8 +7")
        elif pgf < 1.2: s2+=7; d2.append("PEG" + str(round(pgf,2)) + "<1.2 +7")
        elif pgf < 1.5: s2+=4; d2.append("PEG" + str(round(pgf,2)) + "<1.5 +4")
    else: d2.append("PEG無資料 +0")
    # Forward PE Proxy：EPS+Rev雙加速 且 PEG合理
    if fpp:
        s2+=5; d2.append("Forward PE上修(EPS+Rev加速+PEG<1.3) +5")
    elif not pd.isna(peg) and float(peg) < 1.0 and e3q:
        s2+=3; d2.append("Forward PE保守上修空間 +3")
    s2 = min(s2, 20); f2 = s2 >= 12

    # 面向3 財務健康 20分（加入 fcf_proxy）
    s3=0; d3=[]
    dr   = row.get("debt_ratio", np.nan)
    oi1  = row.get("op_income_q1", np.nan)
    oi2  = row.get("op_income_q2", np.nan)
    oi3  = row.get("op_income_q3", np.nan)
    dmf  = row.get("daily_money_flow", np.nan)
    fcf_bool = bool(row.get("fcf_proxy", False))
    # 原始op_income連3季正且成長
    fcf_strict = (not pd.isna(oi1) and not pd.isna(oi2) and not pd.isna(oi3)
                  and float(oi1)>0 and float(oi2)>0 and float(oi3)>0
                  and float(oi1) > float(oi2))
    if fcf_strict and fcf_bool:
        s3+=8; d3.append("FCF連3季正且成長+現金流強健 +8")
    elif fcf_strict:
        s3+=8; d3.append("FCF連3季正向且成長 +8")
    elif fcf_bool:
        s3+=5; d3.append("FCF代理通過(EPS+Rev+負債) +5")
    elif not pd.isna(oi1) and float(oi1) > 0:
        s3+=3; d3.append("營業利益正向 +3")
    if not pd.isna(dr):
        df_ = float(dr)
        if   df_ < 30: s3+=6; d3.append("負債比" + str(round(df_,0)) + "%極低 +6")
        elif df_ < 50: s3+=6; d3.append("負債比" + str(round(df_,0)) + "%<50% +6")
        elif df_ < 65: s3+=3; d3.append("負債比" + str(round(df_,0)) + "%偏高 +3")
        else:               d3.append("負債比" + str(round(df_,0)) + "%高 +0")
    if not pd.isna(dy) and float(dy) > 1.5: s3+=4; d3.append("殖利率" + str(round(float(dy),1)) + "% +4")
    elif not pd.isna(dy) and float(dy) > 0.8: s3+=2; d3.append("殖利率" + str(round(float(dy),1)) + "% +2")
    s3b = min(s3, 20)
    if not pd.isna(dmf) and float(dmf) < 2.5:
        s3b = min(s3b+2, 20); d3.append("資金流" + str(round(float(dmf),1)) + "%<2.5% +2")
    s3 = s3b; f3 = s3 >= 12

    # 面向4 籌碼 25分
    s4=0; d4=[]
    rnk  = int(row.get("foreign_rank_today",999) or 999)
    tr   = str(row.get("foreign_3m_trend",""))
    cd   = int(row.get("foreign_consecutive_days",0) or 0)
    cs   = str(row.get("chip_sync","中性"))
    tnt5 = int(row.get("trust_net_5d",0) or 0)
    tnt  = int(row.get("trust_net_today",0) or 0)
    if rnk<=15 and tr=="up":   s4+=10; d4.append("外資前" + str(rnk) + "名+3M上升 +10")
    elif rnk<=30 and tr=="up": s4+=6;  d4.append("外資前" + str(rnk) + "名+趨勢上 +6")
    elif rnk<=50 and tr=="up": s4+=3;  d4.append("外資前" + str(rnk) + "名 +3")
    if cs == "雙買":    s4+=8; d4.append("雙買(外資+投信) +8")
    elif "單買" in cs:  s4+=4; d4.append(cs + " +4")
    if tnt5>0 and tnt>0:   s4+=7; d4.append("投信5日+今日皆買 +7")
    elif tnt>0 or tnt5>0:  s4+=4; d4.append("投信近期買超 +4")
    s4 = min(s4, 25); f4 = s4 >= 15

    # 面向5 技術 20分
    s5=0; d5=[]
    bull  = bool(row.get("ma_bull_align",False))
    rsi   = float(row.get("rsi14",100) or 100)
    ab60  = bool(row.get("price_above_ma60",False))
    vr    = float(row.get("volume_ratio",1) or 1)
    rel20 = float(row.get("rel_strength_20d",0) or 0)
    wmsig = bool(row.get("weekly_ma13_signal",False))
    if bull:   s5+=8; d5.append("月/季/年線多頭 +8")
    elif ab60: s5+=4; d5.append("站穩季線 +4")
    if   rsi < 50: s5+=5; d5.append("RSI" + str(round(rsi,0)) + "<50 +5")
    elif rsi < 65: s5+=3; d5.append("RSI" + str(round(rsi,0)) + "中性 +3")
    elif rsi < 70: s5+=1; d5.append("RSI" + str(round(rsi,0)) + "偏高 +1")
    if   vr > 1.15: s5+=4; d5.append("月均量放大" + str(round(vr,1)) + "x +4")
    elif vr > 1.0:  s5+=2; d5.append("量比" + str(round(vr,1)) + "x +2")
    if   rel20 > 2: s5+=4; d5.append("相對大盤強+" + str(round(rel20,1)) + "% +4")
    elif rel20 > 0: s5+=3; d5.append("相對大盤略強 +3")
    if wmsig: s5+=2; d5.append("週線MA13多頭 +2")
    s5 = min(s5, 20); f5 = s5 >= 12

    # 面向6 風險 15分
    s6=0; d6=[]
    beta  = row.get("beta_1y", np.nan)
    mdd3  = row.get("max_drawdown_3y", np.nan)
    pfl   = row.get("pct_from_52w_low", np.nan)
    nsco  = int(row.get("news_score",0) or 0)
    if not pd.isna(pfl) and float(pfl) > 30:
        s6+=3; d6.append("距52低" + str(round(float(pfl),0)) + "%>30% +3")
    if not pd.isna(beta):
        bf = float(beta)
        if   bf < 0.8: s6+=6; d6.append("Beta" + str(round(bf,2)) + "低波動 +6")
        elif bf < 1.2: s6+=4; d6.append("Beta" + str(round(bf,2)) + "<1.2 +4")
        else:               d6.append("Beta" + str(round(bf,2)) + "高 +0")
    if not pd.isna(mdd3):
        mf = float(mdd3)
        if   mf > -15: s6+=5; d6.append("回撤" + str(round(mf,0)) + "%極低 +5")
        elif mf > -25: s6+=5; d6.append("回撤" + str(round(mf,0)) + "%可控 +5")
        elif mf > -35: s6+=2; d6.append("回撤" + str(round(mf,0)) + "%偏大 +2")
    s6+=2; d6.append("產業分散 +2")
    s6b = min(s6, 15)
    if nsco >= 2: s6b = min(s6b+2, 15); d6.append("news_score=" + str(nsco) + " +2")
    s6 = s6b; f6 = s6 >= 9

    total = s1+s2+s3+s4+s5+s6
    faces = sum([f1,f2,f3,f4,f5,f6])

    ma20v = row.get("ma20", np.nan); ma60v = row.get("ma60", np.nan)
    if not pd.isna(ma20v):
        mv = float(ma20v); pct = (close/mv-1)*100 if mv > 0 else 0
        entry = (("等回月線" + str(round(mv,0))) if pct > 10
                 else (str(round(mv*0.99,0)) + "~" + str(round(mv*1.03,0))) if pct > 3
                 else (str(round(close*0.99,0)) + "~" + str(round(close*1.01,0))))
    else:
        entry = str(round(close*0.99,0)) + "~" + str(round(close*1.01,0))

    sl1  = (str(round(float(ma20v),0)) if not pd.isna(ma20v) and float(ma20v)<close*0.999 else "月線≈現價")
    sl2  = (str(round(float(ma60v),0)) if not pd.isna(ma60v) and float(ma60v)<close*0.999 else "—")
    tgt  = str(round(close*1.15,0)) + "~" + str(round(close*1.30,0))
    op   = "核心持股" if total >= 110 else ("重點追蹤" if total >= 95 else "穩健型")
    pct20= float(row.get("price_vs_ma20_pct",0) or 0)
    t1g  = str(row.get("t1_grade","—"))
    wfp  = bool(row.get("fresh_priority",False))

    ey1_s  = (str(round(float(ey1),0)) + "%" if not pd.isna(ey1) else "—")
    roe_s  = (str(round(float(roev),1)) + "%" if not pd.isna(roev) else "—")
    pe_s   = (str(round(float(pe),1))   if not pd.isna(pe)  else "—")
    peg_s  = (str(round(float(peg),2))  if not pd.isna(peg) else "—")
    dr_s   = (str(round(float(dr),0)) + "%" if not pd.isna(dr) else "—")
    nsco_s = str(nsco)

    return {"sid":sid,"name":name,"close":close,"total":total,"faces":faces,
            "fp":[f1,f2,f3,f4,f5,f6],
            "s1":s1,"s2":s2,"s3":s3,"s4":s4,"s5":s5,"s6":s6,
            "d1":d1,"d2":d2,"d3":d3,"d4":d4,"d5":d5,"d6":d6,
            "ey1_s":ey1_s,"roe_s":roe_s,"pe_s":pe_s,"peg_s":peg_s,"dr_s":dr_s,
            "entry":entry,"sl1":sl1,"sl2":sl2,"tgt":tgt,"op":op,
            "t1g":t1g,"pct20":pct20,"wfp":wfp,"wmsig":bool(row.get("weekly_ma13_signal",False)),
            "rsi":rsi,"cs":cs,"nsco":nsco_s,"dmf":row.get("daily_money_flow",np.nan)}

def run_p5(df):
    res = []
    for _,row in df.iterrows():
        if float(row.get("close",0) or 0) <= 0: continue
        r = calc_p5(row)
        if r is None: continue
        if r["total"] >= P5_PASS or r["faces"] >= 5: res.append(r)
    if len(res) < 8:
        existing = {r["sid"] for r in res}
        extra = []
        for _,row in df.iterrows():
            if str(row.get("stock_id","")) in existing: continue
            if float(row.get("close",0) or 0) <= 0: continue
            r = calc_p5(row)
            if r and r["total"] >= 60: extra.append(r)
        extra.sort(key=lambda x: -x["total"])
        for r in extra:
            if len(res) >= 8: break
            r["_sup"] = True; res.append(r)
    res.sort(key=lambda x: -x["total"])
    return res[:12]


# ══════════════════════════════════════════════════════════════════


# ── V7 v8.0 鐵三角極強版
# ◄ G. V7 v8.0 鐵三角極強版（短線最高勝率鎖死）
def run_v7_v8(df, min_gates=6, min_score=92):
    passed=[]; backup=[]
    for _,row in df.iterrows():
        sid = str(row["stock_id"]); name = str(row.get("name",sid))
        close = float(row.get("close",0) or 0)
        if close <= 0: continue

        r5 = float(row.get("return_5d", 0) or 0)
        r1y = float(row.get("return_1y", 0) or 0)
        cd = int(row.get("foreign_consecutive_days",0) or 0)
        vr = float(row.get("volume_ratio",0) or 0)
        rsi = float(row.get("rsi14",50) or 50)
        rnk = int(row.get("foreign_rank_today",999) or 999)
        cs = str(row.get("chip_sync","中性"))
        nsco = int(row.get("news_score",0) or 0)
        mry = row.get("monthly_rev_yoy", np.nan)
        qoq = row.get("eps_q1_qoq", np.nan)
        dmf = row.get("daily_money_flow", np.nan)
        pct20= float(row.get("price_vs_ma20_pct",0) or 0)
        ma20v= float(row.get("ma20",close*0.95) or close*0.95)
        rel20= float(row.get("rel_strength_20d",0) or 0)
        fntv = float(row.get("fntv",0) or 0)
        shrink= bool(row.get("volume_shrink",False))
        wmsig= bool(row.get("weekly_ma13_signal",False))
        fresh= bool(row.get("fresh_priority",False))
        cn = str(row.get("cross_note",""))
        t1g = str(row.get("t1_grade","C"))
        sec_m= bool(row.get("sector_momentum",False))

        mryf = float(mry) if not pd.isna(mry) else None
        qoqf = float(qoq) if not pd.isna(qoq) else None
        rev_tank = mryf is not None and mryf < -10

        # G1
        if r5 < 15 and r1y < 250:
            g1s=12; g1ok=True
        elif 5 <= r5 <= 15 and r1y < 250 and (cs=="雙買" or nsco>=2 or rnk<=15):
            g1s=10; g1ok=True
        else:
            g1s=0; g1ok=False

        # G2
        strong_cond = (cs=="雙買" and (mryf is not None and mryf > 10 or qoqf is not None and qoqf > 10 or nsco >= 2) and not rev_tank)
        mid_cond = (cs=="雙買" or "單買" in cs or (mryf is not None and mryf > 0) or nsco == 1)
        weak_cond = (cd >= 3 or rnk <= 30 or (qoqf is not None and qoqf > 0))
        if strong_cond:
            g2s=30; g2ok=True; cat="強"
        elif mid_cond or (cs=="雙買" and rev_tank):
            g2s=22; g2ok=True
            cat = "中(Rev↓)" if (cs=="雙買" and rev_tank) else "中"
        elif weak_cond:
            g2s=15; g2ok=True; cat="弱"
        else:
            g2s=0; g2ok=False; cat="無"

        # G3
        vr_ok = vr >= 1.3 or (1.2 <= vr < 1.3 and rsi < 65)
        blk_k = vr > 3.0 and r5 > 8
        if vr_ok and not blk_k:
            g3_base = 20
            g3_bonus= 5 if (not pd.isna(dmf) and float(dmf) < 2.5) else 0
            g3s = min(g3_base + g3_bonus, 25); g3ok = True
        else:
            g3s=0; g3ok=False

        # G4
        pos_ok = pct20 <= 12; rel_ok = rel20 > 0.5
        if pos_ok and rel_ok:
            g4s=15; g4ok=True
        elif pos_ok:
            g4s=12; g4ok=True
        elif pct20 <= 15 and rnk <= 10:
            g4s=7; g4ok=True
        else:
            g4s=0; g4ok=False

        # G5
        sl = min(ma20v*0.97, close*0.95)
        rr = (close*1.08 - close)/(close - sl) if close > sl else 0
        if rr >= 1.5: g5s=10; g5ok=True
        elif rr >= 1.2: g5s=5; g5ok=True
        else: g5s=0; g5ok=False

        # G6
        grp_ok = sec_m or rnk <= 20
        if grp_ok:
            g6s = 13 + (3 if cs=="雙買" else 0); g6ok=True
        else:
            g6s=0; g6ok=False

        tot = min(g1s+g2s+g3s+g4s+g5s+g6s, 100)
        gates = [g1ok,g2ok,g3ok,g4ok,g5ok,g6ok]
        ng = sum(gates)

        # 鐵三角：fresh + T1標 A將 + 雙買
        iron_triangle = (fresh == True and t1g in ["A+", "A", "A-"] and cs == "雙買")

        reasons = []
        if iron_triangle:
            reasons.insert(0, "🔥鐵三角通過")
        if rnk<=10: reasons.append("外資買超#"+str(rnk)+"名主力積極進場")
        elif rnk<=20: reasons.append("外資排名#"+str(rnk))
        if cs=="雙買": reasons.append("外資+投信雙買(籌碼同向)")
        elif "單買" in cs: reasons.append("籌碼:"+cs)
        if cd>=5: reasons.append("外資連買"+str(cd)+"天籌碼持續累積")
        elif cd>=3: reasons.append("外資連買"+str(cd)+"天")
        if vr>=2.0: reasons.append("量比"+str(round(vr,1))+"x市場關注度高")
        elif vr>=1.5: reasons.append("量比"+str(round(vr,1))+"x")
        if nsco>=2: reasons.append("news_score="+str(nsco)+"(重大事件)")
        if fresh: reasons.append("🔥週線 MA13剛穿("+cn+")趨勢啟動")
        elif wmsig: reasons.append("週線 MA13多頭")
        if rel20>3: reasons.append("相對大盤強+"+str(round(rel20,1))+"%")
        if sec_m: reasons.append("族群有領漲股(G6確認)")
        if rev_tank: reasons.append("⚠️月營收YoY"+str(round(mryf,0))+"%拖累催化")

        fterm = fntv/10000*0.20 if fntv > 0 else 0
        momentum = round(r5*0.30 + vr*10*0.25 + (100-rsi)*0.25 + fterm, 1)

        if tot >= 93: tier = "🔥極強"
        elif tot >= 86: tier = "✅強勢"
        elif tot >= 78: tier = "⚠️備案"
        else: tier = "—"

        tgt1 = round(close*1.08,0); tgt2 = round(close*1.15,0)
        rec = {
            "代號":sid,"名稱":name,"收盤":str(round(close,0)),
            "總分":tot,"分級":tier,
            "G1":g1s,"G2":g2s,"G3":g3s,"G4":g4s,"G5":g5s,"G6":g6s,
            "通過關":str(ng)+"/6","催化":cat,
            "動能分":str(round(momentum,0)),
            "外資排名":"#"+str(rnk),"外資連買":str(cd)+"天",
            "chip_sync":cs,"量比":str(round(vr,2))+"x","RSI":str(round(rsi,0)),
            "週線 MA13":("🔥剛穿" if fresh else ("多頭" if wmsig else "—")),
            "族群":("✅" if sec_m else "—"),
            "進場":str(round(close*0.997,0)) + "~" + str(round(close*1.003,0)),
            "止損":str(round(sl,0)),
            "目標一":str(tgt1),"目標二":str(tgt2),
            "盈虏比":str(round(rr,1)) + ":1",
            "核心理由":"｜".join(reasons) if reasons else "綜合條件",
            "失敗條件":"跌破MA20("+str(round(ma20v,0))+")或14天未達目標",
            "未通關":"/".join([
                ["漲幅","催化","籌碼","位置","盈虏","族群"][i]
                for i,ok in enumerate(gates) if not ok]) if any(not ok for ok in gates) else "—",
            "_tot":tot,"_fresh":fresh,
        }

        core_ok = (cat == "強") or g6ok
        if tot >= 92 and ng >= 6 and core_ok and iron_triangle:
            rec["標記"] = "🔥鐵三角極強"
            passed.append(rec)
        elif 82 <= tot < 92 and ng >= 5 and t1g == "A+":
            rec["標記"] = "【備案】"
            backup.append(rec)

    def vsort(r):
        return (0 if r.get("_fresh", False) and r.get("chip_sync") == "雙買" else 1, -r["_tot"])
    passed.sort(key=vsort); backup.sort(key=lambda r: -r["_tot"])
    for r in passed+backup: r.pop("_tot",None); r.pop("_fresh",None)
    return passed[:6], backup[:3]


# ── T1 v8.2 買點引擎
def run_t1(df, sid_filter=None, force_show=False):
    """
    T1 買點系統（方向B完整重寫版）
    ─────────────────────────────────────────────────────────
    升降級原則（乾淨、無循環漏洞）：
    · 降級優先於升級：先把所有降級跑完，再跑升級
    · 升 A+ 需要多重條件齊備，news或剛穿單獨不足以到 A+
    · 最終過濾：A+ 必須 RSI<55 + chip 不賣超
    · 每天預期 A+：0~4 檔，A：2~8 檔
    """
    res = []
    for _,row in df.iterrows():
        sid  = str(row["stock_id"])
        if sid_filter is not None and sid not in sid_filter: continue
        t1g  = str(row.get("t1_grade","C"))
        if not force_show and t1g not in ["A+","A","A-","B+"]: continue
        name  = str(row.get("name",sid))
        close = float(row.get("close",0) or 0)
        if close <= 0: continue

        # ── 取得所有欄位 ─────────────────────────────────────
        rsi    = float(row.get("rsi14",50)  or 50)
        vr     = float(row.get("volume_ratio",1) or 1)
        shrink = bool(row.get("volume_shrink",False))
        stop_k = bool(row.get("stop_fall_k",False))
        m5m10  = bool(row.get("ma5_gt_ma10",False))
        npbl   = bool(row.get("not_break_prev_low",True))
        bull   = bool(row.get("ma_bull_align",False))
        ab20   = bool(row.get("price_above_ma20",False))
        ma20v  = float(row.get("ma20",close*0.95) or close*0.95)
        pct20  = float(row.get("price_vs_ma20_pct",0) or 0)
        rel20  = float(row.get("rel_strength_20d",0) or 0)
        cd     = int(row.get("foreign_consecutive_days",0) or 0)
        rnk    = int(row.get("foreign_rank_today",999) or 999)
        cs     = str(row.get("chip_sync","中性"))
        nsco   = int(row.get("news_score",0) or 0)
        dmf    = row.get("daily_money_flow", np.nan)
        pfl    = row.get("pct_from_52w_low", np.nan)
        pe     = row.get("pe", np.nan)
        wmsig  = bool(row.get("weekly_ma13_signal",False))
        fresh  = bool(row.get("fresh_priority",False))
        cn     = str(row.get("cross_note",""))
        kd_k      = float(row.get("kd_k", 50) or 50)
        kd_d      = float(row.get("kd_d", 50) or 50)
        kd_golden = bool(row.get("kd_golden_cross", False))
        kd_death  = bool(row.get("kd_death_cross",  False))
        kd_low    = bool(row.get("kd_oversold",     False))
        kd_high   = bool(row.get("kd_overbought",   False))
        chip_sell = cs in ("雙賣","單賣")

        orig = t1g

        # ══ Phase 1：硬降級（無條件，先跑完）══════════════════
        if t1g in ["A+","A","A-","B+"]:

            # D1：RSI 過熱硬降
            #   RSI > 65 → 降一級（初始分級已排掉 >68，這裡補 62~65 的灰色地帶）
            #   RSI > 62 且非月線回踩 → 降一級
            if rsi > 65:
                t1g = {"A+":"A","A":"A-","A-":"B+","B+":"B"}.get(t1g, t1g)
            elif rsi > 62 and pct20 > 3:
                t1g = {"A+":"A","A":"A-","A-":"B+"}.get(t1g, t1g)

            # D2：KD 超買降級
            #   K > 75 → 降一級（不管有沒有死叉）
            if kd_k > 75:
                t1g = {"A+":"A","A":"A-","A-":"B+","B+":"B"}.get(t1g, t1g)

            # D3：籌碼賣超降級（外資+投信同步賣 → 直接降兩級）
            if chip_sell and cd <= -3:
                t1g = {"A+":"B+","A":"B","A-":"B-","B+":"C"}.get(t1g, t1g)
            elif chip_sell:
                t1g = {"A+":"A","A":"A-","A-":"B+","B+":"B"}.get(t1g, t1g)

            # D4：RSI 高檔背離（RSI > 58 + 相對大盤弱 > -1.5%）→ 降兩級
            if rsi > 58 and rel20 < -1.5:
                t1g = {"A+":"B+","A":"B+","A-":"B","B+":"B-"}.get(t1g, t1g)

            # D5：KD 死叉（高檔死叉：D > 65 且死叉）→ 降一級
            if kd_death and kd_d > 65:
                t1g = {"A+":"A","A":"A-","A-":"B+","B+":"B"}.get(t1g, t1g)

            # D6：過度追高（月線% > 7% 且 無週線MA13 且 量比 > 1.5）→ 降一級
            if pct20 > 7 and not wmsig and vr > 1.5:
                t1g = {"A+":"A","A":"A-","A-":"B+"}.get(t1g, t1g)

        # ══ Phase 2：升級（降級跑完後才升，且設硬上限）══════════
        # 升級前先記錄 Phase1 後的等級
        after_down = t1g

        if t1g in ["A+","A","A-","B+"]:

            # U1：KD 低檔金叉（K<30 穿越 D）→ 升一級（最高 A）
            #     注意：KD金叉單獨只能到 A，要到 A+ 需要更多條件
            if kd_golden and kd_low and t1g in ["A-","B+"]:
                t1g = {"A-":"A","B+":"A-"}.get(t1g, t1g)

            # U2：RSI 超低 + KD 金叉 → 升一級（最高 A）
            if rsi < 38 and kd_golden and t1g in ["A-","B+"]:
                t1g = {"A-":"A","B+":"A-"}.get(t1g, t1g)

            # U3：週線MA13剛穿 → 升一級（最高 A+，但需 RSI<58 + chip不賣）
            if fresh and not chip_sell and rsi < 58 and t1g in ["A","A-","B+"]:
                t1g = {"A":"A+","A-":"A","B+":"A-"}.get(t1g, t1g)

            # U4：news >= 3 + chip雙買 → 升一級（最高 A，新聞單獨不足升A+）
            if nsco >= 3 and cs == "雙買" and rsi < 60 and t1g in ["A-","B+"]:
                t1g = {"A-":"A","B+":"A-"}.get(t1g, t1g)

        # ══ Phase 3：A+ 最終關卡（升降後的 A+ 都要過這關）══════
        if t1g == "A+":
            # A+ 必備條件：RSI < 55 + chip 不賣超 + 月線未嚴重偏離
            if rsi >= 55 or chip_sell or pct20 > 5:
                t1g = "A"
            # A+ 加分確認：月線回踩帶 -3%~+2% + 量縮 + (止跌K 或 KD金叉)
            elif not (-3 <= pct20 <= 2 and shrink and (stop_k or kd_golden)):
                # 條件不夠漂亮 → 維持 A（不強制降，但不讓它留在 A+）
                t1g = "A"

        # 計算分級變化說明
        grade_chg = ""
        if t1g != orig:
            grade_chg = " (" + orig + "→" + t1g + ")"

        # ══ 主名單篩選 ══════════════════════════════════════════
        # force_show 模式：全部顯示
        # 一般模式：只取 A+/A（B+以下排除）
        if not force_show and sid_filter is None:
            if t1g not in ["A+","A"]: continue

        # 最低 V7 技術項目數量要求（A+/A 需 ≥ 4 項，A- 需 ≥ 3 項）
        v7_items = [
            ("MA5>MA10",  m5m10),
            ("量縮整理",  shrink),
            ("止跌K",     stop_k),
            ("RSI低檔",   rsi < 55),
            ("未破前低",  npbl and ab20),
            ("月線多頭",  bull),
            ("相對強勢",  rel20 > 0.5),
            ("chip雙買",  cs == "雙買"),
            ("KD低檔",    kd_low or kd_golden),
            ("KD未超買",  not kd_high),
        ]
        v7_pass_items = [lb for lb, ok in v7_items if ok]
        v7_cnt = len(v7_pass_items)
        v7_eff = v7_cnt + (1 if wmsig else 0) + (1 if fresh else 0)

        if not force_show and sid_filter is None:
            min_v7 = 4 if t1g == "A+" else 3
            if v7_eff < min_v7: continue

        # ══ 計算綜合評分（排序用）══════════════════════════════
        grade_pts = {"A+":120, "A":85, "A-":55, "B+":30}.get(t1g, 10)
        t1_score  = (grade_pts
                     + v7_cnt * 4
                     + (20 if fresh else 0)
                     + (12 if wmsig else 0)
                     + nsco * 5
                     + (8 if cs == "雙買" else 0)
                     + (6 if kd_golden else 0)
                     + (5 if rsi < 45 else 0))

        # ══ 計算進場/止損/目標 ══════════════════════════════════
        # 進場帶：月線附近 ±1%；止損：月線 -3%；目標：+10%/+18%
        elo  = round(ma20v * 0.99, 0)
        ehi  = round(ma20v * 1.015, 0)
        sl   = round(ma20v * 0.97, 0)
        tgt1 = round(close * 1.10, 0)
        tgt2 = round(close * (1.22 if fresh else 1.18), 0)

        # ══ 結論文字 ════════════════════════════════════════════
        parts = []
        grade_label = {
            "A+": "T1極佳",
            "A" : "T1良好",
            "A-": "T1觀察",
            "B+": "T1偏強",
        }.get(t1g, "T1")

        parts.append(grade_label + "：月線" + str(round(pct20,1)) + "%，V7" + str(v7_cnt) + "/10" + grade_chg + "。")

        # 技術說明
        if v7_pass_items:
            parts.append("技術：" + "/".join(v7_pass_items[:4]) + "。")

        # RSI + KD
        rsi_str = "RSI" + str(round(rsi,1))
        kd_str  = "KD" + str(round(kd_k,0))
        kd_note = ("低檔金叉🟢" if kd_golden and kd_low
                   else "超買❌" if kd_high
                   else "死叉⚠️" if kd_death
                   else "中性")
        parts.append(rsi_str + " " + kd_str + "(" + kd_note + ")。")

        # 外資籌碼
        if cd >= 5:
            parts.append("外資連買" + str(cd) + "天(#" + str(rnk) + ")強力布局。")
        elif cd >= 3:
            parts.append("外資連買" + str(cd) + "天(#" + str(rnk) + ")。")
        if cs == "雙買":
            parts.append("外資+投信雙買。")

        # news
        if nsco >= 2:
            parts.append("news=" + str(nsco) + "(重大催化)。")
        elif nsco == 1:
            parts.append("news=" + str(nsco) + "(一般)。")

        # 週線MA13
        if fresh:
            parts.append("🔥MA13剛穿(" + cn + ")，趨勢剛啟動。")
        elif wmsig:
            parts.append("⭐週線MA13多頭。")

        # 估值
        if not pd.isna(pe) and float(pe) < 20:
            parts.append("本益比" + str(round(float(pe),1)) + "合理。")

        # 進場建議
        if t1g == "A+":
            parts.append("🎯 建議" + str(int(elo)) + "~" + str(int(ehi)) + "分批進場，止損" + str(int(sl)) + "，目標" + str(int(tgt1)) + "/" + str(int(tgt2)) + "。")
        elif t1g == "A":
            parts.append("建議" + str(int(elo)) + "~" + str(int(ehi)) + "小量進場，止損" + str(int(sl)) + "。")
        else:
            parts.append("觀察為主，暫不進場。")

        wlbl = ("🔥2日剛穿" if fresh else ("⭐週線多頭" if wmsig else "—"))
        res.append({
            "代號":     sid,
            "名稱":     name,
            "T1":       t1g,
            "收盤":     str(round(close, 0)),
            "月線%":    str(round(pct20, 1)) + "%",
            "RSI":      str(round(rsi, 1)),
            "量比":     str(round(vr, 2)) + "x",
            "V7通過":   str(v7_cnt) + "/10(" + "+".join(v7_pass_items[:3]) + ")",
            "chip":     cs,
            "news":     nsco,
            "外資連買": str(cd) + "天",
            "週線MA13": wlbl,
            "進場區間": str(int(elo)) + "~" + str(int(ehi)),
            "止損":     str(int(sl)),
            "目標一":   str(int(tgt1)),
            "目標二":   str(int(tgt2)),
            "結論":     " ".join(parts),
            "_score":   t1_score,
            "_fresh":   fresh,
        })

    # 排序
    res.sort(key=lambda r: -r["_score"])

    if force_show and sid_filter:
        sid_map = {r["代號"]: r for r in res}
        main = [sid_map[s] for s in sid_filter if s in sid_map]
        fresh_l = []
    elif sid_filter:
        main = res[:len(sid_filter)]; fresh_l = []
    else:
        main    = [r for r in res if not r["_fresh"]][:6]
        fresh_l = [r for r in res if     r["_fresh"]][:5]

    for r in res:
        r.pop("_score", None)
        r.pop("_fresh", None)
    return main, fresh_l


# ══════════════════════════════════════════════════════════════════


# ── MAX HTML 工具
def bdg(txt, cls):
    return '<span class="bdg ' + cls + '">' + str(txt) + '</span>'

def wline_bdg():
    return '<span class="wline-bdg">週線</span>'

def sc_cls(s, mx):
    p = s/mx if mx > 0 else 0
    return "c-g" if p>=0.75 else ("c-b" if p>=0.55 else ("c-o" if p>=0.35 else "c-r"))

def grade_cls(g):
    return {"A+":"bg-g","A":"bg-b","A-":"bg-t","B+":"bg-o"}.get(g,"bg-gy")


CSS_MAX_ADDON = """<style>
/* ── MAX 專屬補充 CSS（配合 888 深色主題）── */
/* face-row: P5 面向卡片列 */
.face-row{display:flex;align-items:baseline;gap:6px;margin:4px 0;font-size:11px;padding:2px 0;}
.face-lbl{font-weight:bold;min-width:60px;color:#8b949e;flex-shrink:0;}
.face-sc{font-weight:bold;white-space:nowrap;min-width:46px;color:#f0f6fc;}
.face-det{flex:1;color:#c9d1d9;line-height:1.6;}
/* T1 cards */
.t1card{border-radius:8px;padding:9px 12px;margin:5px 0;background:#161b22;border-left:4px solid #484f58;border:1px solid #30363d;}
.t1card .hd{font-size:12px;font-weight:bold;margin-bottom:4px;color:#f0f6fc;}
.t1card .bd{font-size:10px;color:#adbac7;line-height:1.7;}
/* backup banner */
.bkban{background:#bd5800;color:white;padding:7px 12px;border-radius:6px 6px 0 0;font-size:11px;font-weight:bold;margin-top:10px;}
/* MA13 fresh banner */
.ma13ban{background:linear-gradient(135deg,#6e40c9,#a371f7);color:white;padding:11px 15px;border-radius:9px;margin:10px 0 5px;}
.ma13ban b{font-size:13px;}
.ma13ban small{font-size:10px;opacity:.85;display:block;margin-top:3px;}
/* mini table (cross-system) */
.mini-tbl{margin:6px 0 10px;border-left:3px solid #f0883e;padding-left:8px;background:#161b22;border-radius:0 6px 6px 0;padding:6px 8px;}
.mini-tbl .mini-title{font-size:11px;font-weight:bold;color:#f0883e;margin-bottom:4px;}
/* V7 danger bar */
.danger-bar{background:#da3633;color:white;font-weight:bold;font-size:13px;padding:10px 16px;border-radius:8px;margin:6px 0;border:2px solid #922b21;text-align:center;}
/* wline badge */
.wline-bdg{display:inline-block;background:#bd5800;color:white;padding:1px 5px;border-radius:3px;font-size:9px;font-weight:bold;border:1px solid #f0883e;margin-left:2px;}
/* MAX colour aliases → 888 dark palette */
.c-g{color:#3fb950;font-weight:bold;}
.c-b{color:#58a6ff;font-weight:bold;}
.c-o{color:#f0883e;font-weight:bold;}
.c-r{color:#f85149;font-weight:bold;}
.bg-g{background:#238636;}
.bg-b{background:#1f6feb;}
.bg-o{background:#bd5800;}
.bg-r{background:#da3633;}
.bg-t{background:#1b7c83;}
.bg-dk{background:#21262d;}
.bg-gy{background:#484f58;}
/* P5 card wrapper */
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 15px;margin:6px 0;}
</style>"""

def generate_max_html(df, macro):
    tx_c  = macro.get("taiex_close")
    tx_m  = macro.get("taiex_ma20")
    above = macro.get("taiex_above_ma20", True)
    macro_ok = (above is True)
    if tx_c and tx_m:
        macro_msg = (f"✅ 加權指數 {int(tx_c):,} 點 > MA20 {int(tx_m):,} 點"
                     if macro_ok else
                     f"⚠️ 大盤跌破月線 加權 {int(tx_c):,} < MA20 {int(tx_m):,}")
    else:
        macro_msg = "⚪ 大盤資料未取得"
    fb = macro.get("foreign_b", 0) or 0
    if fb < -500:
        v7_danger = True
        v7_market_msg = f"🛑 外資大幅賣超 {abs(int(fb))} 億 → 今日極度危險"
    elif fb < -300:
        v7_danger = False
        v7_market_msg = f"⚠️ 外資賣超 {abs(int(fb))} 億 → 单檔最夐10%資金"
    elif fb > 100:
        v7_danger = False
        v7_market_msg = f"✅ 大盤外資買超 {int(fb)} 億 → 单檔最夐20%資金"
    else:
        v7_danger = False
        v7_market_msg = f"✅ 大盤外資 {int(fb)} 億 → 中性"
    df_run = df.copy()
    print("  [MAX] 執行 P5/V7/T1 引擎...")
    p5_res            = run_p5(df_run)
    v7_pass, v7_bk    = run_v7_v8(df_run)
    t1_main, t1_fresh = run_t1(df_run)
    p5_top3 = [r["sid"]  for r in p5_res[:3]]
    v7_top3 = [r["代號"] for r in v7_pass[:3]]
    t1_p5_top3, _ = run_t1(df_run, sid_filter=p5_top3, force_show=True)
    t1_v7_top3, _ = run_t1(df_run, sid_filter=v7_top3, force_show=True)
    p5_sids = {r["sid"]  for r in p5_res[:8]}
    v7_sids = {r["代號"] for r in v7_pass[:8]}
    t1_sids = {r["代號"] for r in t1_main[:6]}
    print(f"  [MAX] P5:{len(p5_res)} | V7:{len(v7_pass)}+{len(v7_bk)}備 | T1:{len(t1_main)}剛穿:{len(t1_fresh)}")
    def cross_tag(sid):
        tags = []
        if sid in p5_sids: tags.append("P")
        if sid in v7_sids: tags.append("V")
        if sid in t1_sids: tags.append("T")
        if not tags: return ""
        return '<span class="bdg bg-o" style="font-size:9px;">★' + "+".join(tags) + '</span>'
    def render_mini_t1(t1_list, label=""):
        if not t1_list:
            return '<div style="font-size:10px;color:#999;padding:4px 8px;">（此3檔目前無T1買點資料）</div>'
        cols=[("代號",44),("名稱",54),("T1",32),("收盤",40),("月線%",38),("RSI",30),
              ("V7通過",62),("週線MA13",52),("進場區間",62),("止損",38),("目標一",38)]
        thead="".join('<th style="width:' + str(w) + 'px;">' + h + '</th>'
                      for h,w in cols)
        rows=""
        for i,r in enumerate(t1_list):
            bg = "#1c2333" if i%2==0 else "#161b22"
            gc = grade_cls(r["T1"])
            wc = ("c-r" if "剛穿" in str(r["週線MA13"])
                  else ("c-o" if "週線" in str(r["週線MA13"]) else ""))
            rows+=('<tr style="background:' + bg + ';">'
                   + '<td><b>' + r["代號"] + '</b></td>'
                   + '<td style="text-align:left;">' + r["名稱"] + '</td>'
                   + '<td>' + bdg(r["T1"],gc) + '</td>'
                   + '<td><b>' + str(r["收盤"]) + '</b></td>'
                   + '<td>' + r["月線%"] + '</td>'
                   + '<td>' + r["RSI"] + '</td>'
                   + '<td style="font-size:9px;text-align:left;">' + r["V7通過"] + '</td>'
                   + '<td class="' + wc + '">' + r["週線MA13"] + '</td>'
                   + '<td>' + r["進場區間"] + '</td>'
                   + '<td>' + r["止損"] + '</td>'
                   + '<td>' + r["目標一"] + '</td>'
                   + '</tr>')
        return ('<div class="mini-tbl"><div class="mini-title">📍 '
                + label + ' 前三名 × T1 買點確認</div>'
                + '<div style="overflow-x:auto;">'
                + '<table class="tbl"><thead><tr>' + thead + '</tr></thead>'
                + '<tbody>' + rows + '</tbody></table></div></div>')


    # ══════════════════════════════════════════════════════════════════


    def render_p5(res):
        if not res: return '<p style="color:#888;">P5 無符合條件</p>'

        cols=[("代號",46),("名稱",58),("收盤",44),("總分/130",48),("面向",32),
              ("EPS年增率",46),("淨值報酬率",46),("本益比",36),("PEG",38),("負債比",38),
              ("入場建議",68),("月線止損",48),("季線止損",48),("目標區間",62),("操作/T1",60)]
        thead="".join('<th style="width:' + str(w) + 'px;">' + h + '</th>'
                      for h,w in cols)

        rows=""; sup_inserted=False
        for i,r in enumerate(res):
            bg     = "#1c2333" if i%2==0 else "#161b22"
            tot    = r["total"]; is_sup = r.get("_sup",False)
            tc     = ("c-g" if tot>=110 else "c-b" if tot>=90 else "c-o" if tot>=70 else "c-r")
            fp_n   = r["faces"]
            fp_c   = ("bg-g" if fp_n>=5 else ("bg-b" if fp_n>=4 else "bg-o"))
            wma_tag= (wline_bdg() if (r["wfp"] or r["wmsig"]) else "")
            t1c    = grade_cls(r["t1g"]); op_s = r["op"]

            if is_sup and not sup_inserted:
                rows+=('<tr><td colspan="15" style="background:#1f1d0f;font-size:10px;color:#f0f6fc;border-left:3px solid #f0883e;'
                       + 'padding:4px 8px;text-align:left;">ℹ️ 以下為重點追蹤股（總分未達'
                       + str(P5_PASS) + '分）</td></tr>')
                sup_inserted=True

            rows+=('<tr style="background:' + bg + (';opacity:.85;' if is_sup else '') + ';">'
                   + '<td><b>' + r["sid"] + '</b></td>'
                   + '<td style="text-align:left;">' + r["name"] + wma_tag + '</td>'
                   + '<td><b>' + str(round(r["close"],0)) + '</b></td>'
                   + '<td class="' + tc + '" style="font-weight:bold;font-size:12px;">' + str(tot) + '</td>'
                   + '<td>' + bdg(str(fp_n)+"/6",fp_c) + '</td>'
                   + '<td>' + r["ey1_s"] + '</td>'
                   + '<td>' + r["roe_s"] + '</td>'
                   + '<td>' + r["pe_s"]  + '</td>'
                   + '<td>' + r["peg_s"] + '</td>'
                   + '<td>' + r["dr_s"]  + '</td>'
                   + '<td style="text-align:left;font-size:9px;">' + r["entry"] + '</td>'
                   + '<td>' + r["sl1"]   + '</td>'
                   + '<td>' + r["sl2"]   + '</td>'
                   + '<td style="font-size:9px;">' + r["tgt"] + '</td>'
                   + '<td>' + bdg(op_s,"bg-dk") + ' ' + bdg(r["t1g"],t1c) + '</td>'
                   + '</tr>')

        mc_c = ("" if macro_ok else 'style="color:#c0392b;font-weight:bold;"')
        mc_s = '宏觀：' + ("✅通過" if macro_ok else "⚠️偏弱（僅供參考）")
        sup_cnt = sum(1 for r in res if r.get("_sup"))

        tbl=('<div style="overflow-x:auto;">'
             + '<div style="font-size:10px;color:#8b949e;margin-bottom:3px;">'
             + '<span ' + mc_c + '>' + mc_s + '</span>'
             + '　依總分排序　' + wline_bdg() + '=週線MA13多頭</div>'
             + '<table class="tbl"><thead><tr>' + thead + '</tr></thead>'
             + '<tbody>' + rows + '</tbody></table>'
             + (('<div style="font-size:10px;color:#e67e22;margin-top:3px;">ℹ️ 共'
                 + str(len(res)-sup_cnt) + '檔達P5合格標準，其餘'
                 + str(sup_cnt) + '檔為重點追蹤股</div>') if sup_cnt else "")
             + '</div>')

        # ── 評分細項卡（版面改版）─────────────────────────────────
        # 面向順序：名稱 → 分數（緊接）→ 詳細說明
        face_cfg = [("基本面",30,"d1","s1"),("估值面",20,"d2","s2"),
                    ("財務健康",20,"d3","s3"),("籌碼面",25,"d4","s4"),
                    ("技術面",20,"d5","s5"),("風險面",15,"d6","s6")]
        cards=""
        for r in res[:8]:
            tot = r["total"]
            tc  = ("c-g" if tot>=110 else "c-b" if tot>=90 else "c-o")
            wma_b = ((' ' + bdg("🔥2日剛穿","bg-r")) if r["wfp"]
                     else ((' ' + wline_bdg()) if r["wmsig"] else ""))
            # 入場欄加市價
            entry_info = ('市價：<b>' + str(round(r["close"],0)) + '</b>'
                          + ' ｜ 入場：' + r["entry"]
                          + ' ｜ 止損：' + r["sl1"] + '（月）'
                          + r["sl2"] + '（季）')
            face_rows=""
            for fname,fmax,dk,sk in face_cfg:
                s   = r[sk]; det = r[dk]; sc = sc_cls(s,fmax)
                det_s = "　".join(det) if det else "資料不足"
                # 順序：面向名稱 | 分數 | 詳細說明
                face_rows+=('<div class="face-row">'
                           + '<span class="face-lbl">' + fname + '</span>'
                           + '<span class="face-sc ' + sc + '">' + str(s) + '/' + str(fmax) + '</span>'
                           + '<span class="face-det">' + det_s + '</span>'
                           + '</div>')
            cards+=('<div class="card">'
                   + '<div style="font-size:12px;font-weight:bold;margin-bottom:4px;'
                   + 'border-bottom:1px solid #30363d;padding-bottom:4px;">'
                   + r["sid"] + ' ' + r["name"]
                   + ' <span class="' + tc + '" style="font-size:14px;">' + str(tot) + '/130</span>'
                   + wma_b + '</div>'
                   # 入場止損一行
                   + '<div style="font-size:10px;color:#adbac7;margin-bottom:5px;">'
                   + entry_info + '</div>'
                   + face_rows + '</div>')

        mini = render_mini_t1(t1_p5_top3, "P5長線")
        return tbl + '<div style="margin-top:8px;">' + cards + '</div>' + mini


    # ══════════════════════════════════════════════════════════════════


    def render_v7(passed, backup):
        out=""

        if v7_danger:
            out+=('<div class="danger-bar">🛑 ' + v7_market_msg
                  + ' ｜ 以下選股結果僅供觀察，今日極度不建議實際進場做多！</div>')

        cols=[("標記",50),("代號",44),("名稱",54),("收盤",42),
              ("總分\n/100",36),("分級",44),
              ("G1\n/12",24),("G2\n/30",24),("G3\n/25",24),
              ("G4\n/15",24),("G5\n/10",24),("G6\n/16",24),
              ("通過\n關",30),("催化",36),("動能\n分",30),
              ("族群",30),("chip",44),("RSI",28),("週線\nMA13",46),
              ("進場",56),("止損",38),("目標一",38),("目標二",38)]
        thead="".join('<th style="width:' + str(w) + 'px;">' + h + '</th>'
                      for h,w in cols)

        if passed:
            rows=""
            for i,r in enumerate(passed):
                bg   = "#1c2333" if i%2==0 else "#161b22"
                mk   = r.get("標記","")
                mk_c = "bg-r" if ("最高" in mk or "極強" in mk) else "bg-b"
                tot  = r["總分"]
                tc   = ("c-g" if tot>=93 else "c-b" if tot>=86 else "c-o")
                wc   = ("c-r" if "剛穿" in str(r["週線MA13"])
                        else ("c-g" if "多頭" in str(r["週線MA13"]) else ""))
                cs_v = str(r["chip_sync"])
                cs_c = ("c-g" if cs_v=="雙買" else ("c-b" if "單買" in cs_v else ""))
                cat_v= str(r["催化"])
                cat_c= ("c-g" if cat_v=="強" else ("c-b" if cat_v=="中" else "c-o"))
                rows+=('<tr style="background:' + bg + ';">'
                       + '<td>' + bdg(mk,mk_c) + '</td>'
                       + '<td><b>' + r["代號"] + '</b></td>'
                       + '<td style="text-align:left;">' + r["名稱"] + '</td>'
                       + '<td><b>' + str(r["收盤"]) + '</b></td>'
                       + '<td class="' + tc + '" style="font-weight:bold;font-size:12px;">' + str(tot) + '</td>'
                       + '<td>' + str(r["分級"]) + '</td>'
                       + '<td>' + str(r["G1"]) + '</td>'
                       + '<td>' + str(r["G2"]) + '</td>'
                       + '<td>' + str(r["G3"]) + '</td>'
                       + '<td>' + str(r["G4"]) + '</td>'
                       + '<td>' + str(r["G5"]) + '</td>'
                       + '<td>' + str(r["G6"]) + '</td>'
                       + '<td>' + str(r["通過關"]) + '</td>'
                       + '<td class="' + cat_c + '">' + cat_v + '</td>'
                       + '<td>' + str(r["動能分"]) + '</td>'
                       + '<td>' + str(r["族群"]) + '</td>'
                       + '<td class="' + cs_c + '">' + cs_v + '</td>'
                       + '<td>' + str(r["RSI"]) + '</td>'
                       + '<td class="' + wc + '" style="font-weight:bold;">' + str(r["週線MA13"]) + '</td>'
                       + '<td>' + str(r["進場"]) + '</td>'
                       + '<td>' + str(r["止損"]) + '</td>'
                       + '<td>' + str(r["目標一"]) + '</td>'
                       + '<td>' + str(r["目標二"]) + '</td>'
                       + '</tr>')
            out+=('<div style="overflow-x:auto;">'
                  + '<table class="tbl"><thead><tr>' + thead + '</tr></thead>'
                  + '<tbody>' + rows + '</tbody></table></div>')

            # 核心理由行
            r_rows=""
            for i,r in enumerate(passed):
                bg = "#1c2333" if i%2==0 else "#161b22"
                r_rows+=('<tr style="background:' + bg + ';">'
                         + '<td style="width:44px;font-weight:bold;">' + r["代號"] + '</td>'
                         + '<td style="width:54px;">' + r["名稱"] + '</td>'
                         + '<td style="text-align:left;font-size:10px;color:#c9d1d9;white-space:normal;">' + r["核心理由"] + '</td>'
                         + '<td style="width:200px;font-size:10px;color:#c0392b;white-space:normal;">' + r["失敗條件"] + '</td>'
                         + '</tr>')
            out+=('<div style="overflow-x:auto;margin-top:4px;">'
                  + '<table class="tbl" style="table-layout:auto;">'
                  + '<thead><tr><th style="width:44px;">代號</th><th style="width:54px;">名稱</th>'
                  + '<th>核心理由</th><th style="width:200px;">失敗條件</th></tr></thead>'
                  + '<tbody>' + r_rows + '</tbody></table></div>')
        else:
            out+=('<div style="background:#2c3e50;color:#ecf0f1;padding:10px;'
                  + 'border-radius:8px;">⚡ V7 今日無達86分+4關個股</div>')

        if backup:
            bk_cols=[("代號",44),("名稱",54),("總分",34),("通過關",34),
                     ("G2催化",36),("chip",44),("量比",36),("RSI",28),
                     ("外資連買",44),("族群",30),("未通關",56)]
            bt="".join('<th style="width:' + str(w) + 'px;">' + h + '</th>'
                       for h,w in bk_cols)
            br=""
            for r in backup:
                br+=('<tr style="background:#1e1810;">'
                     + ''.join('<td>' + str(r.get(col,"—")) + '</td>'
                               for col in ["代號","名稱","總分","通過關","催化","chip_sync",
                                           "量比","RSI","外資連買","族群","未通關"])
                     + '</tr>')
            out+=('<div class="bkban" style="margin-top:6px;">⚠️ V7備案（78~85分+T1 A+，需謹慎）</div>'
                  + '<div style="overflow-x:auto;">'
                  + '<table class="tbl"><thead><tr>' + bt + '</tr></thead>'
                  + '<tbody>' + br + '</tbody></table></div>')
            for r in backup:
                out+=('<div style="font-size:10px;padding:3px 8px;'
                      + 'border-left:3px solid #e67e22;margin:2px 0;background:#1e1810;">'
                      + '<b>' + r["代號"] + ' ' + r["名稱"] + '</b>｜' + r["核心理由"] + '</div>')
        elif not passed:
            out+=('<div style="background:#1f1d0f;padding:8px;color:#f0883e;border:1px solid #f0883e;border-radius:6px;'
                  + 'border-radius:6px;font-size:10px;margin-top:4px;">ℹ️ 今日無V7備案股票。</div>')

        out += render_mini_t1(t1_v7_top3, "V7短線")
        return out


    # ══════════════════════════════════════════════════════════════════


    def render_t1(main_l, fresh_l):
        out=""
        cols=[("代號",44),("名稱",54),("T1",32),("收盤",42),
              ("月線%",36),("RSI",28),("量比",34),("V7通過",68),
              ("chip",42),("news",26),("外資連買",44),("週線MA13",52),
              ("進場區間",58),("止損",38),("目標一",38),("目標二",38)]
        thead="".join('<th style="width:' + str(w) + 'px;">' + h + '</th>'
                      for h,w in cols)

        def make_rows(lst):
            h=""
            for i,r in enumerate(lst):
                bg = "#1c2333" if i%2==0 else "#161b22"
                gc = grade_cls(r["T1"])
                wc = ("c-r" if "剛穿" in str(r["週線MA13"])
                      else ("c-o" if "週線" in str(r["週線MA13"]) else ""))
                cs_v = str(r["chip"]); cs_c = ("c-g" if cs_v=="雙買" else ("c-b" if "單買" in cs_v else ""))
                h+=('<tr style="background:' + bg + ';">'
                    + '<td><b>' + r["代號"] + '</b></td>'
                    + '<td style="text-align:left;">' + r["名稱"] + '</td>'
                    + '<td>' + bdg(r["T1"],gc) + '</td>'
                    + '<td><b>' + str(r["收盤"]) + '</b></td>'
                    + '<td>' + r["月線%"] + '</td>'
                    + '<td>' + r["RSI"] + '</td>'
                    + '<td>' + r["量比"] + '</td>'
                    + '<td style="text-align:left;font-size:9px;">' + r["V7通過"] + '</td>'
                    + '<td class="' + cs_c + '">' + cs_v + '</td>'
                    + '<td>' + str(r["news"]) + '</td>'
                    + '<td>' + r["外資連買"] + '</td>'
                    + '<td class="' + wc + '" style="font-weight:bold;">' + r["週線MA13"] + '</td>'
                    + '<td>' + r["進場區間"] + '</td>'
                    + '<td>' + r["止損"] + '</td>'
                    + '<td>' + r["目標一"] + '</td>'
                    + '<td>' + r["目標二"] + '</td>'
                    + '</tr>')
            return h

        if main_l:
            out+=('<div style="overflow-x:auto;">'
                  + '<table class="tbl"><thead><tr>' + thead + '</tr></thead>'
                  + '<tbody>' + make_rows(main_l) + '</tbody></table></div>')
            cards=""
            for r in main_l:
                gc = grade_cls(r["T1"])
                wt = (" " + bdg("🔥剛穿","bg-r") if "剛穿" in str(r["週線MA13"])
                      else (" " + bdg("⭐週線","bg-o") if "週線" in str(r["週線MA13"]) else ""))
                lc = {"A+":"27ae60","A":"2980b9","A-":"16a085","B+":"e67e22"}.get(r["T1"],"aaa")
                cards+=('<div class="t1card" style="border-left-color:#' + lc + ';">'
                        + '<div class="hd">' + r["代號"] + ' ' + r["名稱"] + ' '
                        + bdg(r["T1"],gc) + wt + '</div>'
                        + '<div class="bd">' + r["結論"] + '</div></div>')
            out+='<div style="margin-top:5px;">' + cards + '</div>'
        else:
            out+='<div style="padding:12px;background:#21262d;border-radius:8px;font-size:11px;color:#c9d1d9;">T1 今日無A+~B+買點</div>'

        if fresh_l:
            def make_fresh_rows(lst):
                h=""
                for i,r in enumerate(lst):
                    bg = "#1c2333" if i%2==0 else "#161b22"
                    gc = grade_cls(r["T1"]); sid=r["代號"]; ctag=cross_tag(sid)
                    h+=('<tr style="background:' + bg + ';">'
                        + '<td><b>' + sid + '</b></td>'
                        + '<td style="text-align:left;">' + r["名稱"] + ctag + '</td>'
                        + '<td>' + bdg(r["T1"],gc) + '</td>'
                        + '<td><b>' + str(r["收盤"]) + '</b></td>'
                        + '<td>' + r["月線%"] + '</td>'
                        + '<td>' + r["RSI"] + '</td>'
                        + '<td>' + r["量比"] + '</td>'
                        + '<td style="text-align:left;font-size:9px;">' + r["V7通過"] + '</td>'
                        + '<td>' + str(r["chip"]) + '</td>'
                        + '<td>' + str(r["news"]) + '</td>'
                        + '<td>' + r["外資連買"] + '</td>'
                        + '<td class="c-r" style="font-weight:bold;">' + r["週線MA13"] + '</td>'
                        + '<td>' + r["進場區間"] + '</td>'
                        + '<td>' + r["止損"] + '</td>'
                        + '<td>' + r["目標一"] + '</td>'
                        + '<td>' + r["目標二"] + '</td>'
                        + '</tr>')
                return h

            out+=('<div class="ma13ban">'
                  + '<b>🔥 週線MA13特別推薦（2日內剛穿 · 最高勝率）</b>'
                  + '<small>條件：週線MA13向上+股價剛站上+日線MA5/MA10剛穿MA20（2日內）共'
                  + str(len(fresh_l)) + '檔</small>'
                  + '<small>止損跌破MA20即出｜保守持10~25天｜積極30~50天</small>'
                  + '<small>★P=P5名單 ★V=V7名單 ★T=T1主名單（三系統共振則特別關注）</small>'
                  + '</div>')
            out+=('<div style="overflow-x:auto;">'
                  + '<table class="tbl"><thead><tr>' + thead + '</tr></thead>'
                  + '<tbody>' + make_fresh_rows(fresh_l) + '</tbody></table></div>')

            fr_cards=""
            for r in fresh_l:
                gc=grade_cls(r["T1"]); sid=r["代號"]; ctag=cross_tag(sid)
                fr_cards+=('<div class="t1card" style="border-left-color:#c0392b;">'
                           + '<div class="hd">' + sid + ' ' + r["名稱"] + ctag + ' '
                           + bdg(r["T1"],gc) + ' ' + bdg("🔥2日剛穿","bg-r") + '</div>'
                           + '<div class="bd">' + r["結論"] + '</div></div>')
            out+='<div style="margin-top:4px;">' + fr_cards + '</div>'
        else:
            out+=('<div style="background:#21262d;padding:6px 10px;border-radius:6px;'
                  + 'margin-top:6px;font-size:10px;color:#8b949e;">'
                  + 'ℹ️ 週線MA13剛穿（2日內）：今日無符合條件。</div>')
        return out


    # ══════════════════════════════════════════════════════════════════


    mc_c = "" if macro_ok else "color:#e74c3c;"
    p5_html_inner = render_p5(p5_res)
    v7_html_inner = render_v7(v7_pass, v7_bk)
    t1_html_inner = render_t1(t1_main, t1_fresh)
    sc_row = "".join(
        f'<div class="scard" style="background:{c};"><div class="n">{n}</div>'
        f'<div class="lb">{lb}</div></div>'
        for c, n, lb in [
            ("#c0392b", len(p5_res),   "P5長線"),
            ("#2980b9", len(v7_pass),  "V7鐵三角"),
            ("#e67e22", len(v7_bk),    "V7備案"),
            ("#27ae60", len(t1_main),  "T1買點"),
            ("#922b21", len(t1_fresh), "🔥週線剛穿"),
        ]
    )
    vd_style = "color:#c0392b;font-weight:bold;" if v7_danger else "color:#f39c12;"
    return (
        CSS + CSS_MAX_ADDON
        + '<div class="rep">'
        + '<div class="hdr"><div class="hdr-inner"><div class="hdr-icon">🏆</div>'
        + '<div class="hdr-text"><h1>台股三系統選股報告<span class="ver">MAX</span></h1>'
        + '<div class="sub">P5長線 · V7 v8.0鐵三角極強 · T1 v8.2 買點系統</div>'
        + f'<div class="sub" style="{mc_c}">■ 宏觀：{macro_msg}</div>'
        + f'<div class="sub" style="{vd_style}">■ V7大盤：{v7_market_msg}</div>'
        + '</div></div></div>'
        + f'<div class="sumrow">{sc_row}</div>'
        + '<div class="sec">📊 P5 長線合格清單</div>'
        + p5_html_inner
        + '<div class="sec sec-blue">⚡ V7 v8.0 鐵三角極強（最夐66檔）</div>'
        + v7_html_inner
        + '<div class="sec sec-green">🎯 T1 v8.2 買點系統（超嚴格版）</div>'
        + t1_html_inner
        + '<div class="footer">台股三系統 v8.0 MAX ｜ 僅供參考，不構成投資建議</div>'
        + '</div>'
    )


# ── PART 2：HTML 報告
# ============================================================

CSS = """<style>
*{box-sizing:border-box;}
.rep{font-family:"Microsoft JhengHei","Noto Sans TC","Helvetica Neue",Arial,sans-serif;
     font-size:12px;background:#0d1117;padding:14px;color:#c9d1d9;line-height:1.6;}
.hdr{background:linear-gradient(135deg,#0d1117 0%,#161b22 60%,#1c2333 100%);
     border:1px solid #30363d;padding:22px 26px;border-radius:14px;
     margin-bottom:14px;position:relative;overflow:hidden;}
.hdr::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
             background:linear-gradient(90deg,#ffd700,#f0883e,#f85149,#3fb950,#58a6ff,#ffd700);}
.hdr-inner{display:flex;align-items:center;gap:16px;}
.hdr-icon{font-size:48px;line-height:1;}
.hdr-text h1{margin:0;font-size:24px;color:#ffd700;letter-spacing:2px;}
.hdr-text .sub{font-size:11px;color:#8b949e;margin-top:5px;}
.hdr-text .ver{display:inline-block;padding:1px 8px;border-radius:4px;
               background:#ffd700;color:#000;font-size:10px;font-weight:bold;margin-left:8px;}
.sec{font-size:14px;font-weight:bold;padding:8px 12px;margin:16px 0 8px;
     border-radius:8px;color:#f0f6fc;
     background:linear-gradient(90deg,#161b22,#21262d);border-left:4px solid #ffd700;}
.sec-fire{border-left-color:#f85149;background:linear-gradient(90deg,#1c1017,#21262d);}
.sec-blue{border-left-color:#58a6ff;}
.sec-green{border-left-color:#3fb950;}
.sec-orange{border-left-color:#f0883e;}
.sec-purple{border-left-color:#a371f7;background:linear-gradient(90deg,#16101c,#21262d);}
.sec-down{border-left-color:#58a6ff;background:linear-gradient(90deg,#0a1020,#21262d);}
.grade-aaa{background:linear-gradient(135deg,#b08800,#ffd700);color:#000;font-weight:bold;}
.grade-aa{background:linear-gradient(135deg,#1f6feb,#58a6ff);color:#fff;font-weight:bold;}
.grade-a{background:#21262d;color:#8b949e;border:1px solid #30363d;}
.tbl{width:100%;border-collapse:collapse;background:#161b22;
     border-radius:10px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.4);border:1px solid #30363d;}
.tbl th{background:#21262d;color:#ffd700;font-size:10px;
        padding:8px 4px;text-align:center;white-space:nowrap;
        border-bottom:2px solid #30363d;font-weight:600;}
.tbl td{font-size:11px;padding:6px 4px;border-bottom:1px solid #21262d;text-align:center;color:#c9d1d9;}
.tbl tr:nth-child(even) td{background:#0d1117;}
.tbl tr:hover td{background:#1c2333;}
.bdg{display:inline-block;padding:2px 7px;border-radius:6px;font-size:10px;font-weight:bold;color:white;white-space:nowrap;}
.bg-fire{background:linear-gradient(135deg,#da3633,#f85149);}
.bg-gold{background:linear-gradient(135deg,#b08800,#ffd700);color:#000;}
.bg-green{background:#238636;}.bg-blue{background:#1f6feb;}
.bg-teal{background:#1b7c83;}.bg-orange{background:#bd5800;}.bg-gray{background:#484f58;}
.bg-weekly{background:linear-gradient(135deg,#bd5800,#f0883e);}
.bg-major{background:linear-gradient(135deg,#6e40c9,#a371f7);}
.c-fire{color:#f85149;font-weight:bold;}.c-gold{color:#ffd700;font-weight:bold;}
.c-green{color:#3fb950;font-weight:bold;}.c-blue{color:#58a6ff;font-weight:bold;}
.c-orange{color:#f0883e;font-weight:bold;}.c-gray{color:#8b949e;}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px;margin:6px 0;}
.card-fire{border-left:4px solid #f85149;}.card-gold{border-left:4px solid #ffd700;}
.card-green{border-left:4px solid #3fb950;}.card-purple{border-left:4px solid #a371f7;}
.sumrow{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0;}
.scard{flex:1;min-width:90px;padding:10px;border-radius:10px;text-align:center;border:1px solid #30363d;}
.scard .n{font-size:26px;font-weight:bold;color:#f0f6fc;}
.scard .lb{font-size:10px;color:#8b949e;margin-top:2px;}
.macro-box{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;margin:8px 0;}
.macro-row{display:flex;align-items:center;gap:8px;padding:5px 0;font-size:12px;border-bottom:1px solid #21262d;}
.macro-row:last-child{border-bottom:none;}
.macro-label{font-weight:bold;min-width:100px;color:#8b949e;}
.risk-box{background:linear-gradient(135deg,#1c1017,#161b22);
          border:1px solid #f85149;border-radius:10px;padding:14px 18px;margin:8px 0;}
.risk-item{padding:4px 0;font-size:11px;color:#f0883e;}
.empty-msg{background:#21262d;border:1px solid #30363d;border-radius:8px;padding:14px;text-align:center;color:#8b949e;}
.z1-row{display:flex;gap:6px;align-items:center;padding:3px 0;font-size:11px;}
.src-tag{font-size:9px;padding:1px 5px;border-radius:4px;background:#21262d;color:#8b949e;margin-left:6px;border:1px solid #30363d;}
.footer{text-align:center;color:#484f58;font-size:9px;margin-top:16px;padding-top:8px;border-top:1px solid #21262d;}
</style>"""


def generate_html(t1_top_final, L1_TOP20, L1_full_pool, S1_TOP12, S1_full_pool,
                  t1a_results, t1b_results, z1_results, risk_rules,
                  N_STOCKS, macro, z1_kline_cache, data_today,
                  downtrend_hits=None, df_for_max=None):

    def bdg(txt, cls='bg-gray'):
        return f'<span class="bdg {cls}">{txt}</span>'

    def grade_cls(g):
        return {'A+':'bg-fire','A':'bg-green','A-':'bg-teal','B+':'bg-orange'}.get(str(g),'bg-gray')

    def wk_badge(flag):
        return bdg('週線','bg-weekly') if flag else ''

    def render_t1_top(lst):
        if not lst: return '<div class="empty-msg">🔥 T1-TOP 今日無符合條件</div>'
        cols = [('代號',48),('名稱',65),('T1',40),('週線',36),('收盤',50),
                ('月線%',44),('量比',40),('RSI',36),('籌碼',48),('穿越訊號',82),('止損',58),('目標',58)]
        thead = ''.join(f'<th style="width:{w}px">{h}</th>' for h,w in cols)
        rows  = ''
        for r in lst:
            rows += (f'<tr><td><b>{r["sid"]}</b></td>'
                     f'<td style="text-align:left">{r["name"]}</td>'
                     f'<td>{bdg(r["t1_grade"],grade_cls(r["t1_grade"]))}</td>'
                     f'<td>{wk_badge(r.get("weekly_ma13"))}</td>'
                     f'<td class="c-gold"><b>{r["close"]:.0f}</b></td>'
                     f'<td>{r["pct20"]:.1f}%</td><td>{r["vr"]:.2f}x</td>'
                     f'<td>{r["rsi"]:.0f}</td><td>{r["chip"]}</td>'
                     f'<td style="font-size:9px;text-align:left">{r["cross_note"]}</td>'
                     f'<td class="c-fire">≈{r["sl"]:.0f}</td>'
                     f'<td class="c-green">≈{r["tgt"]:.0f}</td></tr>')
        return f'<div style="overflow-x:auto"><table class="tbl"><thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table></div>'

    def render_macro():
        tx_c = macro.get('taiex_close'); tx_m = macro.get('taiex_ma20')
        ab   = macro.get('taiex_above_ma20'); src = macro.get('taiex_source','未取得')
        tx_s = f'{tx_c:,.2f}' if tx_c else '—'
        tm_s = f'{tx_m:,.2f}' if tx_m else '—'
        ab_s = ('站穩月線 ✅' if ab else '⚠️ 跌破月線') if ab is not None else '—'
        pos  = '偏多' if ab else ('偏空' if ab is not None else '—')
        fb   = macro.get('foreign_b', 0)
        fl   = macro.get('foreign_label', '')
        fc   = 'c-green' if fb>100 else ('c-blue' if fb>=-100 else ('c-orange' if fb>=-300 else 'c-fire'))
        adv  = macro.get('op_advice','')
        adv_c = 'c-green' if '積極' in adv else 'c-orange'
        return (f'<div class="macro-box">'
                f'<div class="macro-row"><span class="macro-label">加權指數</span>'
                f'<span>{tx_s}（{pos}）月線:{tm_s} {ab_s}<span class="src-tag">{src}</span></span></div>'
                f'<div class="macro-row"><span class="macro-label">大盤外資</span>'
                f'<span class="{fc}">{fl}<span class="src-tag">{N_STOCKS}檔樣本估算</span></span></div>'
                f'<div class="macro-row"><span class="macro-label">操作建議</span>'
                f'<span class="{adv_c}" style="font-size:13px;font-weight:bold">{adv}</span></div>'
                f'</div>')

    def render_l1(lst):
        if not lst: return '<div class="empty-msg">L1 今日無通過守門條件的股票</div>'
        cols = [('代號',46),('名稱',88),('收盤',50),('總分/100',52),
                ('EPS年增',52),('ROE',42),('PE',40),('PEG',40),
                ('負債比',44),('入場建議',84),('月線止損',58),('季線止損',58),('目標區間',72),('操作',68)]
        thead = ''.join(f'<th style="width:{w}px">{h}</th>' for h,w in cols)
        rows  = ''
        for r in lst:
            tc  = 'c-gold' if r['total']>=80 else ('c-green' if r['total']>=60 else 'c-orange')
            oc  = 'bg-fire' if '重點' in r['op_label'] else 'bg-green'
            rows += (f'<tr><td><b>{r["sid"]}</b></td>'
                     f'<td style="text-align:left;white-space:nowrap">{r["name"]} {wk_badge(r.get("weekly_ma13"))}</td>'
                     f'<td class="c-gold"><b>{r["close"]:.0f}</b></td>'
                     f'<td class="{tc}" style="font-weight:bold;font-size:13px">{r["total"]}</td>'
                     f'<td>{r["eps_yoy"]}</td><td>{r["roe"]}</td>'
                     f'<td>{r["pe"]}</td><td>{r["peg"]}</td><td>{r["debt_ratio"]}</td>'
                     f'<td style="text-align:left;font-size:10px">{r["entry"]}</td>'
                     f'<td class="c-fire" style="font-size:10px">{r["sl_ma20"]}</td>'
                     f'<td class="c-orange" style="font-size:10px">{r["sl_ma60"]}</td>'
                     f'<td style="font-size:10px">{r["tgt"]}</td>'
                     f'<td>{bdg(r["op_label"],oc)}</td></tr>')
        return f'<div style="overflow-x:auto"><table class="tbl"><thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table></div>'

    def render_s1(lst):
        if not lst: return '<div class="empty-msg">S1 今日無主力/強勢股</div>'
        cols = [('代號',48),('名稱',72),('標記',62),('加強條件',36),('加權分',46),
                ('收盤',50),('量比',40),('RSI',36),('5日漲%',44),('籌碼',48),('止損',58),('目標',58)]
        thead = ''.join(f'<th style="width:{w}px">{h}</th>' for h,w in cols)
        rows  = ''
        for r in lst:
            lc  = 'bg-fire' if '主力' in r['label'] else 'bg-green'
            rows += (f'<tr><td><b>{r["sid"]}</b></td>'
                     f'<td style="text-align:left">{r["name"]} {wk_badge(r.get("weekly_ma13"))}</td>'
                     f'<td>{bdg(r["label"],lc)}</td>'
                     f'<td class="c-gold">{r["cnt"]}/5</td>'
                     f'<td class="c-blue">{r["weighted_score"]:.0f}</td>'
                     f'<td class="c-gold"><b>{r["close"]:.0f}</b></td>'
                     f'<td>{r["vr"]:.2f}x</td><td>{r["rsi"]:.0f}</td>'
                     f'<td>{r["r5"]:.1f}%</td><td>{r["chip"]}</td>'
                     f'<td class="c-fire">≈{r["sl"]:.0f}</td>'
                     f'<td class="c-green">≈{r["tgt"]:.0f}</td></tr>')
        
        return f'<div style="overflow-x:auto"><table class="tbl"><thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table></div>'

    def render_t1a(lst):
        if not lst: return '<div class="empty-msg">T1-A 今日無符合條件</div>'
        cols = [('代號',48),('名稱',65),('T1',40),('週線',36),('收盤',50),
                ('月線%',44),('量比',40),('RSI',36),('籌碼',48),('穿越訊號',82),('止損',58),('目標',58)]
        thead = ''.join(f'<th style="width:{w}px">{h}</th>' for h,w in cols)
        rows  = ''
        for r in lst:
            rows += (f'<tr><td><b>{r["sid"]}</b></td>'
                     f'<td style="text-align:left">{r["name"]}</td>'
                     f'<td>{bdg(r["t1_grade"],grade_cls(r["t1_grade"]))}</td>'
                     f'<td>{wk_badge(r.get("weekly_ma13"))}</td>'
                     f'<td class="c-gold"><b>{r["close"]:.0f}</b></td>'
                     f'<td>{r["pct20"]:.1f}%</td><td>{r["vr"]:.2f}x</td>'
                     f'<td>{r["rsi"]:.0f}</td><td>{r["chip"]}</td>'
                     f'<td style="font-size:9px;text-align:left">{r["cross_note"]}</td>'
                     f'<td class="c-fire">≈{r["sl"]:.0f}</td>'
                     f'<td class="c-green">≈{r["tgt"]:.0f}</td></tr>')
        return f'<div style="overflow-x:auto"><table class="tbl"><thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table></div>'

    def render_t1b(lst):
        if not lst: return '<div class="empty-msg">T1-B 今日無拉回進場機會</div>'
        cols = [('代號',48),('名稱',65),('T1-B',50),('條件',40),
                ('收盤',50),('月線%',44),('量比',40),('RSI',36),('止損',58),('目標',58)]
        thead = ''.join(f'<th style="width:{w}px">{h}</th>' for h,w in cols)
        rows  = ''
        for r in lst:
            gc  = 'bg-green' if r['grade']=='A' else 'bg-teal'
            rows += (f'<tr><td><b>{r["sid"]}</b></td>'
                     f'<td style="text-align:left">{r["name"]}</td>'
                     f'<td>{bdg(r["grade"],gc)}</td>'
                     f'<td class="c-gold">{r["total_conds"]}/5</td>'
                     f'<td class="c-gold"><b>{r["close"]:.0f}</b></td>'
                     f'<td>{r["pct20"]:.1f}%</td><td>{r["vr"]:.2f}x</td>'
                     f'<td>{r["rsi"]:.0f}</td>'
                     f'<td class="c-fire">≈{r["sl"]:.0f}</td>'
                     f'<td class="c-green">≈{r["tgt"]:.0f}</td></tr>')
        for r in lst:
            rows += (f'<tr><td colspan="10" style="text-align:left;font-size:9px;'
                     f'color:#8b949e;padding:2px 8px;background:#0d1117">'
                     f'↳ {r["sid"]} {r["name"]}：{" ✦ ".join(r["conds_met"])}</td></tr>')
        return f'<div style="overflow-x:auto"><table class="tbl"><thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table></div>'

    def render_z1(lst):
        if not lst: return '<div class="empty-msg">Z1 無待檢視持股</div>'
        cards = ''
        for r in lst:
            vc     = ('card-fire' if '出場' in r['verdict'] else ('card-gold' if '減碼' in r['verdict'] else 'card-green'))
            cat    = r.get('category', 'L1長線')
            cat_cls = 'bg-fire' if 'T1-TOP' in cat else ('bg-gold' if 'T1-A' in cat else ('bg-major' if 'S1' in cat else 'bg-blue'))
            checks_s = ''.join(f'<div class="z1-row">{c[0]} {c[1]}</div>' for c in r['checks'])
            b64    = z1_kline_cache.get(r['sid'])
            kline_html = (f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;border:1px solid #30363d;display:block">'
                          if b64 else '<div style="background:#21262d;border:1px solid #30363d;border-radius:8px;height:120px;display:flex;align-items:center;justify-content:center;color:#8b949e;font-size:11px">⚠️ K線載入失敗</div>')
            cards += (f'<div class="card {vc}" style="display:flex;gap:14px;align-items:flex-start;margin-bottom:10px">'
                      f'<div style="min-width:180px;max-width:210px;flex-shrink:0">'
                      f'<div style="font-weight:bold;margin-bottom:6px">'
                      f'<span class="bdg {cat_cls}">{cat}</span> <span style="color:#f0f6fc">{r["sid"]} {r["name"]}</span> '
                      f'<span style="color:#8b949e;font-size:11px">({r["close"]:.0f})</span></div>'
                      f'<div style="font-size:13px;margin-bottom:6px">{r["verdict"]}</div>'
                      f'{checks_s}</div>'
                      f'<div style="flex:1;min-width:0">'
                      f'<div style="font-size:10px;color:#8b949e;margin-bottom:4px">📊 {r["sid"]} {r["name"]} 日K線 MA20🟠 MA60🔵</div>'
                      f'{kline_html}</div></div>')
        return cards

    def render_downtrend(hits):
        hits = hits or []
        if not hits:
            return '<div class="empty-msg">🔌 今日無符合條件（需 6 分以上）— 市場條件未達拉回共振標準</div>'

        # ── 統計列 ────────────────────────────────────────────────
        aaa = sum(1 for r in hits if r['score'] >= _GRADE_AAA)
        aa  = sum(1 for r in hits if _GRADE_AA <= r['score'] < _GRADE_AAA)
        a   = sum(1 for r in hits if _SCORE_MIN <= r['score'] < _GRADE_AA)
        stat_html = (
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;padding:8px 4px 10px">'
            f'<div style="background:#1a1500;border:1px solid #b08800;border-radius:8px;padding:6px 14px;font-size:11px">'
            f'<span style="color:#ffd700;font-weight:bold">⭐⭐⭐ {aaa} 檔</span>'
            f'<span style="color:#8b949e;font-size:9px;margin-left:4px">極強共振 ≥8分</span></div>'
            f'<div style="background:#0a1828;border:1px solid #1f6feb;border-radius:8px;padding:6px 14px;font-size:11px">'
            f'<span style="color:#58a6ff;font-weight:bold">⭐⭐ {aa} 檔</span>'
            f'<span style="color:#8b949e;font-size:9px;margin-left:4px">強力推薦 6.5~7.9分</span></div>'
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:6px 14px;font-size:11px">'
            f'<span style="color:#c9d1d9;font-weight:bold">⭐ {a} 檔</span>'
            f'<span style="color:#8b949e;font-size:9px;margin-left:4px">值得留意 6.0~6.4分</span></div>'
            f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:6px 14px;font-size:11px">'
            f'<span style="color:#8b949e">合計列入 {len(hits)} 檔（最多{_TOP_N}）</span></div>'
            f'</div>'
        )

        # ── 表格 ──────────────────────────────────────────────────
        cols = [('總分',44),('代號',50),('名稱',72),('收盤',50),
                ('推薦等級',64),('符合條件',160),('關鍵數據',200),('建議觀點',130),('標註',110)]
        thead = ''.join(f'<th style="width:{w}px">{h}</th>' for h,w in cols)
        rows  = ''
        for r in hits:
            sc   = r['score']
            # 分數顏色
            if sc >= _GRADE_AAA:
                sc_cls  = 'c-gold'
                sc_bg   = 'background:#1a1500'
                gr_cls  = 'grade-aaa'
            elif sc >= _GRADE_AA:
                sc_cls  = 'c-blue'
                sc_bg   = 'background:#0a1828'
                gr_cls  = 'grade-aa'
            else:
                sc_cls  = 'c-gray'
                sc_bg   = ''
                gr_cls  = 'grade-a'

            # 條件標籤（小badge）
            cond_parts = r['cond_str'].split(' + ')
            cond_html  = ' '.join(
                f'<span class="bdg '
                + ('bg-fire' if '雙買' in p else
                   'bg-teal' if '安全' in p else
                   'bg-blue' if '乖離' in p else 'bg-gray')
                + f'">{p}</span>'
                for p in cond_parts if p
            )

            rows += (
                f'<tr style="{sc_bg}">'
                f'<td class="{sc_cls}" style="font-size:14px;font-weight:bold">{sc:.0f}</td>'
                f'<td><b>{r["sid"]}</b></td>'
                f'<td style="text-align:left">{r["name"]}</td>'
                f'<td class="c-gold"><b>{r["close"]:.1f}</b></td>'
                f'<td><span class="bdg {gr_cls}">{r["grade"]}</span></td>'
                f'<td style="text-align:left">{cond_html}</td>'
                f'<td style="text-align:left;font-size:10px;color:#adbac7">{r["key_data"]}</td>'
                f'<td style="text-align:left;font-size:10px">{r["suggest"]}</td>'
                f'<td style="text-align:left;font-size:10px;color:#f0883e">{r["tags"]}</td>'
                f'</tr>'
            )

        note = ('<div style="font-size:10px;color:#8b949e;padding:2px 4px 8px">'
                '計分：雙買發動股 4分 ｜ 安全回踩 3分 ｜ 乖離黃金點 2分 ｜ 投信新認養 1分'
                '｜ 滿分 10分，6分以上列入 ｜ 本區為下跌震盪補充視角，不影響原主系統推薦</div>')

        table = (f'<div style="overflow-x:auto"><table class="tbl">'
                 f'<thead><tr>{thead}</tr></thead>'
                 f'<tbody>{rows}</tbody></table></div>')
        return stat_html + note + table

    def render_risk(risk_rules, t1_top_final, S1_TOP12, L1_TOP20):
        items = ''.join(f'<div class="risk-item">🔒 {r}</div>' for r in risk_rules)
        alloc = '<div style="margin-bottom:10px;">'
        if t1_top_final: alloc += '<div style="padding:3px 0;font-size:11px">🔥 T1-TOP 每檔最高 10% 資金</div>'
        if S1_TOP12:     alloc += '<div style="padding:3px 0;font-size:11px">⚡ S1/T1 短線每檔 5~10% 資金</div>'
        if L1_TOP20:     alloc += '<div style="padding:3px 0;font-size:11px">📊 L1 長線每檔 10~15% 資金</div>'
        alloc += '<div style="padding:3px 0;font-size:11px">💰 建議保留至少 20% 現金部位</div></div>'
        return f'<div class="risk-box">{alloc}{items}</div>'

    summary_cards = ''.join(
        f'<div class="scard" style="background:{c}"><div class="n">{n}</div><div class="lb">{lb}</div></div>'
        for c, n, lb in [
            ('#da3633', len(t1_top_final), 'T1-TOP'),
            ('#b08800', len(L1_TOP20),     'L1長線'),
            ('#1f6feb', len(S1_TOP12),     'S1主力'),
            ('#238636', len(t1a_results),  'T1-A主力池'),
        ]
    )

    full_html = (
        CSS + '<div class="rep">'
        + '<div class="hdr"><div class="hdr-inner">'
        + '<div class="hdr-icon">📈</div>'
        + '<div class="hdr-text">'
        + '<h1>台股發發發系統<span class="ver">v2.0</span></h1>'
        + f'<div class="sub">{data_today} ｜ L1 長線守門 · S1 主力計數 · T1 進場觸發 · Z1 續抱</div>'
        + '</div></div></div>'
        + f'<div class="sumrow">{summary_cards}</div>'
        + '<div class="sec">🌐 ① 今日宏觀濾網</div>'
        + render_macro()
        + f'<div class="sec sec-blue">⚡ ② S1 主力強勢清單（{len(S1_TOP12)}檔）</div>'
        + '<div style="font-size:10px;color:#8b949e;padding:0 4px 6px">止損：跌破MA20 ｜ 止盈：+15%</div>'
        + render_s1(S1_TOP12)
        + '<div class="sec sec-fire">🔥 ③ T1-TOP（最高勝率突破型・條件最嚴苛）</div>'
        + '<div style="font-size:10px;color:#8b949e;padding:0 4px 6px">止損：跌破MA20 ｜ 止盈：+20%</div>'
        + render_t1_top(t1_top_final)
        + f'<div class="sec sec-fire">🔥 ④ T1-A（短線主力高勝率濾網・已接S1池）</div>'
        + '<div style="font-size:10px;color:#8b949e;padding:0 4px 6px">止損：跌破MA20 ｜ 止盈：+15%</div>'
        + render_t1a(t1a_results)
        + f'<div class="sec sec-blue">📊 ⑤ L1 長線觀察池 前{len(L1_TOP20)}檔（守門通過{len(L1_full_pool)}檔）</div>'
        + '<div style="font-size:10px;color:#8b949e;padding:0 4px 6px">止損：跌破月線MA20 / 季線MA60 ｜ 止盈：+30%~70%</div>'
        + render_l1(L1_TOP20)
        + '<div class="sec sec-purple">🔄 ⑥ Z1 續抱檢視（S1 加權分前5）</div>'
        + render_z1(z1_results)
        + f'<div class="sec sec-down">🔌 ⑦ 下跌趨勢最高勝率拉回共振（全{N_STOCKS}檔掃描・6分以上列入）</div>'
        + render_downtrend(downtrend_hits)
        + f'<div class="footer">台股發發發系統 v2.0 ｜ {data_today} ｜ 僅供參考，不構成投資建議</div>'
        + '</div>'
    )

    html_fname = f'output/taiwan_888_{TODAY.replace("-","")}.html'
    with open(html_fname, 'w', encoding='utf-8') as f:
        f.write(f'<!DOCTYPE html><html><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>台股發發發 v2.0 {data_today}</title></head>'
                f'<body style="margin:0;padding:0">{full_html}</body></html>')
    print(f'✅ HTML 報告已儲存：{html_fname}（{len(full_html)//1024} KB）')
    if df_for_max is not None:
        try:
            max_html = generate_max_html(df_for_max, macro)
            with open(html_fname, 'r', encoding='utf-8') as _fh: _c = _fh.read()
            _c = _c.replace('</body></html>', max_html + '</body></html>')
            with open(html_fname, 'w', encoding='utf-8') as _fh: _fh.write(_c)
            print(f'✅ MAX 三系統已接上 HTML')
        except Exception as _e:
            print(f'⚠️ MAX 生成失敗（不影響 888 報告）: {_e}')
            import traceback; traceback.print_exc()
    return html_fname


# ============================================================
# ── 通知：Telegram + Email
# ============================================================

def send_telegram(t1_top_final, L1_TOP20, S1_TOP12, t1a_results, t1b_results,
                  N_STOCKS, macro, html_fname):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print('⚠️  未設定 Telegram，跳過')
        return
    lines = [
        f'📈 *台股發發發系統 v2.0 — {TODAY}*',
        '',
        f'🔍 掃描標的：{N_STOCKS} 檔',
        f'🔥 T1-TOP：{len(t1_top_final)} 檔',
        f'📊 L1 長線前20：{len(L1_TOP20)} 檔',
        f'⚡ S1 主力：{len(S1_TOP12)} 檔',
        f'🎯 T1-A：{len(t1a_results)} 檔',
        f'📍 T1-B：{len(t1b_results)} 檔',
        '',
    ]
    if t1_top_final:
        lines.append('*🔥 T1-TOP：*')
        for r in t1_top_final[:5]:
            lines.append(f'  {r["sid"]} {r["name"]} | {r["t1_grade"]} | 量比{r["vr"]:.2f}x')
        lines.append('')
    if L1_TOP20:
        lines.append('*📊 L1 Top5：*')
        for r in L1_TOP20[:5]:
            lines.append(f'  {r["sid"]} {r["name"]} | 總分{r["total"]}/100')
        lines.append('')
    adv = macro.get('op_advice', '')
    adv_emoji = '✅' if '積極' in adv else '⚠️'
    lines.append(f'{adv_emoji} 操作建議：{adv}')
    if REPORT_URL:
        lines.append(f'🌐 [完整報告]({REPORT_URL})')
    msg = '\n'.join(lines)
    try:
        resp = requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown',
                  'disable_web_page_preview': False},
            timeout=15)
        if resp.status_code == 200:
            print('✅ Telegram 通知已發送')
        else:
            print(f'⚠️  Telegram 失敗：{resp.text}')
    except Exception as e:
        print(f'⚠️  Telegram 錯誤：{e}')


def send_email(csv_fname, html_fname, t1_top_final, L1_TOP20, S1_TOP12,
               t1a_results, t1b_results, N_STOCKS, macro):
    if not GMAIL_USER or not GMAIL_APP_PASS or not EMAIL_TO:
        print('⚠️  未設定 Email，跳過')
        return
    msg = MIMEMultipart('mixed')
    msg['Subject'] = f'台股發發發 v2.0 — {TODAY} | T1-TOP:{len(t1_top_final)} L1:{len(L1_TOP20)} S1:{len(S1_TOP12)}'
    msg['From']    = GMAIL_USER
    msg['To']      = EMAIL_TO

    body_lines = [
        f'台股發發發系統 v2.0 — {TODAY}',
        f'掃描標的：{N_STOCKS} 檔',
        f'T1-TOP：{len(t1_top_final)} 檔 | L1：{len(L1_TOP20)} 檔 | S1：{len(S1_TOP12)} 檔 | T1-A：{len(t1a_results)} 檔 | T1-B：{len(t1b_results)} 檔',
        f'操作建議：{macro.get("op_advice","")}',
        '',
    ]
    if t1_top_final:
        body_lines.append('T1-TOP（最高勝率）：')
        for r in t1_top_final[:5]:
            body_lines.append(f'  {r["sid"]} {r["name"]} | {r["t1_grade"]} | 量比{r["vr"]:.2f}x | 目標≈{r["tgt"]:.0f}')
        body_lines.append('')
    if L1_TOP20:
        body_lines.append('L1 長線 Top5：')
        for r in L1_TOP20[:5]:
            body_lines.append(f'  {r["sid"]} {r["name"]} | 總分{r["total"]}/100 | {r["eps_yoy"]}')
        body_lines.append('')
    if REPORT_URL:
        body_lines.append(f'完整報告：{REPORT_URL}')
    body_lines.append('\n--- 本郵件由系統自動發送，不構成投資建議 ---')
    msg.attach(MIMEText('\n'.join(body_lines), 'plain', 'utf-8'))

    for fpath in [csv_fname, html_fname]:
        if fpath and os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(fpath))
            msg.attach(part)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_APP_PASS)
            smtp.sendmail(GMAIL_USER, EMAIL_TO.split(','), msg.as_string())
        print('✅ Email 已發送')
    except Exception as e:
        print(f'⚠️  Email 失敗：{e}')


# ============================================================
# 主程式
# ============================================================

def main():
    print('='*58)
    print('台股發發發系統 v2.0 — 開始執行')
    print('='*58)

    # ── Part 1：抓資料
    selected_ids, name_map = load_top500_and_filter()
    price_data, mkt_ret    = fetch_price_data(selected_ids)
    df_tech                = calc_tech_indicators(price_data, mkt_ret)
    df_per                 = fetch_per_data(price_data)
    df_inc_data            = fetch_income_data(price_data)
    df_bal_data            = fetch_balance_data(price_data)
    df_shr                 = fetch_shareholding_data(price_data)
    df_inst                = fetch_inst_data(price_data)
    df_mops                = fetch_monthly_revenue(price_data)
    news_score_map, news_count_map, news_latest_map, news_date_map = fetch_news_data(price_data, name_map)
    df_wma                 = calc_weekly_ma13(price_data)

    # 確保 for col in COL_ORDER 中的月營收欄位存在
    for c in ['monthly_rev_latest','monthly_rev_yoy','monthly_rev_mom','rev_month']:
        if c not in df_mops.columns:
            df_mops[c] = np.nan

    df = build_derived_features(
        df_tech, df_per, df_inc_data, df_bal_data,
        df_shr, df_inst, df_mops, df_wma,
        price_data, mkt_ret, name_map,
        news_score_map, news_count_map, news_latest_map, news_date_map)

    # 族群動能
    sector_map = calc_sector_momentum(df, price_data, FINMIND_TOKEN)
    df['sector_momentum'] = df['stock_id'].astype(str).map(sector_map).fillna(False)

    # 輸出 CSV
    csv_fname, df_out = export_csv(df)

    # ── Part 2：分析 + HTML
    print('='*58)
    print('🏆 Part 2：台股發發發系統 2.0 分析')
    print('='*58)

    df2, N_STOCKS, MARKET_FOREIGN_B, data_today = load_and_clean_csv(csv_fname)

    print('='*58)
    print('🌐 Part2 Step 2: 宏觀濾網（加權指數）')
    print('='*58)
    macro = fetch_taiex()

    print('='*58)
    print('🎯 Part2 Step 3: 篩選（T1/L1/S1/Z1）')
    print('='*58)
    (t1_top_final, L1_TOP20, L1_full_pool, S1_TOP12, S1_full_pool,
     t1a_results, t1b_results, z1_results, risk_rules,
     N_STOCKS, macro) = run_screening(df2, N_STOCKS, MARKET_FOREIGN_B, macro)

    print('='*58)
    print('🔌 Part2 Step 3b: 下跌趨勢最高勝率拉回共振')
    print('='*58)
    downtrend_hits = run_downtrend_plugin(df2)

    print('='*58)
    print('📊 Part2 Step 4: Z1 K線圖')
    print('='*58)
    z1_kline_cache = fetch_z1_klines(z1_results)

    html_fname = generate_html(
        t1_top_final, L1_TOP20, L1_full_pool, S1_TOP12, S1_full_pool,
        t1a_results, t1b_results, z1_results, risk_rules,
        N_STOCKS, macro, z1_kline_cache, data_today,
        downtrend_hits=downtrend_hits,
        df_for_max=df2)

    # 複製為 index.html（GitHub Pages 固定網址）
    import shutil
    shutil.copy(html_fname, 'output/index.html')
    print('✅ output/index.html 已更新')

    # ── 通知
    send_telegram(t1_top_final, L1_TOP20, S1_TOP12, t1a_results, t1b_results,
                  N_STOCKS, macro, html_fname)
    send_email(csv_fname, html_fname, t1_top_final, L1_TOP20, S1_TOP12,
               t1a_results, t1b_results, N_STOCKS, macro)

    print('='*58)
    print(f'✅ 台股發發發系統 v2.0 完成')
    print(f'   CSV ：{csv_fname}')
    print(f'   HTML：{html_fname}')
    print(f'   FinMind calls：{_fm_calls[0]}')
    print('='*58)


if __name__ == '__main__':
    main()
