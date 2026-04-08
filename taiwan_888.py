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
                'high_52w','low_52w','bias_5','bias_20','bias_60','weekly_ma13']
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
            sid   = str(row['stock_id']); name = str(row.get('name', sid))
            close = float(row.get('close', 0) or 0)
            if close <= 0: continue
            fresh = bool(row.get('fresh_priority', False))
            t1g   = str(row.get('t1_grade', 'C'))
            cn    = str(row.get('cross_note', ''))
            vr    = float(row.get('volume_ratio', 0) or 0)
            pct20 = float(row.get('price_vs_ma20_pct', 0) or 0)
            ma20v = float(row.get('ma20', 0) or 0)
            bull  = bool(row.get('ma_bull_align', False))
            if not fresh: continue
            if t1g not in ['A+','A','A-']: continue
            if not has_valid_cross(cn): continue
            if not (bull or bool(row.get('price_above_ma20', False))): continue
            if vr <= 1.2: continue
            if not (-6 <= pct20 <= 3): continue
            results.append({'sid':sid,'name':name,'close':close,'t1_grade':t1g,
                            'cross_note':cn,'vr':vr,'pct20':pct20,'ma20':ma20v,
                            'rsi':float(row.get('rsi14',50) or 50),
                            'chip':str(row.get('chip_sync','')),
                            'weekly_ma13':bool(row.get('weekly_ma13_signal',False)),
                            'sl':round(ma20v,0),'tgt':round(close*1.20,0)})
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
            if cnt >= 4:   label = '🔥主力股'
            elif cnt == 3: label = '✅強勢股'
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

    # ── T1-A
    def run_t1a(df_all):
        results = []
        for _, row in df_all.iterrows():
            sid   = str(row['stock_id']); name = str(row.get('name', sid))
            close = float(row.get('close', 0) or 0)
            if close <= 0: continue
            fresh = bool(row.get('fresh_priority', False))
            t1g   = str(row.get('t1_grade', 'C'))
            cn    = str(row.get('cross_note', ''))
            vr    = float(row.get('volume_ratio', 0) or 0)
            ma20v = float(row.get('ma20', 0) or 0)
            bull  = bool(row.get('ma_bull_align', False))
            ab20  = bool(row.get('price_above_ma20', False))
            if not fresh: continue
            if t1g not in ['A+','A','A-']: continue
            if not has_valid_cross(cn): continue
            if not (bull or ab20): continue
            if vr <= 1.2: continue
            results.append({'sid':sid,'name':name,'close':close,'t1_grade':t1g,
                            'cross_note':cn,'vr':vr,
                            'pct20':float(row.get('price_vs_ma20_pct',0) or 0),
                            'ma20':ma20v,'sl':round(ma20v,0),'tgt':round(close*1.10,0),
                            'rsi':float(row.get('rsi14',50) or 50),
                            'chip':str(row.get('chip_sync','')),
                            'weekly_ma13':bool(row.get('weekly_ma13_signal',False))})
        return results

    # ── T1-B
    def run_t1b(s1_pool):
        results = []
        for r in s1_pool:
            pct20 = r['pct20']; m5m10 = r['ma5_gt_ma10']
            cn = r['cross_note']; vr = r['vr']
            vs = r['volume_shrink']; r5 = r['r5']
            sk = r['stop_fall_k']; ma20v = r['ma20']
            conds_met = []
            c1 = (-6 <= pct20 <= 3)
            if c1: conds_met.append('月線%在-6~+3%')
            c2 = (m5m10 or any(k in str(cn) for k in ['MA5穿','MA10穿']))
            if c2: conds_met.append('MA5>MA10或剛穿')
            c3 = (vr > 1.4 and vs)
            if c3: conds_met.append('量比>1.4+量縮')
            c4 = (1.5 < r5 < 7)
            if c4: conds_met.append(f'漲幅{r5:.1f}%')
            c5 = sk
            if c5: conds_met.append('止跌K')
            total_conds = sum([c1,c2,c3,c4,c5])
            if total_conds == 5:   grade = 'A'
            elif total_conds == 4: grade = 'A-'
            else: continue
            results.append({'sid':r['sid'],'name':r['name'],'close':r['close'],
                            'grade':grade,'conds_met':conds_met,'total_conds':total_conds,
                            'pct20':pct20,'vr':vr,'rsi':r['rsi'],'chip':r['chip'],
                            'ma20':ma20v,'sl':round(ma20v,0),'tgt':round(r['close']*1.10,0)})
        return results

    t1_top_results = run_t1_top(df)
    t1a_results    = run_t1a(df)
    t1b_results    = run_t1b(S1_full_pool)
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

    z1_results = []; _z1_seen = set()

    def _z1_add(sid, name, close, category, df_row=None):
        if sid in _z1_seen: return
        _z1_seen.add(sid)
        if df_row is None:
            _row = df[df['stock_id'].astype(str) == sid]
            if _row.empty: return
            df_row = _row.iloc[0]
        verdict, checks = z1_check(df_row)
        z1_results.append({'sid':sid,'name':name,'close':close,
                           'verdict':verdict,'checks':checks,'category':category})

    for r in t1_top_final:
        row = df[df['stock_id'].astype(str) == r['sid']]
        if row.empty: continue
        _z1_add(r['sid'], r['name'], r['close'], '🔥T1-TOP', row.iloc[0])
    for r in t1a_results:
        row = df[df['stock_id'].astype(str) == r['sid']]
        if row.empty: continue
        _z1_add(r['sid'], r['name'], r['close'], '⭐T1-A', row.iloc[0])
    for r in S1_MAJOR:
        row = df[df['stock_id'].astype(str) == r['sid']]
        if row.empty: continue
        _z1_add(r['sid'], r['name'], r['close'], '💥S1主力', row.iloc[0])
    l1_added = 0
    for r in L1_TOP20:
        if l1_added >= 8: break
        if r['sid'] in _z1_seen: continue
        row = df[df['stock_id'].astype(str) == r['sid']]
        if row.empty: continue
        _z1_add(r['sid'], r['name'], r['close'], 'L1長線', row.iloc[0])
        l1_added += 1

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
                  N_STOCKS, macro, z1_kline_cache, data_today):

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
        for r in lst:
            rows += (f'<tr><td colspan="12" style="text-align:left;font-size:9px;'
                     f'color:#8b949e;padding:2px 8px;background:#0d1117">'
                     f'↳ {r["sid"]} {r["name"]}：{" ✦ ".join(r["conds"])}</td></tr>')
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
            ('#238636', len(t1a_results),  'T1-A獨立'),
            ('#1b7c83', len(t1b_results),  'T1-B拉回'),
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
        + '<div class="sec sec-fire">🔥 ① T1-TOP（最高勝率突破型・條件最嚴苛）</div>'
        + '<div style="font-size:10px;color:#8b949e;padding:0 4px 6px">止損：跌破MA20 ｜ 止盈：+20%</div>'
        + render_t1_top(t1_top_final)
        + '<div class="sec">🌐 ② 今日宏觀濾網</div>'
        + render_macro()
        + f'<div class="sec sec-blue">📊 ③ L1 長線觀察池 前{len(L1_TOP20)}檔（守門通過{len(L1_full_pool)}檔）</div>'
        + '<div style="font-size:10px;color:#8b949e;padding:0 4px 6px">止損：跌破月線MA20 / 季線MA60 ｜ 止盈：+30%~70%</div>'
        + render_l1(L1_TOP20)
        + f'<div class="sec sec-blue">⚡ ④ S1 主力強勢清單（{len(S1_TOP12)}檔）</div>'
        + '<div style="font-size:10px;color:#8b949e;padding:0 4px 6px">止損：跌破MA20 ｜ 止盈：+15%</div>'
        + render_s1(S1_TOP12)
        + f'<div class="sec sec-green">🎯 ⑤ T1-A（獨立高勝率濾網・全市場{N_STOCKS}檔）</div>'
        + '<div style="font-size:10px;color:#8b949e;padding:0 4px 6px">止損：跌破MA20 ｜ 止盈：+10%</div>'
        + render_t1a(t1a_results)
        + f'<div class="sec sec-orange">📍 ⑥ T1-B 今日可進場清單（拉回型・S1 {len(S1_full_pool)}檔內）</div>'
        + '<div style="font-size:10px;color:#8b949e;padding:0 4px 6px">止損：跌破MA20 ｜ 止盈：+10%</div>'
        + render_t1b(t1b_results)
        + '<div class="sec sec-purple">🔄 ⑦ Z1 續抱檢視</div>'
        + render_z1(z1_results)
        + '<div class="sec sec-fire">🛡️ ⑧ 資金配置建議 + 風險控管鐵律</div>'
        + render_risk(risk_rules, t1_top_final, S1_TOP12, L1_TOP20)
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
    print('📊 Part2 Step 4: Z1 K線圖')
    print('='*58)
    z1_kline_cache = fetch_z1_klines(z1_results)

    html_fname = generate_html(
        t1_top_final, L1_TOP20, L1_full_pool, S1_TOP12, S1_full_pool,
        t1a_results, t1b_results, z1_results, risk_rules,
        N_STOCKS, macro, z1_kline_cache, data_today)

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
