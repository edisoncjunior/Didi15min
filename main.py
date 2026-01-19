# Near+5 Didi funcionando (local e web) = 
# coloquei pra rodar local 23h55 sábado 17/01 e funcionou
# coloquei pra rodar web dia 18/01 as 09h09

#proximos testes = 1) colocar log para enviar as 9 e 21h
# antes de fazer os testes -> desligar a versão web ou mudar o grupo?

#!/usr/bin/env python3
"""
Binance Futures 15m scanner -> Telegram alerts

Critérios:
 - Triple SMA crossover: SMA(3), SMA(8), SMA(20) crossing simultaneously.
 - ADX (period=8) crescente com aceleração.
 - Bollinger Bands (period=8, std=2) "abertas".

Env vars in .env:
 - TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
 - BINANCE_API_KEY (optional), BINANCE_API_SECRET (optional)
 - POLL_SECONDS, KLINES_LIMIT
"""

import os
import sys
import time
import signal
import logging
import asyncio
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Config / env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY") or ""
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET") or ""
POLL_SECONDS = int(os.getenv("POLL_SECONDS") or 60)
KLINES_LIMIT = int(os.getenv("KLINES_LIMIT") or 200)

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    LOGGER.error("TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
    sys.exit(1)

BOLLINGER_PERIOD = 8
BOLLINGER_STD = 2
ADX_PERIOD = 8

BOLLINGER_WIDTH_MIN_PCT = float(os.getenv("BOLLINGER_WIDTH_MIN_PCT") or 0.015)  # 1.5%
ADX_MIN = float(os.getenv("ADX_MIN") or 15)
ADX_ACCEL_THRESHOLD = float(os.getenv("ADX_ACCEL_THRESHOLD") or 0.05)  # relative accel

BINANCE_FAPI = "https://fapi.binance.com"   # futures api (perpetual)

# ==========================
# LISTA FIXA DE ATIVOS
# ==========================
FIXED_SYMBOLS = [
#    "APEUSDT", "1INCHUSDT", "ADAUSDT", "ALICEUSDT", "ANKRUSDT", "ARBUSDT", "ATAUSDT", "BANDUSDT", "BCHUSDT",
#    "BELUSDT", "BNBUSDT", "BTCUSDT", "CELOUSDT", "CHZUSDT", "CYBERUSDT", "DASHUSDT", "DOTUSDT", "DYDXUSDT",
#    "EGLDUSDT", "ENJUSDT", "ENSUSDT", "ETCUSDT", "ETHUSDT", "FETUSDT", "FILUSDT", "FLMUSDT", "GMTUSDT",
#    "GRTUSDT", "GTCUSDT", "HBARUSDT", "HOTUSDT", "ICPUSDT", "ICXUSDT", "IMXUSDT", "JASMYUSDT", "JTOUSDT",
#    "JUPUSDT", "KAVAUSDT", "KNCUSDT", "LINKUSDT", "LPTUSDT", "LQTYUSDT", "LTCUSDT", "MASKUSDT", "MTLUSDT",
    "NEARUSDT",
#    "NEOUSDT", "OGNUSDT", "ONEUSDT", "OPUSDT", "PEOPLEUSDT", "RLCUSDT", "RSRUSDT", "RUNEUSDT",
#    "SANDUSDT", "SFPUSDT", "SKLUSDT", "SNXUSDT", "SOLUSDT",
    "STORJUSDT",
#    "SUIUSDT", "SUSHIUSDT", "SXPUSDT", "THETAUSDT", "TIAUSDT", "TONUSDT", "TRBUSDT", "TRXUSDT",
#    "UNIUSDT", "VETUSDT", "WOOUSDT", "XLMUSDT", "XMRUSDT",
    "XRPUSDT", 
    "XTZUSDT",
#    "ZENUSDT",
    "ZILUSDT",
    "ZRXUSDT"
]

LOG_FILE = "scanner_runtime.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

LOGGER = logging.getLogger("scanner")


if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    LOGGER.error("TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
    sys.exit(1)

# --- Utilities

def utc_minus_3_now_str():
    tz = timezone(timedelta(hours=-3))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        LOGGER.exception("Erro enviando Telegram: %s", e)

def fetch_klines(symbol, interval="15m", limit=KLINES_LIMIT):
    url = BINANCE_FAPI + "/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    # convert to dataframe
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.set_index("open_time", inplace=True)
    return df


# --- Indicators (pandas implementations)

def sma(series, period):
    return series.rolling(window=period, min_periods=1).mean()

def true_range(df):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df, period=14):
    tr = true_range(df)
    # Wilder's ATR (exponential smoothing with alpha=1/period)
    return tr.rolling(window=period, min_periods=period).mean()

def bollinger_bands(series, period=BOLLINGER_PERIOD, std_factor=BOLLINGER_STD):
    ma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std()
    upper = ma + std_factor * std
    lower = ma - std_factor * std
    width = (upper - lower) / ma.replace(0, np.nan)
    return upper, lower, width

def adx(df, period=ADX_PERIOD):
    # Returns ADX series (Wilder)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = true_range(df)
    atr_val = tr.rolling(window=period, min_periods=period).mean()

    # smoothed DM
    plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).sum()

    plus_di = 100 * (plus_dm_smooth / atr_val)
    minus_di = 100 * (minus_dm_smooth / atr_val)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.rolling(window=period, min_periods=period).mean()
    return adx_series.fillna(0)

# --- Signal logic

def triple_sma_cross(df):
    """Detect triple SMA crossover on most recent candle.
       Return 'LONG' or 'SHORT' or None.
       Logic: previous candle didn't have order 3>8>20 (for long) and current candle has it (cross happened).
       Similarly for short (3<8<20).
    """
    close = df["close"]
    s3 = sma(close, 3)
    s8 = sma(close, 8)
    s20 = sma(close, 20)
    # take last two rows
    if len(df) < 3:
        return None
    cur_idx = df.index[-1]
    prev_idx = df.index[-2]
    # values
    s3_cur, s8_cur, s20_cur = s3.iloc[-1], s8.iloc[-1], s20.iloc[-1]
    s3_prev, s8_prev, s20_prev = s3.iloc[-2], s8.iloc[-2], s20.iloc[-2]

    # LONG if now s3> s8 > s20 and previously not in that order
    if (s3_cur > s8_cur > s20_cur) and not (s3_prev > s8_prev > s20_prev):
        return "LONG"
    # SHORT if now s3 < s8 < s20 and previously not in that order
    if (s3_cur < s8_cur < s20_cur) and not (s3_prev < s8_prev < s20_prev):
        return "SHORT"
    return None

def adx_accelerating(df):
    adx_series = adx(df, period=ADX_PERIOD)
    if len(adx_series) < 4:
        return False, None
    # take last three adx deltas
    d2 = adx_series.iloc[-1] - adx_series.iloc[-2]  # newest delta
    d1 = adx_series.iloc[-2] - adx_series.iloc[-3]  # previous delta
    cur_adx = adx_series.iloc[-1]
    # require ADX above min and increasing and acceleration (d2 > d1 and relative accel)
    if cur_adx >= ADX_MIN and d2 > 0 and (d2 - d1) > (abs(d1) * ADX_ACCEL_THRESHOLD):
        return True, cur_adx
    # fallback: require simply increasing twice in a row
    if cur_adx >= ADX_MIN and (adx_series.iloc[-1] > adx_series.iloc[-2] > adx_series.iloc[-3]):
        return True, cur_adx
    return False, cur_adx

def bollinger_open(df):
    close = df["close"]
    upper, lower, width = bollinger_bands(close, period=BOLLINGER_PERIOD, std_factor=BOLLINGER_STD)
    # use last width and compare to a dynamic baseline: mean of last 20 widths
    last_width = float(width.iloc[-1]) if len(width) else 0.0
    baseline = float(width.dropna().rolling(window=20, min_periods=1).mean().iloc[-1])
    # Accept if last_width > max(baseline, BOLLINGER_WIDTH_MIN_PCT)
    threshold = max(baseline, BOLLINGER_WIDTH_MIN_PCT)
    return last_width >= threshold, last_width, baseline

def compute_targets(df, side):
    # Use ATR(14) as volatility measure to set TP levels
    atr_val = atr(df, period=14).iloc[-1] if len(df) >= 14 else np.nan
    price = df["close"].iloc[-1]
    if np.isnan(atr_val) or atr_val == 0:
        # fallback: use percentage TPs
        if side == "SHORT":
            tps = [price * (1 - p/100) for p in (0.5, 1.0, 2.0)]
        else:
            tps = [price * (1 + p/100) for p in (0.5, 1.0, 2.0)]
        return price, tps, atr_val
    # set TP distances as multiples of ATR: 0.5, 1.0, 2.0
    if side == "SHORT":
        tps = [price - m * atr_val for m in (0.5, 1.0, 2.0)]
    else:
        tps = [price + m * atr_val for m in (0.5, 1.0, 2.0)]
    return price, tps, atr_val

# --- Main scanning

def analyze_symbol(symbol):
    try:
        df = fetch_klines(symbol, interval="15m", limit=KLINES_LIMIT)
        df = df.iloc[:-1] # remove candles ainda abertos
    except Exception as e:
        LOGGER.debug("Erro ao buscar klines %s: %s", symbol, e)
        return None

    if df is None or df.empty:
        return None

    # Filters/indicators
    bb_open, last_width, baseline = bollinger_open(df)
    if not bb_open:
        return None

    adx_ok, adx_value = adx_accelerating(df)
    if not adx_ok:
        return None

    cross = triple_sma_cross(df)
    if not cross:
        return None

    entry_price, tps, atr_val = compute_targets(df, cross)
    # Compose result
    price_now = float(df["close"].iloc[-1])
    return {
        "symbol": symbol,
        "side": cross,
        "price": price_now,
        "entry_price_calc": entry_price,
        "tps": tps,
        "atr": float(atr_val) if not np.isnan(atr_val) else None,
        "adx": float(adx_value),
        "bb_width": float(last_width),
        "bb_baseline": float(baseline)
    }

# Graceful shutdown: send Telegram message
SHUTDOWN = False
def handle_sigint(sig, frame):
    global SHUTDOWN
    SHUTDOWN = True
    send_telegram(f"🤖 Scanner (NEAR+5) interrompido pelo usuário em {utc_minus_3_now_str()}.")
    LOGGER.info("Interrupção solicitada. Encerrando...")

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

def main_loop():
    send_telegram(f"🤖 Scanner 15min (NEAR+5) iniciado em {utc_minus_3_now_str()} — Binance Futures (15m).")
    LOGGER.info("Iniciado scanner com lista fixa de símbolos.")

    while not SHUTDOWN:
        try:
            symbols = FIXED_SYMBOLS
            LOGGER.info("Verificando %d símbolos fixos: %s", len(symbols), ", ".join(symbols))
            LOGGER.info("Novo ciclo iniciado (%s símbolos)", len(FIXED_SYMBOLS))
            alerts = []
            for sym in symbols:
                try:
                    res = analyze_symbol(sym)
                    if res:
                        alerts.append(res)
                        msg = build_alert_message(res)

                        # 1️⃣ tenta gravar log (NUNCA pode quebrar o fluxo)
                        try:
                            log_signal_to_file(res)
                        except Exception as e:
                            LOGGER.exception("Falha ao registrar log (ignorado): %s", e)

                        # 2️⃣ envia Telegram SEMPRE
                        send_telegram(msg)

                        LOGGER.info("Alerta enviado: %s %s", res["symbol"], res["side"])
                except Exception as e:
                    LOGGER.debug("Erro analisando %s: %s", sym, e)
            if not alerts:
                LOGGER.info("Nenhum sinal encontrado neste ciclo.")
        except Exception as e:
            LOGGER.exception("Erro no loop principal: %s", e)
        # sleep
        for _ in range(int(max(1, POLL_SECONDS))):
            if SHUTDOWN:
                break
            time.sleep(1)
    LOGGER.info("Scanner finalizado.")
    LOGGER.info("Ciclo finalizado")

def build_alert_message(res):
    sym = res["symbol"]
    side = res["side"]
    price = res["price"]
    adxv = res["adx"]
    atrv = res["atr"]
    bbw = res["bb_width"]
    tps = res["tps"]
    now = utc_minus_3_now_str()
    # Compose TPs text
    tps_text = "\n".join([f"TP{i+1}: {tp:.8f}" for i,tp in enumerate(tps)])
    msg = (
        f"🚨 <b>ALERTA 15min (NEAR+5)</b>\n"
        f"Exchange: Binance Futures\n"
        f"Par: <b>{sym}</b>\n"
        f"Horário SP: {now}\n"
        f"Preço atual: <b>{price:.8f}</b>\n"
        f"Sinal: <b>{side}</b>\n"
        f"ADX (period={ADX_PERIOD}): {adxv:.2f}\n"
        f"ATR(14): {atrv:.8f}\n"
        f"BB width: {bbw:.4f}\n\n"
        f"Análise de alvos:\n"
        f"Entry (calc): {res['entry_price_calc']:.8f}\n"
        f"{tps_text}\n\n"
        f"Observação: critérios aplicados -> SMA(3,8,20) crossover, ADX crescente e bandas Bollinger abertas."
    )
    return msg

def get_daily_log_filename(date=None):
    tz = timezone(timedelta(hours=-3))
    if date is None:
        date = datetime.now(tz)

    day_str = date.strftime("%Y-%m-%d")
    return f"telegram_signals_{day_str}.tsv"

def log_signal_to_file(res, timeframe="15m", exchange="Binance Futures"):
    """
    Grava o sinal em arquivo texto tabulado (TSV).
    """
    tz = timezone(timedelta(hours=-3))
    now = datetime.now(tz)

    header = (
        "symbol\talert_date\talert_time\ttimeframe\texchange\tprice\tside\t"
        "adx\tatr\tbb_width\tentry\ttp1\ttp2\ttp3\tbb_baseline\tstrategy\n"
    )

    row = (
        f"{res['symbol']}\t"
        f"{now.strftime('%Y-%m-%d')}\t"
        f"{now.strftime('%H:%M:%S')}\t"
        f"{timeframe}\t"
        f"{exchange}\t"
        f"{res['price']:.8f}\t"
        f"{res['side']}\t"
        f"{res['adx']:.2f}\t"
        f"{res['atr']:.8f}\t"
        f"{res['bb_width']:.6f}\t"
        f"{res['entry_price_calc']:.8f}\t"
        f"{res['tps'][0]:.8f}\t"
        f"{res['tps'][1]:.8f}\t"
        f"{res['tps'][2]:.8f}\t"
        f"{res['bb_baseline']:.6f}\t"
        f"SMA(3,8,20)+ADX+BB\n"
    )

    log_file = get_daily_log_filename()
    file_exists = os.path.isfile(log_file)

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write(header)
            f.write(row)
            LOGGER.info("Log gravado com sucesso: %s", log_file)
    except Exception as e:
        LOGGER.exception("Erro ao gravar log em arquivo (%s): %s", log_file, e)



if __name__ == "__main__":
    main_loop()
