# versão criada a partir do main2local = que funciona localmente.
# colocada no github 01/02 as 12h55 em teste web.


#!/usr/bin/env python3
"""
Scanner Binance Futures 15m
Versão preparada para execução WEB (GitHub + Railway)

- Sem dependência de terminal interativo
- Shutdown correto via SIGTERM (Railway)
- Variáveis via ENV (Railway / GitHub Secrets)
- Loop resiliente (não morre em erro)
"""

import os
import sys
import time
import signal
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

load_dotenv()

# =========================================================
# CONFIGURAÇÃO DE AMBIENTE (Railway / GitHub)
# =========================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", 120))
KLINES_LIMIT = int(os.getenv("KLINES_LIMIT", 200))

BINANCE_FAPI = "https://fapi.binance.com"

# =========================================================
# PARÂMETROS DE ESTRATÉGIA
# =========================================================

BOLLINGER_PERIOD = 8
BOLLINGER_STD = 2
ADX_PERIOD = 8

BOLLINGER_WIDTH_MIN_PCT = float(os.getenv("BOLLINGER_WIDTH_MIN_PCT", 0.015))
ADX_MIN = float(os.getenv("ADX_MIN", 15))
ADX_ACCEL_THRESHOLD = float(os.getenv("ADX_ACCEL_THRESHOLD", 0.05))

# =========================================================
# ATIVOS FIXOS
# =========================================================

FIXED_SYMBOLS = [
    "BCHUSDT", "BNBUSDT", "CHZUSDT", "DOGEUSDT", "ENAUSDT",
    "ETHUSDT", "JASMYUSDT", "SOLUSDT", "UNIUSDT", "XMRUSDT", "XRPUSDT"
]

# =========================================================
# LOGGING (Railway captura STDOUT automaticamente)
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

LOGGER = logging.getLogger("scanner")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    LOGGER.error("Variáveis TELEGRAM_TOKEN e TELEGRAM_CHAT_ID não configuradas.")
    sys.exit(1)

TZ_SP = pytz.timezone("America/Sao_Paulo")

# =========================================================
# UTILITÁRIOS
# =========================================================

def now_sp():
    return datetime.now(TZ_SP)

def now_sp_str():
    return now_sp().strftime("%Y-%m-%d %H:%M:%S")

def send_telegram(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }
        requests.post(url, data=payload, timeout=10).raise_for_status()
    except Exception as e:
        LOGGER.error("Falha ao enviar Telegram: %s", e)

# =========================================================
# DADOS DE MERCADO
# =========================================================

def fetch_klines(symbol, interval="15m", limit=200):
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tb_base","tb_quote","ignore"
    ]
    df = pd.DataFrame(r.json(), columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.iloc[:-1]  # remove candle aberto

# =========================================================
# INDICADORES
# =========================================================

def sma(s, p): return s.rolling(p).mean()

def true_range(df):
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

def atr(df, p=14):
    return true_range(df).rolling(p).mean()

def bollinger_bands(series):
    ma = series.rolling(BOLLINGER_PERIOD).mean()
    std = series.rolling(BOLLINGER_PERIOD).std()
    upper = ma + BOLLINGER_STD * std
    lower = ma - BOLLINGER_STD * std
    width = (upper - lower) / ma
    return upper, lower, width

def adx(df):
    tr = true_range(df)
    atr_v = tr.rolling(ADX_PERIOD).mean()

    up = df["high"].diff()
    down = -df["low"].diff()

    plus = up.where((up > down) & (up > 0), 0)
    minus = down.where((down > up) & (down > 0), 0)

    pdi = 100 * plus.rolling(ADX_PERIOD).sum() / atr_v
    mdi = 100 * minus.rolling(ADX_PERIOD).sum() / atr_v
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.rolling(ADX_PERIOD).mean().fillna(0)

# =========================================================
# LÓGICA DE SINAL
# =========================================================

def triple_sma_cross(df):
    c = df["close"]
    s3, s8, s20 = sma(c,3), sma(c,8), sma(c,20)
    if len(df) < 3: return None

    if s3.iloc[-1] > s8.iloc[-1] > s20.iloc[-1] and not (s3.iloc[-2] > s8.iloc[-2] > s20.iloc[-2]):
        return "LONG"
    if s3.iloc[-1] < s8.iloc[-1] < s20.iloc[-1] and not (s3.iloc[-2] < s8.iloc[-2] < s20.iloc[-2]):
        return "SHORT"
    return None

def analyze_symbol(symbol):
    df = fetch_klines(symbol)
    upper, lower, width = bollinger_bands(df["close"])
    adx_v = adx(df).iloc[-1]
    cross = triple_sma_cross(df)

    if width.iloc[-1] < BOLLINGER_WIDTH_MIN_PCT: return None
    if adx_v < ADX_MIN: return None
    if not cross: return None

    price = df["close"].iloc[-1]
    return {
        "symbol": symbol,
        "side": cross,
        "price": price,
        "adx": adx_v
    }

# =========================================================
# CONTROLE DE CICLO / SHUTDOWN
# =========================================================

SHUTDOWN = False

def shutdown_handler(sig, frame):
    global SHUTDOWN
    SHUTDOWN = True
    send_telegram(f"🛑 Scanner finalizado ({now_sp_str()})")
    LOGGER.info("Encerramento solicitado.")

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

# =========================================================
# LOOP PRINCIPAL
# =========================================================

def main():
    send_telegram(f"🤖 Scanner iniciado ({now_sp_str()})")
    time.sleep(15)

    while not SHUTDOWN:
        try:
            for symbol in FIXED_SYMBOLS:
                res = analyze_symbol(symbol)
                if res:
                    msg = (
                        f"🚨 <b>SINAL 15m</b>\n"
                        f"Par: <b>{res['symbol']}</b>\n"
                        f"Lado: <b>{res['side']}</b>\n"
                        f"Preço: {res['price']:.8f}\n"
                        f"ADX: {res['adx']:.2f}\n"
                        f"Hora: {now_sp_str()}"
                    )
                    send_telegram(msg)
                    LOGGER.info("Alerta enviado: %s", res["symbol"])
        except Exception as e:
            LOGGER.exception("Erro no ciclo principal: %s", e)

        time.sleep(POLL_SECONDS)

    LOGGER.info("Scanner encerrado com sucesso.")

if __name__ == "__main__":
    main()
