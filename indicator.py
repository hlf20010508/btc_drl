import pandas as pd
import talib


def add_indicator(data: pd.DataFrame):
    ema(7, data)
    ema(14, data)
    ema(21, data)

    rsi(7, data)
    rsi(14, data)
    rsi(21, data)

    macd(data)
    kdj(data)
    obv(data)
    bollinger(data)
    vegas(data)


def ema(period: int, df: pd.DataFrame):
    emas = talib.EMA(df["Close"], timeperiod=period)

    df["EMA" + str(period)] = emas


def rsi(period: int, df: pd.DataFrame):
    rsis = talib.RSI(df["Close"], timeperiod=period)

    df["RSI" + str(period)] = rsis


def macd(df: pd.DataFrame):
    df["MACD"], df["MACD-signal"], df["MACD-hist"] = talib.MACD(
        df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )


def kdj(df: pd.DataFrame):
    df["KDJ-K"], df["KDJ-D"] = talib.STOCH(
        df["High"],
        df["Low"],
        df["Close"],
        fastk_period=9,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    df["KDJ-J"] = 3 * df["KDJ-K"] - 2 * df["KDJ-D"]


def obv(df: pd.DataFrame):
    df["OBV"] = talib.OBV(df["Close"], df["Volume"])


def bollinger(df: pd.DataFrame):
    df["BBANDS-upper"], df["BBANDS-middle"], df["BBANDS-lower"] = talib.BBANDS(
        df["Close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )


def vegas(df: pd.DataFrame):
    df["VEGAS-filter"] = talib.EMA(df["Close"], timeperiod=12)
    df["VEGAS-tunnel1-upper"] = talib.EMA(df["Close"], timeperiod=144)
    df["VEGAS-tunnel1-lower"] = talib.EMA(df["Close"], timeperiod=169)
    df["VEGAS-tunnel2-upper"] = talib.EMA(df["Close"], timeperiod=576)
    df["VEGAS-tunnel2-lower"] = talib.EMA(df["Close"], timeperiod=676)
