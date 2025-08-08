import pandas as pd
import ta

def prepare_features(df, prefix):
    df = df.copy()
    df['returns'] = df['Close'].pct_change()
    df['volatility_7d'] = df['returns'].rolling(7).std()
    df['volatility_14d'] = df['returns'].rolling(14).std()
    df['volume_ma7'] = df['Volume'].rolling(7).mean()
    df['momentum_7d'] = df['Close'].diff(7)
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()

    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    df['rsi'] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
    macd = ta.trend.MACD(close_series)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    df = df.add_prefix(f'{prefix}_')

    exclude_cols = {f'{prefix}_Open', f'{prefix}_High', f'{prefix}_Low', f'{prefix}_Close', f'{prefix}_Adj Close', f'{prefix}_Volume'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return df[feature_cols]
