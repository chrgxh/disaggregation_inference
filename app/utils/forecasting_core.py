import os
import numpy as np
import pandas as pd

from app.utils.data_loader import load_power_csv_as_is, power_df_to_hourly_kwh_dt

TZ_NAME = os.getenv("HERON_TZ", "Europe/Athens")


def add_fourier(x: pd.DataFrame, period: int, K: int, col_prefix: str) -> pd.DataFrame:
    for k in range(1, K + 1):
        x[f"{col_prefix}_sin_{k}"] = np.sin(2 * np.pi * k * x["t"] / period)
        x[f"{col_prefix}_cos_{k}"] = np.cos(2 * np.pi * k * x["t"] / period)
    return x


def make_features_v2(df: pd.DataFrame, detrend: bool = True) -> pd.DataFrame:
    x = df.copy()
    x["ts"] = pd.to_datetime(x["ts"], utc=True)

    loc = x["ts"].dt.tz_convert(TZ_NAME)
    x["hour"] = loc.dt.hour
    x["dow"] = loc.dt.dayofweek
    x["is_weekend"] = (x["dow"] >= 5).astype(int)
    x["month"] = loc.dt.month
    x["day"] = loc.dt.day

    x["is_peak_hour"] = ((x["hour"] >= 9) & (x["hour"] <= 21)).astype(int)

    x["t"] = (x["ts"].dt.tz_convert("UTC").view("int64") // 10**9) // 3600
    x = add_fourier(x, period=24, K=4, col_prefix="day")
    x = add_fourier(x, period=168, K=3, col_prefix="week")
    x = add_fourier(x, period=8760, K=2, col_prefix="year")

    x["roll24_mean"] = x["energy"].rolling(24, min_periods=6).mean()
    x["roll24_std"] = x["energy"].rolling(24, min_periods=6).std()
    x["roll168_mean"] = x["energy"].rolling(168, min_periods=24).mean()
    x["roll168_std"] = x["energy"].rolling(168, min_periods=24).std()
    x["roll720_mean"] = x["energy"].rolling(720, min_periods=100).mean()
    x["roll720_std"] = x["energy"].rolling(720, min_periods=100).std()

    if detrend:
        trend = x["roll168_mean"].fillna(x["energy"].mean())
        x["energy_detrended"] = x["energy"] - trend

    for l in range(1, 24):
        x[f"lag_{l}"] = x["energy"].shift(l)
    x["lag_24"] = x["energy"].shift(24)
    x["lag_168"] = x["energy"].shift(168)

    x["lag_24_diff"] = x["energy"].diff(24)
    x["lag_168_diff"] = x["energy"].diff(168)

    x["hour_dow_int"] = x["hour"] * x["dow"]
    x["peak_hour_std"] = x["is_peak_hour"] * x["roll24_std"].fillna(0)

    return x


def feature_columns_v2(xdf: pd.DataFrame) -> list[str]:
    base_cols = [
        "energy_detrended",
        "hour", "dow", "is_weekend", "month", "day",
        "is_peak_hour",
        "roll24_mean", "roll24_std", "roll168_mean", "roll168_std",
        "roll720_mean", "roll720_std",
        "lag_24", "lag_168", "lag_24_diff", "lag_168_diff",
        "hour_dow_int", "peak_hour_std",
    ]

    fourier_cols = sorted([c for c in xdf.columns if c.startswith(("day_", "week_", "year_"))])
    lag_cols = [f"lag_{i}" for i in range(1, 24)]
    return base_cols + fourier_cols + lag_cols


def load_hourly_kwh_history(
    device_id: str,
    csv_path,
    meter_num: int,
    history_hours: int,
) -> pd.DataFrame:
    df_power = load_power_csv_as_is(
        device_id=device_id,
        csv_path=csv_path,
        meter_num=meter_num,
        last_hours=history_hours,
    )
    return power_df_to_hourly_kwh_dt(df_power)


def build_baseline_features(
    df_hourly: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    xdf = make_features_v2(df_hourly).dropna().reset_index(drop=True)
    use_cols = feature_columns_v2(xdf)
    feat = xdf[use_cols].values.astype("float32")
    return feat, xdf


def forecast_baseline_24h_gru(
    feat: np.ndarray,
    model,
    sx,
    sy,
) -> list[float]:
    if getattr(sx, "n_features_in_", None) != feat.shape[1]:
        raise ValueError(
            f"GRU feature count mismatch: scaler expects {sx.n_features_in_}, built {feat.shape[1]}"
        )

    feat_s = sx.transform(feat)

    if len(feat_s) < 48:
        raise ValueError("Not enough history for GRU input window (need 48h).")

    Xseq = feat_s[-48:, :]
    preds = []
    for _ in range(24):
        y_hat_s = model.predict(Xseq[None, ...], verbose=0)
        y_hat = sy.inverse_transform(y_hat_s)[0, 0]
        preds.append(float(y_hat))
        Xseq = np.vstack([Xseq, Xseq[-1, :]])[-48:, :]

    return preds


def make_forecast_timestamps_utc(last_ts_utc: pd.Timestamp) -> list[str]:
    ts_utc = pd.date_range(
        start=pd.to_datetime(last_ts_utc, utc=True) + pd.Timedelta(hours=1),
        periods=24,
        freq="H",
        tz="UTC",
    )
    return [str(t) for t in ts_utc]


def forecast_next_day_baseline_from_csv(
    device_id: str,
    csv_path,
    meter_num: int,
    model,
    scaler_bundle,
    history_hours: int = 900,
) -> dict:
    sx = scaler_bundle["X"]
    sy = scaler_bundle["y"]

    hourly = load_hourly_kwh_history(device_id, csv_path, meter_num, history_hours=history_hours)
    feat, xdf = build_baseline_features(hourly)

    preds = forecast_baseline_24h_gru(feat, model, sx, sy)

    last_ts_utc = pd.to_datetime(xdf["ts"].iloc[-1], utc=True)
    return {
        "last_ts_utc": str(last_ts_utc),
        "ts_utc": make_forecast_timestamps_utc(last_ts_utc),
        "predicted_baseline_kwh": preds,
    }
