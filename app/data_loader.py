import pandas as pd
import numpy as np

def load_main_power_csv_as_is(
    device_id: str,
    csv_path: str,
    ts_col: str = "timestamp",
    power_col: str = "power_data_main",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if ts_col not in df.columns or power_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {ts_col}, {power_col}")

    df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df["power"] = pd.to_numeric(df[power_col], errors="coerce")
    df = df.dropna(subset=["ts", "power"]).copy()

    df["device_id"] = device_id
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df[["device_id", "ts", "power"]]

def power_df_to_hourly_kwh_dt(
    df_power: pd.DataFrame,
    max_dt_seconds: float = 60.0,
) -> pd.DataFrame:
    """
    dt-aware conversion: integrates power(W) over time -> hourly kWh.
    max_dt_seconds caps gaps so missing data doesn't explode energy.
    """
    if not {"device_id", "ts", "power"}.issubset(df_power.columns):
        raise ValueError("df_power must contain columns: device_id, ts, power")

    x = df_power.copy()
    x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
    x["power"] = pd.to_numeric(x["power"], errors="coerce")
    x = x.dropna(subset=["ts", "power"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    if len(x) < 2:
        raise ValueError("Need at least 2 samples to compute energy.")

    dt = x["ts"].diff().dt.total_seconds().to_numpy()

    if np.isnan(dt[0]):
        dt[0] = dt[1] if len(dt) > 1 and np.isfinite(dt[1]) and dt[1] > 0 else 1.0

    dt = np.clip(dt, 0.0, float(max_dt_seconds))
    x["energy_wh"] = x["power"].to_numpy(dtype=float) * dt / 3600.0

    hourly = (
        x.set_index("ts")["energy_wh"]
         .resample("H")
         .sum()
         .reset_index()
         .rename(columns={"energy_wh": "energy_wh"})
    )
    hourly["energy"] = hourly["energy_wh"] / 1000.0
    hourly = hourly.drop(columns=["energy_wh"])
    hourly["device_id"] = x["device_id"].iloc[0]
    hourly = hourly.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    return hourly[["device_id", "ts", "energy"]]
