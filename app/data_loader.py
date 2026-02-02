import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union

def load_main_power_csv_as_is(
    device_id: str,
    csv_path: Union[str, Path],
    meter_num: int = 0,
    ts_col: str = "timestamp",
    last_hours: Optional[int] = None,
    rows_per_hour: int = 5000,
) -> pd.DataFrame:
    power_col = "power_data_main" if meter_num == 0 else f"power_data_meter_{meter_num}"

    csv_path = Path(csv_path)

    if last_hours is None:
        df = pd.read_csv(csv_path, usecols=[ts_col, power_col])
    else:
        tail_rows = rows_per_hour * last_hours
        df = pd.read_csv(csv_path, usecols=[ts_col, power_col]).tail(tail_rows)

    df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df["power"] = pd.to_numeric(df[power_col], errors="coerce")
    df = df.dropna(subset=["ts", "power"]).sort_values("ts").drop_duplicates("ts")

    if last_hours is not None and not df.empty:
        end = df["ts"].iloc[-1]
        start = end - pd.Timedelta(hours=last_hours)
        df = df[df["ts"] >= start]

    df = df.reset_index(drop=True)
    df["device_id"] = device_id
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
