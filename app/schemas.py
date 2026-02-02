from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any

class SeriesOut(BaseModel):
    timestamp: List[str]
    prob: List[float]
    pred_label: List[int]
    pred_w: List[float]

class DisaggregateResponse(BaseModel):
    building: int
    appliance: str
    days_back: int
    meta: Dict[str, Any]
    series: SeriesOut

class ForecastNextDayResponse(BaseModel):
    building: int
    appliance: str
    horizon_hours: int
    last_ts_utc: str
    ts_utc: List[str]
    predicted_baseline_kwh: List[float]