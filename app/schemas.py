from pydantic import BaseModel
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
