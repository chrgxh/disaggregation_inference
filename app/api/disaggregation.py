from fastapi import APIRouter, Query, HTTPException, Request
import numpy as np
import pandas as pd

from app.utils.data_loader import load_power_csv_as_is
from app.utils.inference_core import reconstruct_from_mains
from app.schemas import DisaggregateResponse

router = APIRouter(tags=["disaggregation"])

@router.get("/disaggregate", response_model=DisaggregateResponse,responses={404: {"description": "Building or appliance not found"},})
def disaggregate(
    request: Request,
    building: int = Query(..., ge=1, description="Building id as in config.yaml"),
    appliance: str = Query(..., min_length=1, description="Appliance key as in config.yaml"),
    days_back: int = Query(1, ge=1, le=90, description="How many days back from latest timestamp"),
):
    cfg = request.app.state.cfg

    b_id = str(building)

    if b_id not in cfg.buildings:
        raise HTTPException(status_code=404, detail=f"Unknown building: {building}")

    bcfg = cfg.buildings[b_id]

    if appliance not in bcfg.appliances:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown appliance '{appliance}' for building {building}"
        )

    acfg = bcfg.appliances[appliance]

    #Load last N days from CSV
    last_hours = int(days_back) * 24

    rows_per_hour = acfg.rows_per_hour or bcfg.rows_per_hour
    max_mains_w = acfg.max_mains_w or bcfg.max_mains_w

    #this could read from a live database
    df = load_power_csv_as_is(
        device_id=f"building_{building}",
        csv_path=str(bcfg.csv_path),
        last_hours=last_hours,
        rows_per_hour=rows_per_hour,
    )

    if df.empty:
        raise HTTPException(status_code=422, detail="No data available for requested range.")

    mains_w = df["power"].to_numpy(dtype=np.float32)
    ts = df["ts"].to_numpy()

    # Get cached models
    key = (b_id, appliance)
    models = request.app.state.models.get(key)
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded for this building/appliance.")

    clf = models["classifier"]
    reg = models["regressor"]

    out_df, meta = reconstruct_from_mains(
        mains_w=mains_w,
        ts=ts,
        window_size=acfg.window_size,
        classifier=clf,
        clf_threshold=acfg.clf_threshold,
        regressor=reg,
        regressor_output_unit=acfg.regressor_output_unit,
        max_mains_w=max_mains_w,
    )

    return {
        "building": building,
        "appliance": appliance,
        "days_back": days_back,
        "meta": meta,
        "series": {
            "timestamp": out_df["timestamp"].astype(str).tolist(),
            "prob": out_df["prob"].tolist(),
            "pred_label": out_df["pred_label"].tolist(),
            "pred_w": out_df["pred_regressor_w"].tolist(),
        }
    }
