from fastapi import APIRouter, HTTPException, Request
from app.utils.forecasting_core import forecast_next_day_baseline_from_csv
from app.schemas import ForecastNextDayResponse

router = APIRouter()


@router.get(
    "/forecast/next-day",
    response_model=ForecastNextDayResponse,
    responses={404: {"description": "Building or appliance not found"}},
)
def forecast_next_day(request: Request, building: str, appliance: str):
    cache = request.app.state.forecast_models
    key = (str(building), str(appliance))

    if key not in cache:
        raise HTTPException(status_code=404, detail="Forecast model not found")

    bundle = cache[key]
    sc = bundle["scaler"]

    if not isinstance(sc, dict) or "X" not in sc or "y" not in sc:
        raise HTTPException(status_code=500, detail="Forecast scaler must be a dict with keys 'X' and 'y'.")

    try:
        out = forecast_next_day_baseline_from_csv(
            device_id=str(appliance),
            csv_path=bundle["csv_path"],
            meter_num=bundle["meter_num"],
            model=bundle["model"],
            scaler_bundle=sc,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "building": str(building),
        "appliance": str(appliance),
        "horizon_hours": 24,
        **out,
    }
