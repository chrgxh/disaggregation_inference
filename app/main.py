from fastapi import FastAPI
from tensorflow.keras.models import load_model
import joblib

from app.utils.config_loader import load_config, load_forecasting_config
from app.api.disaggregation import router as disaggregation_router
from app.api.forecasting import router as forecasting_router

app = FastAPI(title="Disaggregation Inference API")

app.include_router(disaggregation_router)
app.include_router(forecasting_router)


@app.on_event("startup")
def startup():
    # Disaggregation
    cfg = load_config("app/configs/disaggregation_config.yaml")
    app.state.cfg = cfg

    model_cache = {}
    for b_id, b in cfg.buildings.items():
        for a_key, a in b.appliances.items():
            key = (str(b_id), str(a_key))
            model_cache[key] = {
                "classifier": load_model(str(a.classifier_path), compile=False),
                "regressor": load_model(str(a.regressor_path), compile=False),
            }

    app.state.models = model_cache

    # Forecasting
    fcfg = load_forecasting_config("app/configs/forecasting_config.yaml")
    app.state.forecasting_cfg = fcfg

    forecast_cache = {}
    for b_id, b in fcfg.buildings.items():
        for a_key, a in b.appliances.items():
            key = (str(b_id), str(a_key))
            forecast_cache[key] = {
                "model": load_model(str(a.model_path), compile=False),
                "scaler": joblib.load(a.scaler_path),
                "meter_num": a.meter_num,
                "csv_path": b.csv_path,
            }

    app.state.forecast_models = forecast_cache


@app.get("/health")
def health():
    return {"status": "ok"}
