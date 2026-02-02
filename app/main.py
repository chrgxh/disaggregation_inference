from fastapi import FastAPI
from tensorflow.keras.models import load_model

from app.utils.config_loader import load_config
from app.api.disaggregation import router as disaggregation_router

app = FastAPI(title="Disaggregation Inference API")
app.include_router(disaggregation_router)

@app.on_event("startup")
def startup():
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

@app.get("/health")
def health():
    return {"status": "ok"}
