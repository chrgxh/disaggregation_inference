from fastapi import FastAPI

app = FastAPI(title="Disaggregation Inference API")

@app.get("/health")
def health():
    return {"status": "ok"}
