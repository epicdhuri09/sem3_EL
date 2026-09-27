from fastapi import FastAPI, HTTPException, Request

app = FastAPI(title="Satellite AI Service")


@app.get("/health")
def health():
    return {
        "status": "running",
        "model_ready": False
    }


@app.post("/agent/segment")
@app.post("/agent/anomaly")
@app.post("/agent/changepoint")
@app.post("/agent/structures")
@app.post("/agent/xai")
@app.post("/agent/cause")
@app.post("/agent/priority")
def pending_analysis(request: Request, payload: dict):
    print(f"Received request: {request.url.path}", flush=True)

    raise HTTPException(
        status_code=501,
        detail="AI endpoint available, analysis not implemented yet."
    )