from fastapi import FastAPI, HTTPException, Request
import asyncio
import msgpack
import msgpack_numpy as mnp
import uvicorn
import torch
import gc
from pathlib import Path

import numpy as np

mnp.patch() # extend msgpack to understand numpy arrays

app = FastAPI(title="E2VID Inference Server", version="1.0")

# Global model and device
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.get("/hello_get")
def root():
    return {"message": "Hello from e2vid inference server!"}


@app.post("/hello_post")
def infer():
    return {"message": "Hello from e2vid 2 inference server!"}


@app.post("/params")
def receive_params(request: Request):
    # Read raw binary body
    body = asyncio.run(request.body())

    # Decode MessagePack into a Python dict
    data = msgpack.unpackb(body, object_hook=mnp.decode, raw=False)

    # Convert lists to numpy arrays for easy numerical work
    x, y, t, p = data["x"], data["y"], data["t"], data["p"]

    return {
        "message": f"ACK for arrays: x({x}), y({y}), t({t}), p({p})"
    }

@app.post("/load_model")
def load_model(model_path: str):
    """
    Loads the E2VID model from the specified path.
    Example:
      curl -X POST "http://localhost:8000/load_model?model_path=pretrained/E2VID_lightweight.pth.tar"
    """
    global model

    if model is not None:
        raise HTTPException(status_code=400, detail="Model already loaded.")

    path = Path(model_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")

    try:
        # TODO: Replace this stub with actual E2VID model loading logic.
        # For now we simulate loading.
        print(f"[INFO] Loading E2VID model from: {model_path}")
        model = torch.load(model_path, map_location=device)
        if hasattr(model, "eval"):
            model.eval()
        print("[INFO] Model successfully loaded.")
        return {"status": "ok", "message": f"Model loaded from {model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload_model")
def unload_model():
    """
    Unloads the currently loaded model from memory.
    Example:
      curl -X POST "http://localhost:8000/unload_model"
    """
    global model

    if model is None:
        raise HTTPException(status_code=400, detail="No model currently loaded.")

    try:
        print("[INFO] Unloading model and freeing GPU memory.")
        del model
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "ok", "message": "Model unloaded and memory cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)