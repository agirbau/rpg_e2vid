from fastapi import FastAPI, HTTPException, Request, Response
from contextlib import asynccontextmanager
import asyncio
import msgpack
import msgpack_numpy as mnp
import uvicorn
import gc
from pathlib import Path

import argparse
import torch
from utils.loading_utils import load_model as e2vid_load_model
from utils.loading_utils import get_device as e2vid_get_device
import numpy as np
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options

mnp.patch() # extend msgpack to understand numpy arrays

parser = argparse.ArgumentParser(
description='Evaluating a trained network')
parser.add_argument('-c', '--path_to_model', default="pretrained/E2VID_lightweight.pth.tar", type=str,
                    help='path to model weights')
parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
parser.set_defaults(fixed_duration=False)
parser.add_argument('-N', '--window_size', default=None, type=int,
                    help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                    help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                    help='in case N (window size) is not specified, it will be \
                            automatically computed as N = width * height * num_events_per_pixel')
parser.add_argument('--skipevents', default=0, type=int)
parser.add_argument('--suboffset', default=0, type=int)
parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
parser.set_defaults(compute_voxel_grid_on_cpu=False)

set_inference_options(parser)

args_parsed, _ = parser.parse_known_args() # ← ignore unrecognized args

# 1. Initialize the server
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INIT] Setting up E2VID inference server...")

    app.state.model = None
    app.state.reconstructor = None
    app.state.device = None

    app.state.args = {
        "args_parsed": args_parsed,
        "device": "cuda",
        "height": 480,
        "width": 640,
        "use_cuda": True,
        "model_path_default": "pretrained/E2VID_lightweight.pth.tar",
    }

    yield

    print("[CLEANUP] Releasing resources...")
    app.state.model = None
    app.state.reconstructor = None

app = FastAPI(lifespan=lifespan, title="E2VID Inference Server", version="1.0")


# 2. Load and unload models
@app.post("/load_model")
def load_model(request: Request, model_path: str = "pretrained/E2VID_lightweight.pth.tar", width: int = 640, height: int = 480):
    """
    Loads the E2VID model from the specified path.
    Example:
      curl -X POST "http://localhost:8000/load_model?model_path=pretrained/E2VID_lightweight.pth.tar"
    """
    app = request.app

    if getattr(app.state, "model", None) is not None:
        raise HTTPException(status_code=400, detail="Model already loaded.")

    path = Path(model_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")

    try:
        # Load the model and initialize reconstructor
        print(f"[INFO] Loading E2VID model from: {model_path}")

        # --- actual model loading logic ---
        model = e2vid_load_model(model_path)
        device = e2vid_get_device(True)
        print(f"[INFO] Using device: {device}")
        model = model.to(device)
        model.eval()

        # --- create image reconstructor ---
        num_bins = getattr(model, "num_bins", 5)
        args = app.state.args.get("args_parsed")
        
        reconstructor = ImageReconstructor(model, height, width, num_bins, args)

        # --- store everything in app.state ---
        app.state.model = model
        app.state.device = device
        app.state.reconstructor = reconstructor

        print("[INFO] Model successfully loaded.")
        return {"status": "ok", "message": f"Model loaded from {model_path}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.post("/unload_model")
def unload_model(request: Request):
    """
    Unloads the currently loaded E2VID model and frees GPU memory.
    Example:
      curl -X POST "http://localhost:8000/unload_model"
    """
    app = request.app

    # Check if a model is loaded
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=400, detail="No model currently loaded.")

    try:
        print("[INFO] Unloading E2VID model...")

        # --- Explicit cleanup ---
        if hasattr(app.state, "reconstructor") and app.state.reconstructor is not None:
            del app.state.reconstructor
            app.state.reconstructor = None

        # Free model and device
        model = app.state.model
        del model
        app.state.model = None

        # If using CUDA, free GPU memory
        if app.state.device == "cuda":
            torch.cuda.empty_cache()

        print("[INFO] Model successfully unloaded.")
        return {"status": "ok", "message": "Model unloaded successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {e}")
    

@app.post("/infer")
def infer(request: Request):
    """
    Performs inference using the loaded E2VID model on the provided event data.
    The event data should be sent as a MessagePack-encoded binary payload with keys 't', 'x', 'y', 'p'.
    """

    app = request.app
    model = app.state.model
    reconstructor = app.state.reconstructor
    device = app.state.device
    width, height = app.state.args.get("width"), app.state.args.get("height")
    args = app.state.args.get("args_parsed")

    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")

    # Read raw binary body
    body = asyncio.run(request.body())

    # Decode MessagePack into a Python dict
    event_window = msgpack.unpackb(body, object_hook=mnp.decode, raw=False)

    if args.compute_voxel_grid_on_cpu:
        event_tensor = events_to_voxel_grid(event_window,
                                            num_bins=model.num_bins,
                                            width=width,
                                            height=height)
        event_tensor = torch.from_numpy(event_tensor)
    else:
        event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                    num_bins=model.num_bins,
                                                    width=width,
                                                    height=height,
                                                    device=device)

    num_events_in_window = event_window.shape[0]
    out = reconstructor.server_update_reconstruction(event_tensor, 0)

    print(f"[INFO] Processed {num_events_in_window} events, output shape: {out.shape}")

    # Encode result as MessagePack (NumPy array preserved)
    resp_payload = msgpack.packb(
        {
            "status": "ok",
            "num_events": num_events_in_window,
            "output": out,
        },
        default=mnp.encode,
    )

    return Response(content=resp_payload, media_type="application/msgpack")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)