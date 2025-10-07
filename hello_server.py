from fastapi import FastAPI, Request
import asyncio
import msgpack
import msgpack_numpy as mnp
import uvicorn

mnp.patch() # extend msgpack to understand numpy arrays

app = FastAPI(title="Hello Inference Server", version="1.0")

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

if __name__ == "__main__":
    # TODO: adapt "run_reconstruction.py" logic to serve requests
    uvicorn.run(app, host="0.0.0.0", port=8000)