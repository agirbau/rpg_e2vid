from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Hello Inference Server", version="1.0")

@app.get("/hello_get")
def root():
    return {"message": "Hello from hello 'get' inference server!"}


@app.post("/hello_post")
def infer():
    return {"message": "Hello from hello 'post' inference server!"}

if __name__ == "__main__":
    # TODO: adapt "run_reconstruction.py" logic to serve requests
    uvicorn.run(app, host="0.0.0.0", port=8000)