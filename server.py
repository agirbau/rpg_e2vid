from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from e2vid 3.0 inference server!"}

if __name__ == "__main__":
    # TODO: adapt "run_reconstruction.py" logic to serve requests
    uvicorn.run(app, host="0.0.0.0", port=8000)