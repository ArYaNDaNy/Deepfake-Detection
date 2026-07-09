from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_headers = ["*"],
    allow_methods = ["*"]
)

@app.get("/")
async def root():
    return {
        "status": "online", 
        "message": "Deepfake Detection API is running.",
        "active_models": ["Vision Transformer", "OpenCV Heuristics", "Audio DSP"]
    }

app.include_router(router)