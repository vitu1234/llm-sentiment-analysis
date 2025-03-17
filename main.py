from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from swiftmind.predictor import create_model, create_tokenizer, predict

model = create_model()
tokenizer = create_tokenizer()


class PredictionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 16
    return_full_text: bool = False


class Prediction(BaseModel):
    content: str


app = FastAPI(
    title="SwiftMind API",
    description="Rest API for serving LLM model predictions",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers[
        "Strict-Transport-Security"
    ] = "max-age=63072000; includeSubDomains"
    # response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


@app.route("/heartbeat")
async def heartbeat():
    return {"status": "healthy"}


@app.post("/predict", response_model=Prediction, status_code=200)
async def make_prediction(request: PredictionRequest):
    try:
        prediction = predict(
            request.prompt,
            model,
            tokenizer,
            request.max_new_tokens,
            request.return_full_text,
        )
        return Prediction(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))