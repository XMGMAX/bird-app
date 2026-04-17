from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from collections import deque
from datetime import datetime
from model import load_model, predict
import base64
import asyncio

app = FastAPI(title="Bird Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

history = deque(maxlen=15)

_model = None
_class_names = None
_model_error = None
_model_lock = asyncio.Lock()


async def ensure_model_loaded(force_retry: bool = False):
    global _model, _class_names, _model_error

    if _model is not None and _class_names is not None:
        return _model, _class_names

    # Avoid repeated blocking downloads on every request after a failure.
    if _model_error is not None and not force_retry:
        raise RuntimeError(_model_error)

    async with _model_lock:
        if _model is not None and _class_names is not None:
            return _model, _class_names
        if _model_error is not None and not force_retry:
            raise RuntimeError(_model_error)

        try:
            _model, _class_names = await asyncio.to_thread(load_model)
            _model_error = None
            return _model, _class_names
        except Exception as exc:
            _model_error = f"Model initialization failed: {exc}"
            raise RuntimeError(_model_error) from exc


@app.on_event("startup")
async def warmup_model():
    # Best effort only: keep API alive even if model download fails.
    try:
        await ensure_model_loaded(force_retry=True)
    except Exception:
        pass


@app.get("/")
def root():
    return {"status": "ok", "message": "Bird Classifier API is running!"}


@app.post("/predict")
async def classify_bird(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 10MB)")

    try:
        model, class_names = await ensure_model_loaded()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is currently unavailable. "
                "Please try again in a minute. "
                f"Reason: {exc}"
            ),
        ) from exc

    results = predict(model, class_names, image_bytes, top_k=5)
    top = results[0]
    wiki = await get_wikipedia(top["raw_label"])
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    history.appendleft(
        {
            "id": len(history) + 1,
            "timestamp": datetime.now().strftime("%H:%M · %d %b"),
            "bird": top["label"],
            "confidence": top["confidence"],
            "image_b64": img_b64,
            "mime": file.content_type,
        }
    )

    return {
        "predictions": results,
        "wikipedia": wiki,
        "top_label": top["label"],
        "top_conf": top["confidence"],
    }


@app.get("/history")
def get_history():
    return {"history": list(history)}


@app.get("/health")
def health():
    loaded = _class_names is not None
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "classes": len(_class_names) if loaded else 0,
        "startup_error": _model_error,
    }


async def get_wikipedia(raw_label):
    query = raw_label.replace("_", " ").title()
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title": data.get("title", query),
                "summary": data.get("extract", "")[:400] + "...",
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "image": data.get("thumbnail", {}).get("source", None),
            }
    except Exception:
        pass
    return {
        "title": query,
        "summary": "Wikipedia info unavailable.",
        "url": f"https://en.wikipedia.org/wiki/{query}",
        "image": None,
    }
