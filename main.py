from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from collections import deque
from datetime import datetime
from model import load_model, predict
import base64

app = FastAPI(title="Bird Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model, CLASS_NAMES = load_model()
history = deque(maxlen=15)

@app.get("/")
def root():
    return {"status": "ok", "message": "Bird Classifier API is running!"}

@app.post("/predict")
async def classify_bird(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 10MB)")

    results = predict(model, CLASS_NAMES, image_bytes, top_k=5)
    top     = results[0]
    wiki    = await get_wikipedia(top["raw_label"])
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    history.appendleft({
        "id":         len(history) + 1,
        "timestamp":  datetime.now().strftime("%H:%M · %d %b"),
        "bird":       top["label"],
        "confidence": top["confidence"],
        "image_b64":  img_b64,
        "mime":       file.content_type,
    })

    return {
        "predictions": results,
        "wikipedia":   wiki,
        "top_label":   top["label"],
        "top_conf":    top["confidence"],
    }

@app.get("/history")
def get_history():
    return {"history": list(history)}

@app.get("/health")
def health():
    return {"status": "healthy", "classes": len(CLASS_NAMES)}

async def get_wikipedia(raw_label):
    query = raw_label.replace("_", " ").title()
    url   = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title":   data.get("title", query),
                "summary": data.get("extract", "")[:400] + "...",
                "url":     data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "image":   data.get("thumbnail", {}).get("source", None)
            }
    except Exception:
        pass
    return {
        "title":   query,
        "summary": "Wikipedia info unavailable.",
        "url":     f"https://en.wikipedia.org/wiki/{query}",
        "image":   None
    }