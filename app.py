from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

print("در حال بارگذاری مدل امبدینگ...")
device = "cpu"
model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device=device)
print(f"مدل با موفقیت روی '{device}' بارگذاری شد.")


class EmbeddingRequest(BaseModel):
    text: str


# --- 3. ساخت اپلیکیشن FastAPI ---
app = FastAPI()


# --- 4. تعریف Endpoint برای امبد کردن ---
@app.post("/embed/")
def embed_sentence(request: EmbeddingRequest):
    """
    یک جمله را دریافت کرده و وکتور امبدینگ آن را برمی‌گرداند.
    """
    # امبد کردن لحظه‌ای کوئری کاربر (این فرآیند روی CPU سریع است)
    embedding = model.encode(request.text)
    
    # تبدیل نتیجه به لیست پایتون برای سازگاری با JSON
    return {"text": request.text, "embedding": embedding.tolist()}

@app.get("/")
def read_root():
    return {"status": "OK", "message": "Embedding server is running."}
