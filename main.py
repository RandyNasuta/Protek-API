from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import numpy as np
import cv2
from spam_detection import predict
from image_ocr import extract_text_from_image
import requests
from bs4 import BeautifulSoup
import spaces
from transformers import pipeline
from urllib.parse import urlparse

app = FastAPI(title="AI Security API")

# inisialisasi modek fake news
try:
    print("Memuat model Fake News...")
    fake_news_classifier = pipeline(
        "text-classification", 
        model="./model_saved/indobert-hoax-model", 
        tokenizer="./model_saved/indobert-hoax-model"
    )
    print("Model Fake News berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model Fake News: {e}")
    fake_news_classifier = None
    
# Schema Payload
class TextPayload(BaseModel):
    text: str

class UrlPayload(BaseModel):
    url: str

# 1 Phishing (Text)
@app.post("/api/detect/phishing/text")
async def detect_phishing_text(payload: TextPayload):
    raw_text = payload.text
    
    try:
        prediction, score = predict(raw_text)
        is_phishing = (prediction == 'spam')
    except Exception as e:
        print(f"Error saat prediksi teks: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "status": "error",
                "message": f"Gagal memproses analisis teks: {str(e)}",
                "data": None
            }
        )
    
    if is_phishing:
        severity = "Tinggi" if score > 0.8 else "Sedang"
        message = "Pesan ini terindikasi kuat sebagai penipuan (Scam/Phishing)."
        recommendation = "Jangan mengklik tautan (link) apa pun atau membagikan data pribadi. Segera hapus atau blokir pengirim."
    else:
        severity = "Aman"
        message = "Pesan ini terlihat aman dan tidak menunjukkan pola manipulatif."
        recommendation = "Anda dapat membaca atau membalas pesan ini seperti biasa. Tetap waspada jika pengirim meminta uang."
    
    return {
        "code": 200,
        "status": "success",
        "message": "Analisis teks berhasil",
        "data": {
            "type": "text",
            "is_phishing": is_phishing,
            "confidence": round(score, 4),
            "severity_level": severity,
            "message": message,
            "recommendation": recommendation
        }
    }

# 2 Phishing (Image)
@app.post("/api/detect/phishing/image")
async def detect_phishing_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            return JSONResponse(
                status_code=400,
                content={
                    "code": 400,
                    "status": "error",
                    "message": "Format file tidak didukung atau gambar rusak.",
                    "data": None
                }
            )
        
        extracted_text = extract_text_from_image(img_cv2)
        
        
        if not isinstance(extracted_text, str) or not extracted_text.strip():
            return JSONResponse(
                status_code=422,
                content={
                    "code": 422,
                    "status": "error",
                    "message": "Tidak ada teks yang dapat diekstrak dari gambar.",
                    "data": None
                }
            )
        
        prediction, score = predict(extracted_text)
        is_phishing = (prediction == 'spam')
    
    except Exception as e:
        print(f"Error saat memproses gambar: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "status": "error",
                "message": f"Terjadi kesalahan saat memproses gambar: {str(e)}",
                "data": None
            }
        )
        
    if is_phishing:
        severity = "Tinggi" if score > 0.8 else "Sedang"
        message = "Pesan ini terindikasi kuat sebagai penipuan (Scam/Phishing)."
        recommendation = "Jangan mengklik tautan (link) apa pun atau membagikan data pribadi. Segera hapus atau blokir pengirim."
    else:
        severity = "Aman"
        message = "Pesan ini terlihat aman dan tidak menunjukkan pola manipulatif."
        recommendation = "Anda dapat membaca atau membalas pesan ini seperti biasa. Tetap waspada jika pengirim meminta uang."
            
    return {
        "code": 200,
        "status": "success",
        "message": "Analisis gambar berhasil",
        "data": {
            "type": "image",
            "is_phishing": is_phishing,
            "confidence": round(score, 4),
            "severity_level": severity,
            "message": message,
            "recommendation": recommendation,
            "extracted_text_preview": extracted_text
        }
    }

# 3 URL Fake News
def scrape_article_text(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15,
        allow_redirects=True,
    )
    response.raise_for_status()

    print("URL tujuan:", response.url)
    print("Status HTTP:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))

@spaces.GPU
def predict_fakenews_gpu(text: str):
    if not fake_news_classifier:
        raise ValueError("Pipeline model Fake News tidak tersedia.")
    
    truncated_text = " ".join(text.split()[:200])
    result = fake_news_classifier(truncated_text)[0]
    return result['label'], result['score']

@app.post("/api/detect/fakenews/url")
async def detect_fake_news_url(payload: UrlPayload):
    try:
        article_text = scrape_article_text(payload.url)
        
        if not article_text or not article_text.strip():
            return JSONResponse(
                status_code=422,
                content={
                    "code": 422,
                    "status": "error",
                    "message": "Teks artikel tidak berhasil diekstrak dari URL.",
                    "data": None
                }
            )
        
        label, confidence = predict_fakenews_gpu(article_text)
        
        is_hoax = (label == 'HOAKS')
        severity = "Tinggi" if is_hoax else "Aman"
        message = "Peringatan: Tautan ini terindikasi kuat memuat Berita Bohong (Hoaks/Disinformasi)." if is_hoax else "Berita/Postingan ini cenderung menyajikan fakta yang sah atau merupakan klarifikasi resmi."
        recommendation = "Jangan sebarkan tautan ini tanpa mencari pembanding dari portal berita kredibel." if is_hoax else "Anda dapat membaca dan membagikan informasi ini."
            
        return {
            "code": 200,
            "status": "success",
            "message": "Analisis berita berhasil",
            "data": {
                "type": "url",
                "is_phishing": is_hoax,
                "confidence": round(confidence, 4),
                "severity_level": severity,
                "message": message,
                "recommendation": recommendation,
                "extracted_text_preview": article_text[:200] + "..."
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "code": 500, 
            "status": "error", 
            "message": f"Terjadi kesalahan saat memproses URL: {str(e)}", 
            "data": None
        })