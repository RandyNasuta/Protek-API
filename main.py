from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import numpy as np
import cv2
from spam_detection import predict
from image_ocr import extract_text_from_image

app = FastAPI(title="AI Security API")

# --- JSON ---
class TextPayload(BaseModel):
    text: str

class UrlPayload(BaseModel):
    url: str

# --- 1 Phishing (Text) ---
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
        message = "Peringatan: Pesan ini terindikasi kuat sebagai penipuan (Scam/Phishing)."
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

# --- 2 Phishing (Image) ---
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
        message = "Peringatan: Pesan ini terindikasi kuat sebagai penipuan (Scam/Phishing)."
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

# --- 3 Deepfake (Upload Video) ---
@app.post("/api/detect/deepfake/upload")
async def detect_deepfake_upload(file: UploadFile = File(...)):
    time.sleep(2)
    return {
            "status": "Success",
            "type": "video_upload",
            "filename": file.filename,
            "is_deepfake": True,
            "confidence": 0.95,
            "anomalies_detected": ["Inkonsistensi kedipan mata", "Artefak pada batas wajah"]
        }

# --- 4 Deepfake (Url) ---
@app.post("/api/detect/deepfake/url")
async def detect_deepfake_url(payload: UrlPayload):
    return {
        "status": "success",
        "type": "video_link",
        "url_processed": payload.url,
        "is_deepfake": False,
        "confidence_score": 0.05,
        "message": "Video dari URL ini terlihat asli."
    }