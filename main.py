from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import numpy as np
import cv2
import requests
import spaces
import ipaddress
import socket
import os
from bs4 import BeautifulSoup
from transformers import pipeline
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from spam_detection import predict
from image_ocr import extract_text_from_image
from google_factcheck import verify_with_google
from public_sources import verify_with_public_sources

load_dotenv()

app = FastAPI(title="AI Security API")

# inisialisasi model fake news
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
    deep_check: bool = False

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
            "is_fraud": is_phishing,
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
            "is_fraud": is_phishing,
            "confidence": round(score, 4),
            "severity_level": severity,
            "message": message,
            "recommendation": recommendation,
            "extracted_text_preview": extracted_text
        }
    }

# 3 URL Fake News
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGES_TO_OCR = 2
MIN_ARTICLE_TEXT_LENGTH = 80
MAX_FAKE_NEWS_IMAGE_BYTES = 5 * 1024 * 1024

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}

def validate_public_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL wajib diisi.")

    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("URL hanya boleh menggunakan HTTP atau HTTPS.")

    if not parsed.hostname:
        raise ValueError("Format URL tidak valid.")

    if parsed.username or parsed.password:
        raise ValueError("URL dengan kredensial tidak diizinkan.")

    if parsed.port not in (None, 80, 443):
        raise ValueError("Port URL tidak diizinkan.")

    hostname = parsed.hostname.rstrip(".").lower()

    if hostname in ("localhost",) or hostname.endswith(".localhost"):
        raise ValueError("URL menuju alamat lokal tidak diizinkan.")

    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            resolved = socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
            addresses = [
                ipaddress.ip_address(item[4][0])
                for item in resolved
            ]
        except socket.gaierror:
            raise ValueError("Domain tidak dapat ditemukan.")

    if not addresses:
        raise ValueError("Domain tidak memiliki alamat IP yang valid.")

    for address in addresses:
        if not address.is_global:
            raise ValueError(
                "URL menuju alamat IP privat atau nonpublik tidak diizinkan."
            )

    return url

def safe_download(url: str):
    current_url = url

    for _ in range(4):
        current_url = validate_public_url(current_url)

        response = requests.get(
            current_url,
            headers=HTTP_HEADERS,
            timeout=(5, 12),
            allow_redirects=False,
            stream=True,
        )

        try:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")

                if not location:
                    raise ValueError("Redirect tidak memiliki tujuan.")

                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Ukuran konten URL terlalu besar.")
                except ValueError as exc:
                    if "terlalu besar" in str(exc):
                        raise
                    raise ValueError("Header ukuran konten tidak valid.")

            chunks = []
            total_size = 0

            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue

                total_size += len(chunk)
                if total_size > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Ukuran konten URL melebihi batas 5 MB.")

                chunks.append(chunk)

            body = b"".join(chunks)
            content_type = (
                response.headers.get("Content-Type", "")
                .split(";")[0]
                .strip()
                .lower()
            )

            return current_url, content_type, body

        finally:
            response.close()

    raise ValueError("URL mengalami terlalu banyak redirect.")

def extract_text_from_html(html: str, page_url: str):
    soup = BeautifulSoup(html, "html.parser")
    
    for element in soup([
        "script", "style", "noscript", "svg",
        "nav", "footer", "header", "aside", "form"
    ]):
        element.decompose()

    text_parts = []
    
    for selector in [
        'meta[property="og:title"]',
        'meta[name="twitter:title"]',
        'meta[name="title"]',
        'meta[name="description"]',
        'meta[property="og:description"]',
    ]:
        tag = soup.select_one(selector)
        if tag and tag.get("content"):
            text_parts.append(tag["content"].strip())

    if soup.title and soup.title.get_text(strip=True):
        text_parts.insert(0, soup.title.get_text(" ", strip=True))
        
    article_container = soup.find("article") or soup.find("main") or soup
    paragraphs = article_container.find_all("p")

    for paragraph in paragraphs:
        paragraph_text = paragraph.get_text(" ", strip=True)
        if paragraph_text:
            text_parts.append(paragraph_text)
            
    unique_parts = []
    seen = set()

    for part in text_parts:
        part = " ".join(part.split())
        if part and part not in seen:
            seen.add(part)
            unique_parts.append(part)

    article_text = " ".join(unique_parts)
    
    image_urls = []
    
    for selector in [
        'meta[property="og:image"]',
        'meta[name="twitter:image"]',
    ]:
        tag = soup.select_one(selector)
        if tag and tag.get("content"):
            image_urls.append(urljoin(page_url, tag["content"]))

    for img in soup.find_all("img"):
        image_url = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy-src")
        )

        if image_url:
            image_urls.append(urljoin(page_url, image_url))
            
    cleaned_image_urls = []
    seen_images = set()

    for image_url in image_urls:
        if image_url in seen_images:
            continue

        seen_images.add(image_url)
        path = urlparse(image_url).path.lower()

        if path.endswith((".svg", ".gif")):
            continue

        cleaned_image_urls.append(image_url)

    return article_text, cleaned_image_urls

def extract_text_from_remote_image(image_url: str) -> str:
    _, content_type, image_bytes = safe_download(image_url)

    if content_type and not content_type.startswith("image/"):
        return ""

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return ""

    result = extract_text_from_image(image)

    if isinstance(result, str):
        return result.strip()

    return ""

def scrape_article_text(url: str) -> str:
    final_url, content_type, body = safe_download(url)
    
    if content_type.startswith("image/"):
        image_array = np.frombuffer(body, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return ""

        extracted = extract_text_from_image(image)
        return extracted.strip() if isinstance(extracted, str) else ""
    
    if content_type and (
        "html" not in content_type
        and "xhtml" not in content_type
        and "text/plain" not in content_type
    ):
        raise ValueError(
            "URL bukan halaman HTML atau gambar yang didukung."
        )

    html = body.decode("utf-8", errors="replace")
    article_text, image_urls = extract_text_from_html(html, final_url)
    
    if len(article_text.strip()) >= MIN_ARTICLE_TEXT_LENGTH:
        return article_text
    
    ocr_parts = []

    for image_url in image_urls[:MAX_IMAGES_TO_OCR]:
        try:
            image_text = extract_text_from_remote_image(image_url)
            if image_text:
                ocr_parts.append(image_text)
        except Exception as exc:
            print(f"OCR gambar dilewati: {exc}")

    combined_text = " ".join(
        part for part in [article_text, *ocr_parts] if part
    ).strip()

    return combined_text

@spaces.GPU
def predict_fakenews_gpu(text: str):
    if fake_news_classifier is None:
        raise ValueError("Model Fake News tidak tersedia.")

    cleaned_text = " ".join(text.split())

    if len(cleaned_text) < 20:
        raise ValueError("Teks terlalu pendek untuk dianalisis.")

    truncated_text = " ".join(cleaned_text.split()[:200])
    result = fake_news_classifier(truncated_text)[0]

    return result["label"], float(result["score"])


@app.post("/api/detect/fakenews/url")
async def detect_fake_news_url(payload: UrlPayload):
    try:
        article_text = scrape_article_text(payload.url)

        if not article_text or len(article_text.strip()) < 20:
            return JSONResponse(
                status_code=422,
                content={
                    "code": 422,
                    "status": "error",
                    "message": (
                        "Konten tidak berhasil diekstrak. "
                        "Situs mungkin memerlukan login, JavaScript, "
                        "atau tidak menyediakan teks yang dapat dibaca."
                    ),
                    "data": None,
                },
            )

        print("URL:", payload.url)
        print("Panjang teks:", len(article_text))
        print("Teks yang dianalisis:", article_text[:1500])
        print("Label model:", fake_news_classifier.model.config.id2label)

        label, confidence = predict_fakenews_gpu(article_text)
        normalized_label = str(label).strip().upper()

        print("Hasil prediksi:", normalized_label)
        print("Confidence:", confidence)

        if normalized_label in ("HOAKS", "LABEL_1"):
            model_says_hoax = True
        elif normalized_label in ("BUKAN_HOAKS", "LABEL_0"):
            model_says_hoax = False
        else:
            raise ValueError(
                f"Label model tidak dikenal: {label}"
            )

        
        if confidence < 0.80:
            is_hoax = None
            severity = "Perlu verifikasi"
            message = (
                "Model belum cukup yakin untuk menentukan apakah "
                "konten artikel ini hoaks atau bukan."
            )
            recommendation = (
                "Periksa sumber asli, tanggal publikasi, dan "
                "bandingkan informasi dengan beberapa sumber berita "
                "tepercaya sebelum mengambil kesimpulan."
            )
        elif model_says_hoax:
            is_hoax = True
            severity = "Tinggi"
            message = (
                "Model mengindikasikan bahwa artikel ini berpotensi "
                "hoaks. Hasil perlu diverifikasi dengan sumber tepercaya."
            )
            recommendation = (
                "Jangan langsung menyebarkan informasi ini. "
                "Bandingkan klaim artikel dengan beberapa sumber "
                "berita kredibel atau klarifikasi resmi."
            )
        else:
            is_hoax = False
            severity = "Perlu verifikasi"
            message = (
                "Model tidak mengklasifikasikan artikel ini sebagai "
                "hoaks. Hasil ini bukan jaminan bahwa informasi benar."
            )
            recommendation = (
                "Periksa tanggal publikasi, sumber asli, dan konfirmasi "
                "informasi dari beberapa sumber tepercaya."
            )
        
        external_verification = {
            "enabled": payload.deep_check,
            "status": "NOT_REQUESTED",
            "message": "Pemeriksaan eksternal tidak diminta.",
            "results": [],
        }
        
        if payload.deep_check:
            article_query = " ".join(article_text.split())

            query = article_query[:300]
            external_result = verify_with_google(query)

            public_query = article_query[:1500]
            public_result = verify_with_public_sources(
                public_query,
                source_url=payload.url,
            )

            external_verification = {
                "enabled": True,
                **external_result,
                "public_sources": public_result,
            }
    
        return {
            "code": 200,
            "status": "success",
            "message": "Analisis konten berhasil",
            "data": {
                "type": "url",
                "is_fraud": is_hoax,
                "confidence": round(confidence, 4),
                "severity_level": severity,
                "message": message,
                "recommendation": recommendation,
                "extracted_text_preview": article_text[:500],
                "external_verification": external_verification,
            },
        }

    except (ValueError, requests.RequestException) as exc:
        return JSONResponse(
            status_code=422,
            content={
                "code": 422,
                "status": "error",
                "message": f"URL tidak dapat dianalisis: {str(exc)}",
                "data": None,
            },
        )

    except Exception as exc:
        print(f"Error saat analisis URL: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "status": "error",
                "message": "Terjadi kesalahan internal saat menganalisis URL.",
                "data": None,
            },
        )

# 4 Fake news dari gambar
@app.post("/api/detect/fakenews/image")
async def detect_fake_news_image(file: UploadFile = File(...), deep_check: bool = Form(False)):
    try:
        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
        }
        
        if file.content_type not in allowed_types:
            return JSONResponse(
                status_code=415,
                content={
                    "code": 415,
                    "status": "error",
                    "message": (
                        "Format gambar tidak didukung. "
                        "Gunakan JPG, PNG, atau WEBP."
                    ),
                    "data": None,
                },
            )

        image_bytes = await file.read(
            MAX_FAKE_NEWS_IMAGE_BYTES + 1
        )

        if not image_bytes:
            return JSONResponse(
                status_code=400,
                content={
                    "code": 400,
                    "status": "error",
                    "message": "File gambar kosong.",
                    "data": None,
                },
            )

        if len(image_bytes) > MAX_FAKE_NEWS_IMAGE_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "code": 413,
                    "status": "error",
                    "message": "Ukuran gambar maksimal 5 MB.",
                    "data": None,
                },
            )
            
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={
                    "code": 400,
                    "status": "error",
                    "message": "File gambar rusak atau tidak valid.",
                    "data": None,
                },
            )

        extracted_text = extract_text_from_image(image)

        if not isinstance(extracted_text, str):
            extracted_text = ""

        extracted_text = " ".join(extracted_text.split())

        if len(extracted_text) < 20:
            return JSONResponse(
                status_code=422,
                content={
                    "code": 422,
                    "status": "error",
                    "message": (
                        "Teks pada gambar tidak cukup terbaca untuk dianalisis. "
                        "Coba unggah gambar yang lebih jelas atau gambar "
                        "yang memuat teks lebih lengkap."
                    ),
                    "data": None,
                },
            )
        
        label, confidence = predict_fakenews_gpu(extracted_text)
        normalized_label = str(label).strip().upper()

        if normalized_label in ("HOAKS", "LABEL_1"):
            model_says_hoax = True
        elif normalized_label in ("BUKAN_HOAKS", "LABEL_0"):
            model_says_hoax = False
        else:
            raise ValueError(
                f"Label model tidak dikenal: {label}"
            )

        if confidence < 0.80:
            is_hoax = None
            severity = "Perlu verifikasi"
            message = (
                "Model belum cukup yakin untuk menentukan apakah "
                "teks pada gambar berpotensi hoaks atau bukan."
            )
            recommendation = (
                "Periksa sumber asli, tanggal, konteks gambar, dan "
                "bandingkan informasi dengan sumber tepercaya."
            )
        elif model_says_hoax:
            is_hoax = True
            severity = "Tinggi"
            message = (
                "Teks pada gambar terindikasi berpotensi hoaks "
                "berdasarkan prediksi model. Hasil perlu diverifikasi."
            )
            recommendation = (
                "Cari sumber asli dan bandingkan klaim dengan "
                "beberapa sumber berita kredibel atau klarifikasi resmi."
            )
        else:
            is_hoax = False
            severity = "Perlu verifikasi"
            message = (
                "Model tidak mengklasifikasikan artikel ini sebagai hoaks. "
                "Hasil ini bukan jaminan bahwa informasi benar."
            )
            recommendation = (
                "Periksa tanggal publikasi, sumber asli, dan konfirmasi "
                "informasi dari beberapa sumber tepercaya."
            )
            
        external_verification = {
            "enabled": deep_check,
            "status": "NOT_REQUESTED",
            "message": "Pemeriksaan eksternal tidak diminta.",
            "results": [],
        }
        
        
        if deep_check:
            image_query = " ".join(extracted_text.split())

            query = image_query[:300]
            external_result = verify_with_google(query)

            public_query = image_query[:1500]
            public_result = verify_with_public_sources(public_query)

            external_verification = {
                "enabled": True,
                **external_result,
                "public_sources": public_result,
            }

        return {
            "code": 200,
            "status": "success",
            "message": "Analisis screenshot berhasil",
            "data": {
                "type": "image",
                "is_fraud": is_hoax,
                "confidence": round(float(confidence), 4),
                "severity_level": severity,
                "message": message,
                "recommendation": recommendation,
                "extracted_text_preview": extracted_text[:500],
                "external_verification": external_verification,
            },
        }

    except Exception as exc:
        print(f"Error saat analisis screenshot hoaks: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "status": "error",
                "message": (
                    "Terjadi kesalahan internal saat "
                    "menganalisis screenshot."
                ),
                "data": None,
            },
        )

    finally:
        await file.close()