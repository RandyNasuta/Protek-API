from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import re
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

# Inisialisasi model fake news
try:
    print("Memuat model Fake News...")
    fake_news_classifier = pipeline(
        "text-classification",
        model="./model_saved/indobert-hoax-model",
        tokenizer="./model_saved/indobert-hoax-model",
    )
    print("Model Fake News berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model Fake News: {e}")
    fake_news_classifier = None


# Schema payload
class TextPayload(BaseModel):
    text: str


class UrlPayload(BaseModel):
    url: str
    deep_check: bool = False


# ---------------------------------------------------------------------
# Fungsi keputusan fake news
# ---------------------------------------------------------------------
def normalize_factcheck_text(text: str) -> set:
    """Normalisasi sederhana untuk membandingkan klaim dengan artikel."""
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9À-ÿ\s]", " ", text)

    stopwords = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk",
        "pada", "dengan", "adalah", "atau", "dalam", "akan", "telah",
        "ada", "sebagai", "oleh", "karena", "dapat", "bisa", "tidak",
        "bukan", "bahwa", "saat", "sudah", "lebih", "juga", "agar",
        "the", "is", "are", "was", "were", "of", "to", "in", "a",
        "an", "and", "for", "on", "by", "with", "from", "as",
    }
    return {word for word in text.split() if len(word) > 2 and word not in stopwords}


def classify_factcheck_rating(rating: str) -> str:
    """Klasifikasikan rating yang eksplisit; rating ambigu tidak dipaksakan."""
    value = str(rating or "").strip().lower()
    value = re.sub(r"[_-]+", " ", value)
    value = " ".join(value.split())

    hoax_ratings = {
        "false", "mostly false", "pants on fire", "hoax", "hoaks",
        "palsu", "tidak benar", "salah", "fabricated", "incorrect",
        "fake", "disinformasi", "misinformasi",
    }
    true_ratings = {
        "true", "mostly true", "benar", "fakta", "correct", "accurate",
    }

    if value in hoax_ratings:
        return "HOAX"
    if value in true_ratings:
        return "NOT_HOAX"
    return "REVIEW"


def evaluate_google_factcheck(query: str, google_result) -> dict:
    """Nilai hasil Google Fact Check yang relevan dengan teks yang dianalisis."""
    review = {
        "decision": "REVIEW",
        "reason": "Belum ada penilaian faktual yang cukup jelas dan relevan.",
        "evidence": [],
    }

    if not isinstance(google_result, dict):
        review["reason"] = "Format hasil Google Fact Check tidak valid."
        return review

    query_words = normalize_factcheck_text(query)
    if len(query_words) < 3:
        review["reason"] = "Teks klaim terlalu pendek untuk dicocokkan."
        return review

    results = google_result.get("results", [])
    if not isinstance(results, list):
        return review

    evidence = []
    for item in results:
        if not isinstance(item, dict):
            continue

        claim = item.get("claim", "")
        claim_words = normalize_factcheck_text(claim)
        if len(claim_words) < 3:
            continue

        overlap = query_words & claim_words
        # Ini pencocokan heuristik, bukan pengukuran semantik.
        similarity = len(overlap) / max(1, min(len(query_words), len(claim_words)))
        if len(overlap) < 3 or similarity < 0.35:
            continue

        rating = item.get("rating", "")
        evidence.append({
            "claim": claim,
            "rating": rating,
            "decision": classify_factcheck_rating(rating),
            "publisher": item.get("publisher"),
            "url": item.get("url"),
            "similarity": round(similarity, 3),
        })

    decisions = {
        item["decision"] for item in evidence
        if item["decision"] in {"HOAX", "NOT_HOAX"}
    }

    if "HOAX" in decisions and "NOT_HOAX" in decisions:
        return {
            "decision": "REVIEW",
            "reason": "Pemeriksaan fakta yang relevan memberikan penilaian bertentangan.",
            "evidence": evidence,
        }
    if decisions == {"HOAX"}:
        return {
            "decision": "HOAX",
            "reason": "Pemeriksaan fakta yang relevan menilai klaim sebagai salah atau hoaks.",
            "evidence": evidence,
        }
    if decisions == {"NOT_HOAX"}:
        return {
            "decision": "NOT_HOAX",
            "reason": "Pemeriksaan fakta yang relevan menilai klaim sebagai benar.",
            "evidence": evidence,
        }

    if evidence:
        review["reason"] = "Hasil pemeriksaan fakta yang cocok memiliki rating ambigu atau tidak dikenal."
        review["evidence"] = evidence
    return review

def finalize_fake_news_result(
    model_says_hoax: bool,
    confidence: float,
    deep_check: bool,
    google_result=None,
    query: str = "",
    public_result=None,
) -> dict:
    """
    Menentukan hasil akhir dari model, Google Fact Check,
    dan sumber pemeriksaan fakta publik.

    Prediksi model tidak dianggap sebagai verifikasi fakta.
    Keputusan eksternal yang bertentangan menghasilkan REVIEW.
    """
    confidence = float(confidence)

    google_decision = {
        "decision": "REVIEW",
        "reason": "Google Fact Check belum memberikan keputusan yang cukup jelas.",
        "evidence": [],
    }

    public_decision = {
        "decision": "REVIEW",
        "reason": "Sumber pemeriksaan fakta publik belum memberikan keputusan yang cukup jelas.",
        "evidence": [],
    }

    if deep_check:
        google_decision = evaluate_google_factcheck(
            query,
            google_result,
        )

        if isinstance(public_result, dict):
            candidate = public_result.get("factcheck_decision")

            if isinstance(candidate, dict):
                candidate_decision = candidate.get("decision")

                if candidate_decision in {
                    "HOAX",
                    "NOT_HOAX",
                    "REVIEW",
                }:
                    public_decision = candidate

    google_verdict = google_decision.get("decision", "REVIEW")
    public_verdict = public_decision.get("decision", "REVIEW")

    decisive_verdicts = {
        verdict
        for verdict in (google_verdict, public_verdict)
        if verdict in {"HOAX", "NOT_HOAX"}
    }

    # Jika sumber eksternal memberikan keputusan yang bertentangan,
    # jangan memilih salah satunya secara otomatis.
    if len(decisive_verdicts) > 1:
        factcheck = {
            "decision": "REVIEW",
            "reason": (
                "Google Fact Check dan sumber pemeriksaan fakta publik "
                "memberikan keputusan yang bertentangan. Klaim perlu "
                "ditinjau lebih lanjut."
            ),
            "evidence": (
                google_decision.get("evidence", [])
                + public_decision.get("evidence", [])
            ),
            "google_decision": google_decision,
            "public_decision": public_decision,
        }

    elif decisive_verdicts:
        selected_verdict = next(iter(decisive_verdicts))

        selected_decisions = []

        if google_verdict == selected_verdict:
            selected_decisions.append(google_decision)

        if public_verdict == selected_verdict:
            selected_decisions.append(public_decision)

        combined_evidence = []

        for selected in selected_decisions:
            evidence = selected.get("evidence", [])
            if isinstance(evidence, list):
                combined_evidence.extend(evidence)

        selected_reasons = [
            str(item.get("reason", "")).strip()
            for item in selected_decisions
            if item.get("reason")
        ]

        factcheck = {
            "decision": selected_verdict,
            "reason": " ".join(selected_reasons),
            "evidence": combined_evidence,
            "google_decision": google_decision,
            "public_decision": public_decision,
        }

    else:
        factcheck = {
            "decision": "REVIEW",
            "reason": (
                "Pemeriksaan eksternal belum menghasilkan keputusan "
                "faktual yang cukup kuat dan konsisten."
            ),
            "evidence": (
                google_decision.get("evidence", [])
                + public_decision.get("evidence", [])
            ),
            "google_decision": google_decision,
            "public_decision": public_decision,
        }

    external_verdict = factcheck["decision"]

    if deep_check and external_verdict == "HOAX":
        is_hoax = True
        severity = "Tinggi"
        message = (
            "Sumber pemeriksaan fakta eksternal yang relevan menilai "
            "klaim ini salah atau hoaks. Periksa bukti dan konteks "
            "sumber yang disertakan."
        )
        recommendation = (
            "Jangan menyebarkan klaim sebagai fakta. Buka sumber "
            "pemeriksaan fakta yang disertakan dan pastikan klaim, "
            "tanggal, serta konteksnya sesuai."
        )
        decision_basis = "external_fact_check"

    elif deep_check and external_verdict == "NOT_HOAX":
        is_hoax = False
        severity = "Terverifikasi"
        message = (
            "Sumber pemeriksaan fakta eksternal yang relevan mendukung "
            "kebenaran klaim. Pastikan penilaian tersebut sesuai dengan "
            "klaim dan konteks yang sedang diperiksa."
        )
        recommendation = (
            "Baca sumber pemeriksaan fakta dan periksa konteks serta "
            "tanggalnya sebelum menyebarkan informasi."
        )
        decision_basis = "external_fact_check"

    elif deep_check:
        is_hoax = None
        severity = "Perlu verifikasi"
        message = (
            "Pemeriksaan eksternal belum memberikan keputusan yang "
            "cukup kuat atau terdapat perbedaan penilaian. Sistem "
            "belum dapat memastikan apakah klaim benar atau hoaks."
        )
        recommendation = (
            "Periksa sumber asli, bukti, tanggal, dan konteks klaim. "
            "Jika sumber eksternal bertentangan, lakukan pemeriksaan "
            "manual sebelum mengambil kesimpulan."
        )
        decision_basis = "external_inconclusive"

    elif confidence < 0.80:
        is_hoax = None
        severity = "Perlu verifikasi"
        message = (
            "Keyakinan model belum cukup tinggi untuk menentukan "
            "apakah konten ini hoaks atau bukan."
        )
        recommendation = (
            "Periksa sumber asli dan bandingkan klaim dengan "
            "sumber pemeriksaan fakta tepercaya."
        )
        decision_basis = "model_low_confidence"

    elif model_says_hoax:
        is_hoax = True
        severity = "Perlu verifikasi"
        message = (
            "Model mengindikasikan konten ini berpotensi hoaks. "
            "Hasil ini adalah prediksi model, bukan verifikasi fakta."
        )
        recommendation = (
            "Periksa sumber asli dan bukti pendukung sebelum "
            "menyimpulkan atau menyebarkan informasi."
        )
        decision_basis = "model"

    else:
        is_hoax = False
        severity = "Perlu verifikasi"
        message = (
            "Model tidak mengklasifikasikan konten ini sebagai hoaks, "
            "tetapi kebenaran informasinya belum terverifikasi."
        )
        recommendation = (
            "Konfirmasikan klaim melalui sumber tepercaya dan "
            "pemeriksaan fakta yang relevan."
        )
        decision_basis = "model"

    return {
        "is_hoax": is_hoax,
        "confidence": round(confidence, 4),
        "severity_level": severity,
        "message": message,
        "recommendation": recommendation,
        "decision_basis": decision_basis,
        "factcheck_decision": factcheck,
    }

def run_external_verification(query: str, source_url=None) -> dict:
    """Jalankan pemeriksaan eksternal; kegagalan layanan tidak menggagalkan API."""
    try:
        google_result = verify_with_google(query[:300])
        if not isinstance(google_result, dict):
            google_result = {
                "status": "UNAVAILABLE",
                "message": "Format hasil Google Fact Check tidak valid.",
                "results": [],
            }
    except Exception as exc:
        print(f"Google Fact Check gagal: {exc}")
        google_result = {
            "status": "UNAVAILABLE",
            "message": "Google Fact Check tidak dapat diakses.",
            "results": [],
        }

    try:
        if source_url:
            public_result = verify_with_public_sources(
                query[:1500], source_url=source_url
            )
        else:
            public_result = verify_with_public_sources(query[:1500])
    except Exception as exc:
        print(f"Pemeriksaan sumber publik gagal: {exc}")
        public_result = {
            "status": "UNAVAILABLE",
            "message": "Pemeriksaan sumber publik tidak dapat diselesaikan.",
            "sources": {
                "fact_checks": [],
                "news_comparisons": [],
                "official_sources": [],
            },
        }

    return {
        "enabled": True,
        **google_result,
        "public_sources": public_result,
    }, google_result


# ---------------------------------------------------------------------
# 1. Phishing (Text)
# ---------------------------------------------------------------------
@app.post("/api/detect/phishing/text")
async def detect_phishing_text(payload: TextPayload):
    raw_text = payload.text
    try:
        prediction, score = predict(raw_text)
        is_phishing = prediction == "spam"
    except Exception as e:
        print(f"Error saat prediksi teks: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "status": "error",
                "message": f"Gagal memproses analisis teks: {str(e)}",
                "data": None,
            },
        )

    if is_phishing:
        severity = "Tinggi" if score > 0.8 else "Sedang"
        message = "Pesan ini terindikasi kuat sebagai penipuan (Scam/Phishing)."
        recommendation = (
            "Jangan mengklik tautan (link) apa pun atau membagikan data pribadi. "
            "Segera hapus atau blokir pengirim."
        )
    else:
        severity = "Aman"
        message = "Pesan ini terlihat aman dan tidak menunjukkan pola manipulatif."
        recommendation = (
            "Anda dapat membaca atau membalas pesan ini seperti biasa. "
            "Tetap waspada jika pengirim meminta uang."
        )

    return {
        "code": 200,
        "status": "success",
        "message": "Analisis teks berhasil",
        "data": {
            "type": "text",
            "is_fraud": is_phishing,
            "confidence": round(float(score), 4),
            "severity_level": severity,
            "message": message,
            "recommendation": recommendation,
        },
    }


# ---------------------------------------------------------------------
# 2. Phishing (Image)
# ---------------------------------------------------------------------
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
                    "data": None,
                },
            )

        extracted_text = extract_text_from_image(img_cv2)
        if not isinstance(extracted_text, str) or not extracted_text.strip():
            return JSONResponse(
                status_code=422,
                content={
                    "code": 422,
                    "status": "error",
                    "message": "Tidak ada teks yang dapat diekstrak dari gambar.",
                    "data": None,
                },
            )

        prediction, score = predict(extracted_text)
        is_phishing = prediction == "spam"

    except Exception as e:
        print(f"Error saat memproses gambar: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "status": "error",
                "message": f"Terjadi kesalahan saat memproses gambar: {str(e)}",
                "data": None,
            },
        )

    if is_phishing:
        severity = "Tinggi" if score > 0.8 else "Sedang"
        message = "Pesan ini terindikasi kuat sebagai penipuan (Scam/Phishing)."
        recommendation = (
            "Jangan mengklik tautan (link) apa pun atau membagikan data pribadi. "
            "Segera hapus atau blokir pengirim."
        )
    else:
        severity = "Aman"
        message = "Pesan ini terlihat aman dan tidak menunjukkan pola manipulatif."
        recommendation = (
            "Anda dapat membaca atau membalas pesan ini seperti biasa. "
            "Tetap waspada jika pengirim meminta uang."
        )

    return {
        "code": 200,
        "status": "success",
        "message": "Analisis gambar berhasil",
        "data": {
            "type": "image",
            "is_fraud": is_phishing,
            "confidence": round(float(score), 4),
            "severity_level": severity,
            "message": message,
            "recommendation": recommendation,
            "extracted_text_preview": extracted_text,
        },
    }


# ---------------------------------------------------------------------
# 3. Fake news dari URL
# ---------------------------------------------------------------------
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
    if hostname == "localhost" or hostname.endswith(".localhost"):
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
            addresses = [ipaddress.ip_address(item[4][0]) for item in resolved]
        except socket.gaierror:
            raise ValueError("Domain tidak dapat ditemukan.")

    if not addresses:
        raise ValueError("Domain tidak memiliki alamat IP yang valid.")
    for address in addresses:
        if not address.is_global:
            raise ValueError("URL menuju alamat IP privat atau nonpublik tidak diizinkan.")
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
        "script", "style", "noscript", "svg", "nav", "footer",
        "header", "aside", "form",
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
    for paragraph in article_container.find_all("p"):
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
        image_url = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
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
    return result.strip() if isinstance(result, str) else ""


def scrape_article_text(url: str) -> str:
    final_url, content_type, body = safe_download(url)
    if content_type.startswith("image/"):
        image_array = np.frombuffer(body, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return ""
        extracted = extract_text_from_image(image)
        return extracted.strip() if isinstance(extracted, str) else ""

    if content_type and not (
        "html" in content_type
        or "xhtml" in content_type
        or "text/plain" in content_type
    ):
        raise ValueError("URL bukan halaman HTML atau gambar yang didukung.")

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

    return " ".join(part for part in [article_text, *ocr_parts] if part).strip()


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


def interpret_model_label(label: str) -> bool:
    normalized_label = str(label).strip().upper()
    if normalized_label in ("HOAKS", "LABEL_1"):
        return True
    if normalized_label in ("BUKAN_HOAKS", "LABEL_0"):
        return False
    raise ValueError(f"Label model tidak dikenal: {label}")


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
                        "Konten tidak berhasil diekstrak. Situs mungkin memerlukan "
                        "login, JavaScript, atau tidak menyediakan teks yang dapat dibaca."
                    ),
                    "data": None,
                },
            )

        print("URL:", payload.url)
        print("Panjang teks:", len(article_text))
        print("Teks yang dianalisis:", article_text[:1500])
        if fake_news_classifier is not None:
            print("Label model:", fake_news_classifier.model.config.id2label)

        label, confidence = predict_fakenews_gpu(article_text)
        model_says_hoax = interpret_model_label(label)
        print("Hasil prediksi:", str(label).strip().upper())
        print("Confidence:", confidence)

        external_verification = {
            "enabled": False,
            "status": "NOT_REQUESTED",
            "message": "Pemeriksaan eksternal tidak diminta.",
            "results": [],
        }
        google_result = None
        article_query = " ".join(article_text.split())

        if payload.deep_check:
            external_verification, google_result = run_external_verification(
                article_query, source_url=payload.url
            )

        final_result = finalize_fake_news_result(
            model_says_hoax=model_says_hoax,
            confidence=confidence,
            deep_check=payload.deep_check,
            google_result=google_result,
            query=article_query[:300],
            public_result=external_verification.get("public_sources"),
        )
        external_verification["factcheck_decision"] = final_result["factcheck_decision"]

        return {
            "code": 200,
            "status": "success",
            "message": "Analisis konten berhasil",
            "data": {
                "type": "url",
                "is_fraud": final_result["is_hoax"],
                "confidence": final_result["confidence"],
                "severity_level": final_result["severity_level"],
                "message": final_result["message"],
                "recommendation": final_result["recommendation"],
                "decision_basis": final_result["decision_basis"],
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


# ---------------------------------------------------------------------
# 4. Fake news dari gambar
# ---------------------------------------------------------------------
@app.post("/api/detect/fakenews/image")
async def detect_fake_news_image(
    file: UploadFile = File(...),
    deep_check: bool = Form(False),
):
    try:
        allowed_types = {"image/jpeg", "image/png", "image/webp"}
        if file.content_type not in allowed_types:
            return JSONResponse(
                status_code=415,
                content={
                    "code": 415,
                    "status": "error",
                    "message": "Format gambar tidak didukung. Gunakan JPG, PNG, atau WEBP.",
                    "data": None,
                },
            )

        image_bytes = await file.read(MAX_FAKE_NEWS_IMAGE_BYTES + 1)
        if not image_bytes:
            return JSONResponse(
                status_code=400,
                content={"code": 400, "status": "error", "message": "File gambar kosong.", "data": None},
            )
        if len(image_bytes) > MAX_FAKE_NEWS_IMAGE_BYTES:
            return JSONResponse(
                status_code=413,
                content={"code": 413, "status": "error", "message": "Ukuran gambar maksimal 5 MB.", "data": None},
            )

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"code": 400, "status": "error", "message": "File gambar rusak atau tidak valid.", "data": None},
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
                        "Coba unggah gambar yang lebih jelas atau gambar yang memuat teks lebih lengkap."
                    ),
                    "data": None,
                },
            )

        label, confidence = predict_fakenews_gpu(extracted_text)
        model_says_hoax = interpret_model_label(label)
        image_query = " ".join(extracted_text.split())

        external_verification = {
            "enabled": False,
            "status": "NOT_REQUESTED",
            "message": "Pemeriksaan eksternal tidak diminta.",
            "results": [],
        }
        google_result = None
        if deep_check:
            external_verification, google_result = run_external_verification(
                image_query
            )

        final_result = finalize_fake_news_result(
            model_says_hoax=model_says_hoax,
            confidence=confidence,
            deep_check=deep_check,
            google_result=google_result,
            query=image_query[:300],
            public_result=external_verification.get("public_sources"),
        )
        external_verification["factcheck_decision"] = final_result["factcheck_decision"]

        return {
            "code": 200,
            "status": "success",
            "message": "Analisis screenshot berhasil",
            "data": {
                "type": "image",
                "is_fraud": final_result["is_hoax"],
                "confidence": final_result["confidence"],
                "severity_level": final_result["severity_level"],
                "message": final_result["message"],
                "recommendation": final_result["recommendation"],
                "decision_basis": final_result["decision_basis"],
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
                "message": "Terjadi kesalahan internal saat menganalisis screenshot.",
                "data": None,
            },
        )
    finally:
        await file.close()
