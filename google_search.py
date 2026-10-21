
import os
import requests
from urllib.parse import urlparse

GOOGLE_SEARCH_URL = (
    "https://www.googleapis.com/customsearch/v1"
)


def search_google(query: str, limit: int = 5) -> dict:
    api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
    engine_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")

    if not api_key or not engine_id:
        return {
            "status": "UNAVAILABLE",
            "message": (
                "Google Custom Search API key atau Search Engine ID belum dikonfigurasi."
            ),
            "results": [],
        }

    if not query or not query.strip():
        return {
            "status": "UNAVAILABLE",
            "message": "Kueri pencarian kosong.",
            "results": [],
        }

    try:
        response = requests.get(
            GOOGLE_SEARCH_URL,
            params={
                "key": api_key,
                "cx": engine_id,
                "q": query[:500],
                "num": max(1, min(limit, 10)),
                "lr": "lang_id",
                "gl": "id",
                "safe": "active",
            },
            timeout=15,
        )

        if response.status_code in (401, 403):
            print(
                "Google Custom Search API menolak akses:",
                response.status_code,
                response.text[:500],
            )
            return {
                "status": "UNAVAILABLE",
                "message": (
                    "Akses Google Custom Search ditolak. Periksa API key, aktivasi API, dan konfigurasi mesin pencari."
                ),
                "results": [],
            }

        if response.status_code == 429:
            return {
                "status": "UNAVAILABLE",
                "message": (
                    "Kuota atau batas permintaan Google Custom Search telah tercapai."
                ),
                "results": [],
            }

        response.raise_for_status()
        data = response.json()

        results = []

        for item in data.get("items", []):
            url = item.get("link", "")
            parsed = urlparse(url)

            # Hanya tampilkan URL web HTTP/HTTPS.
            if parsed.scheme not in ("http", "https"):
                continue

            if not parsed.hostname:
                continue

            results.append({
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", ""),
                "domain": parsed.hostname.lower(),
            })

        return {
            "status": (
                "MATCHES_FOUND" if results else "NO_MATCHES"
            ),
            "message": (
                "Hasil pencarian ditemukan."
                if results
                else "Tidak ditemukan hasil pencarian."
            ),
            "results": results,
        }

    except requests.Timeout:
        return {
            "status": "UNAVAILABLE",
            "message": "Google Custom Search mengalami timeout.",
            "results": [],
        }

    except requests.RequestException as exc:
        print(f"Google Custom Search error: {exc}")
        return {
            "status": "UNAVAILABLE",
            "message": (
                "Google Custom Search tidak dapat diakses."
            ),
            "results": [],
        }

    except ValueError as exc:
        print(f"Google Custom Search response error: {exc}")
        return {
            "status": "UNAVAILABLE",
            "message": "Respons Google Custom Search tidak valid.",
            "results": [],
        }


def verify_with_public_sources(query: str) -> dict:
    if not query or not query.strip():
        return {
            "status": "UNAVAILABLE",
            "message": "Teks untuk pencarian sumber tidak tersedia.",
            "sources": {
                "news_comparisons": [],
                "official_sources": [],
            },
        }

    # Batasi panjang teks yang dikirim sebagai kueri.
    query = " ".join(query.split())[:250]

    news_result = search_google(
        f'{query} (site:kompas.com OR site:tempo.co '
        'OR site:antaranews.com OR site:liputan6.com)',
        limit=5,
    )

    official_result = search_google(
        f'{query} (site:go.id OR site:polri.go.id '
        'OR site:kejaksaan.go.id)',
        limit=5,
    )

    sources = {
        "news_comparisons": news_result["results"],
        "official_sources": official_result["results"],
    }

    statuses = [
        news_result["status"],
        official_result["status"],
    ]

    if all(status == "UNAVAILABLE" for status in statuses):
        status = "UNAVAILABLE"
        message = "Pencarian berita dan sumber resmi gagal."
    elif any(status == "MATCHES_FOUND" for status in statuses):
        status = "MATCHES_FOUND"
        message = (
            "Pencarian selesai. Cocokkan klaim, tanggal, dan isi sumber sebelum menarik kesimpulan."
        )
    elif all(status == "NO_MATCHES" for status in statuses):
        status = "NO_MATCHES"
        message = (
            "Pencarian selesai, tetapi tidak ada hasil yang cocok dari kueri dan mesin pencari ini."
        )
    else:
        status = "PARTIAL"
        message = (
            "Sebagian pencarian tidak berhasil. Hasil yang tersedia tetap ditampilkan."
        )

    return {
        "status": status,
        "message": message,
        "sources": sources,
    }
