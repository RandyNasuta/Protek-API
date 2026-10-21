
import os
import requests

GOOGLE_FACTCHECK_URL = (
    "https://factchecktools.googleapis.com/"
    "v1alpha1/claims:search"
)


def verify_with_google(query: str, limit: int = 5) -> dict:
    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")

    if not api_key:
        return {
            "status": "UNAVAILABLE",
            "message": "Google Fact Check API key belum dikonfigurasi.",
            "results": [],
        }

    if not query or not query.strip():
        return {
            "status": "UNAVAILABLE",
            "message": "Kueri pencarian tidak tersedia.",
            "results": [],
        }

    try:
        response = requests.get(
            GOOGLE_FACTCHECK_URL,
            params={
                "key": api_key,
                "query": query[:500],
                "languageCode": "id",
                "pageSize": limit,
            },
            timeout=15,
        )

        if response.status_code in (401, 403):
            return {
                "status": "UNAVAILABLE",
                "message": (
                    "Google menolak akses. Periksa API key, aktivasi API, dan izin proyek."
                ),
                "results": [],
            }

        response.raise_for_status()
        data = response.json()

        results = []

        for claim in data.get("claims", []):
            for review in claim.get("claimReview", []):
                publisher = review.get("publisher") or {}

                results.append({
                    "claim": claim.get("text"),
                    "claim_date": claim.get("claimDate"),
                    "review_title": review.get("title"),
                    "publisher": publisher.get("name"),
                    "rating": review.get("textualRating"),
                    "review_date": review.get("reviewDate"),
                    "url": review.get("url"),
                })

                if len(results) >= limit:
                    break

            if len(results) >= limit:
                break

        if results:
            return {
                "status": "MATCHES_FOUND",
                "message": (
                    "Ditemukan pemeriksaan fakta yang perlu dicocokkan dengan klaim dalam artikel."
                ),
                "results": results,
            }

        return {
            "status": "NO_MATCHES",
            "message": (
                "Tidak ditemukan pemeriksaan fakta yang cocok dalam hasil Google Fact Check API."
            ),
            "results": [],
        }

    except requests.Timeout:
        return {
            "status": "UNAVAILABLE",
            "message": "Google Fact Check API mengalami timeout.",
            "results": [],
        }

    except requests.RequestException as exc:
        print(f"Google Fact Check API error: {exc}")
        return {
            "status": "UNAVAILABLE",
            "message": "Google Fact Check API tidak dapat diakses.",
            "results": [],
        }

    except ValueError as exc:
        print(f"Google Fact Check response error: {exc}")
        return {
            "status": "UNAVAILABLE",
            "message": "Respons Google Fact Check tidak valid.",
            "results": [],
        }