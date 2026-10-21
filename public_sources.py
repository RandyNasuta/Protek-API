
import re
import requests
import feedparser

from urllib.parse import urlencode, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed


HEADERS = {"User-Agent": "ProtekFactCheck/1.0"}
TIMEOUT = 12

RSS_FEEDS = {
    "fact_checks": [
        {
            "name": "TurnBackHoax",
            "url": "https://turnbackhoax.id/feed",
        },
    ],
    "news_comparisons": [
        {
            "name": "ANTARA News",
            "url": "https://www.antaranews.com/rss/terkini.xml",
        },
    ],
}

NEWS_DOMAINS = [
    "kompas.com",
    "tempo.co",
    "liputan6.com",
]

OFFICIAL_DOMAINS = [
    "ojk.go.id",
    "idx.co.id",
    "msci.com",
    "lpsk.go.id",
    "kejaksaan.go.id",
    "polri.go.id",
    "mahkamahagung.go.id",
    "putusan.mahkamahagung.go.id",
]

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan",
    "ini", "itu", "adalah", "akan", "atau", "dalam", "oleh", "karena",
    "sebagai", "setelah", "terhadap", "hingga", "bagi", "masih", "sudah",
    "telah", "dapat", "menjadi", "tersebut", "para", "secara", "setiap",
    "saat", "serta", "namun", "juga", "agar", "antara", "sejak", "sampai",
    "lebih", "kurang", "bisa", "ada", "jadi", "sangat", "paling", "hanya",
    "the", "a", "an", "is", "of", "to", "in", "and", "for", "on", "with",
    "by", "from", "was", "were", "are", "as", "at", "be",
    "berita", "artikel", "klaim", "menurut", "terbaru", "terkini",
    "hari", "kumpulan", "viral", "media", "sosial", "mengatakan",
    "menyebutkan", "disebut", "terkait", "terjadi", "informasi",
}


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    replacements = {
        "&amp;": " ",
        "&quot;": " ",
        "&#39;": " ",
        "&nbsp;": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_keywords(text, max_words=100):
    words = normalize_text(text).split()
    selected = []
    seen = set()

    for word in words:
        if len(word) < 3 or word in STOPWORDS:
            continue

        if word not in seen:
            selected.append(word)
            seen.add(word)

        if len(selected) >= max_words:
            break

    return selected


def keyword_score(query, title, summary=""):
    query_words = set(extract_keywords(query, max_words=200))
    document_words = set(
        extract_keywords(f"{title} {summary}", max_words=200)
    )

    if not query_words or not document_words:
        return 0.0

    matched = query_words & document_words

    if not matched:
        return 0.0

    query_coverage = len(matched) / len(query_words)
    document_precision = len(matched) / len(document_words)
    denominator = query_coverage + document_precision

    if denominator == 0:
        return 0.0

    return round(
        2 * query_coverage * document_precision / denominator,
        3,
    )


def _valid_http_url(url):
    try:
        parsed = urlparse(str(url or "").strip())
        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
        )
    except (TypeError, ValueError):
        return False


def _clean_snippet(text):
    text = re.sub(r"<[^>]+>", " ", str(text or ""))

    replacements = {
        "&amp;": "&",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return " ".join(text.split())[:700]


def read_rss_feed(feed_config, query, limit=10):
    try:
        response = requests.get(
            feed_config["url"],
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        response.raise_for_status()

        parsed = feedparser.parse(response.content)
        results = []

        for entry in parsed.entries[:60]:
            title = (entry.get("title") or "").strip()
            url = (entry.get("link") or "").strip()
            summary = _clean_snippet(entry.get("summary") or "")
            published = (
                entry.get("published")
                or entry.get("updated")
                or ""
            )

            if not title or not _valid_http_url(url):
                continue

            score = keyword_score(query, title, summary)

            # Sedikit lebih longgar untuk RSS pemeriksaan fakta.
            if score < 0.15:
                continue

            results.append({
                "title": title,
                "url": url,
                "publisher": feed_config["name"],
                "published": published,
                "snippet": summary,
                "relevance_score": score,
                "category": "fact_checks",
            })

        results.sort(
            key=lambda item: item["relevance_score"],
            reverse=True,
        )

        return {
            "status": "MATCHES_FOUND" if results else "NO_MATCHES",
            "results": results[:limit],
        }

    except Exception as exc:
        print(
            f"[PUBLIC SOURCES] RSS error "
            f"({feed_config['name']}): {exc}"
        )
        return {"status": "UNAVAILABLE", "results": []}


def search_google_news_rss(query, domain, limit=5):
    search_terms = " ".join(str(query or "").split()).strip()

    if not search_terms:
        return {"status": "NO_MATCHES", "results": []}

    search_query = f"site:{domain} {search_terms}"

    params = urlencode({
        "q": search_query,
        "hl": "id",
        "gl": "ID",
        "ceid": "ID:id",
    })

    url = f"https://news.google.com/rss/search?{params}"

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        response.raise_for_status()

        parsed = feedparser.parse(response.content)
        results = []

        for entry in parsed.entries[:25]:
            title = (entry.get("title") or "").strip()
            article_url = (entry.get("link") or "").strip()
            summary = _clean_snippet(entry.get("summary") or "")
            published = (
                entry.get("published")
                or entry.get("updated")
                or ""
            )

            if not title or not _valid_http_url(article_url):
                continue

            source_info = entry.get("source", {})
            publisher = ""

            if hasattr(source_info, "get"):
                publisher = (source_info.get("title") or "").strip()

            if not publisher:
                publisher = domain

            score = keyword_score(query, title, summary)

            if score < 0.15:
                continue

            results.append({
                "title": title,
                "url": article_url,
                "publisher": publisher,
                "published": published,
                "snippet": summary,
                "relevance_score": score,
                "category": (
                    "official_sources"
                    if domain in OFFICIAL_DOMAINS
                    else "news_comparisons"
                ),
            })

        results.sort(
            key=lambda item: item["relevance_score"],
            reverse=True,
        )

        return {
            "status": "MATCHES_FOUND" if results else "NO_MATCHES",
            "results": results[:limit],
        }

    except Exception as exc:
        print(
            f"[PUBLIC SOURCES] Google News RSS error "
            f"({domain}): {exc}"
        )
        return {"status": "UNAVAILABLE", "results": []}


def normalize_article_url(url):
    if not _valid_http_url(url):
        return ""

    parsed = urlparse(str(url).strip())
    hostname = (parsed.hostname or "").lower()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return f"{hostname}{parsed.path.rstrip('/')}".lower()


def is_same_article(item, query, source_url=None):
    result_url = normalize_article_url(item.get("url", ""))
    original_url = normalize_article_url(source_url)

    if original_url and result_url and result_url == original_url:
        return True

    title = (item.get("title") or "").strip()

    if " - " in title:
        title = title.rsplit(" - ", 1)[0].strip()

    normalized_title = normalize_text(title)
    normalized_query = normalize_text(query)

    return (
        len(normalized_title) >= 35
        and normalized_query.startswith(normalized_title)
    )


def is_generic_listing_title(title):
    normalized = normalize_text(title)

    generic_patterns = [
        "artikel terbaru terkini",
        "hari ini artikel terbaru",
        "berita terbaru hari ini",
        "kumpulan artikel",
        "indeks berita",
    ]

    return any(
        pattern in normalized
        for pattern in generic_patterns
    )


def deduplicate_items(items, category, query, source_url=None):
    unique_items = []
    seen_urls = set()
    seen_titles = set()

    for item in items:
        title = item.get("title", "") or ""

        if category == "news_comparisons":
            if is_same_article(item, query, source_url):
                continue

            if is_generic_listing_title(title):
                continue

        url_key = normalize_article_url(item.get("url", ""))
        title_key = normalize_text(title)
        publisher_key = normalize_text(item.get("publisher", ""))

        if url_key and url_key in seen_urls:
            continue

        duplicate_key = (title_key, publisher_key)

        if title_key and duplicate_key in seen_titles:
            continue

        if url_key:
            seen_urls.add(url_key)

        if title_key:
            seen_titles.add(duplicate_key)

        unique_items.append(item)

    unique_items.sort(
        key=lambda item: item.get("relevance_score", 0),
        reverse=True,
    )

    return unique_items[:20]


def assess_factcheck_item(item, query):
    """
    Mengidentifikasi hasil pemeriksaan fakta yang memiliki
    label eksplisit dan kemiripan yang memadai.

    Skor kemiripan hanya mengukur relevansi teks, bukan
    probabilitas bahwa suatu klaim benar atau salah.
    """
    title = str(item.get("title") or "")
    snippet = str(item.get("snippet") or "")
    publisher = str(item.get("publisher") or "")

    combined = f"{title} {snippet}"
    normalized_title = normalize_text(title)
    normalized_combined = normalize_text(combined)

    query_keywords = set(extract_keywords(query, max_words=200))
    source_keywords = set(
        extract_keywords(combined, max_words=200)
    )

    matched_keywords = query_keywords & source_keywords

    overlap = (
        len(matched_keywords) / max(len(query_keywords), 1)
    )

    # Sumber harus berasal dari daftar pemeriksaan fakta tepercaya
    # yang dikonfigurasi di atas.
    trusted_publisher = (
        normalize_text(publisher) == normalize_text("TurnBackHoax")
        or "turnbackhoax.id" in str(item.get("url") or "").lower()
    )

    false_patterns = [
        r"\bsalah\b",
        r"\bhoaks\b",
        r"\bhoax\b",
        r"\bkeliru\b",
        r"\bdisinformasi\b",
        r"\bmenyesatkan\b",
        r"\bfaktanya tidak benar\b",
    ]

    true_patterns = [
        r"\bbenar\b",
        r"\bfakta\b",
        r"\bterbukti benar\b",
        r"\btidak menyesatkan\b",
    ]

    mixed_patterns = [
        r"\bsebagian benar\b",
        r"\bsebagian salah\b",
        r"\bkurang tepat\b",
        r"\bperlu diluruskan\b",
    ]

    # Label harus muncul di judul, agar kata "salah" yang kebetulan
    # muncul di isi artikel tidak otomatis menjadi sebuah vonis.
    is_false = any(
        re.search(pattern, normalized_title)
        for pattern in false_patterns
    )

    is_true = any(
        re.search(pattern, normalized_title)
        for pattern in true_patterns
    )

    is_mixed = any(
        re.search(pattern, normalized_title)
        for pattern in mixed_patterns
    )

    # Tolak kecocokan lemah. Harus ada setidaknya tiga kata
    # kunci yang sama dan overlap minimum 0,25.
    relevant = (
        len(matched_keywords) >= 3
        and overlap >= 0.25
    )

    decision = "REVIEW"
    reason = (
        "Sumber belum cukup cocok dengan klaim atau "
        "tidak memiliki label pemeriksaan fakta yang tegas."
    )

    if trusted_publisher and relevant:
        if is_mixed:
            decision = "REVIEW"
            reason = (
                "Artikel pemeriksaan fakta relevan, tetapi "
                "judul menunjukkan penilaian campuran atau parsial."
            )
        elif is_false and not is_true:
            decision = "HOAX"
            reason = (
                "Sumber pemeriksaan fakta tepercaya memiliki label "
                "yang menunjukkan klaim salah dan relevansi teks "
                "memenuhi ambang pencocokan."
            )
        elif is_true and not is_false:
            decision = "NOT_HOAX"
            reason = (
                "Sumber pemeriksaan fakta tepercaya memiliki label "
                "yang menunjukkan klaim benar dan relevansi teks "
                "memenuhi ambang pencocokan."
            )

    return {
        "decision": decision,
        "reason": reason,
        "keyword_overlap": round(overlap, 3),
        "matched_keyword_count": len(matched_keywords),
        "matched_keywords": sorted(matched_keywords)[:20],
        "trusted_publisher": trusted_publisher,
        "explicit_false_label": is_false,
        "explicit_true_label": is_true,
        "explicit_mixed_label": is_mixed,
        "source": {
            "title": title,
            "url": item.get("url", ""),
            "publisher": publisher,
            "published": item.get("published", ""),
            "snippet": snippet,
        },
    }


def evaluate_factcheck_sources(factcheck_items, query):
    """
    Menghasilkan keputusan berdasarkan artikel pemeriksaan fakta
    yang lolos pemeriksaan relevansi.

    Bila ditemukan konflik atau lebih dari satu keputusan yang
    berlawanan, hasilnya REVIEW.
    """
    assessments = []

    for item in factcheck_items:
        assessment = assess_factcheck_item(item, query)
        item["match_assessment"] = (
            "POTENTIAL_CLAIM_MATCH"
            if assessment["decision"] != "REVIEW"
            else "RELATED_TOPIC"
        )
        item["keyword_overlap"] = assessment["keyword_overlap"]
        item["matched_keywords"] = assessment["matched_keywords"]
        item["factcheck_assessment"] = assessment
        assessments.append(assessment)

    decisive = [
        item for item in assessments
        if item["decision"] in ("HOAX", "NOT_HOAX")
    ]

    decisions = {item["decision"] for item in decisive}

    if len(decisions) > 1:
        return {
            "decision": "REVIEW",
            "reason": (
                "Ditemukan hasil pemeriksaan fakta yang memberikan "
                "keputusan berbeda. Klaim perlu ditinjau manual."
            ),
            "evidence": [
                item["source"] for item in decisive
            ],
        }

    if len(decisions) == 1:
        selected_decision = next(iter(decisions))
        selected = [
            item for item in decisive
            if item["decision"] == selected_decision
        ]

        selected.sort(
            key=lambda item: (
                item["keyword_overlap"],
                item["matched_keyword_count"],
            ),
            reverse=True,
        )

        best = selected[0]

        return {
            "decision": selected_decision,
            "reason": best["reason"],
            "evidence": [
                item["source"] for item in selected[:5]
            ],
            "keyword_overlap": best["keyword_overlap"],
        }

    return {
        "decision": "REVIEW",
        "reason": (
            "Belum ditemukan artikel pemeriksaan fakta tepercaya "
            "yang cocok dan memiliki label keputusan eksplisit. "
            "Tidak ditemukannya kecocokan bukan bukti bahwa klaim benar."
        ),
        "evidence": [],
    }


def verify_with_public_sources(query, source_url=None):
    empty_sources = {
        "fact_checks": [],
        "news_comparisons": [],
        "official_sources": [],
    }

    if not query or not str(query).strip():
        return {
            "status": "UNAVAILABLE",
            "verification_level": "NONE",
            "message": "Teks pencarian tidak tersedia.",
            "sources": empty_sources,
            "factcheck_decision": {
                "decision": "REVIEW",
                "reason": "Teks pencarian tidak tersedia.",
                "evidence": [],
            },
        }

    query = " ".join(str(query).split())[:1500]

    search_keywords = extract_keywords(query, max_words=10)
    search_text = " ".join(search_keywords)

    if not search_text:
        return {
            "status": "UNAVAILABLE",
            "verification_level": "NONE",
            "message": "Tidak cukup kata kunci untuk pencarian.",
            "sources": empty_sources,
            "factcheck_decision": {
                "decision": "REVIEW",
                "reason": "Tidak cukup kata kunci untuk pencarian.",
                "evidence": [],
            },
        }

    results_by_category = {
        "fact_checks": [],
        "news_comparisons": [],
        "official_sources": [],
    }

    statuses = []
    jobs = []

    # RSS pemeriksaan fakta menggunakan teks klaim yang tersedia.
    for feed in RSS_FEEDS["fact_checks"]:
        jobs.append((
            "fact_checks",
            feed["name"],
            read_rss_feed,
            (feed, query),
        ))

    # Pencarian berita menggunakan kata kunci yang lebih pendek.
    for domain in NEWS_DOMAINS:
        jobs.append((
            "news_comparisons",
            domain,
            search_google_news_rss,
            (search_text, domain),
        ))

    for domain in OFFICIAL_DOMAINS:
        jobs.append((
            "official_sources",
            domain,
            search_google_news_rss,
            (search_text, domain),
        ))

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(function, *args): (category, name)
            for category, name, function, args in jobs
        }

        for future in as_completed(future_map):
            category, name = future_map[future]

            try:
                result = future.result()
            except Exception as exc:
                print(
                    f"[PUBLIC SOURCES] error={name}: {exc}"
                )
                result = {
                    "status": "UNAVAILABLE",
                    "results": [],
                }

            result_status = result.get("status", "UNAVAILABLE")
            statuses.append(result_status)

            results_by_category[category].extend(
                result.get("results", [])
            )

            print(
                f"[PUBLIC SOURCES] sumber={name}, "
                f"kategori={category}, status={result_status}, "
                f"jumlah_hasil={len(result.get('results', []))}"
            )

    for category, items in results_by_category.items():
        results_by_category[category] = deduplicate_items(
            items=items,
            category=category,
            query=query,
            source_url=source_url,
        )

    # Keputusan eksplisit hanya diambil dari kategori fact_checks.
    factcheck_decision = evaluate_factcheck_sources(
        results_by_category["fact_checks"],
        query,
    )

    total_results = sum(
        len(items) for items in results_by_category.values()
    )

    decisive_factcheck = (
        factcheck_decision["decision"] in ("HOAX", "NOT_HOAX")
    )

    if decisive_factcheck:
        status = "FACTCHECK_DECISION_FOUND"
        verification_level = "FACTCHECK_MATCH"
        message = factcheck_decision["reason"]
    elif total_results:
        status = "RELATED_SOURCES_FOUND"
        verification_level = "TOPIC_ONLY"
        message = (
            "Sumber terkait ditemukan, tetapi belum ada keputusan "
            "pemeriksaan fakta yang cukup kuat. Hasil ini tidak "
            "membuktikan klaim benar ataupun salah."
        )
    elif statuses and all(
        item_status == "UNAVAILABLE"
        for item_status in statuses
    ):
        status = "UNAVAILABLE"
        verification_level = "NONE"
        message = (
            "Sumber eksternal tidak dapat diakses. "
            "Silakan coba lagi nanti."
        )
    elif any(
        item_status == "UNAVAILABLE"
        for item_status in statuses
    ):
        status = "PARTIAL"
        verification_level = "INCOMPLETE"
        message = (
            "Sebagian sumber eksternal tidak dapat diakses. "
            "Hasil pemeriksaan belum konklusif."
        )
    else:
        status = "NO_MATCHES"
        verification_level = "NONE"
        message = (
            "Tidak ditemukan sumber pembanding yang memenuhi "
            "kriteria. Ini bukan bukti bahwa klaim benar."
        )

    print(
        f"[PUBLIC SOURCES] status={status}, "
        f"total={total_results}, "
        f"factcheck_decision={factcheck_decision['decision']}"
    )

    return {
        "status": status,
        "verification_level": verification_level,
        "message": message,
        "sources": results_by_category,
        "factcheck_decision": factcheck_decision,
    }
    