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
    "hari", "kumpulan"
}

MONTH_MAP = {
    "januari": "01",
    "februari": "02",
    "maret": "03",
    "april": "04",
    "mei": "05",
    "juni": "06",
    "juli": "07",
    "agustus": "08",
    "september": "09",
    "oktober": "10",
    "november": "11",
    "desember": "12",
}


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("&amp;", " ")
        .replace("&quot;", " ")
        .replace("&#39;", " ")
        .replace("&nbsp;", " ")
    )
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_keywords(text, max_words=8):
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
    """Skor kemiripan teks, bukan probabilitas kebenaran klaim."""
    query_words = set(extract_keywords(query, max_words=100))
    document_words = set(
        extract_keywords(f"{title} {summary}", max_words=200)
    )

    if not query_words or not document_words:
        return 0.0

    matched = query_words & document_words
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
        return parsed.scheme in ("http", "https") and bool(parsed.hostname)
    except (TypeError, ValueError):
        return False


def _clean_snippet(text):
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    text = (
        text.replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    return " ".join(text.split())[:500]


def read_rss_feed(feed_config, query, limit=5):
    try:
        response = requests.get(
            feed_config["url"],
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
        results = []

        for entry in parsed.entries[:50]:
            title = (entry.get("title") or "").strip()
            url = (entry.get("link") or "").strip()
            summary = entry.get("summary") or ""
            published = entry.get("published") or entry.get("updated") or ""

            if not title or not _valid_http_url(url):
                continue

            score = keyword_score(query, title, summary)
            if score < 0.30:
                continue

            results.append({
                "title": title,
                "url": url,
                "publisher": feed_config["name"],
                "published": published,
                "snippet": _clean_snippet(summary),
                "relevance_score": score,
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
        print(f"[PUBLIC SOURCES] RSS error ({feed_config['name']}): {exc}")
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

    print(f"[SEARCH QUERY] domain={domain}, query={search_query}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
        results = []

        for entry in parsed.entries[:30]:
            title = (entry.get("title") or "").strip()
            article_url = (entry.get("link") or "").strip()
            summary = entry.get("summary") or ""
            published = entry.get("published") or entry.get("updated") or ""

            if not title or not _valid_http_url(article_url):
                continue

            source_info = entry.get("source", {})
            publisher = ""
            if hasattr(source_info, "get"):
                publisher = (source_info.get("title") or "").strip()
            if not publisher:
                publisher = domain

            score = keyword_score(query, title, summary)
            if score < 0.30:
                continue

            results.append({
                "title": title,
                "url": article_url,
                "publisher": publisher,
                "published": published,
                "snippet": _clean_snippet(summary),
                "relevance_score": score,
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
        print(f"[PUBLIC SOURCES] Google News RSS error ({domain}): {exc}")
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
    return any(pattern in normalized for pattern in generic_patterns)


def build_search_queries(text, max_queries=3):
    text = str(text or "")
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bSHARE\s+NOW\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\(Dok\.?\s*Istimewa\)",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    month_names = (
        r"Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|"
        r"September|Oktober|November|Desember"
    )

    english_months = (
        r"January|February|March|April|May|June|July|August|"
        r"September|October|November|December"
    )
    all_months = month_names + "|" + english_months
    text = re.sub(
        r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4}"
        r"\s*[•·]\s*\d{1,2}\s+(?:" + all_months + r")"
        r"\s+\d{4}(?:\s+\d{1,2}[:.]\d{2})?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\b(?:share now|read more|baca juga)\b", " ", text, flags=re.I)
    text = " ".join(text.split())
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    candidates = []
    seen = set()

    for sentence in sentences:
        sentence = sentence.strip(" \t\n\"'")
        if len(sentence.split()) < 5:
            continue

        normalized = normalize_text(sentence)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        keywords = extract_keywords(sentence, max_words=10)
        if len(keywords) < 3:
            continue

        money = re.search(
            r"\bRp\.?\s*\d[\d.,]*",
            sentence,
            flags=re.IGNORECASE,
        )
        percentage = re.search(
            r"(?<!\w)\d+(?:[.,]\d+)?\s*%",
            sentence,
        )
        date = re.search(
            r"\b(?:\d{1,2}\s+(?:" + all_months + r")\s+\d{4}"
            r"|\d{1,2}[/-]\d{1,2}[/-]\d{4})\b",
            sentence,
            flags=re.IGNORECASE,
        )

        facts = []
        if money:
            facts.append(money.group(0))
        if percentage:
            facts.append(percentage.group(0))
        if date:
            facts.append(date.group(0))

        query_parts = keywords[:]
        for fact in facts:
            if fact not in query_parts:
                query_parts.append(fact)

        search_query = " ".join(query_parts).strip()[:180]
        score = (
            3 * int(money is not None)
            + 2 * int(percentage is not None)
            + int(date is not None)
            + min(len(keywords), 10) / 10
        )
        candidates.append({"query": search_query, "score": score})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    queries = []

    for candidate in candidates:
        candidate_query = candidate["query"]
        if candidate_query and candidate_query not in queries:
            queries.append(candidate_query)
        if len(queries) >= max_queries:
            break

    if not queries:
        fallback = extract_keywords(text, max_words=10)
        if fallback:
            queries.append(" ".join(fallback)[:180])

    return queries


def get_title_key(item):
    title = str(item.get("title") or "").strip()
    title = re.sub(
        r"\s*[-|–—]\s*(Kompas\.com|Tempo\.co|Liputan6\.com|ANTARA News)\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    return normalize_text(title)


def get_publisher_key(item):
    publisher = str(item.get("publisher") or "").strip()
    if not publisher:
        title = str(item.get("title") or "")
        match = re.search(
            r"\s*[-|–—]\s*(Kompas\.com|Tempo\.co|Liputan6\.com|ANTARA News)\s*$",
            title,
            flags=re.IGNORECASE,
        )
        if match:
            publisher = match.group(1)
    return normalize_text(publisher)


def deduplicate_items(items, category, query, source_url=None):
    unique_items = []

    for item in items:
        title = item.get("title", "") or ""

        if category == "news_comparisons":
            if is_same_article(item, query, source_url):
                print(
                    "[PUBLIC SOURCES] dilewati, artikel sumber yang sama: "
                    f"{title}"
                )
                continue
            if is_generic_listing_title(title):
                print(
                    "[PUBLIC SOURCES] dilewati, judul generik: "
                    f"{title}"
                )
                continue

        url_key = normalize_article_url(item.get("url", ""))
        title_key = get_title_key(item)
        publisher_key = get_publisher_key(item)
        duplicate_index = None

        for index, existing in enumerate(unique_items):
            existing_url = normalize_article_url(existing.get("url", ""))
            same_url = bool(url_key and existing_url and url_key == existing_url)
            same_title_publisher = (
                bool(title_key)
                and bool(publisher_key)
                and title_key == get_title_key(existing)
                and publisher_key == get_publisher_key(existing)
            )
            if same_url or same_title_publisher:
                duplicate_index = index
                break

        if duplicate_index is None:
            unique_items.append(item)
        elif item.get("relevance_score", 0) > unique_items[duplicate_index].get(
            "relevance_score", 0
        ):
            unique_items[duplicate_index] = item

    unique_items.sort(
        key=lambda item: item.get("relevance_score", 0),
        reverse=True,
    )
    return unique_items[:10]


def _normalize_number(value):
    digits = re.sub(r"\D", "", str(value or ""))
    return (digits.lstrip("0") or "0") if digits else ""


def _extract_money(text):
    matches = re.findall(
        r"\bRp\.?\s*([\d][\d.,]*)",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return {
        _normalize_number(value)
        for value in matches
        if _normalize_number(value)
    }


def _extract_percentages(text):
    matches = re.findall(
        r"(?<!\w)(\d+(?:[.,]\d+)?)\s*%",
        str(text or ""),
    )
    values = set()
    for value in matches:
        try:
            values.add(round(float(value.replace(",", ".")), 4))
        except (ValueError, TypeError):
            continue
    return values


def _extract_dates(text):
    text = str(text or "").lower()
    dates = set()

    month_pattern = (
        r"\b(\d{1,2})\s+"
        r"(januari|februari|maret|april|mei|juni|juli|agustus|"
        r"september|oktober|november|desember|"
        r"january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+(\d{4})\b"
    )
    english_month_map = {
        "january": "01", "february": "02", "march": "03",
        "april": "04", "may": "05", "june": "06",
        "july": "07", "august": "08", "september": "09",
        "october": "10", "november": "11", "december": "12",
    }
    full_month_map = {**MONTH_MAP, **english_month_map}

    for day, month, year in re.findall(month_pattern, text, flags=re.IGNORECASE):
        day_num = int(day)
        if 1 <= day_num <= 31:
            dates.add(
                f"{int(year):04d}-{full_month_map[month.lower()]}-{day_num:02d}"
            )

    numeric_pattern = r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"
    for day, month, year in re.findall(numeric_pattern, text):
        day_num, month_num = int(day), int(month)
        if 1 <= day_num <= 31 and 1 <= month_num <= 12:
            dates.add(f"{int(year):04d}-{month_num:02d}-{day_num:02d}")

    return dates


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
        }

    query = " ".join(str(query).split())[:1500]
    search_queries = build_search_queries(query, max_queries=3)

    if not search_queries:
        fallback = extract_keywords(query, max_words=10)
        search_queries = [" ".join(fallback)] if fallback else []

    if not search_queries:
        return {
            "status": "UNAVAILABLE",
            "verification_level": "NONE",
            "message": "Tidak cukup kata kunci untuk melakukan pencarian.",
            "sources": empty_sources,
        }

    print(f"[PUBLIC SOURCES] jumlah_kueri={len(search_queries)}")
    for search_text in search_queries:
        print(f"[PUBLIC SOURCES] kueri={search_text}")

    results_by_category = {
        "fact_checks": [],
        "news_comparisons": [],
        "official_sources": [],
    }
    statuses = []
    jobs = []

    for category, feeds in RSS_FEEDS.items():
        for feed in feeds:
            jobs.append((
                category,
                feed["name"],
                read_rss_feed,
                (feed, query),
            ))

    for domain in NEWS_DOMAINS:
        for search_text in search_queries:
            jobs.append((
                "news_comparisons",
                domain,
                search_google_news_rss,
                (search_text, domain),
            ))

    for domain in OFFICIAL_DOMAINS:
        for search_text in search_queries:
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
                print(f"[PUBLIC SOURCES] error={name}: {exc}")
                result = {"status": "UNAVAILABLE", "results": []}

            result_status = result.get("status", "UNAVAILABLE")
            statuses.append(result_status)
            items = result.get("results", [])
            print(
                f"[PUBLIC SOURCES] sumber={name}, kategori={category}, "
                f"status={result_status}, jumlah_hasil={len(items)}"
            )
            results_by_category[category].extend(items)

    for category, items in results_by_category.items():
        results_by_category[category] = deduplicate_items(
            items=items,
            category=category,
            query=query,
            source_url=source_url,
        )

    claim_money = _extract_money(query)
    claim_percentages = _extract_percentages(query)
    claim_dates = _extract_dates(query)
    claim_keywords = set(extract_keywords(query, max_words=100))

    potential_matches = 0
    related_sources = 0

    for category, items in results_by_category.items():
        for item in items:
            source_text = f"{item.get('title', '')} {item.get('snippet', '')}"
            source_keywords = set(extract_keywords(source_text, max_words=100))
            keyword_overlap = (
                len(claim_keywords & source_keywords) / max(len(claim_keywords), 1)
            )

            matched_facts = []
            if claim_money & _extract_money(source_text):
                matched_facts.append("nominal_uang")
            if claim_percentages & _extract_percentages(source_text):
                matched_facts.append("persentase")
            if claim_dates & _extract_dates(source_text):
                matched_facts.append("tanggal_peristiwa")

            if matched_facts and keyword_overlap >= 0.15:
                item["match_assessment"] = "POTENTIAL_CLAIM_MATCH"
                item["matched_facts"] = matched_facts
                potential_matches += 1
            else:
                item["match_assessment"] = "RELATED_TOPIC"
                item["matched_facts"] = []
                related_sources += 1

            item["keyword_overlap"] = round(keyword_overlap, 3)

    total_results = sum(len(items) for items in results_by_category.values())

    if potential_matches:
        status = "POTENTIAL_CLAIM_MATCHES"
        verification_level = "POTENTIAL_MATCH"
        message = (
            "Ditemukan sumber yang cocok dengan sebagian fakta dalam klaim. "
            "Kecocokan otomatis bukan bukti bahwa klaim benar atau salah. "
            "Periksa isi sumber dan konteksnya."
        )
    elif total_results:
        status = "RELATED_SOURCES_FOUND"
        verification_level = "TOPIC_ONLY"
        message = (
            "Ditemukan sumber terkait topik, tetapi belum ada kecocokan fakta "
            "yang cukup untuk mengindikasikan kecocokan dengan klaim spesifik. "
            "Ini bukan bukti bahwa klaim benar atau salah."
        )
    elif statuses and all(item_status == "UNAVAILABLE" for item_status in statuses):
        status = "UNAVAILABLE"
        verification_level = "NONE"
        message = "Sumber eksternal tidak dapat diakses. Coba lagi nanti."
    elif any(item_status == "UNAVAILABLE" for item_status in statuses):
        status = "PARTIAL"
        verification_level = "INCOMPLETE"
        message = (
            "Sebagian sumber eksternal tidak dapat diakses dan belum ditemukan "
            "hasil yang lolos penyaringan."
        )
    else:
        status = "NO_MATCHES"
        verification_level = "NONE"
        message = (
            "Tidak ditemukan sumber pembanding yang memenuhi kriteria. "
            "Ini bukan bukti bahwa klaim benar atau salah."
        )

    print(
        f"[PUBLIC SOURCES] status={status}, total={total_results}, "
        f"potential_matches={potential_matches}, related_sources={related_sources}"
    )

    return {
        "status": status,
        "verification_level": verification_level,
        "message": message,
        "sources": results_by_category,
    }