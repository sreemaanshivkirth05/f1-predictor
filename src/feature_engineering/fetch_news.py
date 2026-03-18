"""
fetch_news.py
-------------
Collects F1 news from RSS feeds and NewsAPI, scores sentiment,
and extracts event flags that affect race/championship predictions.

Sources (in priority order — fastest first):
  1. RSS feeds (completely free, no key, instant)
     - Autosport, BBC Sport F1, Sky F1, RaceFans, The Race
  2. NewsAPI (free tier: 100 req/day, needs API key)

Event flags extracted:
  - injury_flag: driver injured or ill
  - upgrade_flag: car upgrade confirmed
  - grid_penalty_flag: grid penalty expected or confirmed
  - tension_flag: driver-team conflict
  - contract_flag: contract uncertainty
  - fia_flag: FIA investigation or protest
  - weather_alert: major weather change for upcoming race

Sentiment scoring:
  - Uses keyword matching (fast, no ML model needed)
  - Scores range from -1.0 (very negative) to +1.0 (very positive)
  - Exponential time decay applied (recent news weighted more)

HOW TO RUN:
    python src/data_collection/fetch_news.py

NEWSAPI KEY (optional but recommended):
    Get a free key at newsapi.org/register
    Set it in the NEWSAPI_KEY variable below or as env variable

OUTPUT:
    data/news_sentiment.csv      — daily sentiment per driver
    data/news_event_flags.csv    — event flags per driver
    data/news_raw.csv            — raw headlines for debugging
"""

import requests
import pandas as pd
import numpy as np
import os
import time
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# NewsAPI key — get free key at newsapi.org/register
# Leave as empty string to use RSS only (still works well)
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")

# How many days of news to pull
LOOKBACK_DAYS = 7

# Exponential decay for news age
NEWS_DECAY_LAMBDA = 0.3

# ─── RSS FEED SOURCES ─────────────────────────────────────────────────────────

RSS_FEEDS = [
    {"name": "Autosport",   "url": "https://www.autosport.com/rss/f1/news/"},
    {"name": "BBC Sport F1","url": "https://feeds.bbci.co.uk/sport/formula1/rss.xml"},
    {"name": "RaceFans",    "url": "https://www.racefans.net/feed/"},
    {"name": "The Race",    "url": "https://the-race.com/feed/"},
    {"name": "Motorsport",  "url": "https://www.motorsport.com/rss/f1/news/"},
]

# ─── DRIVER AND TEAM KEYWORDS ─────────────────────────────────────────────────

DRIVER_KEYWORDS = {
    "max_verstappen":   ["verstappen", "max verstappen", "max v"],
    "norris":           ["norris", "lando norris", "lando"],
    "leclerc":          ["leclerc", "charles leclerc"],
    "russell":          ["russell", "george russell"],
    "hamilton":         ["hamilton", "lewis hamilton"],
    "piastri":          ["piastri", "oscar piastri"],
    "antonelli":        ["antonelli", "kimi antonelli", "andrea antonelli"],
    "sainz":            ["sainz", "carlos sainz"],
    "alonso":           ["alonso", "fernando alonso"],
    "bearman":          ["bearman", "oliver bearman"],
    "hulkenberg":       ["hulkenberg", "hülkenberg", "nico hulkenberg"],
    "gasly":            ["gasly", "pierre gasly"],
    "ocon":             ["ocon", "esteban ocon"],
    "albon":            ["albon", "alex albon", "alexander albon"],
    "lawson":           ["lawson", "liam lawson"],
    "stroll":           ["stroll", "lance stroll"],
    "hadjar":           ["hadjar", "isack hadjar"],
    "colapinto":        ["colapinto", "franco colapinto"],
    "bortoleto":        ["bortoleto", "gabriel bortoleto"],
    "arvid_lindblad":   ["lindblad", "arvid lindblad"],
    "perez":            ["perez", "checo", "sergio perez"],
    "bottas":           ["bottas", "valtteri bottas"],
}

TEAM_KEYWORDS = {
    "red_bull":      ["red bull", "redbull", "oracle red bull"],
    "mercedes":      ["mercedes", "amg petronas"],
    "ferrari":       ["ferrari", "scuderia ferrari"],
    "mclaren":       ["mclaren"],
    "aston_martin":  ["aston martin"],
    "alpine":        ["alpine"],
    "williams":      ["williams"],
    "rb":            ["rb f1", "racing bulls", "vcarb"],
    "haas":          ["haas"],
    "sauber":        ["sauber", "audi f1"],
    "cadillac":      ["cadillac", "andretti"],
}

# ─── SENTIMENT KEYWORDS ───────────────────────────────────────────────────────

POSITIVE_KEYWORDS = [
    "win", "wins", "won", "victory", "champion", "championship",
    "podium", "pole", "fastest", "dominant", "strong", "confident",
    "upgrade", "improvement", "ahead", "leading", "excellent",
    "contract extension", "extended", "renewed", "committed",
    "healthy", "fit", "recovered", "back", "ready",
]

NEGATIVE_KEYWORDS = [
    "crash", "accident", "retire", "dnf", "breakdown", "failure",
    "penalty", "disqualified", "dq", "protest", "investigation",
    "injury", "injured", "ill", "sick", "unwell", "hospital",
    "fired", "dropped", "contract terminated", "leaving", "quit",
    "tension", "conflict", "dispute", "unhappy", "frustrated",
    "struggling", "weak", "behind", "deficit", "gap",
]

# ─── EVENT FLAG KEYWORDS ──────────────────────────────────────────────────────

EVENT_KEYWORDS = {
    "injury_flag": [
        "injured", "injury", "ill", "sick", "unwell", "hospital",
        "medical", "health", "concussion", "fracture", "recover",
    ],
    "upgrade_flag": [
        "upgrade", "update", "new parts", "new floor", "new wing",
        "b-spec", "major update", "significant update", "development",
        "brought upgrades", "package", "new package",
    ],
    "grid_penalty_flag": [
        "grid penalty", "engine penalty", "power unit penalty",
        "gearbox penalty", "take penalty", "back of grid",
        "new engine", "new power unit", "fifth engine",
    ],
    "tension_flag": [
        "tension", "conflict", "dispute", "unhappy", "frustrated",
        "leave", "exit", "contract talks", "uncertain future",
        "team orders", "disagreement", "falling out",
    ],
    "contract_flag": [
        "contract", "future uncertain", "seat uncertain", "no deal",
        "yet to sign", "free agent", "transfer", "moving to",
        "joining", "announcement expected",
    ],
    "fia_flag": [
        "fia investigation", "protest", "stewards", "disqualified",
        "excluded", "appeal", "scrutineering", "illegal", "legal",
        "hearing", "penalty points", "super licence",
    ],
}


# ─── RSS FETCHER ──────────────────────────────────────────────────────────────

def fetch_rss_articles(days_back=LOOKBACK_DAYS):
    """
    Fetches articles from all RSS feeds.

    Returns:
        list: List of article dicts with title, description, date, source
    """
    articles  = []
    cutoff    = datetime.now() - timedelta(days=days_back)

    for feed in RSS_FEEDS:
        try:
            resp = requests.get(feed["url"], timeout=10, headers={
                "User-Agent": "F1Predictor/1.0"
            })
            if resp.status_code != 200:
                continue

            root = ET.fromstring(resp.content)

            # Handle both RSS and Atom formats
            items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")

            for item in items:
                # Get title
                title_el = item.find("title")
                title = title_el.text if title_el is not None else ""
                if not title:
                    continue

                # Get description
                desc_el = item.find("description") or item.find("summary")
                desc    = desc_el.text if desc_el is not None else ""

                # Get pub date
                date_el = item.find("pubDate") or item.find("published")
                date_str = date_el.text if date_el is not None else ""

                articles.append({
                    "source":      feed["name"],
                    "title":       str(title).strip(),
                    "description": str(desc).strip()[:500] if desc else "",
                    "date_str":    date_str,
                    "text":        f"{title} {desc}".lower(),
                })

            time.sleep(0.5)  # Be polite

        except Exception as e:
            print(f"  RSS fetch failed ({feed['name']}): {e}")
            continue

    print(f"  Fetched {len(articles)} articles from RSS feeds")
    return articles


def fetch_newsapi_articles(days_back=LOOKBACK_DAYS):
    """
    Fetches F1 articles from NewsAPI.
    Only runs if NEWSAPI_KEY is set.

    Returns:
        list: Article dicts
    """
    if not NEWSAPI_KEY:
        print("  NewsAPI key not set — skipping (set NEWSAPI_KEY env var to enable)")
        return []

    articles = []
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    queries = ["Formula 1", "F1 Grand Prix", "F1 2026"]

    for query in queries:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":          query,
                    "from":       from_date,
                    "sortBy":     "publishedAt",
                    "language":   "en",
                    "pageSize":   50,
                    "apiKey":     NEWSAPI_KEY,
                },
                timeout=10
            )

            if resp.status_code != 200:
                continue

            data = resp.json()
            for art in data.get("articles", []):
                title = art.get("title", "")
                desc  = art.get("description", "") or ""
                articles.append({
                    "source":      art.get("source", {}).get("name", "NewsAPI"),
                    "title":       str(title).strip(),
                    "description": str(desc).strip()[:500],
                    "date_str":    art.get("publishedAt", ""),
                    "text":        f"{title} {desc}".lower(),
                })

            time.sleep(1)

        except Exception as e:
            print(f"  NewsAPI query failed: {e}")

    print(f"  Fetched {len(articles)} articles from NewsAPI")
    return articles


# ─── SENTIMENT SCORING ────────────────────────────────────────────────────────

def score_sentiment(text):
    """
    Scores text sentiment using keyword matching.
    Fast — no ML model needed.

    Returns:
        float: Sentiment score -1.0 to +1.0
    """
    text  = str(text).lower()
    pos   = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg   = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    total = pos + neg

    if total == 0:
        return 0.0

    return round((pos - neg) / total, 3)


def extract_event_flags(text):
    """
    Extracts binary event flags from article text.

    Returns:
        dict: {flag_name: 0 or 1}
    """
    text  = str(text).lower()
    flags = {}
    for flag, keywords in EVENT_KEYWORDS.items():
        flags[flag] = int(any(kw in text for kw in keywords))
    return flags


def days_ago(date_str):
    """
    Computes how many days ago an article was published.
    Returns 7 (max lookback) if date cannot be parsed.
    """
    if not date_str:
        return 7
    try:
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(date_str[:25], fmt[:len(date_str[:25])])
                return max(0, (datetime.now() - dt.replace(tzinfo=None)).days)
            except Exception:
                continue
    except Exception:
        pass
    return 3  # Default to 3 days ago if parsing fails


# ─── PROCESS ARTICLES ─────────────────────────────────────────────────────────

def process_articles(articles):
    """
    Processes all articles to extract per-driver sentiment and event flags.

    For each driver:
      - Finds articles mentioning them
      - Scores sentiment of each article
      - Applies time decay (recent = more weight)
      - Aggregates into a single score per driver

    Returns:
        tuple: (sentiment_df, flags_df, raw_df)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    rows  = []

    for art in articles:
        text    = art.get("text", "")
        title   = art.get("title", "")
        sentiment = score_sentiment(text)
        flags     = extract_event_flags(text)
        d_ago     = days_ago(art.get("date_str", ""))
        weight    = np.exp(-NEWS_DECAY_LAMBDA * d_ago)

        # Find which drivers and teams this article mentions
        mentioned_drivers = []
        for driver_id, keywords in DRIVER_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                mentioned_drivers.append(driver_id)

        mentioned_teams = []
        for team_id, keywords in TEAM_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                mentioned_teams.append(team_id)

        for driver_id in mentioned_drivers:
            row = {
                "date":       today,
                "driver_id":  driver_id,
                "source":     art.get("source", ""),
                "title":      title[:200],
                "sentiment":  sentiment,
                "weight":     round(weight, 3),
                "days_ago":   d_ago,
            }
            row.update(flags)
            rows.append(row)

    raw_df = pd.DataFrame(rows)

    if raw_df.empty:
        print("  No relevant F1 articles found")
        empty_drivers = list(DRIVER_KEYWORDS.keys())
        sentiment_df  = pd.DataFrame({
            "date":                 [today] * len(empty_drivers),
            "driver_id":           empty_drivers,
            "sentiment_score_7d":  [0.0] * len(empty_drivers),
            "positive_mentions":   [0] * len(empty_drivers),
            "negative_mentions":   [0] * len(empty_drivers),
            "article_count":       [0] * len(empty_drivers),
        })
        flags_df = pd.DataFrame({
            "date":               [today] * len(empty_drivers),
            "driver_id":          empty_drivers,
            **{f: [0] * len(empty_drivers) for f in EVENT_KEYWORDS.keys()}
        })
        return sentiment_df, flags_df, raw_df

    # Aggregate per driver
    sentiment_rows = []
    flag_rows      = []
    flag_cols      = list(EVENT_KEYWORDS.keys())

    for driver_id in raw_df["driver_id"].unique():
        drv_df = raw_df[raw_df["driver_id"] == driver_id]

        # Weighted sentiment score
        weighted_sentiment = (
            (drv_df["sentiment"] * drv_df["weight"]).sum()
            / drv_df["weight"].sum()
        ) if drv_df["weight"].sum() > 0 else 0.0

        positive_count = (drv_df["sentiment"] > 0).sum()
        negative_count = (drv_df["sentiment"] < 0).sum()

        sentiment_rows.append({
            "date":               today,
            "driver_id":          driver_id,
            "sentiment_score_7d": round(weighted_sentiment, 3),
            "positive_mentions":  int(positive_count),
            "negative_mentions":  int(negative_count),
            "article_count":      len(drv_df),
        })

        # Event flags — any article triggering = flag is 1
        flag_row = {"date": today, "driver_id": driver_id}
        for flag in flag_cols:
            if flag in drv_df.columns:
                flag_row[flag] = int(drv_df[flag].max())
            else:
                flag_row[flag] = 0
        flag_rows.append(flag_row)

    sentiment_df = pd.DataFrame(sentiment_rows)
    flags_df     = pd.DataFrame(flag_rows)

    return sentiment_df, flags_df, raw_df


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_news_pipeline():
    """
    Full news pipeline:
      1. Fetch from RSS feeds (fast, free)
      2. Fetch from NewsAPI if key available
      3. Score sentiment and extract flags
      4. Save results
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("F1 NEWS SENTIMENT PIPELINE")
    print("=" * 60)
    print(f"Fetching last {LOOKBACK_DAYS} days of F1 news...")
    print(f"NewsAPI: {'enabled' if NEWSAPI_KEY else 'disabled (no key)'}")
    print("=" * 60)

    # Fetch articles
    rss_articles  = fetch_rss_articles()
    api_articles  = fetch_newsapi_articles()
    all_articles  = rss_articles + api_articles

    print(f"\nTotal articles collected: {len(all_articles)}")

    if not all_articles:
        print("No articles found — creating empty output files")
        today = datetime.now().strftime("%Y-%m-%d")
        drivers = list(DRIVER_KEYWORDS.keys())
        pd.DataFrame({
            "date": [today]*len(drivers),
            "driver_id": drivers,
            "sentiment_score_7d": [0.0]*len(drivers),
            "positive_mentions": [0]*len(drivers),
            "negative_mentions": [0]*len(drivers),
            "article_count": [0]*len(drivers),
        }).to_csv(os.path.join(DATA_DIR, "news_sentiment.csv"), index=False)
        return

    # Process
    print("\nProcessing articles...")
    sentiment_df, flags_df, raw_df = process_articles(all_articles)

    # Append to existing (keep last 30 days)
    for filename, df in [
        ("news_sentiment.csv", sentiment_df),
        ("news_event_flags.csv", flags_df),
        ("news_raw.csv", raw_df),
    ]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            existing = pd.read_csv(path)
            combined = pd.concat([existing, df], ignore_index=True)
            # Keep last 30 days
            if "date" in combined.columns:
                cutoff  = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                combined = combined[combined["date"] >= cutoff]
            combined = combined.drop_duplicates(
                subset=["date","driver_id"] if "driver_id" in combined.columns else None,
                keep="last"
            )
        else:
            combined = df

        combined.to_csv(path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("NEWS SENTIMENT SUMMARY")
    print("=" * 60)
    if not sentiment_df.empty:
        top = sentiment_df.sort_values("sentiment_score_7d", ascending=False).head(10)
        for _, row in top.iterrows():
            bar   = "█" * int(abs(row["sentiment_score_7d"]) * 10)
            sign  = "+" if row["sentiment_score_7d"] >= 0 else "-"
            print(f"  {row['driver_id']:<20} {sign}{abs(row['sentiment_score_7d']):.2f}  {bar}")

    if not flags_df.empty:
        print("\nActive event flags:")
        for flag in list(EVENT_KEYWORDS.keys()):
            if flag in flags_df.columns:
                flagged = flags_df[flags_df[flag] == 1]["driver_id"].tolist()
                if flagged:
                    print(f"  {flag:<25}: {', '.join(flagged)}")

    print(f"\nSaved:")
    print(f"  {os.path.join(DATA_DIR, 'news_sentiment.csv')}")
    print(f"  {os.path.join(DATA_DIR, 'news_event_flags.csv')}")
    print(f"  {os.path.join(DATA_DIR, 'news_raw.csv')}")


# ─── LOAD HELPERS (called by feature engineering) ─────────────────────────────

def load_news_features():
    """
    Loads the latest news sentiment and flags as features.
    Called by build_dataset.py.

    Returns:
        pd.DataFrame: Latest news features per driver
    """
    sentiment_path = os.path.join(DATA_DIR, "news_sentiment.csv")
    flags_path     = os.path.join(DATA_DIR, "news_event_flags.csv")

    if not os.path.exists(sentiment_path):
        return pd.DataFrame()

    sentiment_df = pd.read_csv(sentiment_path)
    flags_df     = pd.read_csv(flags_path) if os.path.exists(flags_path) else pd.DataFrame()

    # Get most recent entry per driver
    latest_sentiment = (
        sentiment_df.sort_values("date", ascending=False)
        .drop_duplicates("driver_id")
        [["driver_id","sentiment_score_7d","positive_mentions",
          "negative_mentions","article_count"]]
    )

    if not flags_df.empty:
        latest_flags = (
            flags_df.sort_values("date", ascending=False)
            .drop_duplicates("driver_id")
            .drop(columns=["date"], errors="ignore")
        )
        combined = latest_sentiment.merge(latest_flags, on="driver_id", how="left")
    else:
        combined = latest_sentiment

    return combined


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_news_pipeline()