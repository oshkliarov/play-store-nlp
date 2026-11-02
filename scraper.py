import re
import requests
from google_play_scraper import reviews, Sort
import urllib.parse as urlparse
from urllib.parse import parse_qs


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-ЯіІїЇєЄґҐ0-9\s.,!?]', '', text)
    return text.strip()


def is_valid_text(text: str) -> bool:
    if len(text) < 4:
        return False
    vowels = "аеєиіїоуюяaeiouy"
    if not any(char.lower() in vowels for char in text):
        return False
    words = text.lower().split()
    if len(words) < 2:
        return False
    if len(words) > 1 and len(set(words)) == 1:
        return False
    return True


def scrape_google_play(url: str, count: int) -> list:
    parsed_url = urlparse.urlparse(url)
    app_id = parse_qs(parsed_url.query).get('id', [None])[0]
    if not app_id:
        raise ValueError("Invalid Google Play URL. Make sure it contains '?id='")
    result, _ = reviews(app_id, lang='uk', country='ua', sort=Sort.NEWEST, count=count)
    return [{"Review": r['content'], "Rating": r['score'], "Date": r['at']} for r in result] if result else []


def scrape_app_store(url: str, count: int) -> list:
    match = re.search(r'id(\d+)', url)
    if not match:
        raise ValueError("Invalid App Store URL. Make sure it contains '/id...'")
    app_id = match.group(1)
    api_url = f"https://itunes.apple.com/ua/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"
    response = requests.get(api_url)
    data = response.json()

    if 'entry' not in data.get('feed', {}):
        return []

    reviews_data = []
    for entry in data['feed']['entry'][1:count + 1]:
        reviews_data.append({
            "Review": entry['content']['label'],
            "Rating": int(entry['im:rating']['label']),
            "Date": entry['updated']['label']
        })
    return reviews_data