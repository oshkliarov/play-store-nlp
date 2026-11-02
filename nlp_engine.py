from transformers import pipeline
import google.genai as genai
from google.genai import types
import json
import os
import streamlit as st


@st.cache_resource
def load_local_models():
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    zero_shot_model = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
    return sentiment_model, zero_shot_model


def map_sentiment_local(label: str) -> str:
    stars = int(label.split()[0])
    if stars <= 2:
        return "Negative"
    elif stars == 3:
        return "Neutral"
    else:
        return "Positive"


def analyze_with_local(reviews_list: list) -> list:
    sentiment_model, zero_shot_model = load_local_models()
    texts = [str(t)[:512] for t in reviews_list]

    sentiments = sentiment_model(texts)
    categories = ["Bugs/Crashes", "Ads/Pricing", "UI/UX", "Customer Support", "Updates/Speed", "Privacy/Permissions",
                  "Other"]

    results = []
    for i, text in enumerate(texts):
        sentiment = map_sentiment_local(sentiments[i]['label'])
        category = "No Issue"
        if sentiment == "Negative":
            zs = zero_shot_model(text[:300], candidate_labels=categories)
            if zs['scores'][0] > 0.30:
                category = zs['labels'][0]
            else:
                category = "Other"
        results.append({"index": i, "sentiment": sentiment, "category": category})
    return results


def analyze_with_gemini(reviews_list: list) -> list:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key is missing. Please check your .env file or Streamlit Secrets.")

    client = genai.Client(api_key=api_key)
    payload = [{"index": i, "text": text} for i, text in enumerate(reviews_list)]

    prompt = f"""
    Analyze the following list of app reviews. For each review, determine the sentiment and the core issue category.
    Categories for Negative reviews: "Bugs/Crashes", "Ads/Pricing", "UI/UX", "Customer Support", "Updates/Speed", "Privacy/Permissions", "Other".
    If the review is Positive or Neutral, use category "No Issue".

    Return a JSON array of objects with keys: "index" (int), "sentiment" ("Positive", "Neutral", "Negative"), and "category" (str).

    Data to analyze:
    {json.dumps(payload, ensure_ascii=False)}
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return []


def analyze_reviews_batch(reviews_list: list, engine: str) -> list:
    if engine == "Gemini API (Cloud)":
        return analyze_with_gemini(reviews_list)
    else:
        return analyze_with_local(reviews_list)