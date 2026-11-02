import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os

from scraper import scrape_google_play, scrape_app_store, clean_text, is_valid_text
from nlp_engine import analyze_reviews_batch

load_dotenv()

st.set_page_config(page_title="App Intelligence Dashboard", layout="wide")
st.title("App Intelligence Dashboard")
st.markdown("Automated sentiment analysis and issue categorization for mobile app reviews.")

with st.sidebar:
    st.header("Analysis Settings")

    # Вибір рушія аналізу
    engine_choice = st.radio("Select NLP Engine:", ["Local Model (BERT)", "Gemini API (Cloud)"])

    platform = st.radio("Select Platform:", ["Google Play", "App Store"])

    if platform == "Google Play":
        default_url = "https://play.google.com/store/apps/details?id=com.viber.voip"
    else:
        default_url = "https://apps.apple.com/ua/app/viber-messenger-chats-calls/id382617920"

    app_url = st.text_input("App URL:", default_url)
    num_reviews = st.slider("Number of reviews to fetch:", 10, 200, 50)
    analyze_btn = st.button("Analyze Data", use_container_width=True)

if analyze_btn:
    df = pd.DataFrame()

    with st.spinner("Fetching data..."):
        try:
            if platform == "Google Play":
                raw_data = scrape_google_play(app_url, num_reviews)
            else:
                raw_data = scrape_app_store(app_url, num_reviews)

            if not raw_data:
                st.warning("No reviews found for this application.")
                st.stop()

            df = pd.DataFrame(raw_data)
            df['Date'] = pd.to_datetime(df['Date']).dt.date

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    with st.spinner(f"Running analysis using {engine_choice}..."):
        df['Clean_Review'] = df['Review'].apply(clean_text)
        df = df[df['Clean_Review'].apply(is_valid_text)].copy()
        df.reset_index(drop=True, inplace=True)

        if df.empty:
            st.warning("No valid reviews left after spam filtering.")
            st.stop()

        texts_to_analyze = df['Clean_Review'].tolist()

        # Передаємо вибраний рушій у бекенд
        analysis_results = analyze_reviews_batch(texts_to_analyze, engine=engine_choice)

        if not analysis_results:
            st.error("Failed to parse NLP response.")
            st.stop()

        sentiments = []
        categories = []
        for i in range(len(df)):
            match = next((item for item in analysis_results if item["index"] == i), None)
            if match:
                sentiments.append(match.get("sentiment", "Neutral"))
                categories.append(match.get("category", "No Issue"))
            else:
                sentiments.append("Neutral")
                categories.append("Other")

        df['Sentiment'] = sentiments
        df['Issue_Category'] = categories

        st.success("Analysis complete.")

        avg_rating = round(df['Rating'].mean(), 2)
        neg_count = len(df[df['Sentiment'] == 'Negative'])
        neg_percent = round((neg_count / len(df)) * 100, 1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Rating", f"{avg_rating} / 5.0")
        col2.metric("Negative Feedback", f"{neg_percent}%")
        col3.metric("Valid Reviews Processed", str(len(df)))

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Overview", "Issue Analysis", "Raw Database"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(df, names='Sentiment', title='Sentiment Distribution',
                                 color='Sentiment', color_discrete_map={'Positive': '#2ca02c', 'Neutral': '#1f77b4',
                                                                        'Negative': '#d62728'})
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                trend_df = df.groupby(['Date', 'Sentiment']).size().reset_index(name='Count')
                fig_line = px.line(trend_df, x='Date', y='Count', color='Sentiment', title='Sentiment Trend',
                                   color_discrete_map={'Positive': '#2ca02c', 'Neutral': '#1f77b4',
                                                       'Negative': '#d62728'})
                st.plotly_chart(fig_line, use_container_width=True)

        with tab2:
            neg_df = df[df['Sentiment'] == 'Negative']
            if not neg_df.empty:
                fig_bar = px.bar(neg_df['Issue_Category'].value_counts().reset_index(),
                                 x='Issue_Category', y='count', title="Negative Reviews by Category",
                                 color='count', color_continuous_scale='Reds')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No negative reviews detected.")

        with tab3:
            filter_val = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Neutral", "Negative"])
            display_df = df if filter_val == "All" else df[df['Sentiment'] == filter_val]
            st.dataframe(display_df[['Date', 'Rating', 'Sentiment', 'Issue_Category', 'Review']],
                         use_container_width=True)