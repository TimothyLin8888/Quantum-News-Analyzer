from newsapi import NewsApiClient
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()
api_key = os.getenv("NEWS_API_KEY")
newsapi = NewsApiClient(api_key=api_key)

nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

def fetch_news(sources=None, query=None, from_date=None, to_date=None, language="en", page_size=100):
    """
    Fetches news articles using NewsAPI.
    You can specify sources (e.g. 'bbc-news, cnn'), or a query keyword (e.g. 'climate', 'election').
    """
    articles_data = []

    response = newsapi.get_everything(
        q=query,
        sources=sources,
        language=language,
        from_param=from_date,
        to=to_date,
        sort_by="relevancy",
        page_size=page_size
    )

    for article in response.get("articles", []):
        articles_data.append({
            "source": article["source"]["name"],
            "author": article.get("author"),
            "title": article.get("title"),
            "description": article.get("description"),
            "content": article.get("content"),
            "url": article.get("url"),
            "publishedAt": article.get("publishedAt")
        })
    return pd.DataFrame(articles_data)

def save_news(df, filename="news_data.csv"):
    """Saves the collected news data into /data/raw/"""
    # os.makedirs("data/raw", exist_ok=True)
    data_folder = os.path.join(os.pardir, 'data')
    filepath = os.path.join(data_folder, filename)
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"\033[1;32mSaved {len(df)} articles to {filepath}\033[0m")


def analyze_sentiment(text):
    """Return sentiment label from compound VADER score."""
    if not isinstance(text, str):
        return "neutral"
    score = sid.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"
    

def save_sentiment(df, filename="news_with_sentiment.csv"):
    """Apply sentiment analysis and save the dataset."""
    # Combine title + description for better signal
    data_folder = os.path.join(os.pardir, 'data')
    filepath = os.path.join(data_folder, filename)
    df["text"] = df["title"].fillna('') + ". " + df["description"].fillna('')

    # Apply sentiment function
    df["sentiment"] = df["text"].apply(analyze_sentiment)

    # Save as CSV
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"\033[1;32mSaved sentiment-labeled news to {filepath}\033[0m")
    print(df[["title", "sentiment"]].head())


if __name__ == "__main__":
    today = datetime.now().strftime("2025-11-03")

    # Example: Fetch top political or tech news for today
    news_df = fetch_news(query="AI OR politics OR economy", from_date=today)
    print(news_df.head())

    save_news(news_df, filename=f"news_{today}.csv")
    save_sentiment(news_df, filename=f"news_with_sentiment_{today}.csv")