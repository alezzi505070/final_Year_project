#!/usr/bin/env python3

import os
import datetime
import requests
import openai
from anthropic import Anthropic
import google.generativeai as genai
import numpy as np
import pandas as pd
import logging
import csv
from newsapi import NewsApiClient
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from typing import List
import time

from flask import Flask, render_template, request, redirect, url_for

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
 # Disable template caching in development (for debugging only)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ------------------------------------------------------------------------------
#  API Keys & Configurations
# ------------------------------------------------------------------------------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "ce75402c3e1045f1a1f2d4b2b1cf6f43")  
openai.api_key = os.getenv("OPENAI_API_KEY", "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")
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "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"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyD8gEQXCxgkNoL4rzlB8uDKaEx1ktUYF00"))

# ------------------------------------------------------------------------------
#  Model Definitions
# ------------------------------------------------------------------------------
SENTIMENT_MODEL_NAME = "ft:gpt-4o-mini-2024-07-18:personal:kol:A6JutyNw"
JUDGMENT_MODELS = {
    "o1_preview": "gpt-4o-2024-11-20",
    "anthropic_claude": "claude-3-5-sonnet-20241022",
    "google_gemini": "gemini-2.0-flash-exp"
}

MODEL_PATH = "best_stock_prediction_model_GOLD.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

today = datetime.date.today()
past_month = today - datetime.timedelta(days=30)
DECISIONS_CSV = "stock_decisions_log.csv"

# ------------------------------------------------------------------------------
#  Data Preparation / Prediction
# ------------------------------------------------------------------------------
def prepare_data(stock, time_steps=30):
   try:
        data = yf.download(stock, period="1y", interval="1d", progress=False) # added progress=False
        if data.empty:
            raise ValueError(f"No data found for stock {stock}")
        data.reset_index(inplace=True)

        close_scaler = StandardScaler()
        feature_scaler = StandardScaler()

        data_features = data[['Close', 'High', 'Low', 'Open', 'Volume']]
        scaled_close = close_scaler.fit_transform(data_features[['Close']])
        scaled_features = feature_scaler.fit_transform(data_features[['High', 'Low', 'Open', 'Volume']])
        scaled_data = np.column_stack((scaled_close, scaled_features))

        x_pred = []
        for i in range(time_steps, len(scaled_data)):
            x_pred.append(scaled_data[i - time_steps:i])

        return np.array(x_pred), close_scaler, data
   except Exception as e:
     logger.error(f"Error Preparing data for stock {stock}: {e}")
     raise

def get_next_day_price_prediction(stock_symbol):
    try:
        time_steps = 30
        x_pred, close_scaler, _ = prepare_data(stock_symbol, time_steps=time_steps)
        if len(x_pred) == 0:
            return 0
        # Warm-up prediction
        _ = model.predict(x_pred)
        # Final next-day
        next_day_sequence = x_pred[-1]
        next_day_prediction = model.predict(next_day_sequence[np.newaxis, :, :])
        return float(close_scaler.inverse_transform(next_day_prediction)[0, 0])
    except Exception as e:
       logger.error(f"Error getting price prediction for {stock_symbol}: {e}")
       return 0

# ------------------------------------------------------------------------------
#  Fetch News
# ------------------------------------------------------------------------------
def get_newsapi_articles(stock_symbol, from_date, to_date, top_n=4):
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    try:
        articles_data = newsapi.get_everything(
            q=stock_symbol,
            language='en',
            from_param=str(from_date),
            to=str(to_date),
            sort_by='relevancy',
            page_size=top_n
        )
        newsapi_articles = []
        if articles_data and articles_data.get('articles'):
            for article in articles_data['articles']:
                title = article.get('title') or ""
                description = article.get('description') or ""
                content = article.get('content') or ""
                combined_text = f"{title}\n\n{description}\n\n{content}".strip()
                newsapi_articles.append(combined_text)
        return newsapi_articles
    except Exception as e:
        logger.error(f"Error fetching news from NewsAPI: {e}")
        return []


def get_yahoo_news_articles(stock_symbol, top_n=4):
    url = f"https://news.search.yahoo.com/search?p={stock_symbol}"
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(r.text, 'html.parser')
        articles = soup.select('.NewsArticle')
        yahoo_articles = []
        for a in articles[:top_n]:
            h = a.select_one('h4.title')
            s = a.select_one('p')
            headline = h.get_text(strip=True) if h else ""
            snippet = s.get_text(strip=True) if s else ""
            yahoo_articles.append(f"{headline}\n\n{snippet}".strip())
        return yahoo_articles
    except Exception as e:
        logger.error(f"Error fetching news from Yahoo: {e}")
        return []

# ------------------------------------------------------------------------------
#  Sentiment Analysis
# ------------------------------------------------------------------------------
def get_sentiment_for_article(article_text):
    prompt = (
        "Analyze the sentiment (positive, negative, or neutral) of the following text. "
        "Respond with 'yes' if positive, 'no' if negative, 'natural' if neutral.\n\n"
        f"Text:\n{article_text}\n\nSentiment:"
    )
    try:
        r = openai.ChatCompletion.create(
            model=SENTIMENT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return r.choices[0].message['content'].strip().lower()
    except Exception as e:
        logger.error(f"Error getting sentiment from OpenAI: {e}")
        return "neutral"

def get_article_sentiments(articles: List[str]) -> List[str]:
    sentiments = []
    for article in articles:
        if article.strip():
            sentiments.append(get_sentiment_for_article(article))
    return sentiments

def filter_nonneutral_articles(articles: List[str], sentiments: List[str]) -> List[tuple]:
    nonneutral = []
    for art, sent in zip(articles, sentiments):
        s = sent.strip().lower()
        if s not in ["neutral", "natural"]:
            nonneutral.append((art, s))
    return nonneutral

# ------------------------------------------------------------------------------
#  Explanation
# ------------------------------------------------------------------------------
def get_explanation_from_model(model_name: str, prompt_content: str) -> str:
     if model_name not in JUDGMENT_MODELS:
         logger.error(f"Invalid model name for explanation: {model_name}")
         return "Error: unknown model"

     try:
         if model_name.startswith(("openai_", "o1_")):
             response = openai.ChatCompletion.create(
                 model=JUDGMENT_MODELS[model_name],
                 messages=[{"role": "user", "content": prompt_content}],
                 temperature=0.0
             )
             return response.choices[0].message['content'].strip()
         elif model_name.startswith("anthropic_"):
             response = anthropic_client.messages.create(
                 model=JUDGMENT_MODELS[model_name],
                 max_tokens=300,
                 messages=[{"role": "user", "content": prompt_content}]
             )
             if response and hasattr(response, "content") and response.content:
                 return response.content[0].text.strip() if response.content[0].text else ""
             else:
                 return "Error: No content from Anthropic"
         elif model_name.startswith("google_"):
             model = genai.GenerativeModel(JUDGMENT_MODELS[model_name])
             resp = model.generate_content(prompt_content)
             return resp.text.strip() if resp.text else ""
         else:
             return "Error: Unknown model type"
     except Exception as e:
         logger.error(f"Error getting explanation from {model_name}: {str(e)}")
         return f"Error from {model_name}: {str(e)}"

def explain_buy_or_sell_decision_3models(stock_symbol: str, articles: List[str], sentiments: List[str], final_decision: str) -> str:
    nonneutral = filter_nonneutral_articles(articles, sentiments)
    relevant_lines = []
    for i, (art, s) in enumerate(nonneutral, start=1):
        snippet = art[:200].replace('\n', ' ')
        relevant_lines.append(f"{i}. [{s.upper()}] {snippet}...")
    relevant_joined = "\n".join(relevant_lines)

    prompt = f"""
We concluded '{final_decision.upper()}' for {stock_symbol}.

Below are the NON-NEUTRAL (positive or negative) articles that influenced this decision:
{relevant_joined if relevant_joined else 'None. All articles were neutral or unavailable.'}

Please provide a brief (2-3 sentences) justification for why we reached '{final_decision.upper()}',
based on the articles above.
"""

    explanations = []
    for mk in JUDGMENT_MODELS:
        text = get_explanation_from_model(mk, prompt)
        explanations.append(f"--- Explanation from {mk} ---\n{text}\n")

    return (
        f"=== Combined Justifications for {final_decision.upper()} on {stock_symbol} ===\n"
        + "\n".join(explanations)
    )


# ------------------------------------------------------------------------------
#  Buy/Sell Decision
# ------------------------------------------------------------------------------
def get_judgment_from_model(model_name: str, prompt_content: str) -> str:
    def clean_response(res: str) -> str:
        if not res:
            return None
        c = res.strip().lower()
        first_word = c.split()[0] if c else ''
        return first_word if first_word in ["buy", "sell"] else None

    if model_name not in JUDGMENT_MODELS:
        logger.error(f"Invalid model name: {model_name}")
        return None

    try:
        if model_name.startswith(("openai_", "o1_")):
            response = openai.ChatCompletion.create(
                model=JUDGMENT_MODELS[model_name],
                messages=[{"role": "user", "content": f"{prompt_content}\nIMPORTANT: Respond with only 'buy' or 'sell'."}],
                temperature=0.0
            )
            return clean_response(response.choices[0].message['content'])
        elif model_name.startswith("anthropic_"):
            def call_anthropic():
                try:
                    r = anthropic_client.messages.create(
                        model=JUDGMENT_MODELS[model_name],
                        max_tokens=100,
                        messages=[{"role": "user", "content": f"{prompt_content}\nIMPORTANT: Respond with only 'buy' or 'sell'."}]
                    )
                    if (hasattr(r, "content") and isinstance(r.content, list) and r.content
                        and hasattr(r.content[0], "text")):
                        return clean_response(r.content[0].text)
                except Exception as e:
                    logger.error(f"Anthropic API call error: {e}")
                    return None

            return call_anthropic()
        elif model_name.startswith("google_"):
            logger.debug(f"Calling Google Generative AI with model={JUDGMENT_MODELS[model_name]}")
            try:
                model = genai.GenerativeModel(JUDGMENT_MODELS[model_name])
                r = model.generate_content(f"{prompt_content}\nIMPORTANT: Respond with only 'buy' or 'sell'.")
                logger.debug(f"Google model response: {r}")
                return clean_response(r.text)
            except Exception as e:
                logger.error(f"Error in Google Generative AI call: {str(e)}")
                return None
    except Exception as e:
        logger.error(f"Error getting judgment from {model_name}: {str(e)}")
        return None

def ensemble_buy_or_sell(prompt_content: str) -> str:
    votes = []
    for mk in JUDGMENT_MODELS:
        d = get_judgment_from_model(mk, prompt_content)
        logger.debug(f"Model: {mk}, Decision: {d}")
        if d:
            votes.append(d)
    logger.debug(f"Votes from all models: {votes}")

    buys = votes.count("buy")
    sells = votes.count("sell")
    return "buy" if buys > sells else "sell"

def get_buy_or_sell_judgment(stock_symbol: str) -> str:
    newsapi_arts = get_newsapi_articles(stock_symbol, from_date=past_month, to_date=today, top_n=4)
    yahoo_arts = get_yahoo_news_articles(stock_symbol, top_n=4)
    articles = newsapi_arts + yahoo_arts
    sentiments = get_article_sentiments(articles)
    predicted_price = get_next_day_price_prediction(stock_symbol)

    s_text = "\n".join([f"- {s}" for s in sentiments])
    prompt_content = f"""
You have the following information for {stock_symbol}:

News Sentiments (each from an article in the past month):
{s_text}

Model’s Next-Day Price Prediction: {predicted_price:.2f}

Based on this information, decide strictly whether to BUY or SELL the stock.
Provide your answer as either the single word 'buy' or 'sell', with no additional explanation.
""".strip()

    return ensemble_buy_or_sell(prompt_content)
 # ------------------------------------------------------------------------------
#  Logging
# ------------------------------------------------------------------------------
def log_decision_to_csv(stock: str, decision: str, predicted_price: float, sentiments: List[str]):
     t = datetime.datetime.now().isoformat()
     row = [t, stock, decision, predicted_price, ";".join(sentiments)]
     file_exists = os.path.isfile(DECISIONS_CSV)
     with open(DECISIONS_CSV, 'a', newline='', encoding='utf-8') as f:
         w = csv.writer(f)
         if not file_exists:
             w.writerow(["Timestamp", "Stock", "Decision", "PredictedPrice", "Sentiments"])
         w.writerow(row)
     logger.info(f"Logged decision: {row}")


# ------------------------------------------------------------------------------
#  Flask Routes
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """
    Displays a landing page with a form for the user to enter a stock symbol.
    """
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    When the user submits the stock symbol, perform the analysis
    and display the results on a separate page.
    """
    try:
        stock_symbol = request.form.get("stock_symbol", "BTC-USD").strip() # added strip()
        logger.info(f"User requested analysis for symbol: {stock_symbol}")

        if not stock_symbol:
            return "<h2>Error</h2><p>Stock Symbol cannot be empty.</p>"

        # 1) Fetch articles & sentiments
        newsapi_arts = get_newsapi_articles(stock_symbol, from_date=past_month, to_date=today, top_n=4)
        yahoo_arts = get_yahoo_news_articles(stock_symbol, top_n=4)
        articles = newsapi_arts + yahoo_arts
        sentiments = get_article_sentiments(articles)

        # 2) Final decision
        decision = get_buy_or_sell_judgment(stock_symbol)

        # 3) Explanation (via all 3 models)
        explanation_text = explain_buy_or_sell_decision_3models(stock_symbol, articles, sentiments, decision)

        # 4) Next day price
        predicted_price = None
        try:
            predicted_price = get_next_day_price_prediction(stock_symbol)
            logger.info(f"Predicted price: {predicted_price}")
        except Exception as e:
            logger.error(f"Price prediction error: {str(e)}")

        # 5) Log
        try:
            log_decision_to_csv(stock_symbol, decision, predicted_price, sentiments)
        except Exception as e:
            logger.error(f"Error logging results: {str(e)}")

        # 6) Render the results page
        return render_template(
            "results.html",
            stock_symbol=stock_symbol.upper(),
            decision=decision.upper(),
            predicted_price=predicted_price,
            explanation_text=explanation_text
        )

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return f"<h2>Oops, an error occurred</h2><p>{str(e)}</p>"

if __name__ == "__main__":
    app.run(debug=True)