import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from textblob import TextBlob

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Set the page title and layout
st.set_page_config(page_title="Stock Movement Prediction", layout="wide")

st.title("Stock Price Movement Prediction")
st.markdown("""
This interactive web application predicts stock price movements based on **user-generated content** and **historical stock data**. 
It combines sentiment analysis with market trends to forecast price directions.
""")

# Sidebar Inputs
st.sidebar.header("User Inputs")

stock_symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL, TSLA, etc.)", value="AAPL")

st.sidebar.header("Select Date Range")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

st.sidebar.header("Analyze Custom Sentiment")
user_text = st.sidebar.text_area("Enter a snippet of text (e.g., a tweet or discussion):", "")
analyze_sentiment_button = st.sidebar.button("Analyze Sentiment")

st.sidebar.header("Overall Sentiment")
user_sentiment = st.sidebar.radio("Sentiment on Social Media", ("Positive", "Negative", "Neutral"))

if analyze_sentiment_button and user_text:
    blob = TextBlob(user_text)
    sentiment_polarity = blob.sentiment.polarity
    if sentiment_polarity > 0:
        st.sidebar.success(f"Sentiment Analysis Result: Positive (Score: {sentiment_polarity:.2f})")
        sentiment_score = 1
    elif sentiment_polarity < 0:
        st.sidebar.error(f"Sentiment Analysis Result: Negative (Score: {sentiment_polarity:.2f})")
        sentiment_score = -1
    else:
        st.sidebar.info(f"Sentiment Analysis Result: Neutral (Score: {sentiment_polarity:.2f})")
        sentiment_score = 0
else:
    sentiment_score = 1 if user_sentiment == "Positive" else (-1 if user_sentiment == "Negative" else 0)

st.header(f"Stock Data for {stock_symbol}")
try:
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    if not stock_data.empty:
        st.write(stock_data.tail())
        st.line_chart(stock_data["Close"], use_container_width=True)
    else:
        st.warning("No stock data found for the given date range.")
except Exception as e:
    st.error(f"Error fetching data: {e}")

st.header("Stock Movement Prediction")
if st.button("Predict Stock Movement"):
    try:
        features = np.array([[sentiment_score]])
        prediction = model.predict(features)
        confidence = max(model.predict_proba(features)[0])

        if prediction[0] == 1:
            st.success(f"Prediction: The stock price is likely to go **UP**.")
        else:
            st.error(f"Prediction: The stock price is likely to go **DOWN**.")
        
        st.markdown(f"**Model Confidence:** {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
