import streamlit as st
import analysis as al
from scraper import extract_data
import pandas as pd
import matplotlib.pyplot as plt
import time

st.markdown('<style>{}</style>'.format(open("style.css").read()), unsafe_allow_html=True)


def basic_insights(df):
    total_review = df['reviewBody'].nunique()
    average_rating = df['ratingValue'].mean().round(2)
    sentiment_counts = df['Sentiment'].value_counts()
    positive_comments = sentiment_counts[0]
    negative_comments = sentiment_counts[1]
    neutral_comments = sentiment_counts[2]
    total_reviews, avg_rating, sentiments = st.columns(3, border=True)
    total_reviews.title(f'Total Reviews: {total_review}')
    avg_rating.title(f'Avg Rating: {average_rating}')
    sentiments.write(f"Positive Comments: {positive_comments}")
    sentiments.write(f'Negative Comments: {negative_comments}')
    sentiments.write(f'Neutral Comments: {neutral_comments}')

def positive_vs_negative(df,df_words):
    fig1,ax1=plt.subplots()
    al.positive_words(df_words,ax1)
    fig2,ax2 = plt.subplots()
    al.negative_Words(df_words,ax2)
    sentiment = al.sentiment_counts(df)
    positive, negative, sentiments = st.columns(3,border=True)
    positive.subheader('Top 20 Most Common Positive Words')
    positive.pyplot(fig1)
    negative.subheader('Top 20 Most Common Negative Words')
    negative.pyplot(fig2)
    sentiments.subheader('Sentiment Analysis Distribution')
    sentiments.pyplot(sentiment)

def rating_suspicious(df):
    ratings = al.rating_counts(df)
    suspicious_reviews = al.suspicious_comment(df)
    rating, suspicious = st.columns(2,border=True)
    rating.subheader('Break Down of Star Rating')
    rating.pyplot(ratings)
    suspicious.subheader('Suspicious Comments')
    suspicious.pyplot(suspicious_reviews)


container = st.container()
container.title('Trust Pilot Review Analysis')


with st.sidebar:
    keywords = st.text_input("Enter Keyword to Search")

if keywords:
    with st.status("Scraping in progress...⏳", expanded=True) as status:
        scrape = extract_data(keywords)
        df = pd.DataFrame(scrape)
          # Call your actual scraping function here
        status.update(label="Scraping Completed ✅", state="complete")

    if not df.empty:
        #feature engineering
        company_name = st.container()
        company_name.title(df['Company Review'][0])
        df = al.feature_data(df)
        df = al.cleaned_data(df)
        df = al.sentiment_analysis(df)
        basic_container = st.container()
        with basic_container:
            basic_insights(df)
        feedback_container = st.container(border=True)
        feedback_container.subheader('Latest Review:')
        feedback_container.write(f'"{df['reviewBody'][0]}"')
        feedback_container.write(f"Written by: {df['Author'][0]}    |      Date:{df['Dates'][0]}")
        df_words = al.sentiment_intensity(df)
        positive_vs_negative(df, df_words)
        rating_suspicious(df)
        review_over_time = st.container()
        review_over_time.plotly_chart(al.number_of_reviews_over_time(df))
        st.write(df)
