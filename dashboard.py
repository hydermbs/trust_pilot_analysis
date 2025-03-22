import streamlit as st
import analysis as al
from scraper import extract_data
import pandas as pd
import matplotlib.pyplot as plt
import time

@st.cache_data
def get_scraped_data(keywords):
    return extract_data(keywords)

def basic_insights(df):
    total_review = df['reviewBody'].nunique()
    average_rating = df['ratingValue'].mean().round(2)
    sentiment_counts = df['Sentiment'].value_counts()
    try:
        positive_comments = sentiment_counts[0]
    except:
        positive_comments = st.write('Positive comments are 0')
    try:
        negative_comments = sentiment_counts[1]
    except IndexError:
        negative_comments = st.write('Negative comments are 0')
    try:
        neutral_comments = sentiment_counts[2]
    except IndexError:
        neutral_comments = st.write('Nuteral Commments are 0')
    total_reviews, avg_rating, sentiments = st.columns(3, border=True)
    total_reviews.title(f'Total Reviews: {total_review}')
    avg_rating.title(f'Avg Rating: {average_rating}')
    sentiments.write(f"Positive Comments: {positive_comments}")
    sentiments.write(f'Negative Comments: {negative_comments}')
    sentiments.write(f'Neutral Comments: {neutral_comments}')

def positive_vs_negative(df):

    sentiment = al.sentiment_counts(df)
    negative_feedback, sentiments = st.columns([2,1],border=True)
    negative_feedback.subheader("📊 Issues & Suggestions")
    negative_feedback.write(al.suggestion(df))

    
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

st.markdown('<style>{}</style>'.format(open("style.css").read()), unsafe_allow_html=True)
container = st.container()
container.title('Trust Pilot Review Analysis')


with st.sidebar:
    keywords = st.text_input("Enter Keyword to Search")
if keywords:
    with st.status("Scraping in progress...⏳", expanded=True) as status:
       
        scrape = get_scraped_data(keywords)
        df = pd.DataFrame(scrape)
          # Call your actual scraping function here
        status.update(label="Scraping Completed ✅", state="complete")

    if not df.empty:
        #feature engineering
        df = al.feature_data(df)
        unique_years = sorted(df['Year'].unique())
        year_options = ['All']+list(unique_years)
    
        with st.sidebar:
            selected_year = st.selectbox("Select a Year",year_options)

        filtered_df = df if selected_year == "All" else df[df['Year'] == selected_year]

        
        df = al.clean_text(filtered_df)
        df = al.sentiment_analysis(filtered_df)
        
        basic_container = st.container()
        with basic_container:
            basic_insights(filtered_df)
        try:
            feedback_container = st.container(border=True)
            feedback_container.subheader('Latest Review:')
            feedback_container.write(f'"{filtered_df['reviewBody'][0]}"')
            feedback_container.write(f"Written by: {filtered_df['Author'][0]}    |      Date:{filtered_df['Dates'][0]}")
        except KeyError:
            pass
        df_words = al.sentiment_intensity(filtered_df)
        positive_vs_negative(filtered_df, df_words)
        rating_suspicious(filtered_df)
        review_over_time = st.container()
        review_over_time.plotly_chart(al.number_of_reviews_over_time(filtered_df))
        st.write(filtered_df)


