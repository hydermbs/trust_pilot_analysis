import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.express as px


def feature_data(df):
    df['ratingValue'] = pd.to_numeric(df['ratingValue'],errors='coerce')
    df[['Dates','Time']] = df['Date'].str.split('T',expand=True)
    df.drop(columns=['Date','Time'],inplace=True)
    return df

def clean_text(df):
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

    def process_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]','',text)
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    df['cleaned_text'] = df['reviewBody'].apply(process_text)
    return df

def sentiment_analysis(df):
    # get sentiment using TextBlob
    def get_sentiment(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity >0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else: return 'Neutral'

    df['Sentiment'] = df['cleaned_text'].apply(get_sentiment)
    return df

def sentiment_counts(df):
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    fig,ax = plt.subplots(figsize=(10,6))
    ax.pie(sentiment_counts['count'],labels=sentiment_counts['Sentiment'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('tab10'), shadow=True, textprops={'color':'w'})
    ax.legend(sentiment_counts['Sentiment'])
    return fig

def sentiment_intensity(df):
    #get most Common Positive and Negative Words
    analyzer = SentimentIntensityAnalyzer()
    vectorizer = CountVectorizer(ngram_range=(2,3),stop_words='english')
    texts = df['cleaned_text'].astype('str').to_list()
    X = vectorizer.fit_transform(texts)
    phrases = vectorizer.get_feature_names_out()
    positive_words = []
    negative_words = []

    for phrase in phrases:
        score = analyzer.polarity_scores(phrase)['compound']
        if score >0.2:
            positive_words.append(phrase)
        elif score < -0.2:
            negative_words.append(phrase)
    most_common_positive_words = Counter(positive_words).most_common(20)
    most_common_negative_words = Counter(negative_words).most_common(20)
    positive_word, positive_words_frequencies = zip(*most_common_positive_words)
    negative_word, negative_words_frequencies = zip(*most_common_negative_words)
    df_words = pd.DataFrame({'Positive_words':positive_word, 'positive_words_frequencies':positive_words_frequencies,'negative_words':negative_word,'negative_words_frequencies':negative_words_frequencies})
    return df_words

#Positive Word Chart
def positive_words(df_words, ax=None):
    sns.set(style='whitegrid')

    # Create a new figure only if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    else:
        fig = ax.figure  # Get the figure from the provided Axes

    sns.barplot(x=df_words['positive_words_frequencies'], 
                y=df_words['Positive_words'], 
                palette='viridis', 
                orient='h', 
                ax=ax)

    # Add text labels to bars
    for i, value in enumerate(df_words['positive_words_frequencies']):
        ax.text(value, i, f"{value}", va='center', ha='left', fontsize=12)

    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Words', fontsize=12)
    
    plt.tight_layout()  # Ensure proper layout

    return fig 


#Negative Words Chart
def negative_Words(df_words, ax=None):
    sns.set(style='whitegrid')

    # Create a new figure only if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    else:
        fig = ax.figure  # Get the figure from the provided Axes

    sns.barplot(x=df_words['negative_words_frequencies'], 
                y=df_words['negative_words'], 
                palette='rocket', 
                orient='h', 
                ax=ax)

    # Add text labels to bars
    for i, value in enumerate(df_words['negative_words_frequencies']):
        ax.text(value, i, f"{value}", va='center', ha='left', fontsize=12)

    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Words', fontsize=12)
    
    plt.tight_layout()  # Ensure proper layout

    return fig  # Return the figure instead of showing it

def rating_counts(df):
    rating_counts = df['ratingValue'].value_counts().sort_index()
    sns.set(style='whitegrid')
    fig,ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=rating_counts.values, y=rating_counts.index,palette='viridis',orient='h')
    ax.set_xlabel('Star Rating')
    ax.set_ylabel('Number of Reviews')
    return fig

def number_of_reviews_over_time(df):
    df['Dates'] = pd.to_datetime(df['Dates'])
    review_per_day = df.groupby(df['Dates'].dt.date).size()
    fig = px.line(
        review_per_day,
        title='Number of Reviews Over Time',
        labels={'Dates': 'Date', 'Number of Reviews': 'Number of Reviews'},
        template='plotly_white'
    )
    
    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def suspicious_comment(df):
    df['Dates'] = pd.to_datetime(df['Dates'])
    review_per_day = df.groupby(df['Dates'].dt.date).size()
    #Analyze text
    review_per_day_zscore = zscore(review_per_day)
    spikes = review_per_day[review_per_day_zscore>3]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfid_matrix = vectorizer.fit_transform(df['reviewBody'])
    cosine_sim = cosine_similarity(tfid_matrix,tfid_matrix)
    duplicate_pairs = np.argwhere(cosine_sim >0.9)
    duplicate_pairs = duplicate_pairs[duplicate_pairs[:,0] != duplicate_pairs[:,1]]
    df['is_suspicious'] = False

    df.loc[df['Dates'].dt.date.isin(spikes.index),'is_suspicious']=True
    df.loc[np.unique(duplicate_pairs.flatten()),'is_suspicious']=True
    suspicious_counts = df['is_suspicious'].value_counts().reset_index()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.pie(suspicious_counts['count'],labels=suspicious_counts['is_suspicious'], startangle=90, autopct='%1.1f%%',colors=sns.color_palette('viridis'), shadow=True, textprops={'color':'w'})
    ax.legend(suspicious_counts['is_suspicious'])
    return fig

