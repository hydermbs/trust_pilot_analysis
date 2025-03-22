import nltk
from textblob import TextBlob
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
import re
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.express as px


def feature_data(df):
    df[['Dates','Time']] = df['Date'].str.split('T',expand=True)
    df['Dates'] = pd.to_datetime(df['Dates'])
    df['Year'] = df['Dates'].dt.year
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




def suggestion(df):
    reviews = df['cleaned_text'].tolist()
    token = "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"
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,huggingfacehub_api_token=token)

    def summarize_reviews(review_list):
        template = """
        Analyze the following customer reviews and identify the top 5 issues or areas summerize them in 6 words that need improvement. Provide a concise and actionable list of suggestions for the company to address these issues. Focus on recurring themes, specific pain points, and areas where customer satisfaction is lacking. Format the output as a numbered list, with each item being a clear and actionable recommendation not more than 6 words.
        also suggest the frequency of each issue like how many times that comes in complain
        Reviews:
        {reviews}
        """
        prompt = PromptTemplate(template=template, input_variables=["reviews"])
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        # Join the reviews into a single string
        reviews_text = "\n".join(review_list)
        return llm_chain.invoke({"reviews": reviews_text})

    # Summarize all reviews together
    all_reviews_summary = summarize_reviews(reviews)
    return all_reviews_summary['text']




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

