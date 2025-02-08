import pandas as pd 
from bs4 import BeautifulSoup as bs 
import requests
import json
import random
import time

def extract_data(query):
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    ]
    session = requests.Session()
    def get_response(url):
        headers = {'User-Agent': random.choice(user_agents)}
        response = session.get(url, headers=headers,timeout=10)
        response.raise_for_status()
        return response
    rest_data = []
    page = 1
    while True:
        print(page)
        try:
            url = f"https://uk.trustpilot.com/review/{query}?page={page}"
            response = get_response(url)
            soup = bs(response.content,'html.parser')
            element = soup.find_all('script',{"type":"application/ld+json"})
            data = element[0].text
            json_file = json.loads(data)
            company_name = json_file['@graph'][6]['name']
            data_element = json_file['@graph']
            for item in data_element:
                if item.get('@type') =='Review':
                    author = item.get('author')
                    author_name = author['name']
                    date_published = item.get('datePublished')
                    headline = item.get('headline')
                    review_body = item.get('reviewBody')
                    rating = item.get('reviewRating')
                    rating_value = rating['ratingValue']
                    dict_1 = {'Company Review':company_name,'Author':author_name,'Date':date_published,'headline':headline,'reviewBody':review_body,'ratingValue':rating_value}
                    rest_data.append(dict_1)
            page= page+1
            time.sleep(1)
        except requests.exceptions.HTTPError as e:
            break
    return rest_data
