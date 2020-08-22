# CS521: Statistical Natural Language Processing
# University of Illinois at Chicago
# Spring 2020
# Stock Market Prediction using Deep Learning
# =========================================================================================================

from utils import *

# Function to retreive multiple columns in the dataset into one
# Arguments:
# data: A list rows with single columns as news 
# Returns: new_text (list)
# Where, new_text (list) is a list of news
def retreiveNews(data):
    new_text = []
    for r in data:
        for field, values in r.items():
            if ("text") in field :
                news += values+"\n"
        new_text.append(news)
    return new_text

# Function to concatenate multiple columns in the dataset into one
# Arguments:
# data: A list rows with multiple columns as news
# Returns: new_text (list)
# Where, new_text (list) is a list of text of each row
def concatenateNews(data):
    new_text = []
    for r in data:
        news = ""
        for field, values in r.items():
            if ("Top") in field :
                news += values+"\n"
        new_text.append(news)
    return new_text

# Function to preprocess the documents 
# Arguments:
# cleanWords: A list rows of the dataset 
# Returns: cleanWords (list)
# Where, cleanWords (list) is a list of cleaned, preprocessed tokens
def preprocess(textDocs):     
    tokens = []
    for doc in textDocs:
        tokens.append(get_tokens(doc))
    
    tokensWOStopWords = []
    for doc in tokens:
        tokensWOStopWords.append(removeStopWords(doc))

    cleanWords = []
    for doc in tokensWOStopWords:
        cleanWords.append(cleanText(doc))
    
    return cleanWords
