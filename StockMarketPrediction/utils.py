# CS521: Statistical Natural Language Processing
# University of Illinois at Chicago
# Spring 2020
# Stock Market Prediction using Deep Learning
# =========================================================================================================

import csv
import numpy as np
import string
import re
import spacy
spacy_nlp = spacy.load('en_core_web_sm')

# dict to keep track of all the unique tokens
unigrams = {}

# Function to load the CSV file i.e the dataset
# Arguments:
# filepath: A string containing the file path of the input document
# Returns: data (list)
# Where, data (list) is a list of rows
def load_data(filepath):
    data = []
    with open(filepath, encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        for row in reader:
            record = {}
            for idx, field in enumerate(header):
                if idx<len(row):
                    record[field] = row[idx]
                else:
                    record[field] = ""
            data.append(record)
    return data

# Function to load the glove embeddings
# Returns: embeddings_dict (dict)
# Where, embeddings_dict (dict) is a dict of words as keys & the vector as values 
def load_glove():
    embeddings_dict= {}
    with open("Embeddings/glove.6B.300d.txt", 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

# Function to split a document into a list of tokens
# Arguments:
# doc: A string containing input document
# Returns: tokens (list)
# Where, tokens (list) is a list of tokens that the document is split into
def get_tokens(doc):
    whiteSpaceToken = re.split("\\W+", doc)
    tokens = []
    # replacing a list of special characters by blank space
    transtable = str.maketrans('', '', string.punctuation)
    for word in whiteSpaceToken:
        tokens.append(word.lower().translate(transtable))
    return tokens


# Function to remove the stopwords & return tokens in the document
# Arguments:
# tokens: A list of tokens that the document is split into
# Returns: tokenList (list)
# Where, tokenList (list) is a list of non stop word tokens in the document
def removeStopWords(tokens):
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    tokenList= []
    for word in tokens:
        if(word not in spacy_stopwords):
            tokenList.append(word)
    return tokenList 

# Function to count all the unigrams
# Arguments:
# tokens: A list of tokens that the document is split into
# Returns: unigrams (dict)
# Where, unigrams (dict) is a dict with the key as the word and the value as the number of occurences 
def countUnigrams(word):
    # finding unique words
    unigrams[word] = unigrams.get(word, 0)+1
    return unigrams

# Function to clean the input text
# Arguments:
# tokens: A list of tokens that the document is split into
# Returns: cleanText (list)
# Where, cleanText (list) is a list of cleaned, preprocessed tokens 
def cleanText(tokens):
    cleanText=[]
    for word in tokens:
        # cleaning text by keeping only alphabetical words & words keep with more than 2 letters
        if (word.isalpha() and len(word)>2):
            cleanText.append(word)
            countUnigrams(word)
    return cleanText