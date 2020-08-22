# CS521: Statistical Natural Language Processing
# University of Illinois at Chicago
# Spring 2020
# Stock Market Prediction using Deep Learning
# =========================================================================================================

import preprocessing as pp 
import NN as nn 
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer     
from keras.preprocessing.sequence import pad_sequences


def main():
   
    # Load dataset by uncommenting any of the required dataset. 
    # Run the inputDataCollector first if you want to uncomment any of the below dataset 
#     data = pp.load_data('Dataset/stocknews/RedditNews.csv')
#     data = pp.load_data('Dataset/tweets.csv')
#     data = pp.load_data('Dataset/news.csv')
    data = pp.load_data('Dataset/news.csv')
    data = pd.DataFrame(data)
    
    # This line is to concatenate the title & summary in a news articles for the export dataset
    text = list(data["TITLE"]+ data["SUMMARY"])
    
    # Uncomment the below line for all the other datasets
    #     text = list(data["Text"])
    
    # calculate the labels for the task
    labels = np.array(list(data["Label"]))
    
    # preprocess the dataset
    text = pp.preprocess(text)

    #tokenize the text
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(text) 
    sequences = tokenizer.texts_to_sequences(text) 
    word_index = tokenizer.word_index 

    # padding sequences to ensure all rows are of equal length
    max_review_length = 1000
    text = pad_sequences(sequences, maxlen=max_review_length, padding='pre', truncating='pre')
    
    #split the data into 80-20 train-test
    X_train = text[:int(0.8*len(text))]
    X_val = text[-int(0.2*len(text)):]
    y_train = labels[:int(0.8*len(text))]
    y_val = labels[-int(0.2*len(text)):]
    
    y_train = [int(i) for i in y_train] 
    y_val = [int(i) for i in y_val] 

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    
    # build the model & validate the results
    nn.build(word_index, X_train, y_train, X_val, y_val)

if __name__ == '__main__':
    main()