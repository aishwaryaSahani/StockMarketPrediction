# CS521: Statistical Natural Language Processing
# University of Illinois at Chicago
# Spring 2020
# Stock Market Prediction using Deep Learning
# =========================================================================================================

from newsapi import NewsApiClient
import yfinance as yf
import pandas as pd
import tweepy as tw
import praw
import datetime as dt

def fetchTweets(name, companydf, tickerSymbol):
    consumer_key = "9L3hNv8VYRnwITJIbTZXM0TQd"
    consumer_secret = 'MMCv15gDOw7sgR5BOI86imOx8Kdvd6j7OHRyowHzjQmfJDKBca'
    access_token = '335826064-jG7JAhvn99aqKyMHjfv3xjotMk1N3CHsyBEQp95u'
    access_token_secret = 'UZYRv5KKd1LYFUj9E5BGZ9DiRQodWW5BGu7GJnwPScSYl'
    
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    
    tweets = []
    
    dateList = companydf.index.values
    # Collect tweets
    for date in dateList:
        index = 0
        tweetdate = dt.datetime.strptime(str(date)[:10], '%Y-%m-%d').date()
        for tweet in tw.Cursor(api.search, q=name, until=tweetdate , lang="en").items(50):
            parsed_tweet = {}
            parsed_tweet["Date"] = dt.datetime.strptime(str(tweetdate)[:10], '%Y-%m-%d').date().strftime('%m/%d/%Y')
            parsed_tweet["Tweet"] = tweet.text.encode('utf-8')
            index+=1
            tweets.append(parsed_tweet)

    tweetDf = pd.DataFrame(tweets)
    helperValues = [None]*len(tweetDf)
    labelValues = companydf["Label"]
    index = 0
    for i in (list(tweetDf.index.values)):
        date_time = tweetDf["Date"][i]
        helperValues[index] = str(date_time)+tickerSymbol
        index+=1
    tweetDf.insert(1, "Helper",helperValues, True)
    tweetDf.insert(4, "Label",labelValues, True)
    return tweetDf
        
def fetchStockValue(tickerSymbol, startDate):
    #get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)
#     #get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start=startDate.date(), end=dt.date.today())
    tickerDf = pd.DataFrame(tickerDf)
    values = [tickerSymbol]*len(tickerDf)
    tickerDf.insert(0, "Symbol",values, True)
    
    helperValues = [None]*len(tickerDf)
    values = [None]*len(tickerDf)
    index =0
    for i in (list(tickerDf.index.values)):
        date_time = dt.datetime.strptime(str(i)[:10], '%Y-%m-%d').date().strftime('%Y-%m-%d')
        if abs(tickerDf["Open"][i]-tickerDf["Close"][i]) <0.01*tickerDf["Open"][i]:
            values[index] = 1 
        elif tickerDf["Open"][i]<tickerDf["Close"][i] :
            values[index] = 2
        else: 
            values[index] = 0
        
        helperValues[index] = str(date_time)+tickerSymbol
        index+=1
    tickerDf.insert(1, "Helper",helperValues, True)
    tickerDf.insert(2, "Label",values, True)
    
    return pd.DataFrame(tickerDf)

def fetchNewsArticles(name, companydf, tickerSymbol, headers_row_list):
    dateList = companydf.index.values
    newsapi = NewsApiClient(api_key='b7de039e294740bb84d8dff8c2bbf97d')

    all_rows =[]   
    for date in dateList:
        response = newsapi.get_everything(q=name,
                              sources= "bloomberg,reuters,cnn,business-insider,fox-news,google-news,msnbc,nbc-news,the-huffington-post,the-wall-street-journal",
                              language='en',
                              from_param= str(date)[:10],
                              to= str(date)[:10],
                              sort_by='publishedAt') 
        articles_list = response['articles']
        date_time = dt.datetime.strptime(str(date)[:10], '%Y-%m-%d').date().strftime('%m/%d/%Y')
        
        row_list = []  
        index = 0
        for article in articles_list:
            row_list = [] 
            title = article["title"] if article["title"]!=None else ""
            description = article["description"] if article["description"]!=None else ""
            row_list.append(date_time)
            row_list.append(title+" "+description)
            
            all_rows.append(row_list)
    
    newsdf = pd.DataFrame(all_rows, columns=headers_row_list)
    
    values = [tickerSymbol]*len(newsdf)
    newsdf.insert(0, "Symbol",values, True)
    
    helperValues = [None]*len(newsdf)
    values = [None]*len(newsdf)
    labelValues = companydf["Label"]
    index =0
    for i in (list(newsdf.index.values)):
        date_time = newsdf["Date"][i]
        helperValues[index] = str(date_time)+tickerSymbol
        index+=1
    newsdf.insert(2, "Helper",helperValues, True)
    newsdf.insert(4, "Label",labelValues, True)
    return newsdf
    
def main():
    
    stockdf = pd.read_csv("Dataset/US-Stock-Symbols_v2.0.csv")
    existTickerDf = pd.read_csv("Dataset/stockInfo.csv")
    existNewsDf = pd.read_csv("Dataset/news.csv")
    existTweetsDf = pd.read_csv("Dataset/tweets.csv")
    
    headers_row_list = [] 
    headers_row_list.append("Date")
    headers_row_list.append("Text")
    df = pd.DataFrame(columns=headers_row_list)
    tickerDf = pd.DataFrame()
    twitterDf = pd.DataFrame()
    for company in range(len(stockdf)): 
        name = stockdf["Name"][company]
        symbol =  stockdf["Symbol"][company]  
        if symbol not in set(existNewsDf["Symbol"]):
            print("Collection of input data for the company "+symbol+" -> "+name+" started")
            startDate = pd.to_datetime('now') + pd.offsets.DateOffset(months=-1)
            companydf = fetchStockValue(symbol, startDate)
            newsdf = fetchNewsArticles(name, companydf, symbol, headers_row_list)
            df = df.append(newsdf)
            tickerDf = tickerDf.append(companydf)
            
#             tweetsDF = fetchTweets(name, companydf, symbol)
#             twitterDf = twitterDf.append(tweetsDF)
    existTickerDf =  tickerDf
    pd.DataFrame(existTickerDf).to_csv("Dataset/stockInfo_2.csv")
    
    existNewsDf =  existNewsDf.append(df)
    pd.DataFrame(existNewsDf).to_csv('Dataset/news_2.csv')

#     existTweetsDf =  existTweetsDf.append(twitterDf) 
#     pd.DataFrame(existTweetsDf).to_csv("Dataset/tweets_2.csv")
    
    print("Collection of input data completed")
    
if __name__ == '__main__':
    main()