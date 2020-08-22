# READ_ME 
## Stock Market Prediction using Deep Learning
#### Name: Aishwarya Sahani
#### UIN: 652324475

The scope would be to retrieve, utilize & compare the different sources of publicly available information to predict the performance of a stock. The hurdle would be filter out what is relevant to determine the prediction from the various sources. Since there all multiple sources of information. We will try to incorporate the best sources as possible to learn & predict the stock. The filtered text would be represented as word embeddings.. These embeddings will be passed to a Convolution Neural Network to determine the prediction.
The Neural Network Classifier will predict 3 classes as output – 
- Up 
- Down
- Stay 

The Neural Network will compare multiple sources like 
- Company related News articles 
- Company related Twitter articles
- Company related Reddit articles

The developed model can be used to determine the potential investments and risks to aid the investors utilizing publicly available information. The purpose of the project is to predict the stock market performance of companies and provide an insight to the investors on where to invest.

### Steps to run:
1. Download the Datasets from the link [Dataset](https://drive.google.com/drive/folders/1Maau1Uiyq2bpNtfkpgPENta0ExBPXtmO?usp=sharing)
2. Store the "Dataset" in the project folder
3. Download the 300 dimensional [Glove](http://nlp.stanford.edu/data/glove.6B.zip) embeddings
4. Place the file in the Embeddings folder
5. You can run the main.py file 
6. To change the dataset, uncomment the line which imports the data in main.py
7. Run the file & wait for results
8. Check Result.xlsx for a summary of results

### Creating datasets:
1. Check the file - "US-Stock-Symbols" to find the companies to consider. Add or remove the companies based on TICKER symbol & Company Name.
2. Run the InputDataCollector, this file will collect data from Stock Ticker Information, Twiter, News Articles about the company in the files stockInfo, tweets & news respectively.
3. The data in these files collected has to be annotated. This can be done using VLOOKUP function in Excel. Based on the date field, we will have to fetch the corresponding label value for the date from the stockinfo file.

Libraries used:
- numpy:
 for large, multi-dimensional arrays and matrices, and mathematical functions to operate on these arrays
- pandas:
for data manipulation and analysis
- keras:
for neural-network library
- re:
for regular expression operation
- csv:
for handling CSV files
- spacy:
for stop word list
- string:
for string punctuations during cleaning
- tweepy:
for fetching twitter data
- praw:
for fetching reddit data
- datetime:
for date handling & formatting
- yfinance:
for fetching ticker information of stocks
- newsapi:
for fetching news based on query
- matplotlib:
for plotting the model's accuracy & loss
- sklearn:
for evaluating the performance
- tensorflow:
for dataflow and differentiable programming across a range of tasks

### Contents:
The folder consists of 
1. Dataset:
The folder consists of the multiple input datasets like:
    1. company_news: 
    used the existing dataset to check the news related to different companies in different files. The file export.csv consists of the aggregate of news articles from all the companies.
    2. stocknews:
    existing Dow Jones dataset, used to validate Reddit articles
    3. news:
    created the dataset by collecting news articles related to different companies 
    4. stockInfo:
    ticker information for the company_news dataset
    5. tweets
    collected the twitter dataset related to different companies 
    6. US-Stock-Symbols
    The number of companies under consideration
    7. US-Stock-Symbols_v2.0
    The number of companies under consideration for another dataset 
2. Embeddings:
The folder consists of the 300 dimensional Glove embedding file
3. inputDataCollector.py
Run this file to collect data for various sources and store it in the dataset folder. The file collects the stock info, tweet data & news data based on the companies defined in the file "US-Stock-Symbols".
4. main.py
Run this file to run the model. The file calls the preprocess() method & call the build() method.
5. NN.py
The file contains the neural network model 
6. preprocessing.py
The file contains the preprocessing unit
7. utils.py
The file contains the utility methods used during the implementation