# load the libraries
from openai import OpenAI
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import seaborn as sns
sns.set_style('darkgrid')
from collections import Counter
from wordcloud import WordCloud

# this library causes the output printed on sys.stdout, which is usually the console, to be encoded  using UTF-8. 
# This is useful when you want to print non-ASCII characters without running into encoding errors.
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def load_api_key_from_file(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config.get('api_key')

def load_transcriptions_from_csv(file_path, column_name):
    # Load the CSV file into a dataframe
    df = pd.read_csv(file_path)
    # let's extract texts from specified column
    transcriptions = df[column_name].astype(str).tolist()
    return transcriptions

api_key = load_api_key_from_file('api_key.json')

client = OpenAI(
    api_key=api_key
)

def sentiment_analysis(transcriptions):
    print("start")
    results = []

    for index, transcription in enumerate(transcriptions):
        try:
            print(index)

            if index % 100 == 0:
                time.sleep(5)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiments of the following sentences. Please consider the tone of the discussion of every sentence, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is positive, negative, or neutral, and provide brief explanations for your analysis where possible for every sentence."
                    },
                    {
                        "role": "user",
                        "content": transcription
                    }
                ]
            )
            # let's extract feeling and reason from the answer
            message_content = response.choices[0].message.content
            sentiment_start = message_content.find("Sentiment:")
            if sentiment_start != -1:
                sentiment_start += len("Sentiment:")
                sentiment_end = message_content.find("\n", sentiment_start)
                sentiment = message_content[sentiment_start:sentiment_end].strip()
                reason_start = message_content.find("Explanation:") + len("Explanation:")
                reason = message_content[reason_start:].strip()
            else:
                sentiment_end = message_content.find(".")
                sentiment = message_content[:sentiment_end].split()[-1]
                if sentiment not in ["positive", "negative"]:
                    sentiment = "neutral"
                reason = message_content[sentiment_end+1:].strip()
            results.append({
                "tweet": transcription,
                "sentiment": sentiment,
                "reason": reason.replace("\n", " ")
            })
            print(transcription)
            print(sentiment)
            print(reason)
            print("\n\n")
        except Exception as e:
            print(e)
            results.append({
                "tweet": transcription,
                "sentiment": "error",
                "reason": "error"
            })
    return results

# Specify CSV file name and column name
file_path = "twitter_ElonMusk_restricted_cleaned.csv"
column_name = "tweet"  

# Load transcripts from CSV file
transcriptions = load_transcriptions_from_csv(file_path, column_name)

# let's perform the sentiment analysis on the extracted texts
results = sentiment_analysis(transcriptions)

# let's create a new dataframe with the results
results_df = pd.DataFrame(results)

# let's specify the path to the new CSV file to save
output_file_path = "sentiment_analysis_results_ChatGPT_ElonMusk.csv"

# let's save the dataframe to a new CSV file
results_df.to_csv(output_file_path, index=False)

print("Risultati salvati con successo in:", output_file_path)

# ----------------------------------------------------------------------------
# CASE STUDY: Sentiment predicition to dataset of reply to a Elon Musk's tweet
# ----------------------------------------------------------------------------

# Load the output dataset created by ChatGPT
sentiment_results_GPT = pd.read_csv('sentiment_analysis_results_ChatGPT_ElonMusk.csv')

# checking my columns in my data
print(sentiment_results_GPT.keys())

# looking into my data
print(sentiment_results_GPT.head())

# checking last 5 entries
print(sentiment_results_GPT.tail())

# checking info our data
print(sentiment_results_GPT.info())

# describing our data
print(sentiment_results_GPT.describe())

# checking unique values 
print(sentiment_results_GPT.nunique())

# there are several duplicate tuples in the dataset, so I display them
print(sentiment_results_GPT['tweet'].value_counts())

# let's eliminate the duplicates, keeping only the first occurrence of each unique value
sentiment_results_GPT = sentiment_results_GPT.drop_duplicates(subset='tweet', keep='first')

# checking that there are no more duplicates
print(sentiment_results_GPT.nunique())

# checking null values in my data
print(sentiment_results_GPT.isnull().sum())
# if there were any, remove rows or columns with null values
print(sentiment_results_GPT.dropna(inplace=True))

# checking type values in my data
print(sentiment_results_GPT.dtypes)

# get Dimension of the Data
print(sentiment_results_GPT.shape)

# EDA

# let's convert all values ​​in the 'sentiment' column to lowercase
sentiment_results_GPT['sentiment'] = sentiment_results_GPT['sentiment'].str.lower()

# visualize the heatmap of null data values 
plt.figure(figsize=(10,10))
sns.heatmap(sentiment_results_GPT.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.title('Null data values ')
plt.show()

# Rimuovere righe con valori NaN nella colonna 'tweet'
sentiment_results_GPT = sentiment_results_GPT.dropna(subset=['tweet'])

# Bar plot of positives, negatives and neutrals tweets 
positive_count = sentiment_results_GPT[sentiment_results_GPT['sentiment'] == 'positive'].shape[0]
negative_count = sentiment_results_GPT[sentiment_results_GPT['sentiment'] == 'negative'].shape[0]
neutral_count = sentiment_results_GPT[sentiment_results_GPT['sentiment'] == 'neutral'].shape[0]
plt.bar(['Positivi', 'Neutrali', 'Negativi'], [positive_count, neutral_count, negative_count], color=['green', 'yellow', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Numero di Tweet')
plt.title('Distribuzione dei Tweet dopo Sentiment Prediction con ChatGPT')
plt.show()

print() 
# let's print the percentage of each target variable values after cleaning
print('Positivi', round(sentiment_results_GPT['sentiment'].value_counts()['positive']/len(sentiment_results_GPT) * 100),'% -->', sentiment_results_GPT['sentiment'].value_counts()['positive'])
print('Neutrali', round(sentiment_results_GPT['sentiment'].value_counts()['neutral']/len(sentiment_results_GPT) * 100),'% -->', sentiment_results_GPT['sentiment'].value_counts()['neutral'])
print('Negativi', round(sentiment_results_GPT['sentiment'].value_counts()['negative']/len(sentiment_results_GPT) * 100),'% -->', sentiment_results_GPT['sentiment'].value_counts()['negative'])
print() 

# PLOT THE WORDCLOUD
    # wordcloud positivo
        # Most Common Words in Positive Tweets
positive = sentiment_results_GPT[sentiment_results_GPT['sentiment'] == 'positive']
positive_tweets_list = positive['tweet'].tolist()
positive_tweets_str = " ".join(positive_tweets_list)

plt.figure(figsize = (10,10))
plt.imshow(WordCloud().generate(positive_tweets_str))
plt.title('Word Cloud positiva')
plt.show()

    # wordcloud neutrale
        # Most Common Words in Neutral Tweets
neutral = sentiment_results_GPT[sentiment_results_GPT['sentiment'] == 'neutral']
neutral_tweets_list = neutral['tweet'].tolist()
neutral_tweets_str = " ".join(neutral_tweets_list)

plt.figure(figsize = (10,10))
plt.imshow(WordCloud().generate(neutral_tweets_str))
plt.title('Word Cloud neutrale')
plt.show()

    # wordcloud negativo
        # Most Common Words in Negative Tweets
negative = sentiment_results_GPT[sentiment_results_GPT['sentiment'] == 'negative']
negative_tweets_list = negative['tweet'].tolist()
negative_tweets_str = " ".join(negative_tweets_list)

plt.figure(figsize = (10,10))
plt.imshow(WordCloud().generate(negative_tweets_str))
plt.title('Word Cloud negativa')
plt.show()


# Put all tweets into a large list and generic wordcloud 
sentences = sentiment_results_GPT['tweet'].tolist()
len(sentences)
sentences_as_one_string =" ".join(sentences)
plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(sentences_as_one_string))
plt.title('Word Cloud generica')
plt.show()
