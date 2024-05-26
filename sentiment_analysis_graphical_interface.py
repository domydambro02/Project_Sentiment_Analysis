import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from wordcloud import WordCloud
from collections import Counter


st.set_option('deprecation.showPyplotGlobalUse', False)

# let's set the width of the page
st.set_page_config(layout="wide")

# Create a Streamlit title
st.markdown("<h1 style='text-align: center; font-size: 60px'>Real-Time Sentiment Analysis</h1>", unsafe_allow_html=True)

# Load and center an image
# Creating a central column
col1, col2, col3 = st.columns([2, 1.2, 2])  # Regola le proporzioni delle colonne come desideri

# Center column for image
with col2:
    image_url = "sentiment_emoji.png" 
    st.image(image_url, width=400)

# Sidebar title
st.sidebar.title('Carica i file CSV')

# Upload the first CSV file (ML)
uploaded_file_1 = st.sidebar.file_uploader('Carica il primo CSV (ML)', type='csv')

# Upload the second CSV file (GPT)
uploaded_file_2 = st.sidebar.file_uploader('Carica il secondo CSV (GPT)', type='csv')

# let's add spacing for better readability
st.sidebar.markdown('---')

# Tip for the user
st.sidebar.write("Assicurati che i file siano corretti prima di proseguire.")

if uploaded_file_1 is not None and uploaded_file_2 is not None:
    # let's divide the interface into two columns
    col1, col2 = st.columns(2)

    with col1.container():
        st.markdown("<h1 style='text-align: center; color: red'>Machine Learning <br> SUPPORT VECTOR MACHINE</h1>", unsafe_allow_html=True)
        data1 = pd.read_csv(uploaded_file_1)
    
        # Checking columns in the data
        st.subheader("Numero di colonne nei tuoi dati:")
        st.write(data1.columns)

        # Looking into the data
        st.subheader("Testa dei tuoi dati:")
        st.write(data1.head())

        # Checking last 5 entries
        st.subheader("Tail of your data:")
        st.write(data1.tail())

        # Checking unique values
        st.subheader("Unique values in your data:")
        st.write(data1.nunique())

        # Checking for duplicate tuples
        st.subheader("Duplicate tuples in your data:")
        st.write(data1['tweet'].value_counts())

        # Removing duplicates
        data1.drop_duplicates(subset='tweet', keep='first', inplace=True)

        # Checking for null values
        st.subheader("Null values in your data:")
        st.write(data1.isnull().sum())

        # Dropping rows with null values
        data1.dropna(subset=['tweet'], inplace=True)

        # Checking data types
        st.subheader("Data types in your data:")
        st.write(data1.dtypes)

        # Get dimension of the data
        st.subheader("Dimension of your data:")
        st.write(data1.shape)

        # Convert all values in the 'sentiment' column to lowercase
        data1['sentiment_prediction'] = data1['sentiment_prediction'].str.lower()

        with st.container():
            # let's remove rows with Nan values in the 'tweet' column
            data1.dropna(subset=['tweet'])

            # Bar plot of positive and negative tweets
            st.subheader("Distribution of tweets by sentiment:")
            plt.figure(figsize=(8, 6))
            positive_count = data1[data1['sentiment_prediction'] == 'positive'].shape[0]
            negative_count = data1[data1['sentiment_prediction'] == 'negative'].shape[0]
            neutral_count = data1[data1['sentiment_prediction'] == 'neutral'].shape[0]
            plt.bar(['Positivi', 'Neutrali', 'Negativi'], [positive_count, neutral_count, negative_count], color=['green', 'yellow', 'red'])
            plt.xlabel('Sentiment')
            plt.ylabel('Number of Tweets')
            st.pyplot()

            # Wordcloud of positive tweets
            st.subheader("Positive Word Cloud:")
            plt.figure(figsize=(8, 6))
            positive_tweets_list = data1[data1['sentiment_prediction'] == 'positive']['tweet'].tolist()
            positive_tweets_str = " ".join(positive_tweets_list)
            plt.figure(figsize=(10,10))
            plt.imshow(WordCloud().generate(positive_tweets_str))
            plt.axis('off')
            st.pyplot()

            # Wordcloud of neutral tweets
            st.subheader("Neutral Word Cloud:")
            plt.figure(figsize=(8, 6))
            neutral_tweets_list = data1[data1['sentiment_prediction'] == 'neutral']['tweet'].tolist()
            neutral_tweets_str = " ".join(neutral_tweets_list)
            plt.figure(figsize=(10,10))
            plt.imshow(WordCloud().generate(neutral_tweets_str))
            plt.axis('off')
            st.pyplot()

            # Wordcloud of negative tweets
            st.subheader("Negative Word Cloud:")
            plt.figure(figsize=(8, 6))
            negative_tweets_list = data1[data1['sentiment_prediction'] == 'negative']['tweet'].tolist()
            negative_tweets_str = " ".join(negative_tweets_list)
            plt.figure(figsize=(10,10))
            plt.imshow(WordCloud().generate(negative_tweets_str))
            plt.axis('off')
            st.pyplot()

            # Wordcloud of all tweets
            st.subheader("Generic Word Cloud:")
            plt.figure(figsize=(8, 6))
            sentences = data1['tweet'].tolist()
            sentences_as_one_string = " ".join(sentences)
            plt.figure(figsize=(10,10))
            plt.imshow(WordCloud().generate(sentences_as_one_string))
            plt.axis('off')
            st.pyplot()

    with col2.container():
        st.markdown("<h1 style='text-align: center; color: gold'>Generative Pre-Trained Transformer <br> GPT-3.5-TURBO</h1>", unsafe_allow_html=True)
        data2 = pd.read_csv(uploaded_file_2)

        # Checking columns in the data
        st.subheader("Columns in your data:")
        st.write(data2.columns)

        # Looking into the data
        st.subheader("Head of your data:")
        st.write(data2.head())

        # Checking last 5 entries
        st.subheader("Tail of your data:")
        st.write(data2.tail())

        # Checking unique values
        st.subheader("Unique values in your data:")
        st.write(data2.nunique())

        # Checking for duplicate tuples
        st.subheader("Duplicate tuples in your data:")
        st.write(data2['tweet'].value_counts())

        # Removing duplicates
        data2.drop_duplicates(subset='tweet', keep='first', inplace=True)

        # Checking for null values
        st.subheader("Null values in your data:")
        st.write(data2.isnull().sum())

        # Dropping rows with null values
        data2.dropna(subset=['tweet'], inplace=True)

        # Checking data types
        st.subheader("Data types in your data:")
        st.write(data2.dtypes)

        # Get dimension of the data
        st.subheader("Dimension of your data:")
        st.write(data2.shape)

        # Convert all values in the 'sentiment' column to lowercase
        data2['sentiment'] = data2['sentiment'].str.lower()

        # Container per i grafici
        with st.container():
            # let's remove rows with Nan values in the 'tweet' column
            data2.dropna(subset=['tweet'])
            plt.figure(figsize=(8, 6))
            # Bar plot of positives, negatives and neutrals tweets 
            st.subheader("Distribution of tweets by sentiment:")
            positive_count = data2[data2['sentiment'] == 'positive'].shape[0]
            negative_count = data2[data2['sentiment'] == 'negative'].shape[0]
            neutral_count = data2[data2['sentiment'] == 'neutral'].shape[0]
            plt.bar(['Positivi', 'Neutrali', 'Negativi'], [positive_count, neutral_count, negative_count], color=['green', 'yellow', 'red'])
            plt.xlabel('Sentiment')
            plt.ylabel('Number of Tweets')
            st.pyplot()

            # Wordcloud of positive tweets
            st.subheader("Positive Word Cloud:")
            positive_tweets_list = data2[data2['sentiment'] == 'positive']['tweet'].tolist()
            positive_tweets_str = " ".join(positive_tweets_list)
            plt.figure(figsize=(10,10))
            plt.imshow(WordCloud().generate(positive_tweets_str))
            plt.axis('off')
            st.pyplot()

            # Wordcloud of neutral tweets
            st.subheader("Neutral Word Cloud:")
            neutral_tweets_list = data2[data2['sentiment'] == 'neutral']['tweet'].tolist()
            neutral_tweets_str = " ".join(neutral_tweets_list)
            plt.figure(figsize=(10,10))
            plt.imshow(WordCloud().generate(neutral_tweets_str))
            plt.axis('off')
            st.pyplot()

            # Wordcloud of negative tweets
            st.subheader("Negative Word Cloud:")
            negative_tweets_list = data2[data2['sentiment'] == 'negative']['tweet'].tolist()
            negative_tweets_str = " ".join(negative_tweets_list)
            plt.figure(figsize=(10,10))
            plt.imshow(WordCloud().generate(negative_tweets_str))
            plt.axis('off')
            st.pyplot()

            # Wordcloud of all tweets
            st.subheader("Generic Word Cloud:")
            sentences = data2['tweet'].tolist()
            sentences_as_one_string = " ".join(sentences)
            plt.figure(figsize=(10,10))
            plt.imshow(WordCloud().generate(sentences_as_one_string))
            plt.axis('off')
            st.pyplot()