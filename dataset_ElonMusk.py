# load the libraries
import time
import joblib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from string import punctuation
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk # Natural Language tool kit 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
stopwords.words('english')
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer,ToktokTokenizer
import re
import html
import emoji
from sklearn.decomposition import TruncatedSVD
import matplotlib.patches as mpatches
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Preparing the data
    # Preprocessing 

# Read the data
data = pd.read_csv('twitter_ElonMusk.csv')

# checking my columns in my data
print(data.keys())

# looking into my data
print(data.head())

# checking last 5 entries
print(data.tail())

# checking the content of the columns
# ATTENTION: id, created_at, media, screen_name, name, profile_image_url, in_reply_to, 
    # retweeted_status, quoted_status, favorite_count, retweet_count, bookmark_count, quote_count,
    # reply_count views_count,favorited, retweeted, bookmarked, metadata IRRELEVANT INFORMATION --> let's eliminate them

# let's select only the desired column (the one containing the tweets)
selected_columns = ['full_text']

# let's modify the dataset with only the desired column
data = data[selected_columns]

# let's rename the 'full_text' column to 'tweet'
data.rename(columns={'full_text': 'tweet'}, inplace=True)

# checking info our data
print(data.info())

# looking into my data
print(data.head())

# checking last 5 entries
print(data.tail())

# describing our data
print(data.describe())

# checking unique values 
print(data.nunique())

# there are several duplicate tuples in the dataset, so I display them
print(data['tweet'].value_counts())

# let's eliminate the duplicates, keeping only the first occurrence of each unique value
data = data.drop_duplicates(subset='tweet', keep='first')

# checking that there are no more duplicates
print(data.nunique())

# checking null values in my data
print(data.isnull().sum())

# if there were any, remove rows or columns with null values
print(data.dropna(inplace=True))

# checking type values in my data
print(data.dtypes)

# get Dimension of the Data
print(data.shape)


# Exploratory Data Analysis (EDA)

# visualize the heatmap of null data values 
plt.figure(figsize=(10,10))
sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.title('Valori di dati nulli')
plt.show()

# let's get the length of the messages
data['length'] = data['tweet'].apply(len)

# let's view the shortest message 
min_length_index = data['length'].idxmin()  
tweet_with_min_length = data.loc[min_length_index, 'tweet'] 
print(tweet_with_min_length)  
print()
# let's view the message with mean length 
avg_length = data['length'].mean()
closest_to_average_index = (data['length'] - avg_length).abs().idxmin()
tweet_with_avg_length = data.loc[closest_to_average_index, 'tweet']
print(tweet_with_avg_length)
print()
# let's view the longest message 
max_length_index = data['length'].idxmax()
tweet_with_max_length = data.loc[max_length_index, 'tweet'] 
print(tweet_with_max_length)

# plot the histogram of the length column before cleaning up
data['length'].plot(bins = 100, kind='hist')
plt.title('Distribuzione delle lunghezze dei tweet prima del filtraggio')
plt.xlabel('Lunghezza')
plt.ylabel('Frequenza')
plt.show()

# checking Most Common Words 
count = Counter(" ".join(data['tweet']).split()).most_common(20)
data_tweet = pd.DataFrame.from_dict(count)
data_tweet = data_tweet.rename(columns={0: "common_words", 1 : "count"})
plt.barh(data_tweet['common_words'], data_tweet['count'], color='skyblue')
plt.title('Parole comuni nel testo')
plt.xlabel('Conteggio')
plt.ylabel('Parola')
plt.show()

# let's store the modified DataFrame in a new CSV file
data.to_csv('twitter_ElonMusk_restricted.csv', index=False)

print(data.iloc[9,0])
# From the result above, most common words in target class variable contains so many stop words like "to, the, a, and, i, of, you, for, in, is" and etc. 
# Which mean that the data_tweet still have untidy data. There are so many punctuation, constraction, special characters, elongation words, twitter tags and hastag. 
# Before we go to the modelling steps we need to removing those all, and make the text become tidy data text.


# Text Mining
    # In this text mining process we will exploring and analyzing unstructured text data

# let's assign the columns of the DataFrame to variables to make it easier to work with
tweet = data['tweet']

print(len(tweet))

# Let's define a pipeline to clean up all the messages 
    # The pipeline performs the following:
        # (1) expand contractions
        # (2) remove tags and links (NO hastag)        
        # (3) remove HTML
        # (4) uppercase to lowercase
        # (5) remove special character, punctuation and Non-ASCII Character
        # (6) remove emojis
        # (7) replace elongated words
        # (8) tokenization
        # (9) remove stopwords
        # (10) lemmatization
        # (11) drop numbers

# (1) Expanding Contraction
    # Contractions is kind of word like I'm, ain't, who's, etc. By expanding the contractions "I'm" will become "I am", "ain't" will become "am not", and so on.
contractions_dict = {   
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "iit will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"\\bu\\b": "you", # solo quando la "u" è una parola singola
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}
def expand_contractions(text, contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match) \
            if contractions_dict.get(match) \
            else contractions_dict.get(match.lower())
        expanded_contraction = expanded_contraction
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
def cons(text):
    text=expand_contractions(text,contractions_dict)
    return text
tweet = tweet.apply(cons)

# (2) Removing tags and links 
def remove_usernames_links(text):
   # Remove usernames (starting with '@')
  text = re.sub(r'@\w+', '', text)
  # Remove links (http or www protocol, followed by non-whitespace characters)
  text = re.sub(r'http\S+|www\S+', '', text)
  return text.strip()
tweet = tweet.apply(remove_usernames_links)

# (3) Remove HMTL 
def remove_html(text):
    # Use the html.unescape function to convert HTML entities
    text = html.unescape(text)
    return text
tweet = tweet.apply(remove_html)

# (4) To lowercase
    # Change all uppercase character to be lowercase character. For example "Pretty" to be "pretty" or "BEAUTY" to be "beauty"
def to_lower(text):
    return ' '.join([w.lower() for w in word_tokenize(text)])
tweet = tweet.apply(to_lower)

# (5) Remove Special Character, Punctuation and Non-ASCII characters
    # Removing all special character like .?!/@# etc.
def remove_special_characters(text, remove_digits = True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
    # Removing all punctuation from text
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
    # Remove Non-ASCII Character like â ï ½ ð ª ³ æ
def strip_non_ascii(text):
    return ''.join([c for c in text if ord(c) < 128])

def clean_text(text):
    text = remove_special_characters(text)
    text = strip_punctuation(text)
    text = strip_non_ascii(text)
    return text
tweet = tweet.apply(clean_text)

# (6) Remove Emojis
def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')
tweet = tweet.apply(remove_emoji)

# (7) Replace Elongated Words
    # Replace all elongated words with appropriate words with 3 or + repeated consecutive characters, 
    # so as to avoid eliminating natural doubles. For example "soooooo" to be "so" or "looooong" to be "long"
def replaceElongated(word):
    repeat_regexp = re.compile(r'(\w*)(\w)\2{2,}(\w*)')
    repl = r'\1\2\3'
    if wordnet.synsets(word):
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:      
        return replaceElongated(repl_word)
    else:       
        return repl_word
tweet = tweet.apply(replaceElongated)

# (8) Tokenization
    # Tokenization is splitting sentences into smaller unit, such as terms or word.
def tokenize_text(text):
    tokenizer = ToktokTokenizer()
    tokenized_text = tokenizer.tokenize(text)
    return ' '.join(tokenized_text)
tweet = tweet.apply(tokenize_text)

# (9) Removing Stopwords
   # Remove stopwords like "is, the, with, etc" since they don't have usefull information
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    words_without_stopwords = [w for w in text.split() if w.lower() not in stopword_list]
    return ' '.join(words_without_stopwords)
tweet = tweet.apply(lambda x : remove_stopwords(x))

# (10) Stemming
#    # Stemming is the process of reducing a word to its word stem. For example "Consulting" to be "consult"
# def stem_text(text):
#     stemmer = PorterStemmer()
#     tokenized_text = word_tokenize(text)
#     stemmed_text = [stemmer.stem(word) for word in tokenized_text]
#     return ' '.join(stemmed_text)
# tweet = tweet.apply(stem_text)

# (10) Lemmatization
    # Lemmatization uses a lexical database and linguistic rules to obtain the correct basic form of a word, called a lemma.
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokenized_text = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in tokenized_text]
    return ' '.join(lemmatized_text)
tweet = tweet.apply(lemmatize_text)

# (11) Drop Numbers
    # Remove numbers from text, since numbers doesn't give much importance to get the main words.
def remove_numbers_and_cleanup(text):
    # Utilizza un'espressione regolare per rimuovere i numeri e pulisce gli spazi extra
    text_without_numbers = re.sub(r'\b\d+\b', '', text)  # Rimuove i numeri
    cleaned_text = re.sub(r'\s+', ' ', text_without_numbers)  # Pulisce gli spazi extra
    return cleaned_text.strip()
tweet = tweet.apply(remove_numbers_and_cleanup)

# let's create the new clean dataset
data_tweet_clean = pd.concat([tweet],axis=1)

# Sostituire valori vuoti con NaN nella colonna 'tweet'
data_tweet_clean['tweet'] = data_tweet_clean['tweet'].replace('', np.nan)
# Drop rows
data_tweet_clean.dropna(subset=['tweet'], inplace=True)

# Plot the histogram of the length column after cleaning up
data_tweet_clean['length'] = data_tweet_clean['tweet'].apply(len)
data_tweet_clean['length'].plot(bins = 100, kind='hist')
plt.title('Distribuzione delle lunghezze dei tweet dopo il filtraggio')
plt.xlabel('Lunghezza del testo')
plt.ylabel('Frequenza')
plt.show()

# let's compare the length before (after balancing) and after cleaning up
    # setting the figure and axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot for balanced dataset
data['length'].plot(ax=axes[0], bins=100, kind='hist', color='blue')
axes[0].set_title('Distribuzione delle lunghezze dei tweet prima del filtraggio')
axes[0].set_xlabel('Lunghezza')
axes[0].set_ylabel('Frequenza')

# Plot for cleaned dataset
data_tweet_clean['length'].plot(ax=axes[1], bins=100, kind='hist', color='green')
axes[1].set_title('Distribuzione delle lunghezze dei tweet dopo il filtraggio')
axes[1].set_xlabel('Lunghezza')
axes[1].set_ylabel('Frequenza')

plt.tight_layout()
plt.show()

# show the original version
print('==========ORIGINAL VERSION======================')
print(data.head()) 
print()
# show the cleaned up version
print('==========CLEANED UP VERSION====================')
print(data_tweet_clean.head()) 


# PLOT THE WORDCLOUD
# Put all tweets into a large list and generic wordcloud 
sentences = data_tweet_clean['tweet'].tolist()
len(sentences)
sentences_as_one_string =" ".join(sentences)
plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(sentences_as_one_string))
plt.title('Word Cloud generica')
plt.show()

# let's store the cleaned up DataFrame in a new CSV file
data_tweet_clean.to_csv('twitter_ElonMusk_restricted_cleaned.csv', index=False)

