# load the libraries
import time
import joblib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm, model_selection
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from string import punctuation
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk # Natural Language tool kit 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stopwords.words('english')
from nltk.tokenize import word_tokenize,ToktokTokenizer
import re
import html
from sklearn.decomposition import TruncatedSVD
import matplotlib.patches as mpatches
import matplotlib

# this library causes the output printed on sys.stdout, which is usually the console, to be encoded  using UTF-8. 
# This is useful when you want to print non-ASCII characters without running into encoding errors.
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Sentiment Analysis on a tweets dataset

# The dataset and the problem to be solved:

# The dataset includes 27481 tweets, taken from Twitter, stored in a CSV file.
# The Textblob module was used to assign a sentiment to each tweet.
# The dataset contains 4 columns (textID, text, selected_text, sentiment) to analyze the tweet and identify whether it is a positive, negative or neutral tweet.
# Link dataset: https:///www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset
# Formulation of the problem: the goal is to train the ML model (SVM) trying to get consistent scores and up to expectations, for Tweet Sentiment prediction. 

# Preparing the data
    # Preprocessing 

# Read the data
data = pd.read_csv('Tweets_3sentiment.csv')
# print(data)

# checking my columns in my data
print(data.keys())

# looking into my data
print(data.head())

# checking last 5 entries
print(data.tail())

# checking info our data
print(data.info())

# describing our data
print(data.describe())

# checking unique values 
print(data.nunique())
# there are no duplicate tuples in the dataset

# checking null values in my data
print(data.isnull().sum())
# there is one null values, so I remove it
print(data.dropna(inplace=True))

# checking type values in my data
print(data.dtypes)

# let's drop the 'id' and 'selected_text' columns
data = data.drop(['textID', 'selected_text'], axis=1)

# get Dimension of the Data
print(data.shape)

# Exploratory Data Analysis (EDA)

# let's rename the 'text' column to 'tweet'
data.rename(columns={'text':'tweet'},inplace=True)

# visualize the heatmap of null data values 
plt.figure(figsize=(10,10))
sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.title('Null data values')
plt.show()

# let's get the length of the messages
data['length'] = data['tweet'].apply(len)
print(data["length"])

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

# plot the histogram of the length column 
data['length'].plot(bins = 100, kind='hist')
plt.title('Length of column before cleaning up')
plt.show()

# Bar plot of positives, negatives and neutral tweets 
positive_count = data[data['sentiment'] == 'positive'].shape[0]
negative_count = data[data['sentiment'] == 'negative'].shape[0]
neutral_count = data[data['sentiment'] == 'neutral'].shape[0]
# Creating the bar graph
plt.bar(['Positivi', 'Neutrali', 'Negativi'], [positive_count, neutral_count, negative_count], color=['green', 'yellow', 'red'])
plt.xlabel('Sentimento')
plt.ylabel('Numero di Tweet')
plt.title('Distribuzione dei Tweet per Sentiment')
plt.show()
print()

print('Positivi', round(data['sentiment'].value_counts()['positive']/len(data) * 100),'% -->', data['sentiment'].value_counts()['positive'])
print('Neutrali', round(data['sentiment'].value_counts()['neutral']/len(data) * 100),'% -->', data['sentiment'].value_counts()['neutral'])
print('Negativi', round(data['sentiment'].value_counts()['negative']/len(data) * 100),'% -->', data['sentiment'].value_counts()['negative'])


# Checking Most Common Words in Each Target Variable Values
    # checking most common positive words 
count1 = Counter(" ".join(data[data['sentiment'] == 'positive']['tweet']).split()).most_common(20)
data_tweet_1 = pd.DataFrame.from_dict(count1)
data_tweet_1 = data_tweet_1.rename(columns={0: "common_words", 1 : "count"})
plt.barh(data_tweet_1['common_words'], data_tweet_1['count'], color='skyblue')
plt.title('Common Positive Words in Text')
plt.xlabel('Count')
plt.ylabel('Word')
plt.show()

    # checking most common neutral words 
count2 = Counter(" ".join(data[data['sentiment'] == 'neutral']['tweet']).split()).most_common(20)
data_tweet_2 = pd.DataFrame.from_dict(count2)
data_tweet_2 = data_tweet_2.rename(columns={0: "common_words", 1 : "count"})
plt.barh(data_tweet_2['common_words'], data_tweet_2['count'], color ='skyblue')
plt.title('Common Neutral Words in Text')
plt.xlabel('Word')
plt.ylabel('Count')
plt.show()

    # checking most common negative words 
count3 = Counter(" ".join(data[data['sentiment'] == 'negative']['tweet']).split()).most_common(20)
data_tweet_3 = pd.DataFrame.from_dict(count3)
data_tweet_3 = data_tweet_3.rename(columns={0: "common_words", 1 : "count"})
plt.barh(data_tweet_3['common_words'], data_tweet_3['count'], color ='skyblue')
plt.title('Common Negative Words in Text')
plt.xlabel('Word')
plt.ylabel('Count')
plt.show()

# let's store the modified DataFrame in a new CSV file
data.to_csv('Tweets_3sentiment_balanced.csv', index=False)

print(data.iloc[15,0])
# From the result above, most common words in target class variable contains so many stop words like "to, the, a, and, i, of, you, for, in, is" and etc. 
# Which mean that the data_tweet still have untidy data. There are so many punctuation, constraction, special characters, elongation words, twitter tags and hastag. 
# Before we go to the modelling steps we need to removing those all, and make the text become tidy data text.


# Text Mining
    # In this text mining process we will exploring and analyzing unstructured text data

# let's assign the columns of the DataFrame to variables to make it easier to work with
tweet = data['tweet']
sentiment = data['sentiment']

print(len(tweet))
print(len(sentiment))

# Let's define a pipeline to clean up all the messages 
    # The pipeline performs the following:
        # (1) expand contractions
        # (2) remove tags and links (NO hastag)        
        # (3) remove HTML
        # (4) uppercase to lowercase
        # (5) remove special character, punctuation and Non-ASCII Character
        # (6) replace elongated words
        # (7) tokenization
        # (8) removing stopwords
        # (9) lemmatization
        # (10) drop numbers

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

# (5) Remove Special Character and Punctuation
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

# (6) Replace Elongated Words
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

# (7) Tokenization
    # Tokenization is splitting sentences into smaller unit, such as terms or word.
def tokenize_text(text):
    tokenizer = ToktokTokenizer()
    tokenized_text = tokenizer.tokenize(text)
    return ' '.join(tokenized_text)
tweet = tweet.apply(tokenize_text)

# (8) Removing Stopwords
   # Remove stopwords like "is, the, with, etc" since they don't have usefull information
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    words_without_stopwords = [w for w in text.split() if w.lower() not in stopword_list]
    return ' '.join(words_without_stopwords)
tweet = tweet.apply(lambda x : remove_stopwords(x))

# (9) Stemming
#    # Stemming is the process of reducing a word to its word stem. For example "Consulting" to be "consult"
# def stem_text(text):
#     stemmer = PorterStemmer()
#     tokenized_text = word_tokenize(text)
#     stemmed_text = [stemmer.stem(word) for word in tokenized_text]
#     return ' '.join(stemmed_text)
# tweet = tweet.apply(stem_text)

# (9) Lemmatization
    # Lemmatization uses a lexical database and linguistic rules to obtain the correct basic form of a word, called a lemma.
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokenized_text = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in tokenized_text]
    return ' '.join(lemmatized_text)
tweet = tweet.apply(lemmatize_text)

# (10) Drop Numbers
    # Remove numbers from text, since numbers doesn't give much importance to get the main words.
def remove_numbers_and_cleanup(text):
    # Utilizza un'espressione regolare per rimuovere i numeri e pulisce gli spazi extra
    text_without_numbers = re.sub(r'\b\d+\b', '', text)  # Rimuove i numeri
    cleaned_text = re.sub(r'\s+', ' ', text_without_numbers)  # Pulisce gli spazi extra
    return cleaned_text.strip()
tweet = tweet.apply(remove_numbers_and_cleanup)

def process_and_plot_vocabulary(tokens_list):
    # Calcolare le frequenze del vocabolario
    all_tokens = [token for sublist in tokens_list for token in sublist]
    vocabulary_frequencies = Counter(all_tokens)
    
    # Ottenere le frequenze
    frequencies = list(vocabulary_frequencies.values())
    
    # Tracciare l'istogramma delle frequenze
    plt.figure(figsize=(10, 6))
    plt.hist(frequencies, bins=100, edgecolor='k', log=True)  # log scale per asse y
    plt.xlabel('Frequenza delle parole')
    plt.ylabel('Conteggio')
    plt.title('Word frequency')
    plt.show()
    
    # Calcolare il numero totale di parole uniche
    num_unique_words = len(vocabulary_frequencies)
    print(f"Numero totale di parole uniche nel vocabolario: {num_unique_words}")
process_and_plot_vocabulary(tweet)

# let's create the new clean dataset
data_clean = pd.concat([tweet, sentiment],axis=1)

# Replace empty strings with np.nan
data_clean['tweet'].replace('', np.nan, inplace=True)
# Drop rows with np.nan
data_clean.dropna(subset=['tweet'], inplace=True)

# Plot the histogram of the length column after cleaning up
data_clean['length'] = data_clean['tweet'].apply(len)
data_clean['length'].plot(bins = 100, kind='hist')
plt.title('Length of column after cleaning up')
plt.show()

# let's compare the length before (after balancing) and after cleaning up
    # setting the figure and axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot for balanced dataset
data['length'].plot(ax=axes[0], bins=100, kind='hist', color='blue')
axes[0].set_title('Length of column before cleaning up')

# Plot for cleaned dataset
data_clean['length'].plot(ax=axes[1], bins=100, kind='hist', color='green')
axes[1].set_title('Length of column after cleaning up')

plt.tight_layout()
plt.show()

# show the original version
print('==========ORIGINAL VERSION======================')
print(data.head()) 
print()
# show the cleaned up version
print('==========CLEANED UP VERSION====================')
print(data_clean.head()) 


# PLOT THE WORDCLOUD
    # wordcloud positivo
        # Most Common Words in Positive Tweets
positive = data_clean[data_clean['sentiment'] == 'positive']

positive_tweets_list = positive['tweet'].tolist()
positive_tweets_str = " ".join(positive_tweets_list)

plt.figure(figsize = (10,10))
plt.imshow(WordCloud().generate(positive_tweets_str))
plt.title('Positive Word Cloud')
plt.show()

    # wordcloud neutrale
            # Most Common Words in Neutral Tweets
neutral = data_clean[data_clean['sentiment'] == 'neutral']
# print(neutral)

neutral_tweets_list = neutral['tweet'].tolist()
neutral_tweets_str = " ".join(neutral_tweets_list)

plt.figure(figsize = (10,10))
plt.imshow(WordCloud().generate(neutral_tweets_str))
plt.title('Neutral Word Cloud')
plt.show()

    # wordcloud negativo
        # Most Common Words in Negative Tweets
negative = data_clean[data_clean['sentiment'] == 'negative']
# print(negative)

negative_tweets_list = negative['tweet'].tolist()
negative_tweets_str = " ".join(negative_tweets_list)

plt.figure(figsize = (10,10))
plt.imshow(WordCloud().generate(negative_tweets_str))
plt.title('Negative Word Cloud')
plt.show()

# Put all tweets into a large list and generic wordcloud 
sentences = data_clean['tweet'].tolist()
len(sentences)
sentences_as_one_string =" ".join(sentences)
plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(sentences_as_one_string))
plt.title('Generic Word Cloud')
plt.show()

# let's store the cleaned up DataFrame in a new CSV file
data_clean.to_csv('Tweets_3sentiment_balanced_cleaned.csv', index=False)


# Text Representation (Bag of words)
    # Classifiers and learning algorithms expect numerical feature vectors rather than raw text documents.
    # This is why we need to turn our tweet text into numerical vectors.
        # let's use bag of words (BOW) since the matter is the frequency of the words in text reviews;
        # however, the order of words is irrelevant. 
        # Two common ways to represent bag of words are:
            # CountVectorizer 
            # Term Frequency, Inverse Document Frequency (TF-IDF). 

# We want to identify unique/representative words for positive and negative tweets, so we’ll choose the TF-IDF. 
# To turn text data into numerical vectors with TF-IDF

# FYI LSA -> Latent semantic analysis 
def plot_LSA(test_data, test_labels):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)

    colors = ['orange','blue', 'green']
    plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
    orange_patch = mpatches.Patch(color='orange', label='Positive')
    blue_patch = mpatches.Patch(color='blue', label='Negative')
    green_patch = mpatches.Patch(color='green', label='Neutral')
    plt.legend(handles=[orange_patch, blue_patch, green_patch], prop={'size': 10})

# CountVectorizer 
# x = data_clean.tweet
# y = data_clean.sentiment
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(data_clean['sentiment'])
# # The data is split in the standard 80,20 ratio
# x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, random_state=42)
# # instantiate the vectorizer
# vect = CountVectorizer()
# vect.fit(x_train)
# # Use the trained to create a document-term matrix from train and test sets
# x_train = vect.transform(x_train)
# x_test = vect.transform(x_test)

# fig = plt.figure(figsize=(12, 12))          
# plot_LSA(x_train, y_train)
# plt.title('LSA Plot CountVectorizer')
# plt.show()

# Term Frequency, Inverse Document Frequency (TF-IDF)
# Creating object of TF-IDF vectorizer
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
vect = vectorizer.fit_transform(data_clean.tweet)

# returns the feature names obtained from TF-IDF vector fitting
print(vectorizer.get_feature_names_out())

# returns a representation of the same data transformed into a numpy array.
# This array shows the TF-IDF frequencies for each term in the documents
print(vect.toarray())  

# returns the transformed matrix size
print(vect.shape)

# Transformed data is converted to a DataFrame pandas 
X = pd.DataFrame(vect.toarray())
print(X)

# Encoding target labels
    # because 'sentiment' has categorical values ​​('positive','negative' and 'neutral'), but evaluation metrics such as
    # roc_auc_score, precision_recall_curve, and confusion_matrix require the target to be binary, i.e. with values ​​in {0, 1, 2} or {-1, 0, 1}.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data_clean['sentiment'])
# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(vect, y_encoded, random_state=42)

fig = plt.figure(figsize=(12, 12))          
plot_LSA(x_train, y_train)
plt.title('LSA Plot TF-IDF')
plt.show()

# Modelling

# Fitting and Evaluating the model
def check_scores(clf, x_train, x_test, y_train, y_test, algorithm_name):
    t0_train = time.time()
    model = clf.fit(x_train, y_train)
    t1_train = time.time()
    time_train = t1_train - t0_train

    t0_predict = time.time()
    predicted_class = model.predict(x_test)
    t1_predict = time.time()
    time_predict = t1_predict - t0_predict

    predicted_class_train = model.predict(x_train)
    test_probs = model.predict_proba(x_test)
    
    # For multiclass classification, calculate metrics for each class
    lr_f1 = f1_score(y_test, predicted_class, average='macro')
    lr_auc = roc_auc_score(y_test, test_probs, multi_class='ovr')

    cm_train = confusion_matrix(y_train, predicted_class_train)
    cm_test = confusion_matrix(y_test, predicted_class)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Plotting train confusion matrix
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Paired', cbar=False, linewidths=0.1, annot_kws={'size': 25}, ax=ax1)
    ax1.set_title(f'Train Confusion Matrix ({algorithm_name})')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    # Plotting test confusion matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Paired', cbar=False, linewidths=0.1, annot_kws={'size': 25}, ax=ax2)
    ax2.set_title(f'Test Confusion Matrix ({algorithm_name})')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    plt.show()

    print(classification_report(y_test, predicted_class))
    print()
    train_accuracy = accuracy_score(y_train, predicted_class_train)
    test_accuracy = accuracy_score(y_test, predicted_class)

    print("Train accuracy score:", train_accuracy)
    print("Test accuracy score:", test_accuracy)
    print()

    # Calculate error rates
    train_error = 1 - train_accuracy
    test_error = 1 - test_accuracy

    print("Train error rate:", train_error)
    print("Test error rate:", test_error)
    print()

    train_auc = roc_auc_score(y_train, clf.predict_proba(x_train), multi_class='ovr')
    test_auc = roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovr')

    print("Train ROC-AUC score:", train_auc)
    print("Test ROC-AUC score:", test_auc)

    print("Training time: %fs; Prediction time: %fs" % (time_train, time_predict))

    # let's Create a DataFrame to store the results
    results = pd.DataFrame(columns=['Num_Tweets', 'Acc_Pos', 'Acc_Neg', 'Acc_Neu', 'Err_Pos', 'Err_Neg', 'Err_Neu'])

    # let’s define the tweet intervals to consider
    tweet_intervals = [50, 200, 500, 1000, 2000, 3000, 4017]

    for num_tweets in tweet_intervals:
        # Riduci i tuoi dati a 'num_tweets' tweet
        x_train_reduced = x_train[:num_tweets]
        y_train_reduced = y_train[:num_tweets]

        # let's train the model on reduced data
        model = clf.fit(x_train_reduced, y_train_reduced)

        # let's calculate the forecast
        predicted_class = model.predict(x_test)

        # let's calculate accuracy and error for positive, negative and neutral tweets
        acc_pos = accuracy_score(y_test[y_test == 1], predicted_class[y_test == 1])
        err_pos = 1 - acc_pos
        acc_neg = accuracy_score(y_test[y_test == 0], predicted_class[y_test == 0])
        err_neg = 1 - acc_neg
        acc_neu = accuracy_score(y_test[y_test == 2], predicted_class[y_test == 2])
        err_neu = 1 - acc_neu

        acc_avg = (acc_pos + acc_neg + acc_neu) / 3
        err_avg = (err_pos + err_neg + err_neu) / 3

        # let's add results to DataFrame
        new_row = pd.DataFrame({'Num_Tweets': [num_tweets], 'Acc_Pos': [acc_pos], 'Acc_Neg': [acc_neg], 'Acc_Neu': [acc_neu], 'Err_Pos': [err_pos], 'Err_Neg': [err_neg], 'Err_Neu': [err_neu], 'Accuracy': [acc_avg], 'Error': [err_avg]})
        results = pd.concat([results, new_row], ignore_index=True)

    # Stampa la tabella dei risultati
    print(results)

    # Crea i grafici
    for column, label, color in zip (['Acc_Pos', 'Acc_Neg', 'Acc_Neu', 'Accuracy'], ['Positive', 'Negative', 'Neutral', 'Average'], ['green', 'red', 'yellow', 'black']):
        plt.plot(results['Num_Tweets'], results[column], label=label, color=color)
    plt.xlabel('Number of Tweets')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    for column, label, color in zip (['Err_Pos', 'Err_Neg', 'Err_Neu', 'Error'], ['Positive', 'Negative', 'Neutral', 'Average'], ['green', 'red', 'yellow', 'black']):
        plt.plot(results['Num_Tweets'], results[column], label=label, color=color)
    plt.xlabel('Number of Tweets')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    
    return train_accuracy, test_accuracy, train_auc, test_auc, train_error, test_error

# Using Support Vector Machine (SVM)
SVM = svm.SVC(probability=True)
s_train_accuracy, s_test_accuracy, s_train_auc, s_test_auc, s_train_error, s_test_error = check_scores(SVM, x_train, x_test, y_train, y_test, "Support Vector Machine")

# Save the trained model
joblib.dump(SVM, 'sentiment_model_SVM.pkl')

# With increase in FPR, TPR also increases.
# With increase in recall, precision decreases.


# ----------------------------------------------------------------------------
# CASE STUDY: Sentiment predicition to dataset of reply to a Elon Musk's tweet
# ----------------------------------------------------------------------------

# Load the trained model
model = joblib.load('sentiment_model_SVM.pkl')

# Load the new test dataset 
new_data = pd.read_csv('twitter-ElonMusk_restricted_cleaned.csv')

# Check for NaN values and replace them with a string placeholder if necessary
new_data['tweet'] = new_data['tweet'].fillna('')

# Transform the text of the new dataset into numerical vectors using the same vectorizer
# Ensure that the vectorizer is the one used during the training phase
new_data_vectors = vectorizer.transform(new_data['tweet'])

# Make predictions on the new dataset
predictions = model.predict(new_data_vectors)

# Add the predictions as a new column to the dataset
new_data['sentiment_prediction'] = predictions

new_data['sentiment_prediction'] = new_data['sentiment_prediction'].replace({0: 'positive', 1: 'negative', 2: 'neutral'})

# Save the dataset with the added predictions
new_data.to_csv('sentiment_analysis_results_SVM_ElonMusk.csv', index=False)

new_data_prediction = pd.read_csv('sentiment_analysis_results_SVM_ElonMusk.csv')

# Bar plot of positives and negatives tweets after prediction
positive_count = new_data_prediction[new_data_prediction['sentiment_prediction'] == 'positive'].shape[0]
negative_count = new_data_prediction[new_data_prediction['sentiment_prediction'] == 'negative'].shape[0]
neutral_count = new_data_prediction[new_data_prediction['sentiment_prediction'] == 'neutral'].shape[0]
# Creating the bar graph
plt.bar(['Positivi', 'Neutrali', 'Negativi'], [positive_count, neutral_count, negative_count], color=['green', 'yellow', 'red'])
plt.xlabel('Sentimento')
plt.ylabel('Numero di Tweet')
plt.title('Distribuzione dei Tweet per Sentiment after prediction SVM')
plt.show()
print()

print('Positivi', round(new_data_prediction['sentiment_prediction'].value_counts()['positive']/len(new_data_prediction) * 100),'% -->', new_data_prediction['sentiment_prediction'].value_counts()['positive'])
print('Neutrali', round(new_data_prediction['sentiment_prediction'].value_counts()['neutral']/len(new_data_prediction) * 100),'% -->', new_data_prediction['sentiment_prediction'].value_counts()['neutral'])
print('Negativi', round(new_data_prediction['sentiment_prediction'].value_counts()['negative']/len(new_data_prediction) * 100),'% -->', new_data_prediction['sentiment_prediction'].value_counts()['negative'])

# PLOT THE WORDCLOUD
    # wordcloud positivo
        # Most Common Words in Positive Tweets
positive = new_data_prediction[new_data_prediction['sentiment_prediction'] == 'positive']
positive_tweets_list = positive['tweet'].tolist()
positive_tweets_str = " ".join(positive_tweets_list)

plt.figure(figsize = (10,10))
plt.imshow(WordCloud().generate(positive_tweets_str))
plt.title('Positive Word Cloud')
plt.show()

    # wordcloud neutrale
        # Most Common Words in Neutral Tweets
neutral = new_data_prediction[new_data_prediction['sentiment_prediction'] == 'neutral']
neutral_tweets_list = neutral['tweet'].tolist()
neutral_tweets_str = " ".join(neutral_tweets_list)

plt.figure(figsize = (10,10))
plt.imshow(WordCloud().generate(neutral_tweets_str))
plt.title('Neutral Word Cloud')
plt.show()

    # wordcloud negativo
        # Most Common Words in Negative Tweets
negative = new_data_prediction[new_data_prediction['sentiment_prediction'] == 'negative']
negative_tweets_list = negative['tweet'].tolist()
negative_tweets_str = " ".join(negative_tweets_list)

plt.figure(figsize = (10,10))
plt.imshow(WordCloud().generate(negative_tweets_str))
plt.title('Negative Word Cloud')
plt.show()


# Put all tweets into a large list and generic wordcloud 
sentences = new_data_prediction['tweet'].tolist()
len(sentences)
sentences_as_one_string =" ".join(sentences)

plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(sentences_as_one_string))
plt.title('Generic Word Cloud')
plt.show()

