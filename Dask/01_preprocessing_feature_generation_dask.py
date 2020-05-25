# Importing basic libraries
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()

# Loading the dataset
df = pd.read_csv("train.csv")

# Converting df to a dask dataframe
df = dd.from_pandas(df, npartitions=1000)

#### DATA PRE-PROCESSING

# Deleting records with null in qid1 or qid2
df = df.dropna(subset=['question1','question2'])

# Converting to lower case
df['question1'] = df['question1'].str.lower()
df['question2'] = df['question2'].str.lower()

# Function to perform word replacements
def word_replace(x):
  x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                            .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                            .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                            .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                            .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                            .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                            .replace("€", " euro ").replace("'ll", " will")
  return x

# Applying word replacements
df["question1"] = df["question1"].fillna("").apply(word_replace, meta=('question1', 'object'))
df["question2"] = df["question2"].fillna("").apply(word_replace, meta=('question2', 'object'))

# Removing punctuation
df['question1'] = df['question1'].str.replace(r'[^\w\s]+', '')
df['question2'] = df['question2'].str.replace(r'[^\w\s]+', '')

df = df.compute()

# Lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

df['question1'] = df['question1'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]),
                                        meta=('question1', 'object'))
df['question2'] = df['question2'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]),
                                        meta=('question2', 'object'))

#### FEATURE EXTRACTION

# Calculate words_diff_q1_q2 = difference of the word counts between question1 and question2
df['words_diff_q1_q2'] = abs(df['question1'].apply(lambda row: len(row.split(" ")), meta=('question1','int64')) - df['question2'].apply(lambda row: len(row.split(" ")), meta=('question2','int64')))

# Calculate word_common = count of unique words in question1 and question2
def normalized_word_common(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)
df['word_common'] = df.apply(normalized_word_common, axis=1, meta=(None, 'int64'))

# Calculate word_total = total number of words in Question 1 + total number of words in Question 2
def normalized_word_total(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))
df['word_total'] = df.apply(normalized_word_total, axis=1, meta=(None, 'int64'))

# Calculate word_share = word_common / word_total
def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2) / (len(w1) + len(w2))
df['word_share'] = df.apply(normalized_word_share, axis=1, meta=(None, 'float64'))

# Vectorizing question1 and question2 and doing distance matrix computations
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

def convert(lst):
    return (lst[0].split())

q1_q2_vectors = {}

# Function to return the vectors for q1 and q2
def get_q1_q2_vectors(row):
    if row['id'] not in q1_q2_vectors:
        vocab = set()
        q1_list = []
        q1_list.append(row['question1'].replace("\n", " "))
        vocab = vocab.union(set(convert(q1_list)))

        q2_list = []
        q2_list.append(row['question2'].replace("\n", " "))
        vocab = vocab.union(set(convert(q2_list)))

        # integer encode sequences of words
        tokenizer = Tokenizer(filters=[])
        tokenizer.fit_on_texts(vocab)

        q1_seq = tokenizer.texts_to_sequences(q1_list)
        q2_seq = tokenizer.texts_to_sequences(q2_list)

        PAD_LENGTH = len(q1_seq[0]) + len(q2_seq[0])
        q1_seq_padded = pad_sequences(q1_seq, maxlen=PAD_LENGTH)
        q2_seq_padded = pad_sequences(q2_seq, maxlen=PAD_LENGTH)

        q1_seq_sorted = np.sort(q1_seq_padded[0])
        q2_seq_sorted = np.sort(q2_seq_padded[0])

        q1_q2_list = []
        q1_q2_list.append(q1_seq_sorted)
        q1_q2_list.append(q2_seq_sorted)

        q1_q2_vectors[row['id']] = q1_q2_list

    return q1_q2_vectors[row['id']][0], q1_q2_vectors[row['id']][1]

def get_cosine_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return cosine(q1_q2_vectors[0], q1_q2_vectors[1])

def get_cityblock_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return cityblock(q1_q2_vectors[0], q1_q2_vectors[1])

def get_jaccard_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return jaccard(q1_q2_vectors[0], q1_q2_vectors[1])

def get_canberra_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return canberra(q1_q2_vectors[0], q1_q2_vectors[1])

def get_euclidean_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return euclidean(q1_q2_vectors[0], q1_q2_vectors[1])

def get_minkowski_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return minkowski(q1_q2_vectors[0], q1_q2_vectors[1])

def get_braycurtis_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return braycurtis(q1_q2_vectors[0], q1_q2_vectors[1])

df['cosine_distance'] = df.apply(get_cosine_distance, axis=1, meta=(None, 'float64'))
df['cityblock_distance'] = df.apply(get_cityblock_distance, axis=1, meta=(None, 'float64'))
df['jaccard_distance'] = df.apply(get_jaccard_distance, axis=1, meta=(None, 'float64'))
df['canberra_distance'] = df.apply(get_canberra_distance, axis=1, meta=(None, 'float64'))
df['euclidean_distance'] = df.apply(get_euclidean_distance, axis=1, meta=(None, 'float64'))
df['minkowski_distance'] = df.apply(get_minkowski_distance, axis=1, meta=(None, 'float64'))
df['braycurtis_distance'] = df.apply(get_braycurtis_distance, axis=1, meta=(None, 'float64'))

# FuzzyWuzzy
# FuzzyWuzzy is a library of Python which is used for string matching.
# Fuzzy string matching is the process of finding strings that match a given pattern.
# Basically it uses Levenshtein Distance to calculate the differences between sequences.
# !pip install fuzzywuzzy
# !pip install python-Levenshtein
from fuzzywuzzy import fuzz

df['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1, meta=(None, 'int64'))
df['fuzz_WRatio'] = df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1, meta=(None, 'int64'))
df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1, meta=(None, 'int64'))
df['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1, meta=(None, 'int64'))
df['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1, meta=(None, 'int64'))
df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1, meta=(None, 'int64'))
df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1, meta=(None, 'int64'))

# N-grams
# A combination of multiple words together are called N-Grams.
# N grams (N > 1) are generally more informative as compared to words, and can be used as features for language modelling.
# N-grams can be easily accessed in TextBlob using the ngrams function, which returns a tuple of n successive words.
# Here I'm calculating the bigrams and trigrams for question1 and question2 and comparing how many bigrams/trigrams match. I am, thus, generating 2 new features common_bigrams and common_trigrams.
# Generating N-grams

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from textblob import TextBlob, Word

# Function to generate bigrams for every q1 and q2 and count the common bigrams.
def get_common_bigrams_count(row):
  q1_bigrams = TextBlob(row['question1'].replace("\n", " ")).ngrams(2)
  q2_bigrams = TextBlob(row['question2'].replace("\n", " ")).ngrams(2)

  common=[]
  for bigrams in q1_bigrams:
      if bigrams in q2_bigrams:
        common.append(bigrams)

  return len(common)

# Function to generate trigrams for every q1 and q2 and count the common trigrams.
def get_common_trigrams_count(row):
  q1_trigrams = TextBlob(row['question1'].replace("\n", " ")).ngrams(3)
  q2_trigrams = TextBlob(row['question2'].replace("\n", " ")).ngrams(3)

  common=[]
  for trigrams in q1_trigrams:
      if trigrams in q2_trigrams:
        common.append(trigrams)

  return len(common)

df['common_bigrams'] = df.apply(get_common_bigrams_count, axis=1, meta=(None, 'int64'))
df['common_trigrams'] = df.apply(get_common_trigrams_count, axis=1, meta=(None, 'int64'))

# Readability Score
# This library measures the readability of a given text using measures that are basically
# linear regressions based on the number of words, syllables, and sentences.
# Here I'm calculating the readability scores of question1 and question2 to generate 2 new features.

# !pip install https://github.com/andreasvc/readability/tarball/master
import readability
def get_q1_readability_score(row):
  text = row['question1']
  try:
    results = readability.getmeasures(text, lang='en')
  except:
    return 0.00
  return results['readability grades']['FleschReadingEase']

def get_q2_readability_score(row):
  text = row['question2']
  try:
    results = readability.getmeasures(text, lang='en')
  except:
    return 0.00
  return results['readability grades']['FleschReadingEase']

df['q1_readability_score'] = df.apply(get_q1_readability_score, axis=1, meta=(None, 'float64'))
df['q2_readability_score'] = df.apply(get_q2_readability_score, axis=1, meta=(None, 'float64'))

df = df.compute()

# [########################################] | 100% Completed | 21min 59.0s

df = pd.read_pickle("df_quora_all_feat.pkl")
df = dd.from_pandas(df, npartitions=1)


# Word2Vec
# A word embedding is a class of approaches for representing words and documents using a dense vector representation.
# It is an improvement over more the traditional bag-of-words model encoding schemes where large sparse vectors were
# used to represent each word or to score each word within a vector to represent an entire vocabulary.
# These representations were sparse because the vocabularies were vast and a given word or document would be represented
# by a large vector comprised mostly of zero values. Two popular examples of methods of learning word embeddings from text
# are : Word2Vec and GloVe
# Here using Word2Vec, I'm calculating the similarity scores of q1 and q2 against and capturing the similarity score difference in the column word_2_vec_diff. The logic is that the more similar the questions, the closer the word_2_vec_diff value would be to 0.

# Build a dictionary of unique qids and question tokens
from keras.preprocessing.text import text_to_word_sequence
from gensim.models import KeyedVectors
import dill


filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

# Building a dictionary of unique question ids as keys and questions as the values
unique_questions = {}
for index, row in df.iterrows():
    if not row["qid1"] in unique_questions.keys():
        unique_questions[row["qid1"]]  = text_to_word_sequence(row["question1"])
    if not row["qid2"] in unique_questions.keys():
        unique_questions[row["qid2"]]  = text_to_word_sequence(row["question2"])

# Building a dictionary of unique question ids as keys and question vectors as the values
try:
    unique_vectors = pd.read_pickle("questions_word2vec_dictionary.pkl")
except:
    unique_vectors = {}

for qid in unique_questions:
    if not qid in unique_vectors.keys():
        tokens = [w for w in unique_questions[qid] if w in model.vocab]
        try :
            unique_vectors[qid] = model.most_similar(positive=tokens, topn=1)
        except :
            unique_vectors[qid] = 999
        if (len(unique_vectors) % 1000 == 0):
            print(len(unique_vectors))
            with open('questions_word2vec_dictionary.pkl', 'wb') as file:
                dill.dump(unique_vectors, file)

with open('questions_word2vec_dictionary.pkl', 'wb') as file:
    dill.dump(unique_vectors, file)

def get_word2vec_similarity_score(row):
    q1_tokens = text_to_word_sequence(row['question1'])
    q1_tokens = [w for w in q1_tokens if w in model.vocab]

    q2_tokens = text_to_word_sequence(row['question2'])
    q2_tokens = [w for w in q2_tokens if w in model.vocab]

    try:
        q1_similarity_score = model.most_similar(positive=q1_tokens, topn=1)
        q2_similarity_score = model.most_similar(positive=q2_tokens, topn=1)
    except:
        return 999

    return abs(q1_similarity_score[0][1] - q2_similarity_score[0][1])


df['word_2_vec_diff'] = df.apply(get_word2vec_similarity_score, axis=1, meta=(None, 'float64'))

df = df.compute()

# Save the dataframe
import dill
with open('df_quora_all_feat_w2v.pkl','wb') as file:
    dill.dump(df, file)