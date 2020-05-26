import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# PRE-PROCESSING
def pre_process(df):

    # Converting to lower case
    df['question1'] = df['question1'].str.lower()
    df['question2'] = df['question2'].str.lower()
    # Applying word replacements
    df["question1"] = df["question1"].fillna("").apply(word_replace)
    df["question2"] = df["question2"].fillna("").apply(word_replace)
    # Removing punctuation
    df['question1'] = df['question1'].str.replace(r'[^\w\s]+', '')
    df['question2'] = df['question2'].str.replace(r'[^\w\s]+', '')
    # Lemmatization
    df['question1'] = df['question1'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
    df['question2'] = df['question2'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

    return df

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