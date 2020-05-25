import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import dask.array as da
from dask import compute

# Load the pickled pandas dataframe created in the feature generation stop
df_pandas = pd.read_pickle("df_quora_all_feat.pkl")

# Drop any rows with Nan values
df_pandas.dropna(inplace=True)

# Sampling of 140000 duplicate and 70000 non-duplicate questions
df_not_duplicate = df_pandas[df_pandas.is_duplicate == 0].sample(n = 70000)
df_duplicate = df_pandas[df_pandas.is_duplicate == 1].sample(n = 140000)
df_pandas = df_not_duplicate.append(df_duplicate)

# Converting df to a dask dataframe
df = dd.from_pandas(df_pandas, npartitions=10)

# Separating the features and the labels
X = df[['words_diff_q1_q2', 'word_common', 'word_total', 'word_share', 'cosine_distance', 'cityblock_distance',
        'jaccard_distance', 'canberra_distance', 'euclidean_distance', 'minkowski_distance', 'braycurtis_distance',
        'fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio',
        'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio', 'common_bigrams',
        'common_trigrams', 'q1_readability_score', 'q2_readability_score']]
y = df['is_duplicate']

# creating a function to convert the dask dataframe to dask array
def to_dask_array(df):
    partitions = df.to_delayed()
    shapes = [part.values.shape for part in partitions]
    dtype = partitions[0].dtype

    results = compute('float64', *shapes)  # trigger computation to find shape
    dtype, shapes = results[0], results[1:]

    chunks = [da.from_delayed(part.values, shape, dtype)
              for part, shape in zip(partitions, shapes)]
    return da.concatenate(chunks, axis=0)

# Test-train split
from dask_ml.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(to_dask_array(X), to_dask_array(y), random_state=99)

###################################################################################

# Fitting the Logistic Regression Classifier
from dask_ml.linear_model import LogisticRegression
lr = LogisticRegression()

with ProgressBar():
    lr.fit(X_train, y_train)

print('Logistic Regression Score : ', lr.score(X_test, y_test).compute())
##### OUTPUT --------> Logistic Regression Score :  0.70025

#####################################################################################

# Fitting the Naive Bayes Classifier
from sklearn.naive_bayes import BernoulliNB
from dask_ml.wrappers import Incremental

nb = BernoulliNB()

parallel_nb = Incremental(nb)

with ProgressBar():
    parallel_nb.fit(X_train, y_train, classes=np.unique(y_train.compute()))

print('\n\nNaive Bayes Classifier Score : ', parallel_nb.score(X_test, y_test))
##### OUTPUT --------> Naive Bayes Classifier Score :  0.65

######################################################################################

# Performing GridSearch on the Logistic Regression Classifier
from dask_ml.model_selection import GridSearchCV

parameters = {'penalty': ['l1', 'l2'], 'C': [0.5, 1, 2]}

lr = LogisticRegression()

tuned_lr = GridSearchCV(lr, parameters)

with ProgressBar():
    tuned_lr.fit(X_train, y_train)

print('\n\nGrid Search Results for Logistic Regression')
print(pd.DataFrame(tuned_lr.cv_results_)[['params', 'mean_test_score']])

#### OUTPUT
#### Grid Search Results for Logistic Regression
####                         params  mean_test_score
#### 0  {'C': 0.5, 'penalty': 'l1'}         0.700778
#### 1  {'C': 0.5, 'penalty': 'l2'}         0.700306
#### 2    {'C': 1, 'penalty': 'l1'}         0.700806
#### 3    {'C': 1, 'penalty': 'l2'}         0.700500
#### 4    {'C': 2, 'penalty': 'l1'}         0.700972
#### 5    {'C': 2, 'penalty': 'l2'}         0.700944

# ####################################################################################
# Many Scikit-Learn algorithms are written for parallel execution using Joblib, which natively provides thread-based and
# process-based parallelism. Joblib is what backs the n_jobs= parameter in normal use of Scikit-Learn.
# Dask can scale these Joblib-backed algorithms out to a cluster of machines by providing an alternative Joblib backend.
# To use the Dask backend to Joblib you have to create a Client,
# and wrap your code with joblib.parallel_backend('dask').

import joblib
from dask.distributed import Client, progress

client = Client(processes=False, threads_per_worker=4,
                n_workers=1, memory_limit='8GB')

####################################################################################

# Fitting the Decision Tree Classifier
from sklearn import tree, metrics
dtc = tree.DecisionTreeClassifier(random_state=99)

# To use the Dask backend to Joblib you have to create a Client,
# and wrap your code with joblib.parallel_backend('dask').
with joblib.parallel_backend('dask'):
    dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
print('\n\nDecision Tree Classifier Score : ', {metrics.accuracy_score(y_test, y_pred)})
##### OUTPUT --------> Decision Tree Classifier Score :  {0.6754390304229533}

#####################################################################################
# Fitting the Random Forest Classifier
from sklearn import ensemble, metrics
import dill

rfc = ensemble.RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=99)

# To use the Dask backend to Joblib you have to create a Client,
# and wrap your code with joblib.parallel_backend('dask').
with joblib.parallel_backend('dask'):
    rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
print('\n\nRandom Forest Classifier Score : ', {metrics.accuracy_score(y_test, y_pred)})
with open('random_forest_model.pkl','wb') as file:
    dill.dump(rfc, file)
# OUTPUT --------> Random Forest Classifier Score :  {0.7445}

#####################################################################################

# Deep Learning Model with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import dill

model = Sequential()
model.add(Dense(18, input_dim=X.shape[1], activation='tanh'))
model.add(Dense(9, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'], )

es_callback = EarlyStopping(monitor='val_loss', mode='min', patience=0)

X = df[['words_diff_q1_q2', 'word_common', 'word_total', 'word_share', 'cosine_distance', 'cityblock_distance',
        'jaccard_distance', 'canberra_distance', 'euclidean_distance', 'minkowski_distance', 'braycurtis_distance',
        'fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio',
        'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio', 'common_bigrams',
        'common_trigrams', 'q1_readability_score', 'q2_readability_score', 'is_duplicate']]


def dask_data_generator(df, fraction=0.01):
    while True:
        batch = df.sample(frac=fraction)
        X = batch.iloc[:, :-1]
        y = batch.iloc[:, -1]
        yield (X.compute(), y.compute())


from dask_ml.model_selection import train_test_split
X_train, X_test = train_test_split(X, random_state=99)

history = model.fit_generator(generator=dask_data_generator(X_train), validation_data=dask_data_generator(X_test), epochs=100,
                              steps_per_epoch=100, validation_steps=10, callbacks=[es_callback])

print("Best Accuracy on Validation Set =", max(history.history['val_accuracy']))
print("Lowest error on Training Set =", min(history.history['loss']))
print("Lowest error on Validation Set =", min(history.history['val_loss']))

with open('deep_learning_model.pkl','wb') as file:
    dill.dump(model, file)

# Epoch 1/100
#
#   1/100 [..............................] - ETA: 1:42 - loss: 0.7355 - accuracy: 0.4786
#   2/100 [..............................] - ETA: 1:29 - loss: 0.7015 - accuracy: 0.5272
#   3/100 [..............................] - ETA: 1:26 - loss: 0.6837 - accuracy: 0.5591
#   4/100 [>.............................] - ETA: 1:23 - loss: 0.6746 - accuracy: 0.5783
# ...
# Epoch 4/100
#
#   1/100 [..............................] - ETA: 55s - loss: 0.5455 - accuracy: 0.7626
#   2/100 [..............................] - ETA: 1:09 - loss: 0.5470 - accuracy: 0.7610

# ...
#  98/100 [============================>.] - ETA: 0s - loss: 0.5579 - accuracy: 0.7483
#  99/100 [============================>.] - ETA: 0s - loss: 0.5578 - accuracy: 0.7483
# 100/100 [==============================] - 44s 437ms/step - loss: 0.5576 - accuracy: 0.7486 - val_loss: 0.5387 - val_accuracy: 0.7560

# Best Accuracy on Validation Set = 0.7559808492660522
# Lowest error on Training Set = 0.5576349037885666
# Lowest error on Validation Set = 0.5223293304443359
