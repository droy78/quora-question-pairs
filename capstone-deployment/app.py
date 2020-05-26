import pandas as pd
import dill
import pre_processing as pp
import feature_generation as fg
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
with open("random_forest_model.pkl", "rb") as f:
    model_rf = dill.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    questions = []
    questions.append([str(x) for x in request.form.values()])
    df = pd.DataFrame(questions, columns=['question1','question2'])
    df['id'] = 1
    df = pp.pre_process(df)
    df = fg.generate_features(df)
    X = df[['words_diff_q1_q2', 'word_common', 'word_total', 'word_share', 'cosine_distance', 'cityblock_distance',
            'jaccard_distance', 'canberra_distance', 'euclidean_distance', 'minkowski_distance', 'braycurtis_distance',
            'fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio',
            'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio', 'common_bigrams',
            'common_trigrams', 'q1_readability_score', 'q2_readability_score']]

    prediction = 'duplicate' if (model_rf.predict(X)[0] == 1) else 'not duplicate'
    return render_template('index.html', prediction_text=f'The questions are {prediction}')

if __name__ == "__main__":
    app.run(debug=True)