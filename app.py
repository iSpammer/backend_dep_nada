from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback
# from flask_restful import reqparse
import h5py
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
@app.route("/", methods=['GET'])
def hello():
    return "hey"
@app.route('/predict', methods=['POST'])
def predict():

    df = pd.read_csv('trial3.csv',
                 encoding='ISO-8859-1', 
                 names=[
                        'sentences',
                        'label',
                        'souce' 
                         ])
    df['sentences'] = df['sentences'].str.lower()
    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 10000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 100
    import string
    # Define the function to remove the punctuation
    def remove_punctuations(text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        return text
    # Apply to the DF series
    df['sentences'] = df['sentences'].apply(remove_punctuations)
    model = load_model('CNN(Word2vec).hdf5', compile=False)


    tokenizer  = Tokenizer(num_words = MAX_NB_WORDS)

    # lr = joblib.load("model.pkl")
    if model:
        try:
            json = request.get_json()  
            model_columns = ["neutral", "Magnifying/minimizing", "Personalization", "overgeneralization", "should statements"]
            print(json)
            tokenizer  = Tokenizer(num_words = MAX_NB_WORDS)
            tokenizer.fit_on_texts(df['sentences'])
            sequences =  tokenizer.texts_to_sequences(df['sentences'])


            tmpr=(json['arr'])
            tmp = [tmpr]
            tokenizer.fit_on_texts(tmp)
            test_sequences =  tokenizer.texts_to_sequences(tmp)
            test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)



            print(test_data)
            prediction = model.predict(test_data)
            print("here:",prediction)        
            return jsonify({'prediction': str(prediction)})
        except:        
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)