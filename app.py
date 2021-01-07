import numpy as np
from flask import Flask, request, jsonify, render_template, json
import pickle
import requests
from flask_cors import CORS
import pandas as pd

import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *


app = Flask(__name__)
CORS(app)
model = pickle.load(open('filename.pkl', 'rb'))
dictionary = pickle.load(open('dict.pkl', 'rb'))
bow_corpus = pickle.load(open('bow_corpus.pkl', 'rb'))


stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# @app.route('/predict/<inputx>',methods=['POST'])
# def predict(inputx):

@app.route('/predict',methods=['POST'])
def predict():

    req_data = request.get_json()

    inputx = req_data['inputx']
    firebaseid = req_data['firebaseid']

    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

 #   inputx = request.form['username']
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print(inputx)
    #unseen_document = 'Pain or burning during urination, Pain in the back'

    bow_vector = dictionary.doc2bow(preprocess(inputx))

    result = model[bow_vector]

    newarr=[None] * 10

    i=0
    for index, score in sorted(result, key=lambda tup: -1*tup[1]):
        newarr[i] = ("Score: {}\t Topic: {}".format(score, model.print_topic(index, 5)))
        i=i+1


    i=0
    vec_lda_topics=[None]*20
    for x in bow_corpus:
        vec_lda_topics[i]=model[x]
        i=i+1
    print ('final : ', vec_lda_topics)

    print ('******')

    myinput = model[dictionary.doc2bow(preprocess(inputx))]
    print ('myinput: ', myinput)

#find the similarity between my input and each doctors inputs
    i=0
    r=len(vec_lda_topics)-1
    similarityArr=[None] * r
    finalArr=[None] * r

    for x in vec_lda_topics[:len(vec_lda_topics)-1]:  
        simmilarity = gensim.matutils.cossim(x, myinput)
        print('similarity ' + str(i), simmilarity)
        similarityArr[i]=simmilarity
        i=i+1

#get the sorted array indexes
    print([i[0] for i in sorted(enumerate(similarityArr), key=lambda x:x[1])])

    finalArr = [i[0] for i in sorted(enumerate(similarityArr), key=lambda x:x[1])]
    print(finalArr[-1])
    arrayindex = finalArr[-1]
    print(arrayindex)

    #read data csv file

    data = pd.read_csv('data002.csv', error_bad_lines=False)
    data_text = data[['doctor_name']]
    data_text['index'] = data_text.index
    documents = data_text

    print(len(documents))
    #print(documents[:30])

    #set the index as last element of the finalarr and increment it by one because of excel numbering
    doc_sample = int(documents[documents['index'] == arrayindex].values[0][0])
    print('$$$$')
    print(doc_sample)
    print(firebaseid)

    ##connect with Spring boot

    url = 'http://ec2-54-84-245-121.compute-1.amazonaws.com:8080/patient/updatepatient'
    #url = 'http://localhost:8080/patient/updatepatient'

    # data = {
    #   "firebaseid": "abc123abc123",
    #   "doctor": {
    #       "did": 2001
    #     }
    # }

    data = {
      "firebaseid": firebaseid,
      "doctor": {
          "did": doc_sample
        }
    }

    # data = {
    #   "pid": 1,
    #   "firebaseid": "abc123abc123",
    #   "name": "patient 1"
    # }


    json_data = json.dumps(data)

    response = requests.post(url, data=json_data)
    print('&&&')
    print(response)
   


    # return str(newarr) 
    return json_data
  #  return "done"

if __name__ == "__main__":
    app.run(debug=True)