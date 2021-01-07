import pandas as pd
import pickle
#dfs = pd.read_excel(data001.csv, sheet_name=None)
#data = pd.read_excel('data001.xlsx', error_bad_lines=False);
data = pd.read_csv('data002.csv', error_bad_lines=False)
data_text = data[['patient_description']]
data_text['index'] = data_text.index
documents = data_text

print(len(documents))
print(documents[:30])

import gensim

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


doc_sample = documents[documents['index'] == 1].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = documents['patient_description'].map(preprocess)
processed_docs[:10]

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

#dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[1]

bow_doc_x = bow_corpus[8]
for i in range(len(bow_doc_x)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], dictionary[bow_doc_x[i][0]], bow_doc_x[i][1]))


from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break


#TF-IDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
# for idx, topic in lda_model_tfidf.print_topics(-1):
#     print('Topic: {} Word: {}'.format(idx, topic))

############################################

pickle.dump(lda_model_tfidf,open('filename.pkl','wb'))

lda_copy = pickle.load(open('filename.pkl', "rb"))

print(lda_copy)

pickle.dump(dictionary,open('dict.pkl','wb'))
pickle.dump(bow_corpus,open('bow_corpus.pkl','wb'))

############################################

#unsean doc
#unseen_document = 'urinating often.blurred vision'
unseen_document = 'Pain or burning during urination, Pain in the back'

bow_vector = dictionary.doc2bow(preprocess(unseen_document))
print(preprocess(unseen_document))



#print(lda_copy[bow_vector])

for index, score in sorted(lda_copy[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))

# def predictx():
#     i=0
#     vec_lda_topics=[None]*20
#     for x in bow_corpus:
#         vec_lda_topics[i]=lda_model_tfidf[x]
#         i=i+1
#     print ('final : ', vec_lda_topics)
#     return vec_lda_topics