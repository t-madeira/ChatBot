
import socket
import select
import sys
import pandas as pd
import spacy
import os
from sklearn.naive_bayes import GaussianNB

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

from string import punctuation

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from random import randrange

from bs4 import BeautifulSoup
import requests

import collections

from scipy import spatial

from sklearn.model_selection import cross_val_score

def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token

def max_length(words):
  return(len(max(words, key = len)))

def encoding_doc(token, words):
  return(token.texts_to_sequences(words))

def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

##daq pra baixo eh tudo meu
def clean(sentences):
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        words.append([i.lower() for i in w])

    return words

def clean_sentence (sentence, nlp):
    # nltk.download("stopwords")
    # nltk.download("punkt") # Preciso ver novamente o que isso faz
    sw = stopwords.words('portuguese') + list(punctuation)

    sentence = re.sub(r'[^ a-z A-Z 0-9]', " ", sentence)  # Misterioso, porem necessario
    sentence = word_tokenize(sentence)

    # Lower case e remove stopword
    indexes_to_drop = []
    for i in range(0, len(sentence)):
        sentence[i] = sentence[i].lower()
        if sentence[i] in sw:
            indexes_to_drop.append(i)

    indexes_to_drop.reverse()
    for index in indexes_to_drop:
        del sentence[index]

    # Verbos para o infinitivo
    doc = nlp(str(sentence))
    for token in doc:
        if token.pos_ == 'VERB':
            indexes = [i for i, x in enumerate(sentence) if
                       x == str(token)]  # Pega todos indices de ocorrencia de token em sentence
            for index in indexes:
                sentence[index] = str(token.lemma_)

    return sentence

def clean_sentences (sentences, nlp):
    cleanned_sentences = []

    for sentence in sentences:
        sentence = clean_sentence(sentence, nlp)
        cleanned_sentences.append(sentence)

    return clanned_sentences

def mapping_and_cleanning_1_x_1(sentences, nlp):
    cleanned_sentences = []
    df_mapping = pd.read_csv('mapping.csv')

    for sentence in sentences:
        sentence = clean_sentence(sentence, nlp)

        # Salva na tabela
        #print ("Adicionando: ", sentence)
        for word in sentence:
            if word not in list(df_mapping['word']):
                df_mapping.at[len(df_mapping['word']), 'word'] = word
                df_mapping.at[len(df_mapping['word'])-1, 'id'] = int(len(df_mapping['word']))
                df_mapping.to_csv('mapping.csv', index=False)
        cleanned_sentences.append(sentence)
    return df_mapping, cleanned_sentences

def cleanning (sentences, nlp):
    cleanned_sentences = []

    for sentence in sentences:
        sentence = clean_sentence(sentence, nlp)
        cleanned_sentences.append(sentence)

    return cleanned_sentences

def substituting_words_by_ids(sentences):
    """
    :param sentences: set of sentences
    :return: set of sentences substituted by their integer mapping ids
    """
    for sentence in sentences:
        for word in sentence:
            sentence [sentence.index(word)] = get_id(word)+1
    return sentences

def substituting_words_by_ids_in_one_sentence(sentence):
    """
    :param sentences: one sentence
    :return: the sentence substituted by integer mapping ids
    """
    for word in sentence:
            sentence [sentence.index(word)] = get_id(word)+1
    return sentence

def get_id(word):
    df_mapping = pd.read_csv('mapping.csv')
    lst = list(df_mapping['word'])
    return lst.index(word)

def size_of_biggest_sentence(sentences):
    biggest_sentence = 0
    for sentence in sentences:
        if len(sentence) >= biggest_sentence:
            biggest_sentence = len(sentence)
    return biggest_sentence

def predict_intent(message, nlp, biggest_sentence, model):
    message = clean_sentence(message, nlp)

    message = re.sub(r'[^ a-z A-Z 0-9]', " ", str(message))  # Misterioso, porem necessario
    message = word_tokenize(message)
    message = substituting_words_by_ids_in_one_sentence(message)
    message = list(message)
    while len(message) < biggest_sentence:
        message.append(0)
    message = np.array(message)
    intent = model.predict(message.reshape(1, -1))

    print("intent: ", intent)
    return intent

def user_train():
    df_intents = pd.read_csv('intents.csv')
    # sentences = df_intents["sentence"]
    intent = df_intents["intent"]
    unique_intent = list(set(intent))

    train_msg = "Desculpe, ainda nao sei responder. Mas voce pode me ajudar a aprender. " \
                "Qual destas era a sua intencao com a sua ultima mensagem?\n"
    for i in range(1, len(unique_intent)+1):
        train_msg += str(unique_intent[i-1]) + "\n"
    return train_msg

def save_msg(message_to_save, message_intent):
    df_intents = pd.read_csv('intents.csv')
    df_intents.at[len(df_intents['intent']) + 1, 'sentence'] = message_to_save
    df_intents.at[len(df_intents['intent']) , 'intent'] = message_intent
    df_intents.to_csv('intents.csv', index=False)

def train_models_1_x_1(nlp, sentences, intents):
    # 1. Mapeia as palavras da base e retorna sentencas tratadas
    df_mapping, sentences = mapping_and_cleanning_1_x_1(sentences, nlp)

    # 2. Cria matriz de representacao de frases por inteiros
    sentences = substituting_words_by_ids(sentences)

    # 3. Pega tamanho da maior sentence apos o tratamento
    biggest_sentence = size_of_biggest_sentence(sentences)

    # 4. Preenche com zeros as sentences menores
    for sentence in sentences:
        while len(sentence) < biggest_sentence:
            sentence.append(0)

    # 5. Prepara dados para o treinamento
    data = [[], []]
    data[0] = sentences
    data[1] = list(intents)

    # 6. Treina o modelo
    model = DecisionTreeClassifier()
    GaussianNB_model = GaussianNB()
    knn_model = KNeighborsClassifier(n_neighbors=15)
    svm_model = svm.SVC(gamma='scale')

    model.fit(data[0], data[1])
    GaussianNB_model.fit(data[0], data[1])
    knn_model.fit(data[0], data[1])
    svm_model.fit(data[0], data[1])
    random_forest = RandomForestClassifier()

    scores = cross_val_score(model, data[0], data[1], cv=5)
    print("Accuracy DecisionTreeClassifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(GaussianNB_model, data[0], data[1], cv=5)
    print("Accuracy GaussianNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(knn_model, data[0], data[1], cv=5)
    print("Accuracy KNeighborsClassifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(svm_model, data[0], data[1], cv=5)
    print("Accuracy SVM: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(random_forest, data[0], data[1], cv=5)
    print("Accuracy Random Forest: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return model, biggest_sentence

def train_models_tf_idf(nlp, sentences, intents):
    # 1. Prepara dados para o treinamento
    data = [[], []]
    data[0] = sentences
    data[1] = list(intents)

    # 2. Treina o modelo
    model = DecisionTreeClassifier()
    GaussianNB_model = GaussianNB()
    knn_model = KNeighborsClassifier(n_neighbors=15)
    svm_model = svm.SVC(gamma='scale')
    random_forest = RandomForestClassifier()#max_depth=3, random_state=0

    model.fit(data[0], data[1])
    GaussianNB_model.fit(data[0], data[1])
    knn_model.fit(data[0], data[1])
    svm_model.fit(data[0], data[1])
    random_forest.fit(data[0], data[1])

    scores = cross_val_score(model, data[0], data[1], cv=5)
    print("Accuracy DecisionTreeClassifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(GaussianNB_model, data[0], data[1], cv=5)
    print("Accuracy GaussianNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(knn_model, data[0], data[1], cv=5)
    print("Accuracy KNeighborsClassifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(svm_model, data[0], data[1], cv=5)
    print("Accuracy SVM: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(random_forest, data[0], data[1], cv=5)
    print("Accuracy Random Forest: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return model

def answer(_class):
    df_answers = pd.read_csv('answers.csv')
    if _class != "desconhecido":
        df_answers = df_answers.sort_values(by=[_class], ascending=False)
        str = df_answers.iloc[randrange(len(df_answers[_class]))][_class]
    return str

def didYouMean(message):
    url = "https://www.google.com/search?q=" + message
    req = requests.get (url)
    if req.status_code == 200:
        content = req.content
    soup = BeautifulSoup(content, 'html.parser')
    full_text = soup.get_text()
    begin = full_text.find("Exibindo resultados para")
    if begin > 0:
        full_text = full_text[begin:]
        end = full_text.find("(")
        print (message + "corrigido para: " + full_text[25:end])
        message = full_text[25:end]
        return message
    return message

def document_frequency (word, sentences):
    """Returns how many times word appeared in the sentences"""
    count = 1 # starts with 1 to avoid 0 ocurrences
    for sentence in sentences:
        for w in sentence:
            if w == word:
                count += 1
    return count

def inverse_document_frequency(words, sentences):
    df = {}
    for w in words:
        df[w]=document_frequency(w, sentences)
    idf= {}
    for d in df:
        num = len(sentences)+1
        den = df[d]
        idf[d] = np.log(num/den) + 1
    return idf

def term_frequency (sentence, idf):

    cnt = collections.Counter()
    for word in sentence:
        cnt[word] += 1

    i=0
    for word in sentence:
        sentence[i] = cnt[word] * idf[word]
        i+=1

    return sentence

def tfidf_mapping(nlp, sentences):

    df_mapping, sentences = mapping_and_cleanning_1_x_1(sentences, nlp)
    words = sorted(df_mapping['word'])
    print(sentences)
    idf = inverse_document_frequency(words, sentences)

    i = 0
    for sentence in sentences:
        sentences[i] = term_frequency(sentence, idf)
        i += 1
    sentences = euclidean_norm(sentences)

    # Preenche com zeros as sentences menores
    biggest_sentence = size_of_biggest_sentence(sentences)
    for sentence in sentences:
        while len(sentence) < biggest_sentence:
            sentence.append(0)

    return sentences

def euclidean_norm(sentences):
    for sentence in sentences:
        _squaresum = np.sqrt(square_sum(sentence))
        for i in range (0, len(sentence)):
            sentence[i] = sentence[i] / _squaresum
            i+=1
    return sentences

def square_sum(sentence):
    sum = 0
    for word in sentence:
        sum += word**2
    return sum

def cosin_similarity(sentences, intents):
    print (sentences)
    print ("intents: ", intents)
    message_received = sentences[-1]
    sentences.pop()
    print(sentences)
    biggest_similarity = 0
    index_of_biggest_similarity = 0
    i = 0
    for sentence in sentences:
        similarity = 1 - spatial.distance.cosine(sentence, message_received)
        if similarity > biggest_similarity:
            biggest_similarity = similarity
            index_of_biggest_similarity = i
        i += 1
    print(index_of_biggest_similarity)
    print (intents[index_of_biggest_similarity])
    return (intents[index_of_biggest_similarity])
