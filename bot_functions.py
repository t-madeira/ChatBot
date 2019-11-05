
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

from random import randrange

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


def clean(sentences):
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        # stemming
        words.append([i.lower() for i in w])

    return words

def clean_sentence (sentence, nlp):
    nltk.download("stopwords")
    nltk.download("punkt") # Preciso ver novamente o que isso faz
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

def mapping_and_cleanning(sentences, nlp):
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

def train_model(nlp):
    # 1. Pega os dados do intents.csv
    df_intents = pd.read_csv('intents.csv')
    sentences = df_intents["sentence"]
    # intent = df_intents["intent"]
    # unique_intent = list(set(intent))

    # 2. Mapeia as palavras da base e retorna sentencas tratadas
    df_mapping, sentences = mapping_and_cleanning(sentences, nlp)

    # 3. Cria matriz de representacao de frases por inteiros
    sentences = substituting_words_by_ids(sentences)

    # 4. Pega tamanho da maior sentence apos o tratamento
    biggest_sentence = size_of_biggest_sentence(sentences)

    # 5. Preenche com zeros as sentences menores
    for sentence in sentences:
        while len(sentence) < biggest_sentence:
            sentence.append(0)

    # 6. Prepara dados para o treinamento
    data = [[], []]
    i = 0
    for sentence in sentences:
        data[0].append(sentence)
        data[1].append(df_intents.iloc[i]['intent'])
        i += 1
    #train_X, val_X, train_Y, val_Y = train_test_split(data[0], data[1], shuffle=True, test_size=0.1)

    # 7. Treina o modelo
    model = DecisionTreeClassifier()
    model.fit(data[0], data[1])

    return model, biggest_sentence

def answer(_class):
    df_answers = pd.read_csv('answers.csv')
    if _class != "desconhecido":
        df_answers = df_answers.sort_values(by=[_class], ascending=False)
        str = df_answers.iloc[randrange(len(df_answers[_class]))][_class]
    return str