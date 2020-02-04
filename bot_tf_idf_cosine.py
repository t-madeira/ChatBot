import socket
import select
import sys
import pandas as pd
import spacy
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz

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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint

import bot_functions
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial

from random import randrange

# Coisas de conexao do chat
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
IP_address = "172.18.135.225"
Port = 8081
server.connect((IP_address, Port))

# Carregando modelo do portugues, isso demora pra cacete
print("Carregando modelo em portugues do spacy...", end="")
# nlp = spacy.load('pt_core_news_sm')
nlp = spacy.load('en')
print("pronto!")

df_intents = pd.read_csv('intents.csv')
sentences = list(df_intents["sentence"])
intents = df_intents["intent"]
sentences = bot_functions.tfidf_mapping(nlp, sentences)
model = bot_functions.train_models_tf_idf(nlp, sentences, intents)


print("****TELA DO BOT****")

sockets_list = [sys.stdin, server]
train_mode = False
while True:
    read_sockets, write_socket, error_socket = select.select(sockets_list, [], [])

    for socks in read_sockets:
        if socks == server:
            if not train_mode:
                message_received = socks.recv(2048)
                message_received = message_received.decode()[17:]
                try:
                    print(message_received)
                    # message_received = bot_functions.clean_sentence(message_received, nlp)
                    # print(message_received)
                    sentences.append(message_received)
                    sentences = bot_functions.tfidf_mapping(nlp, sentences)
                    print(sentences)
                    intent = bot_functions.cosin_similarity(sentences, intents)
                    message_to_send = bot_functions.answer(intent)
                except ValueError:
                    message_received = bot_functions.didYouMean(message_received)
                    try:
                        intent = bot_functions.predict_intent(message_received, nlp, biggest_sentence, model)
                        message_to_send = bot_functions.answer(intent[0])
                    except ValueError:
                        intent = ['desconhecido']
                        message_to_save = message_received
                        message_to_send = bot_functions.user_train()
                        train_mode = True
            else:
                message_received = socks.recv(2048)
                message_received = message_received.decode()[17:]
                message_intent = message_received

                bot_functions.save_msg(message_to_save[:-1], message_intent[:-1]) #[:-1] tirar o \n
                model, biggest_sentence = bot_functions.train_model(nlp)
                message_to_send = "Certo, entendi! Obrigado por contribuir com o meu treinamento! :)\n"
                train_mode = False

            print("\nUsuario enviou: " + message_received)
            server.send(message_to_send.encode())
            sys.stdout.write("Bot: ")
            sys.stdout.write(message_to_send + "\n__________________________________")
            sys.stdout.flush()