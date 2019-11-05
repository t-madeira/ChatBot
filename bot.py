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

from sklearn.model_selection import train_test_split

import bot_functions

from random import randrange



# Coisas de conexao do chat
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
IP_address = "172.18.135.225"
Port = 8081
server.connect((IP_address, Port))

# Carregando modelo do portugues, isso demora pra cacete
nlp = spacy.load('pt_core_news_sm')

#
df_answers = pd.read_csv('answers.csv')

model, biggest_sentence = bot_functions.train_model(nlp)

# Validacao
# predito = model.predict(val_X)
# print("Frase\t\t\tValor predito\t\tValor esperado")
# contador = 0
# for i in range (len(val_Y)):
#     if predito[i] == val_Y[i]:
#         print (str(val_X[i])+"\t\t"+predito[i]+"\t\t\t"+str(val_Y[i])+ ' '+ u'\u2713')
#         contador += 1
#     else:
#         print (str(val_X[i])+"\t\t"+predito[i] + "\t\t\t" + str(val_Y[i]) + ' x')
# print ("acc: ", contador/len(val_Y))

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
                message_to_send = ""
                try:
                    intent = bot_functions.predict_intent(message_received, nlp, biggest_sentence, model)
                    message_to_send = bot_functions.answer(intent[0])
                except ValueError:
                    intent = ['desconhecido']
                    message_to_save = message_received
                    message_to_send += bot_functions.user_train()
                    train_mode = True
            else:
                message_received = socks.recv(2048)
                message_received = message_received.decode()[17:]
                message_intent = message_received
                bot_functions.save_msg(message_to_save[:-1], message_intent[:-1]) #[:-1] tirar o \n
                model, biggest_sentence = bot_functions.train_model(nlp)
                message_to_send = "Certo, entendi! Obrigado por contribuir com o meu treinamento! :)"
                train_mode = False



            print("\nUsuario enviou: " + message_received)
            server.send(message_to_send.encode())
            sys.stdout.write("Bot: ")
            sys.stdout.write(message_to_send + "\n__________________________________")
            sys.stdout.flush()