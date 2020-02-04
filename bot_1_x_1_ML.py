import socket
import select
import sys
import spacy
import pandas as pd
import bot_functions

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

# Leitura dos dados
df_answers = pd.read_csv('answers.csv')
df_intents = pd.read_csv('intents.csv')
sentences = df_intents["sentence"]
intents = df_intents["intent"]

# Treinamento
model, biggest_sentence = bot_functions.train_models_1_x_1(nlp, sentences, intents)

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
                    intent = bot_functions.predict_intent(message_received, nlp, biggest_sentence, model)
                    message_to_send = bot_functions.answer(intent[0])
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