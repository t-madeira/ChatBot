# ChatBot

1. Instalação manual do pacote português do spacy
```
python -m spacy download pt_core_news_sm
```

2. Settar o ip local nos arquivos bot.py, usuario.py e chat_server.py
```
IP_address = "172.18.135.225" 
```
3. Na função clean_sentence(), do arquivo bot_functions, as duas primeiras linhas de comando:
```
nltk.download("stopwords")
nltk.download("punkt")
```
Só precisam ser executas na primeira execução. O downlaod dos pacotes vai ser realizado e depois estas linhas podem ser comentadas/removidas
