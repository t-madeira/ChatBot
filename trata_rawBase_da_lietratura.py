f = open('base.txt', 'r')

sentences = []
for sentence in f.readlines():
    sentences.append(sentence[:-1])

for sentence in sentences:
    index = sentence.rfind(',')
    sentence = sentence[:index] + "\t" + sentence[index + 1:]
    print (sentence)
