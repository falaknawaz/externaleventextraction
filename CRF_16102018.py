__author__ = 'Falak'

import pandas as pd
import numpy as np
import string
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import matplotlib.pyplot as plt
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from nltk.tokenize import sent_tokenize, word_tokenize


data = pd.read_csv("dataset/ner_dataset_updated_without_pos_tab_sep.csv", sep="\t", encoding="latin1", error_bad_lines=False)
data = data.fillna(method="ffill")
data.tail(10)
words = list(set(data["Word"].values))
words.append("ENDPAD")
n_words = len(words); n_words
tags = list(set(data["Tag"].values))
n_tags = len(tags); n_tags

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                     s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)

sent = getter.get_next()

print(sent)

sentences = getter.sentences

max_len = 75
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
#word2idx["Iraq"]
#tag2idx["B-neg"]


X = [[word2idx[w[0]] for w in s] for s in sentences]

X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)

y = [[tag2idx[w[1]] for w in s] for s in sentences]

y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

y = [to_categorical(i, num_classes=n_tags) for i in y]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

# build bilstm-crf model
input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=20, input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)

model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=3,
                    validation_split=0.1, verbose=1)

hist = pd.DataFrame(history.history)

plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.show()

test_pred = model.predict(X_te, verbose=1)
idx2tag = {i: w for w, i in tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out
    
pred_labels = pred2label(test_pred)
test_labels = pred2label(y_te)

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))

print(classification_report(test_labels, pred_labels))

# sentence 1
i = 19
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_te[i], -1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_te[i], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(words[w-1], tags[t], tags[pred]))

# sentence 2    
test_sentence = ["Hawking", "was","not", "a", "Fellow", "of", "the", "Royal", "Society", ",", "a", "lifetime", "member",
                 "of", "the", "Pontifical", "Academy", "of", "Sciences", ",", "and", "a", "recipient", "of",
                 "the", "Presidential", "Medal", "of", "Freedom", ",", "the", "highest", "civilian", "award",
                 "in", "the", "United", "States", "."]

x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=0, maxlen=max_len)

print(x_test_sent[0])

p = model.predict(np.array([x_test_sent[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentence, p[0]):
    print("{:15}: {:5}".format(w, tags[pred]))

# sentence 3
test_sentence = ["A","labor", "strike","is", "to","be","observed", "tomorrow", "in", "Sydney", "."]

x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=0, maxlen=max_len)

#print(x_test_sent[0])

p = model.predict(np.array([x_test_sent[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentence, p[0]):
    print("{:15}: {:5}".format(w, tags[pred]))    

# sentence 4
test_sentence = ["There","is", "possibility","of", "strike", "tomorrow", "."]

x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=0, maxlen=max_len)

#print(x_test_sent[0])

p = model.predict(np.array([x_test_sent[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentence, p[0]):
    print("{:15}: {:5}".format(w, tags[pred]))    

# sentence 5
text = "strike and protests are being observed today in Canberra"
test_sentence = word_tokenize(text)
x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=0, maxlen=max_len)

#print(x_test_sent[0])

p = model.predict(np.array([x_test_sent[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentence, p[0]):
    print("{:15}: {:5}".format(w, tags[pred]))  

# sentence 6
text = "freezing rain and snow are making driving extremely dangerous"
test_sentence = word_tokenize(text)
#test_sentence = ["strike","rain", "strike","rain", "weeeee"]

x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=0, maxlen=max_len)

#print(x_test_sent[0])

p = model.predict(np.array([x_test_sent[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentence, p[0]):
    print("{:15}: {:5}".format(w, tags[pred]))



# all tweets
'''
data = pd.read_csv('final-training-data-nov.txt', delimiter="<>", header=0)
data = data.fillna(method="ffill")
data.tail(10)
data.insert(0,"my_event_terms","")
i=0
#words = list(set(data["tweet_text"].values))
#print(data["tweet_text"])
#f = open('helloworld.txt','w', encoding='utf-8')
for text in list(data["tweet_text"].values):
    test_sentence = word_tokenize(text)
    #test_sentence = ["strike","rain", "strike","rain", "weeeee"]
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                                padding="post", value=0, maxlen=max_len)
    #print(x_test_sent[0])
    p = model.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
    print("{:15}||{}".format("Word", "Prediction"))
    print(30 * "=")
    a_events=""
    for w, pred in zip(test_sentence, p[0]):
        print("{:15}: {:5}".format(w, tags[pred]))
        #f.write("{:15}: {:5}\n".format(w, tags[pred]))
        if(tags[pred]=='B-event' or tags[pred]=='I-event'):
            if not a_events:
                a_events = w
            else:
                a_events = a_events + ";" + w 
    data.at[i, 'my_event_terms'] = a_events
    i+=1
del data["entities"]
del data["event_terms"]
data.to_csv("dataset/annotated_events.csv", sep="\t", index=False) 
#f.close()
'''



# all tweets old - discard below
'''
data = pd.read_csv('dataset\tweets.txt', encoding="latin1", header=0)
data = data.fillna(method="ffill")
data.tail(10)
words = list(data["Word"].values)
#print(words)
#print("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in list(set(data["Word"].values))).strip())
all_tweets = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in words]).strip()
#print(all_tweets)
sentences = sent_tokenize(all_tweets)
f = open('helloworld.txt','w', encoding='utf-8')
for text in list(sentences):
    #test_sentence = word_tokenize(sent)
    test_sentence = word_tokenize(text)
    #test_sentence = ["strike","rain", "strike","rain", "weeeee"]
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                                padding="post", value=0, maxlen=max_len)
    #print(x_test_sent[0])
    p = model.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
    print("{:15}||{}".format("Word", "Prediction"))
    print(30 * "=")
    for w, pred in zip(test_sentence, p[0]):
        print("{:15}: {:5}".format(w, tags[pred]))
        f.write("{:15}: {:5}\n".format(w, tags[pred]))
f.close()
'''