from flask import Flask,request,render_template
import pandas as pd
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Reading the data in the csv file.

app = Flask(__name__)

# requesting for the dataset.
df = pd.read_csv("templates/medium_data.csv")

df.dropna(axis=1, inplace=True)
df.select_dtypes("object").nunique()

df_title = df['title'] = df['title'].head(300).apply(lambda x: x.replace(u'\xa0', u' '))
df_title = df['title'] = df['title'].head(300).apply(lambda x: x.replace('\u200a', ' '))

tokenizer = Tokenizer(oov_token='<oov>')  # For those words which are not found in word_index
tokenizer.fit_on_texts(df_title)
total_words = len(tokenizer.word_index) + 1

print("Total number of words: ", total_words)
print("Word: ID")
print("------------")
print("<oov>: ", tokenizer.word_index['<oov>'])
print("Strong: ", tokenizer.word_index['strong'])
print("And: ", tokenizer.word_index['and'])
# print("Consumption: ", tokenizer.word_index['consumption'])
input_sequences = []
for line in df_title:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # print(token_list)

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# print(input_sequences)
print("Total input sequences: ", len(input_sequences))

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences[1]

# create features and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

print(xs[5])
print(labels[5])
print(ys[5][14])

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# Reducing the epochs size to make it run faster.
history = model.fit(xs, ys, epochs=100, verbose=1)
# print model.summary()
print(model)


@app.route("/",methods = ['GET'])
def home():

    return render_template("index.html")


@app.route('/send_data',methods = ['POST'])

def get_data_from_html():


    global history
    global model
    global next_words
    seed_text = request.form['pay']
    wds =int(request.form.get('pay1'))
    print(wds)
    next_words = wds
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predict = np.argmax(model.predict(token_list, verbose=0))
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predict:
                output_word = word
                break
        seed_text += " " + output_word

    print(seed_text)
    return render_template("index2.html", value=seed_text)



if __name__ == "__main__":
    app.run(host="localhost", port=int("5000"))
    
