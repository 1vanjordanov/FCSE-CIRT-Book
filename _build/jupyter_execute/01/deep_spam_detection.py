## SPAM detection - MKD

Importing libraries:

import pandas as pd
import re
import string
import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

Read dataset and mount google drive:

data = pd.read_csv("/content/drive/MyDrive/Spam_detection/spam2.csv",encoding = "'latin'")

from google.colab import drive
drive.mount('/content/drive')

data.head()

data["text"] = data.v2
data["spam"] = data.v1

## 1. Splitting data

from sklearn.model_selection import train_test_split
emails_train, emails_test, target_train, target_test = train_test_split(data.text,data.spam,test_size = 0.2)  # 80-20

data.info

emails_train.shape

## 2. Preprocessing

Python functions for cleaning text:

def remove_hyperlink(word):
    return  re.sub(r"http\S+", "", word)

def to_lower(word):
    result = word.lower()
    return result

def remove_number(word):
    result = re.sub(r'\d+', '', word)
    return result

def remove_punctuation(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result

def remove_whitespace(word):
    result = word.strip()
    return result

def replace_newline(word):
    return word.replace('\n','')



def clean_up_pipeline(sentence):
    cleaning_utils = [remove_hyperlink,
                      replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuation,remove_whitespace]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence

x_train = [clean_up_pipeline(o) for o in emails_train]
x_test = [clean_up_pipeline(o) for o in emails_test]

# x_train[0]

Load LabelEncoder and fit/transform the data:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_y = le.fit_transform(target_train.values)
test_y = le.transform(target_test.values)

train_y

## 3. Tokenize

## some config values 
embed_size = 100 # how big is each word vector
max_feature = 50000 # how many unique words to use (i.e num rows in embedding vector)
max_len = 2000 # max number of words in a question to use

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=max_feature)

tokenizer.fit_on_texts(x_train)

x_train_features = np.array(tokenizer.texts_to_sequences(x_train))
x_test_features = np.array(tokenizer.texts_to_sequences(x_test))

x_train_features[0]

## 4. Padding

from keras.preprocessing.sequence import pad_sequences
x_train_features = pad_sequences(x_train_features,maxlen=max_len)
x_test_features = pad_sequences(x_test_features,maxlen=max_len)
x_train_features[0]

## 5. Model

Load models:

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional
from keras.models import Model

# create the model
import tensorflow as tf
embedding_vecor_length = 32

model = tf.keras.Sequential()
model.add(Embedding(max_feature, embedding_vecor_length, input_length=max_len))
model.add(Bidirectional(tf.keras.layers.LSTM(64)))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train_features, train_y, batch_size=512, epochs=10, validation_data=(x_test_features, test_y))

from  matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()


y_predict  = [1 if o>0.5 else 0 for o in model.predict(x_test_features)]

y_predict

cf_matrix =confusion_matrix(test_y,y_predict)

Confusion matrix:

import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, ax = ax,cmap='Blues',fmt=''); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Not Spam', 'Spam']); ax.yaxis.set_ticklabels(['Not Spam', 'Spam']);

Metrics (Precision, Recall, F1 score):

from sklearn.metrics import confusion_matrix,f1_score, precision_score,recall_score

tn, fp, fn, tp = confusion_matrix(test_y,y_predict).ravel()

print("Precision: {:.2f}%".format(100 * precision_score(test_y, y_predict)))
print("Recall: {:.2f}%".format(100 * recall_score(test_y, y_predict)))
print("F1 Score: {:.2f}%".format(100 * f1_score(test_y,y_predict)))

np.round(f1_score(test_y,y_predict), 3)

## 6. Predict Macedonian mail

Install Google API translator with pip and load MKD language:

!pip install googletrans

!pip uninstall googletrans
!pip install googletrans==3.1.0a0

from googletrans import Translator, constants
from pprint import pprint

# init the Google API translator
translator = Translator()

# print all available languages
print("Total supported languages:", len(constants.LANGUAGES))
print("Languages:")
pprint(constants.LANGUAGES)  # Long list 

Test on Macedonian mail:

mk_mail = "Не, мислам дека не оди кај нас, сепак живее овде"

translation = translator.translate(mk_mail, src="mk")
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")

text_mkd = [translation.text]
text_mkd

ynew = model.predict(np.array(tokenizer.texts_to_sequences(text_mkd)))  # 0 или 1

text = "Spam" if np.round(ynew[0][0]) == 1 else "Not spam"

print(f"My sample mail was:\n\n-----------{text}-----------")