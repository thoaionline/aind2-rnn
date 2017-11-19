import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

def window_transform_series(series, window_size):
    X = list(map(lambda i:series[i:i+window_size],range(0,len(series) - window_size)))
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

### Build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))

    return model


### Return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    punctuation_codes = list(map(ord,punctuation))
    character_codes = list(range(ord('a'),ord('z')+1))

    whitelist = list(map(chr,(punctuation_codes+character_codes)))+[' ']
    clean_text = ''.join([c for c in text if c in whitelist])

    return clean_text

### Fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[start:start+window_size] for start in range(0,len(text) - window_size, step_size)]
    outputs = [text[end] for end in range(window_size,len(text),step_size)]

    return inputs,outputs

### Build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    
    model.add(LSTM(200, input_shape=(window_size,num_chars),))
    model.add(Dense(num_chars, activation='softmax'))
   
    return model
