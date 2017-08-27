import os
import sys
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *

path = "biblia.txt"
char_idx_file = 'char_idx.pickle'

# this is the length for the seed phrase (in characters) we want the model to predict
maxlen = 25

char_idx = None
if os.path.isfile(char_idx_file):
  print('Loading previous char_idx')
  file = open(char_idx_file, 'rb')
  char_idx = pickle.load(file, encoding='utf8')

# create sequences from tweets
X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

# store char index
pickle.dump(char_idx, open(char_idx_file,'wb'))

# initialize neural net, forward seq to seq (LSTM)
g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model')

m.load("model_saved.tfl")

for i in range(50):
    seed = random_sequence_from_textfile(path, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=128,
          n_epoch=1, run_id='tweets')
    print("TESTING...")
    print("-- Test with temperature of 1.0 --")
    print(m.generate(140, temperature=1.0, seq_seed=seed))
    m.save("model_saved.tfl")
    print("Finished interation " + str(i))

# Save the model
m.save("model_final.tfl")