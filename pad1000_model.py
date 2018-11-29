import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout

NUM_GAMES = 24978 # Events 625,1541,2962,3904,5415,6698,7386,8655 were taken out (empty games)
PAD_LEN = 329

data = pd.read_csv("data.csv") # contains white/black elo and stockfish scores for each game
move_scores_df = data["MoveScores"]
white_elo = data["WhiteElo"]
black_elo = data["BlackElo"]

move_scores = [] # training stockfish data
temp = []

# get move scores from each game and pad to PAD_LEN
for line in move_scores_df:
	for s in line.split(' '):
		if s == 'NA':
			temp.append(temp[-1])
		elif s == '':
			continue
		else:
			temp.append(s)
	
	if int(temp[-1]) > 0:
		temp += [1000]*(PAD_LEN-len(temp))
	elif int(temp[-1]) < 0:
		temp += [-1000]*(PAD_LEN-len(temp))
	else:
		temp += [0]*(PAD_LEN-len(temp))
	
	temp = list(map(int,temp))
	move_scores.append(temp)
	temp = []

# get labels as tuple: (white elo, black elo)
elos = []
for i in range(NUM_GAMES):
	elos.append([int(white_elo[i]),int(black_elo[i])])

X = np.asarray(move_scores)
Y = np.asarray(elos)

# split into training, validation, and test set
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.33)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size = 0.1)

# model
model = Sequential()
model.add(Dense(1024, input_dim=PAD_LEN, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='relu'))
model.summary()

# train
model.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val), verbose=1)
prediction = model.predict(x_test)

error = np.mean((prediction-y_test)**2,axis=0) # mean squared error
print(error)
# print(prediction[0:10])
# print(y_test[0:10])