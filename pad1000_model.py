import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout

#try: conv2d, coord conv

data = pd.read_csv("data.csv")
move_scores_df = data["MoveScores"]
white_elo = data["WhiteElo"]
black_elo = data["BlackElo"]

move_scores = [] #training stockfish data
temp = []

for line in move_scores_df:
	for s in line.split(' '):
		if s == 'NA':
			temp.append(temp[-1])
		elif s == '':
			continue
		else:
			temp.append(s)
	
	if int(temp[-1]) > 0:
		temp += [1000]*(329-len(temp))
	elif int(temp[-1]) < 0:
		temp += [-1000]*(329-len(temp))
	else:
		temp += [0]*(329-len(temp))
	
	temp = list(map(int,temp))
	move_scores.append(temp)
	temp = []

elos = []
for i in range(24978):
	elos.append([int(white_elo[i]),int(black_elo[i])])

X = np.asarray(move_scores)
Y = np.asarray(elos)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.33)

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.1)


model = Sequential()
model.add(Dense(512, input_dim=329, activation='relu'))
model.add(Dense(2, activation='relu'))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

prediction = model.predict(X_test)

print(prediction[0:10])
print(y_test[0:10])