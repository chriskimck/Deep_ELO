#KAGGLEREADY

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout
from google.colab import files

NUM_GAMES = 24978 # Events 625,1541,2962,3904,5415,6698,7386,8655 were taken out (empty games)
PAD_LEN = 330
PAD_SCORE = 850

dataurl = 'https://raw.githubusercontent.com/chriskimck/Deep_ELO/master/kaggledata.csv'
data = pd.read_csv(dataurl) # contains white/black elo and stockfish scores for each game

testurl = 'https://raw.githubusercontent.com/chriskimck/Deep_ELO/master/kaggletest.csv'
test = pd.read_csv(testurl) # contains white/black elo and stockfish scores for each game

move_scores_df = data["MoveScores"]
white_elo = data["WhiteElo"]
black_elo = data["BlackElo"]

move_scores_test_df = test["MoveScores"]
event_test_df = test["Event"]

move_scores = [] #training stockfish data
move_scores_test = []
temp = []
elos = []
i = 0

for line in move_scores_df:
	for s in line.split(' '):
		if s == '':
			continue
		elif s == 'NA':
			temp.append(temp[-1])
		else:
			temp.append(s)
	if temp == []:
		temp += [PAD_SCORE]*(PAD_LEN-len(temp))
	else:
		if int(temp[-1]) > 0:
			temp += [PAD_SCORE]*(PAD_LEN-len(temp))
		
		elif int(temp[-1]) < 0:
			temp += [-PAD_SCORE]*(PAD_LEN-len(temp))	
		else:
			temp += [0]*(PAD_LEN-len(temp))
	elos.append([int(white_elo[i]),int(black_elo[i])])
	i+=1
	temp = list(map(int,temp))
	move_scores.append(temp)
	temp = []
  
for line in move_scores_test_df:
	for s in line.split(' '):
		if s == '':
			continue
		elif s == 'NA':
			temp.append(temp[-1])
		else:
			temp.append(s)
	if temp == []:
		temp += [PAD_SCORE]*(PAD_LEN-len(temp))
	else:
		if int(temp[-1]) > 0:
			temp += [PAD_SCORE]*(PAD_LEN-len(temp))
		
		elif int(temp[-1]) < 0:
			temp += [-PAD_SCORE]*(PAD_LEN-len(temp))	
		else:
			temp += [0]*(PAD_LEN-len(temp))
	temp = list(map(int,temp))
	move_scores_test.append(temp)
	temp = []


X = np.asarray(move_scores)
X_test = np.asarray(move_scores_test)
Y = np.asarray(elos)

X_train, X_val, y_train, y_val = train_test_split(X,Y, test_size = 0.1)

# model
model = Sequential()
model.add(Dense(512, input_dim=PAD_LEN, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(.35))
model.add(Dense(2, activation='relu'))
model.summary()

# train
optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optim, loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)
prediction = model.predict(X_test)

print(prediction[0:10])

submission = pd.DataFrame()

submission['Event'] = event_test_df
submission['WhiteElo'] = prediction[:,0]
submission['BlackElo'] = prediction[:,1]
submission.to_csv('submission_1.csv',index=False)
print("white:", error[0], ", black:", error[1], "\naverage:", np.mean(error))