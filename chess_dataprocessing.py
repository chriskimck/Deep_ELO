import chess.pgn
import chess
import pandas as pd
import numpy as np

stockfish_df = pd.read_csv("stockfish.csv")

pgn = open("data.pgn")

Event = []
WhiteElo = []
BlackElo = []
moves = []
movescores = []
Event_Test = []
moves_Test = []
movescores_Test = []
emptygames = [625,1541,2962,3904,5415,6698,7386,8655,8916,10505,10549,12872,13338,14483,17029,17051,18215,18829,20919,21585,21673,21676]

for i in range(50000):
	game = chess.pgn.read_game(pgn)
	if i < 25000:
		Event.append(int(game.headers["Event"]))
		WhiteElo.append(int(game.headers["WhiteElo"]))
		BlackElo.append(int(game.headers["BlackElo"]))
		exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
		pgn_string = game.accept(exporter)
		temp = pgn_string.split('\n\n')[1]
		temp = temp.replace('\n','')
		moves.append(temp)
		movescores.append(stockfish_df['MoveScores'][i])
	else:
		Event_Test.append(int(game.headers["Event"]))
		exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
		pgn_string = game.accept(exporter)
		temp = pgn_string.split('\n\n')[1]
		temp = temp.replace('\n','')
		moves_Test.append(temp)
		movescores_Test.append(stockfish_df['MoveScores'][i])

data_df = pd.DataFrame(index = range(0,len(Event)), columns = ['Event', 'WhiteElo', 'BlackElo', 'Moves', 'MoveScores'])
data_df["Event"] = pd.DataFrame(Event)
data_df["WhiteElo"] = pd.DataFrame(WhiteElo)
data_df["BlackElo"] = pd.DataFrame(BlackElo)
data_df["Moves"] = pd.DataFrame(moves)
data_df["MoveScores"] = pd.DataFrame(movescores)

datatest_df = pd.DataFrame(index = range(0,len(Event_Test)), columns = ['Event', 'Moves', 'MoveScores'])
datatest_df["Event"] = pd.DataFrame(Event_Test)
datatest_df["Moves"] = pd.DataFrame(moves_Test)
datatest_df["MoveScores"] = pd.DataFrame(movescores_Test)

data_df = data_df[['Event','WhiteElo','BlackElo','Moves','MoveScores']]
datatest_df = datatest_df[['Event','Moves','MoveScores']]

data_df.to_csv("kaggledata.csv", index=False)
datatest_df.to_csv("kaggletest.csv",index=False)

