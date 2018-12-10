import chess
import chess.pgn
import chess.uci
import pandas as pd
import numpy as np

classicalpgn = open("classical.pgn")
rapidpgn = open("rapid.pgn")

evaltime = 1000 #1 second
handler = chess.uci.InfoHandler()
engine = chess.uci.popen_engine('.\stockfish-10-win\Windows\stockfish_10_x64') #give correct address of your engine here
engine.info_handlers.append(handler)

classical_movescores = []
temp = []
event = []
event_num = 1
while event_num < 13:
	classical_games = chess.pgn.read_game(classicalpgn)
	#node = classical_games.variations[0]
	#board = classical_games.board()
	#classical_games = node
	i = 1
	while not classical_games.is_end():
		node = classical_games.variations[0]
		board = classical_games.board()
		engine.position(board)
		evaluation = engine.go(movetime=evaltime)
		if i % 2 == 0:
			temp.append(handler.info["score"][1].cp*-1)
		else:
			temp.append(handler.info["score"][1].cp)
		classical_games = node
		i+=1
	classical_movescores.append(temp)
	print(len(temp))
	event.append(event_num)
	event_num += 1
	temp = []
	i = 1

classical_df = pd.DataFrame(index = range(0,len(event)), columns = ['Event', 'MoveScores'])
classical_df["Event"] = pd.DataFrame(event)
test = pd.Series(classical_movescores)
classical_df["MoveScores"] = pd.DataFrame(test)

print(classical_df)

classical_df.to_csv("worlds_classical.csv", index=False)