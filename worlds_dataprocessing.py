import chess
import chess.pgn
import chess.uci
import pandas as pd

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
	while not classical_games.is_end():
		node = classical_games.variations[0]
		board = classical_games.board()
		engine.position(board)
		evaluation = engine.go(movetime=evaltime)
		temp.append(handler.info["score"][1].cp)
		classical_games = node
	classical_movescores.append(temp)
	event.append(event_num)
	event_num += 1
	temp = []

classical_df = pd.DataFrame(index = range(0,len(event)), columns = ['Event', 'MoveScores'])
classical_df["Event"] = pd.DataFrame(event)
classical_df["MoveScores"] = pd.DataFrame(classical_movescores)

classical_df.to_csv("worlds_classical.csv", index=False)