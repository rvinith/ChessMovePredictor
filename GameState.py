import chess
import chess.pgn
from torch import nn
import numpy as np

class GameState(object):
    def __init__(self, gameBoard):
        if gameBoard is None:
            self.board = chess.Board()
        else:
            self.board = gameBoard

    def convert(self):
        #convert board state to bit representaiton for network
        self.netInput = self.board.shredder_fen()
        return self.netInput
    
    def legalMoves(self):
        return list(self.board.legal_moves)

    def rank(self):
        #ranks moves in network (weights)
        return 1


pgn_handle = open("tactics.pgn")
game = chess.pgn.read_game(pgn_handle)
game = chess.pgn.read_game(pgn_handle)
game = chess.pgn.read_game(pgn_handle)
board = game.board()
CurrentGame = GameState(board)
moves = game.mainline_moves()
result = {'1-0' : 1, '0-1' : -1, '1/2-1/2' : 0}[game.headers['Result']]
print(board)
print(board.fen())
print(CurrentGame.legalMoves())
print(CurrentGame.convert())