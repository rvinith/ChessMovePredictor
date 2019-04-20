import chess
from torch import nn
import numpy as np

class GameState(object):
    def __init__(self):
        self.board = chess.Board()

    def convert(self):
        #convert board state to bit representaiton for network
        pass
    
    def legalMoves(self):
        return list(self.board.legal_moves)

    def rank(self):
        #ranks moves in network (weights)
        return 1

print(GameState.legalMoves)