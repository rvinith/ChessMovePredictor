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