import chess
import chess.pgn
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle

class PredictMove:
    def __init__(self, pool_size = 2)
class CNN(nn.Module):
    def __init__(self):
        super(DigitsConvNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d
        self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
        self.lin1 = nn.Linear(16, 1)

    # def set_parameters(self):

    def forward(self, inputs):
        # inputs = inputs.view(-1, )
        x = x.relu(self.conv1(inputs))
        x = x.relu(self.conv2(inputs))
        x = x.relu(self.conv3(inputs))

        return x


def load_puzzle(pgn_handle):
    """
    Intended use case seen in fit():
    @param pgn_handle: file handle for your training file
    """
    board = chess.Board()
    game = chess.pgn.read_game(pgn_handle)
    if game is None:
        return None, None
    fen = game.headers['FEN']
    board.set_fen(fen)
    move = None
    for j, mv in enumerate(game.mainline_moves()):
        if j == 0:
            board.push(mv)
        if j == 1:
            return board, mv

def fit():
    """
    This is just a snippet for reading board-move pairs you might use for training
    """
    with open('tactics.pgn') as pgn_handle:
        b, m = load_puzzle(pgn_handle)
        b, m = load_puzzle(pgn_handle)
        b, m = load_puzzle(pgn_handle)
        # b = torch.Tensor(b)
        print(b)
        print(m)
        while b is not None:
            b, m = load_puzzle(pgn_handle)

def move(board):
    """
    @param board: a chess.Board
    @return mv: a chess.Move
    """
    #TODO: prediction here

    pass

# filenames = ['tactics.pgn']
# trainFile = open(filenames[0], 'r')
# board, move = load_puzzle(trainFile)
# # board = torch.tensor(board) 
# print(board)
# print(move)

fit()