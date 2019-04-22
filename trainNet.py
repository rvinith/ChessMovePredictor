import chess
import chess.pgn
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GameState import GameState

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_Size = 2)
        self.conv2 = nn.Conv2d(64, 2, kernel_size = 3, padding = 1)
        self.lin1 = nn.Linear(2, 1)

    # def set_parameters(self):

    def forward(self, inputs):
        out = F.relu(self.conv1(inputs))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = out.view(-1, 2)
        out = self.lin1(out)

        return out

def generateInputData():
    pgn_handle = open("tactics.pgn")
    game = chess.pgn.read_game(pgn_handle)
    while game is not None:
        game = chess.pgn.read_game(pgn_handle)
        board = game.board()
        CurrentGame = GameState(board)
        moves = game.mainline_moves()
        result = {'1-0' : 1, '0-1' : -1, '1/2-1/2' : 0}[game.headers['Result']]
        print(board)
        print(board.fen())
        print(CurrentGame.legalMoves())
        print(CurrentGame.convert())

generateInputData()