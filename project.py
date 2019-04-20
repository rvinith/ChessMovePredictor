import chess
import chess.pgn
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# class PredictMove:
#     def __init__(self, pool_size = 2)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_Size = 2)
        self.conv2 = nn.Conv2d(64, 2, kernel_size = 3, padding = 1)
        self.lin1 = nn.Linear(2, 1)

    # def set_parameters(self):

    def forward(self, inputs):
        # out = inputs.view(-1, )
        out = out.relu(self.conv1(out))
        out = self.pool(out)
        out = out.relu(self.conv2(out))
        out = self.lin1(out)

        return out


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
        movesList = torch.Tensor(b.legal_moves)
        # b = torch.Tensor(b)
        print(b)
        print(m)
        print(movesList)
        while b is not None:
            b, m = load_puzzle(pgn_handle)

def fit_and_validate(net, optimizer, loss_func, train, val, n_epochs, batch_size=1):
    net.eval()

    torch.save(net.cpu().state_dict(), "chessModel.pb")

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