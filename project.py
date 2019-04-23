import chess
import chess.pgn
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size = 1, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size = 1, padding = 1)
        self.lin1 = nn.Linear(2, 1)

    # def set_parameters(self):

    def forward(self, inputs):
        out = F.relu(self.conv1(inputs))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = out.view(-1, 2)
        out = self.lin1(out)

        return out

def convertBoard(board):
    flatBoard = np.zeros(64)
    for i in range(64):
        val = board.piece_at(i)        
        if val is None:
            flatBoard[i] = 0
        else:
            flatBoard[i] = {"P": 1, "N" : 2, "B" : 3, "R" : 4, "Q" : 5, "K" : 6, "p" : 7, "n" : 8, "b" : 9, "r" : 10, "q" : 11, "k" : 12}[val.symbol()]
    print(flatBoard.reshape(8,8))

    # return flatBoard
    return flatBoard.reshape(64, 1, 1)

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
        while b is not None:        
            in_channels = np.zeros((2, 64, 1, 1)) # white, black, legal moves
            legal = list(b.legal_moves)
            # current state
            in_channels[0] = torch.Tensor(convertBoard(b))
            b.push(m)
            # next state
            in_channels[1] = torch.Tensor(convertBoard(b))
            # layers[0][2] = torch.Tensor(legal)
            in_channels = torch.Tensor(in_channels)

            ChessNet = ConvNet()
            ChessNet.train()
            ChessNet(in_channels)
            b.pop()

            for move in legal:
                b.push(move)
                # new next state
                in_channels[1] = torch.Tensor(convertBoard(b))
                in_channels = torch.Tensor(in_channels)
                ChessNet(in_channels)
                b.pop()

            # update board and move
            b, m = load_puzzle(pgn_handle)
    
    torch.save(ChessNet.state_dict(), 'model.pb')

def move(board):
    """
    @param board: a chess.Board
    @return mv: a chess.Move
    """
    ChessNet = ConvNet()


    pass

# convertBoard(chess.Board())
fit()