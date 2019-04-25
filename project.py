import chess
import chess.pgn
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import random

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.lin1 = nn.Linear(128, 32)
        self.lin2 = nn.Linear(32, 8)
        self.lin3 = nn.Linear(8, 1)

    def forward(self, inputs):
        out = self.lin1(inputs)
        out = self.lin2(out)
        out = torch.sigmoid(self.lin3(out))

        return out

def convertBoard(board):
    """
    converts board into numerical representation
    """
    flatBoard = np.zeros(64)
    for i in range(64):
        val = board.piece_at(i)        
        if val is None:
            flatBoard[i] = 0
        else:
            flatBoard[i] = {"P": 1, "N" : 2, "B" : 3, "R" : 4, "Q" : 5, "K" : 6, "p" : 7, "n" : 8, "b" : 9, "r" : 10, "q" : 11, "k" : 12}[val.symbol()]

    return flatBoard

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

def loss_batch(net, loss_fn, X, Y, opt = None):
    loss = loss_fn(net(X), Y)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(X)


def train(net, optimizer, loss_fn, X, Y, n_epochs):
    net.eval()
    data = torch.Tensor(X)
    labels = torch.Tensor(Y)
    with torch.no_grad():
        epoch_loss = [loss_fn(net(data), labels)]
    for i in range(n_epochs):
        loss = loss_fn(net(data), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss.append(loss)

    torch.save(net.state_dict(), 'model.pb')

def fit():
    """
    Training moves and board combos
    """
    files = ['tactics.pgn', 'tactics_depth8.pgn']
    ChessNet = NeuralNet()
    X = []
    Y = []
    game = 1
    loss_fn = nn.MSELoss()
    opt = optim.Adam(ChessNet.parameters(), lr = 0.004)
    n_epochs = 20
    ChessNet.train()
    for file in range(len(files)):
        with open(files[file]) as pgn_handle:    
            b, m = load_puzzle(pgn_handle)    
            while b is not None:      
                in_channels = np.zeros((2, 64)) # board, next state, legal moves
                legal = list(b.legal_moves)
                # current state
                in_channels[0] = convertBoard(b)

                for i in range(7):
                    move = random.choice(legal)
                    b.push(move)
                    # new next state
                    in_channels[1] = convertBoard(b)
                    inp = in_channels.flatten()
                    X.append(inp)
                    if move == m:
                        Y.append(1)
                    else:
                        Y.append(0)
                    # print("yeet ", game)
                    game += 1
                    b.pop()

                # update board and move
                b, m = load_puzzle(pgn_handle)

    train(ChessNet, opt, loss_fn, X, Y, n_epochs)

def move(board):
    """
    @param board: a chess.Board
    @return mv: a chess.Move
    """
    ChessNet = NeuralNet()
    in_channels = np.zeros((2, 64)) # board, next state, legal moves    
    legal = list(board.legal_moves)
    outputs = []
    ChessNet.load_state_dict(torch.load('model.pb'))
    ChessNet.eval()

    in_channels[0] = convertBoard(board)
    for move in legal:
        board.push(move)
        # new next state
        in_channels[1] = convertBoard(board)
        inp = torch.Tensor(in_channels.flatten())
        netOutput = ChessNet(inp)
        outputs.append(netOutput.data[0])
        board.pop()

    print(outputs)
    outputs = np.asarray(outputs)
    idx = np.argmax(outputs)
    # print(idx)
    mv = legal[idx]
    # print(mv)

    return mv

fit()
# with open('tactics.pgn') as pgn_handle:
#     b, m = load_puzzle(pgn_handle)
#     move(b)
#     b, m = load_puzzle(pgn_handle)
#     move(b)