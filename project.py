import chess
import chess.pgn
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size = 3, padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(64, 2, kernel_size = 3, padding = 1)
        self.lin1 = nn.Linear(2, 1)

    # def set_parameters(self):

    def forward(self, inputs):
        out = F.relu(self.conv1(inputs))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = out.view(-1, 2)
        out = self.lin1(out)

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

def train(ChessNet, X, Y, optimizer, loss_func, n_epochs, bs = 256):
    loss = 0
    numLoss = 0
    data_X = torch.stack(X)
    data_Y = torch.Tensor(Y)
    data = utils.TensorDataset(data_X, data_Y)
    trainingSet = utils.DataLoader(data, batch_size = bs, shuffle = True)
    ChessNet.train()

    for epoch in range(n_epochs):
        for i, (data_set, label_set) in enumerate(trainingSet):
            output = ChessNet(data_set)
            l = loss_func(output, label_set)
            l.backward()
            optimizer.step()
            loss += l.item()
            numLoss += 1

    torch.save(ChessNet.state_dict(), 'model.pb')

def fit():
    """
    Training moves and board combos
    """
    ChessNet = ConvNet()
    opt = optim.Adam(ChessNet.parameters())
    loss = nn.CrossEntropyLoss()
    epochs = 20
    boardPairs = []
    labels = []

    with open('tactics.pgn') as pgn_handle:    
        b, m = load_puzzle(pgn_handle)    
        while b is not None:        
            in_channels = np.zeros((64, 2, 3, 3)) # board, next state, legal moves
            legal = list(b.legal_moves)
            # current state
            in_channels[:, 0] = torch.Tensor(convertBoard(b))

            for move in legal:
                b.push(move)
                # new next state
                in_channels[:, 1] = torch.Tensor(convertBoard(b))
                in_channels = torch.Tensor(in_channels)
                boardPairs.append(in_channels)
                if m == move:
                    labels.append(1)
                else:
                    labels.append(0)
                b.pop()

            # update board and move
            b, m = load_puzzle(pgn_handle)
    
    train(ChessNet, boardPairs, labels, opt, loss, epochs)

def move(board):
    """
    @param board: a chess.Board
    @return mv: a chess.Move
    """
    ChessNet = ConvNet()
    in_channels = np.zeros((64, 2, 3, 3)) # board, next state, legal moves    
    legal = list(board.legal_moves)
    outputs = []
    # opt = optim.SGD(ConvNet.parameters(), lr = 0.004, momentum = 0)
    # loss = nn.CrossEntropyLoss()
    # n_epochs = 20
    ChessNet.load_state_dict(torch.load('model.pb'))
    ChessNet.eval()

    in_channels[:, 0] = convertBoard(board)
    for move in legal:
        board.push(move)
        # new next state
        in_channels[:, 1] = convertBoard(board)
        
        netOutput = ChessNet(torch.Tensor(in_channels)).data[1]
        outputs.append(netOutput)
        board.pop()

    outputs = np.asarray(outputs)
    idx = np.argmax(outputs)
    mv = legal[idx]
    print(mv)

    return mv

fit()