import chess
import chess.pgn
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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
        moves = list(b.legal_moves)
        print(b)
        print(m)
        print(moves)
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