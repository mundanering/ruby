import math
import os
import random
import random as rn
import sys

import chess
import chess.svg
import torch
import torch.optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SIDES = ['white', 'black']

GAMES = 1
H_DIM = 512
L_DIM = 4


class RubyCore(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=H_DIM, layers_dim=L_DIM, output_dim=1):
        super(RubyCore, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers_dim = layers_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layers_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def get_legal_moves(_board):
    legal_moves = []
    for moves in _board.legal_moves:
        legal_moves.append(moves)
    random.shuffle(legal_moves)
    return legal_moves


def create_inputs(_ll, _history, _board, _player):
    ll_ref = [str(move) for move in _ll]
    ll_input = '|'.join(ll_ref)
    tensor_ll = torch.tensor([ord(c) for c in ll_input])

    his_input = '>'.join(_history)
    tensor_history = torch.tensor([ord(c) for c in his_input])

    tensor_board = torch.tensor([ord(c) for c in _board.board_fen()])

    tensor_player = torch.tensor([ord(c) for c in _player])

    padded_sequences = pad_sequence([tensor_ll, tensor_history, tensor_board, tensor_player])

    inputs = torch.stack([padded_sequences.to(torch.float32)])

    return inputs


def get_output(_model, _inputs, _ll):
    _model = _model.to(device)
    _inputs = _inputs.to(device)
    output = _model(_inputs)

    res = output.item()
    index = math.floor(res * 1e8)
    index = index % len(_ll)

    return index


def game_state(_board):
    if _board.is_checkmate():
        if _board.turn:
            return 'black'
        else:
            return 'white'


def play(_model):
    board = chess.Board()
    print(board)
    print('-----------------')
    if os.path.isfile('model.pth'):
        _model = torch.load('model.pth')
    else:
        print('model not found')
        sys.exit(1)
    for game in range(GAMES):
        opponent = RubyCore()
        opponent.load_state_dict(_model.state_dict())

        p1, p2 = rn.sample(SIDES, 2)

        history = []

        while True:
            if p1 == 'white':
                # p1
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, history, board, p1)
                index = get_output(_model, inputs, ll)

                history.append(str(ll[index]))
                board.push(ll[index])
                print(board)
                print('-----------------')
                if board.is_game_over():
                    break

                # p2
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, history, board, p2)
                index = get_output(opponent, inputs, ll)

                history.append(str(ll[index]))
                board.push(ll[index])
                print(board)
                print('-----------------')
                if board.is_game_over():
                    break

            if p1 == 'black':
                # p2
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, history, board, p2)
                index = get_output(opponent, inputs, ll)

                history.append(str(ll[index]))
                board.push(ll[index])
                print(board)
                print('-----------------')
                if board.is_game_over():
                    break

                # p1
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, history, board, p1)
                index = get_output(_model, inputs, ll)

                history.append(str(ll[index]))
                board.push(ll[index])
                print(board)
                print('-----------------')
                if board.is_game_over():
                    break

        winner = game_state(board)

        print('winner:', winner)


if __name__ == '__main__':
    model = RubyCore()

    play(model)
