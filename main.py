import json
import math
import os
import random as rn

import chess
import chess.svg
import torch
import torch.optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence

SIDES = ['white', 'black']

SAVE_HISTORY = False
EPOCHS = 10000
H_DIM = 512
L_DIM = 1


class RubyCore(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=H_DIM, layers_dim=L_DIM, output_dim=1):
        super(RubyCore, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers_dim = layers_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layers_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def get_legal_moves(_board):
    legal_moves = []
    for moves in _board.legal_moves:
        legal_moves.append(moves)
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
    output = _model(_inputs)

    res = output.item()
    index = math.floor((res * 100))
    index = index % len(_ll)

    return index


def game_state(_board):
    info = _board.outcome()
    return info.winner


def calculate_loss(_win):
    if _win:
        return -1.00
    else:
        return 10.00


def train(_model, _optimiser):
    if os.path.isfile('model.pth'):
        _model = torch.load('model.pth')
    for epoch in range(EPOCHS):
        opponent = RubyCore()
        opponent.load_state_dict(_model.state_dict())

        p1, p2 = rn.sample(SIDES, 2)

        history = []

        board = chess.Board()

        while True:
            if p1 == 'white':
                # p1
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, history, board, p1)
                index = get_output(_model, inputs, ll)

                history.append(str(ll[index]))
                board.push(ll[index])
                if board.is_game_over():
                    break

                # p2
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, history, board, p2)
                index = get_output(opponent, inputs, ll)

                history.append(str(ll[index]))
                board.push(ll[index])
                if board.is_game_over():
                    break

            if p1 == 'black':
                # p2
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, history, board, p2)
                index = get_output(opponent, inputs, ll)

                history.append(str(ll[index]))
                board.push(ll[index])
                if board.is_game_over():
                    break

                # p1
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, history, board, p1)
                index = get_output(_model, inputs, ll)

                history.append(str(ll[index]))
                board.push(ll[index])
                if board.is_game_over():
                    break

        winner = game_state(board)

        if SAVE_HISTORY:
            try:
                with open('match_history.json', 'r') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            data = {'player': p1, 'winner': winner, 'history': history}
            existing_data.append(data)

            with open('match_history.json', 'w') as f:
                json.dump(existing_data, f, indent=4)

        if p1 == winner:
            l_val = calculate_loss(True)
            print('win')
        else:
            l_val = calculate_loss(False)
            print('loss')
        loss = torch.tensor([l_val], requires_grad=True)

        _optimiser.zero_grad()
        loss.backward()
        _optimiser.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}]'.format(epoch + 1, EPOCHS))
            torch.save(_model, 'model.pth')


if __name__ == '__main__':
    model = RubyCore()

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model, optimiser)
