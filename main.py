import json
import math
import os
import random
import random as rn

import chess
import chess.svg
import torch
import torch.optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SIDES = ['white', 'black']

SAVE_HISTORY = False
GAMES = 100000000
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
    # print('res: ', res)
    index = math.floor(res * 1e8)
    index = index % len(_ll)

    return index


def game_state(_board):
    if _board.is_checkmate():
        if _board.turn:
            return 'black'
        else:
            return 'white'


def calculate_loss(_win, _board, colour):
    if _win == 'yes':
        if colour == 'white':
            if 'Q' in _board.board_fen():
                if _board.fullmove_number <= 20:
                    return 1.00
                else:
                    return 1.00 + (_board.fullmove_number * 0.001)
            else:
                if _board.fullmove_number <= 20:
                    return -1.00
                else:
                    return 0.01 + (_board.fullmove_number * 0.001)
        else:
            if 'q' in _board.board_fen():
                if _board.fullmove_number <= 20:
                    return 1.00
                else:
                    return 1.00 + (_board.fullmove_number * 0.001)
            else:
                if _board.fullmove_number <= 20:
                    return -1.00
                else:
                    return 0.01 + (_board.fullmove_number * 0.001)
    elif _win == 'no':
        if colour == 'white':
            if 'Q' in _board.board_fen():
                if _board.fullmove_number <= 20:
                    return 10.00
                else:
                    return 10.00 + (_board.fullmove_number * 0.01)
            else:
                if _board.fullmove_number <= 20:
                    return 7.00
                else:
                    return 7.00 + (_board.fullmove_number * 0.01)
        else:
            if 'q' in _board.board_fen():
                if _board.fullmove_number <= 20:
                    return 10.00
                else:
                    return 10.00 + (_board.fullmove_number * 0.01)
            else:
                if _board.fullmove_number <= 20:
                    return 7.00
                else:
                    return 7.00 + (_board.fullmove_number * 0.01)
    elif _win == 'draw':
        if colour == 'white':
            if 'Q' in _board.board_fen():
                if _board.fullmove_number <= 20:
                    return 7.00
                else:
                    return 7.00 + (_board.fullmove_number * 0.01)
            else:
                if _board.fullmove_number <= 20:
                    return 5.00
                else:
                    return 5.00 + (_board.fullmove_number * 0.01)
        else:
            if 'q' in _board.board_fen():
                if _board.fullmove_number <= 20:
                    return 7.00
                else:
                    return 7.00 + (_board.fullmove_number * 0.01)
            else:
                if _board.fullmove_number <= 20:
                    return 5.00
                else:
                    return 5.00 + (_board.fullmove_number * 0.01)


def train(_model, _optimiser):
    board = chess.Board()
    if os.path.isfile('model.pth'):
        _model = torch.load('model.pth')
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

        print(board)
        print(history)
        print(p1, winner)
        if p1 == winner:
            l_val = calculate_loss('yes', board, p1)
            print('win')
        elif p2 == winner:
            l_val = calculate_loss('no', board, p1)
            print('loss')
        else:
            l_val = calculate_loss('draw', board, p1)
        loss = torch.tensor([l_val], requires_grad=True)

        print('training loss:', l_val)

        _optimiser.zero_grad()
        loss.backward()
        _optimiser.step()
        board.reset()

        if (game + 1) % 10 == 0:
            print('Games [{}/{}]'.format(game + 1, GAMES))
            torch.save(_model, 'model.pth')


if __name__ == '__main__':
    model = RubyCore()

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, optimiser)
