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
H_DIM = 256


class RubyCore(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=H_DIM, output_dim=64):
        super(RubyCore, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=8, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out_space = self.linear(out)
        out_space = self.softmax(out_space)
        return out_space


def get_legal_moves(_board):
    legal_moves = []
    for moves in _board.legal_moves:
        legal_moves.append(moves)
    return legal_moves


def create_inputs(_ll, _board, _colour):
    ll_ref = [str(move) for move in _ll]
    ll_input = '|'.join(ll_ref)
    tensor_ll = torch.tensor([ord(c) for c in ll_input])

    bb = [ord(c) for c in str(_board)]
    bb = [v for v in bb if v != 32]
    bb = [v for v in bb if v != 10]

    tensor_board = torch.tensor(bb)
    tensor_board = torch.chunk(tensor_board, 64)
    tensor_board = list(tensor_board)

    tensors_castle = torch.tensor([_board.has_kingside_castling_rights(chess.WHITE),
                                   _board.has_queenside_castling_rights(chess.WHITE),
                                   _board.has_kingside_castling_rights(chess.BLACK),
                                   _board.has_queenside_castling_rights(chess.BLACK)])

    tensors_castle = torch.chunk(tensors_castle, 4)
    tensors_castle = list(tensors_castle)

    tensor_colour = torch.tensor([0]) if _colour == 'white' else torch.tensor([1])
    tensor_move = torch.tensor([_board.fullmove_number])
    tensor_clock = torch.tensor([_board.halfmove_clock])

    tl = [tensor_board, tensors_castle]
    tl = [item for sublist in tl for item in sublist]
    tl.append(tensor_ll)
    tl.append(tensor_colour)
    tl.append(tensor_move)
    tl.append(tensor_clock)

    padded_sequences = pad_sequence(tl)

    inputs = torch.stack([padded_sequences.to(torch.float32)])

    return inputs


def get_output(_model, _inputs, _ll, _epsilon):
    if random.random() < _epsilon:
        index = random.randint(0, len(_ll) - 1)
    else:
        with torch.no_grad():
            _model = _model.to(device)
            _inputs = _inputs.to(device)
            output = _model(_inputs)
            index = torch.argmax(output).item()
            index = index % len(_ll)
    return index


def game_state(_board, _pl):
    res = _board.outcome().termination
    if _board.is_checkmate():
        if _board.turn:
            return 'black'
        else:
            return 'white'
    else:
        return res


def calculate_loss(_win, _board, _colour, _penalty, _ll):
    queen_present = 'Q' in _board.board_fen() if _colour == 'white' else 'q' in _board.board_fen()
    base_loss = 0.5 if queen_present else 0.00
    loss_divider = 1 if queen_present else 2
    exp_term = math.exp((_board.fullmove_number - math.exp(math.tau - 2.5)) / (math.exp(math.exp(1)) - 10) / math.pi)

    if _win == 'yes':
        return (base_loss + (exp_term / 3) / loss_divider) + _penalty
    elif _win == 'no':
        return base_loss + 8.0 + (exp_term / loss_divider) + _penalty
    elif _win == 'draw':
        return base_loss + 4.0 + (exp_term / loss_divider) + _penalty
    elif _win == 'progress':
        return base_loss + (exp_term / loss_divider) + _penalty


def move_penalty(_board, _p):
    _p += 10.0 if _board.halfmove_clock == 70.0 else 0.0
    return _p


def train(_model, _optimiser):
    board = chess.Board()
    if os.path.isfile('model.pth'):
        _model = torch.load('model.pth')
    num_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    print('number of trainable parameters:', num_params)
    epsilon = 0.05
    for game in range(GAMES):
        opponent = RubyCore()
        opponent.load_state_dict(_model.state_dict())

        p1, p2 = rn.sample(SIDES, 2)

        history = []
        penalty = 0.0
        # if epsilon < 0.05:
        #     epsilon = 0.05
        # else:
        #     epsilon = epsilon * 0.9999

        while True:
            if p1 == 'white':
                # p1
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, board, p1)
                index = get_output(_model, inputs, ll, epsilon)

                history.append(str(ll[index]))
                board.push(ll[index])
                penalty = move_penalty(board, penalty)
                loss = torch.tensor([calculate_loss('progress', board, p1, penalty, ll)], requires_grad=True)
                _optimiser.zero_grad()
                loss.backward()
                _optimiser.step()
                if board.is_game_over():
                    break

                # p2
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, board, p2)
                index = get_output(opponent, inputs, ll, epsilon)

                history.append(str(ll[index]))
                board.push(ll[index])
                if board.is_game_over():
                    break

            if p1 == 'black':
                # p2
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, board, p2)
                index = get_output(opponent, inputs, ll, epsilon)

                history.append(str(ll[index]))
                board.push(ll[index])
                if board.is_game_over():
                    break

                # p1
                ll = get_legal_moves(board)
                inputs = create_inputs(ll, board, p1)
                index = get_output(_model, inputs, ll, epsilon)

                history.append(str(ll[index]))
                board.push(ll[index])
                penalty = move_penalty(board, penalty)
                loss = torch.tensor([calculate_loss('progress', board, p1, penalty, ll)], requires_grad=True)
                _optimiser.zero_grad()
                loss.backward()
                _optimiser.step()
                if board.is_game_over():
                    break

        state = game_state(board, p1)

        if SAVE_HISTORY:
            try:
                with open('match_history.json', 'r') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            data = {'player': p1, 'state': state, 'history': history}
            existing_data.append(data)

            with open('match_history.json', 'w') as f:
                json.dump(existing_data, f, indent=4)

        # print(board)
        # print(history)
        # print('player:', p1, ', state:', state)
        penalty += 5.00 if (state == chess.Termination(4) or
                            state == chess.Termination(5) or
                            state == chess.Termination(6) or
                            state == chess.Termination(7)) else 0.0
        if p1 == state:
            l_val = calculate_loss('yes', board, p1, penalty, ll)
            # print('win')
        elif p2 == state:
            l_val = calculate_loss('no', board, p1, penalty, ll)
            # print('loss')
        else:
            l_val = calculate_loss('draw', board, p1, penalty, ll)
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

    optimiser = torch.optim.Adam(model.parameters(), lr=7e-4, weight_decay=3e-5)

    train(model, optimiser)
