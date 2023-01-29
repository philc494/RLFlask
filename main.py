from flask import Flask, render_template, request, session
from joblib import dump, load
import numpy as np
import random
import collections.abc
from math import e
from collections import defaultdict
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__, static_folder='/Users/philcrawford/PycharmProjects/RLFlask/static')
app.secret_key = 'dljsaklqk24e21cjn!Ew@@dsa5'

@app.route('/', methods=['GET', 'POST'])
def show_tables():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('landing.html')


@app.route('/form', methods=['GET', 'POST'])
def form():
    return render_template('form.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    patt_inp = request.form['Pattern']
    iter_inp = request.form['Iterations']

    alph1_inp = request.form['Alpha1']
    expr1_inp = request.form['Exprate1']
    expdec1_inp = request.form['Expdecay1']

    alph2_inp = request.form['Alpha2']
    expr2_inp = request.form['Exprate2']
    expdec2_inp = request.form['Expdecay2']

    session['pattern'] = patt_inp
    session['iterations'] = iter_inp
    session['alph1'] = alph1_inp
    session['alph2'] = alph2_inp
    session['expr1'] = expr1_inp
    session['expr2'] = expr2_inp
    session['expdec1'] = expdec1_inp
    session['expdec2'] = expdec2_inp
    session.modified = True

    form_data = request.form
    run_rl(str(patt_inp) * 6, float(alph1_inp), float(expr1_inp),
           float(expdec1_inp), int(iter_inp), '1')
    run_rl(str(patt_inp) * 6, float(alph2_inp), float(expr2_inp),
           float(expdec2_inp), int(iter_inp), '2')

    visualize_tables(2, load('model1.joblib'), '1')
    visualize_tables(2, load('model2.joblib'), '2')
    return render_template('data.html', form_data=form_data)


@app.route('/plot', methods=['GET', 'POST'])
def plot():
    return render_template('plot.html',
                           url1a='static/AAint1.jpg',
                           url1b='static/AAint2.jpg',
                           url2a='static/BBint1.jpg',
                           url2b='static/BBint2.jpg',
                           url3a='static/CCint1.jpg',
                           url3b='static/CCint2.jpg',
                           url4a='static/DDint1.jpg',
                           url4b='static/DDint2.jpg',
                           patt_inp=session.get('pattern'),
                           iter_inp=session.get('iterations'),
                           alph1=session.get('alph1'),
                           alph2=session.get('alph2'),
                           expr1=session.get('expr1'),
                           expr2=session.get('expr2'),
                           expdec1=session.get('expdec1'),
                           expdec2=session.get('expdec2'))


def run_rl(trainpattern, alph, exprate, expval, iterations, label):
    train_pattern = trainpattern
    test_pattern = train_pattern

    train_iterations = iterations
    test_iterations = train_iterations
    exploration_rate = exprate

    alpha = alph
    exp_val = expval
    model_list = ['2']
    base_reward = 100

    parameters = {'train_pattern': train_pattern, 'test_pattern': test_pattern,
                              'train_iterations': train_iterations,
                              'test_iterations': test_iterations, "alpha": alpha, "exp_rate": exploration_rate,
                              "ExpVal": exp_val}

    # actual program

    train_seq = train_pattern * train_iterations
    games = len(train_seq)
    game = 0
    moves_per_train = []
    game_num_train = []
    scenario_per_train = []
    dist_list = []

    int_move_counter = 0
    act_move_counter = 0
    game = 0
    start_pos = (2, 2)
    current_pos = start_pos
    board_rows = 5
    board_cols = 5
    win_obj_A = (0, 0)
    win_obj_B = (0, 4)
    win_obj_C = (4, 0)
    win_obj_D = (4, 4)
    int_action_list = []
    act_action_list = []
    rewards_A = {}
    rewards_B = {}
    rewards_C = {}
    rewards_D = {}
    rewards_A_int = {}
    rewards_B_int = {}
    rewards_C_int = {}
    rewards_D_int = {}
    blank = {}
    for i in range(board_rows):
        for j in range(board_cols):
            rewards_A[(i, j)] = 0
            rewards_B[(i, j)] = 0
            rewards_C[(i, j)] = 0
            rewards_D[(i, j)] = 0
            rewards_A_int[(i, j)] = 0
            rewards_B_int[(i, j)] = 0
            rewards_C_int[(i, j)] = 0
            rewards_D_int[(i, j)] = 0
            blank[(i, j)] = 0

    for i in rewards_A:
        rewards_A[i] = {(-1, 0): 0, (1, 0): 0, (0, -1): 0, (0, 1): 0,
                        (-1, -1): 0, (-1, 1): 0, (1, -1): 0, (1, 1): 0, (0, 0): 0}
        rewards_B[i] = {(-1, 0): 0, (1, 0): 0, (0, -1): 0, (0, 1): 0,
                        (-1, -1): 0, (-1, 1): 0, (1, -1): 0, (1, 1): 0, (0, 0): 0}
        rewards_C[i] = {(-1, 0): 0, (1, 0): 0, (0, -1): 0, (0, 1): 0,
                        (-1, -1): 0, (-1, 1): 0, (1, -1): 0, (1, 1): 0, (0, 0): 0}
        rewards_D[i] = {(-1, 0): 0, (1, 0): 0, (0, -1): 0, (0, 1): 0,
                        (-1, -1): 0, (-1, 1): 0, (1, -1): 0, (1, 1): 0, (0, 0): 0}
        rewards_A_int[i] = {(-1, 0): 0, (1, 0): 0, (0, -1): 0, (0, 1): 0,
                            (-1, -1): 0, (-1, 1): 0, (1, -1): 0, (1, 1): 0, (0, 0): 0}
        rewards_B_int[i] = {(-1, 0): 0, (1, 0): 0, (0, -1): 0, (0, 1): 0,
                            (-1, -1): 0, (-1, 1): 0, (1, -1): 0, (1, 1): 0, (0, 0): 0}
        rewards_C_int[i] = {(-1, 0): 0, (1, 0): 0, (0, -1): 0, (0, 1): 0,
                            (-1, -1): 0, (-1, 1): 0, (1, -1): 0, (1, 1): 0, (0, 0): 0}
        rewards_D_int[i] = {(-1, 0): 0, (1, 0): 0, (0, -1): 0, (0, 1): 0,
                            (-1, -1): 0, (-1, 1): 0, (1, -1): 0, (1, 1): 0, (0, 0): 0}
        blank[i] = {(-1, 0): 0, (1, 0): 0, (0, -1): 0, (0, 1): 0,
                    (-1, -1): 0, (-1, 1): 0, (1, -1): 0, (1, 1): 0, (0, 0): 0}


    act_actions = [
        "up",
        "down",
        "left",
        "right",
        "uleft",
        "uright",
        "dleft",
        "dright",
        "stay"]
    int_actions = [
        "up",
        "down",
        "left",
        "right",
        "uleft",
        "uright",
        "dleft",
        "dright",
        "stay"]
    act_action_trans = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (
        0, 1), "uleft": (-1, -1), "uright": (-1, 1), "dleft": (1, -1), "dright": (1, 1), "stay": (0, 0)}
    int_action_trans = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (
        0, 1), "uleft": (-1, -1), "uright": (-1, 1), "dleft": (1, -1), "dright": (1, 1), "stay": (0, 0)}

    def get_exp(n):
        return e ** n


    def set_win_pos(letter):
        if letter == "A":
            winning_pos = win_obj_A
        elif letter == "B":
            winning_pos = win_obj_B
        elif letter == "C":
            winning_pos = win_obj_C
        else:
            winning_pos = win_obj_D
        return winning_pos

    def take_next_move(action):
        if action == "up":
            next_pos = (current_pos[0] - 1, current_pos[1])
        elif action == "down":
            next_pos = (current_pos[0] + 1, current_pos[1])
        elif action == "left":
            next_pos = (current_pos[0], current_pos[1] - 1)
        elif action == "right":
            next_pos = (current_pos[0], current_pos[1] + 1)
        elif action == "stay":
            next_pos = (current_pos[0], current_pos[1])
        elif action == "uleft":
            next_pos = (current_pos[0] - 1, current_pos[1] - 1)
        elif action == "uright":
            next_pos = (current_pos[0] - 1, current_pos[1] + 1)
        elif action == "dleft":
            next_pos = (current_pos[0] + 1, current_pos[1] - 1)
        else:
            next_pos = (current_pos[0] + 1, current_pos[1] + 1)
        if (next_pos[0] >= 0) and (next_pos[0] <= (board_rows - 1)):
            if (next_pos[1] >= 0) and (next_pos[1] <= (board_cols - 1)):
                return next_pos
        return current_pos

    def pick_act_move(win_target, exprate):
        exp_rate = exprate
        next_act_action = ""
        while True:
            random.shuffle(act_actions)
            if np.random.uniform(0, 1) <= exp_rate:
                next_act_action = np.random.choice(act_actions)
            else:
                best_reward = -1000000
                if win_target == "A":
                    for a in act_actions:
                        poss_reward = rewards_A[current_pos][act_action_trans[a]]
                        if poss_reward > best_reward:
                            next_act_action = a
                            best_reward = poss_reward
                elif win_target == "B":
                    for a in act_actions:
                        poss_reward = rewards_B[current_pos][act_action_trans[a]]
                        if poss_reward > best_reward:
                            next_act_action = a
                            best_reward = poss_reward
                elif win_target == "C":
                    for a in act_actions:
                        poss_reward = rewards_C[current_pos][act_action_trans[a]]
                        if poss_reward > best_reward:
                            next_act_action = a
                            best_reward = poss_reward
                else:
                    for a in act_actions:
                        poss_reward = rewards_D[current_pos][act_action_trans[a]]
                        if poss_reward > best_reward:
                            next_act_action = a
                            best_reward = poss_reward
            new_position = take_next_move(next_act_action)
            if new_position == current_pos and next_act_action != "stay":
                continue
            else:
                return next_act_action

    def pick_int_move(prev_target):
        next_int_action = ""
        while True:
            random.shuffle(int_actions)
            if np.random.uniform(0, 1) <= exploration_rate:
                next_int_action = np.random.choice(int_actions)
            else:
                best_reward = -1000000
                if prev_target == "A":
                    for a in int_actions:
                        poss_reward = rewards_A_int[current_pos][int_action_trans[a]]
                        if poss_reward > best_reward:
                            next_int_action = a
                            best_reward = poss_reward
                elif prev_target == "B":
                    for a in int_actions:
                        poss_reward = rewards_B_int[current_pos][int_action_trans[a]]
                        if poss_reward > best_reward:
                            next_int_action = a
                            best_reward = poss_reward
                elif prev_target == "C":
                    for a in int_actions:
                        poss_reward = rewards_C_int[current_pos][int_action_trans[a]]
                        if poss_reward > best_reward:
                            next_int_action = a
                            best_reward = poss_reward
                else:
                    for a in int_actions:
                        poss_reward = rewards_D_int[current_pos][int_action_trans[a]]
                        if poss_reward > best_reward:
                            next_int_action = a
                            best_reward = poss_reward
            new_position = take_next_move(next_int_action)
            if new_position == current_pos and next_int_action != "stay":
                continue
            else:
                return next_int_action

    def dict_update(d, update):
        for k, v in update.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = dict_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def update_act_rewards(reward_apply):
        reversed_act_list = unique(list(reversed(act_action_list)))
        for a, b in reversed_act_list:
            if scenario == "A":
                rewards_A[a][b] = round(
                    rewards_A[a][b] + alpha * (reward_apply - rewards_A[a][b]), 2)
            elif scenario == "B":
                rewards_B[a][b] = round(
                    rewards_B[a][b] + alpha * (reward_apply - rewards_B[a][b]), 2)
            elif scenario == "C":
                rewards_C[a][b] = round(
                    rewards_C[a][b] + alpha * (reward_apply - rewards_C[a][b]), 2)
            else:
                rewards_D[a][b] = round(
                    rewards_D[a][b] + alpha * (reward_apply - rewards_D[a][b]), 2)

    def update_int_rewards(reward_apply):
        if game == 0:
            return
        reversed_int_list = unique(list(reversed(int_action_list)))
        for a, b in reversed_int_list:
            if prev_scenario == "A":
                rewards_A_int[a][b] = round(
                    rewards_A_int[a][b] + alpha * (reward_apply - rewards_A_int[a][b]), 2)
            elif prev_scenario == "B":
                rewards_B_int[a][b] = round(
                    rewards_B_int[a][b] + alpha * (reward_apply - rewards_B_int[a][b]), 2)
            elif prev_scenario == "C":
                rewards_C_int[a][b] = round(
                    rewards_C_int[a][b] + alpha * (reward_apply - rewards_C_int[a][b]), 2)
            else:
                rewards_D_int[a][b] = round(
                    rewards_D_int[a][b] + alpha * (reward_apply - rewards_D_int[a][b]), 2)

    def check_to_int():
        if int_move_counter < 4 and into_int_state:
            return True
        return False

    def unique(sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    def dist_calc(currentpos, next_target):
        if next_target == 'A':
            dist = max(abs(currentpos[0] - win_obj_A[0]), abs(currentpos[1] - win_obj_A[1]))
        elif next_target == 'B':
            dist = max(abs(currentpos[0] - win_obj_B[0]), abs(currentpos[1] - win_obj_B[1]))
        elif next_target == 'C':
            dist = max(abs(currentpos[0] - win_obj_C[0]), abs(currentpos[1] - win_obj_C[1]))
        else:
            dist = max(abs(currentpos[0] - win_obj_D[0]), abs(currentpos[1] - win_obj_D[1]))
        return dist

    prev_scenario = 'X'
    into_int_state = False
    dist_recorded = False

    while game < games:
        scenario = train_seq[game]
        win_pos = set_win_pos(scenario)
        go_to_int = check_to_int()
        if go_to_int:
            int_action = pick_int_move(prev_scenario)
            int_action_coord = int_action_trans[int_action]
            int_action_list.append((current_pos, int_action_coord))
            current_pos = take_next_move(int_action)
            int_move_counter += 1
        else:
            into_int_state = False
            if not dist_recorded:
                int_distance = dist_calc(current_pos, scenario)
                dist_list.append(int_distance)
                dist_recorded = True
            if current_pos == win_pos:
                reward = base_reward * (get_exp(exp_val * act_move_counter))
                update_act_rewards(reward)
                update_int_rewards(reward)
                moves_per_train.append(act_move_counter)
                game_num_train.append(game + 1)
                scenario_per_train.append(scenario)
                game += 1
                act_move_counter = 0
                int_move_counter = 0
                int_action_list = []
                act_action_list = []
                into_int_state = True
                dist_recorded = False
                prev_scenario = scenario
            else:
                act_action = pick_act_move(scenario, exploration_rate)
                act_action_coord = act_action_trans[act_action]
                act_action_list.append((current_pos, act_action_coord))
                current_pos = take_next_move(act_action)
                act_move_counter += 1

    rewards_return = {'A': rewards_A, 'B': rewards_B, 'C': rewards_C, 'D': rewards_D, 'Aint': rewards_A_int,
            'Bint': rewards_B_int, 'Cint': rewards_C_int,
            'Dint': rewards_D_int}
    info_return = {'train_moves': moves_per_train, 'games_train': game_num_train,
            'scen_train': scenario_per_train}

    # Train visualization

    reward_results = {}
    for model in model_list:
        reward_results[model] = {}
        for x in range(0, 1):
            reward_results[model][x] = rewards_return

    set_list = []
    table_list = []
    pos_list = []
    act_list = []
    resultsfinal = defaultdict(dict)
    reward_tempdic = defaultdict(dict)

    for a in reward_results[model]:
        set_list.append(a)
    for a in reward_results[model][0]:
        table_list.append(a)
    for a in reward_results[model][0]['A']:
        pos_list.append(a)
    for a in reward_results[model][0]['A']:
        for b in reward_results[model][0]['A'][a]:
            act_list.append(b)

    for a in model_list:
        reward_tempdic[a] = defaultdict(dict)
        resultsfinal[a] = defaultdict(dict)
        for b in table_list:
            reward_tempdic[a][b] = defaultdict(dict)
            resultsfinal[a][b] = defaultdict(dict)
            for c in pos_list:
                reward_tempdic[a][b][c] = defaultdict(dict)
                resultsfinal[a][b][c] = defaultdict(dict)
                for d in act_list:
                    reward_tempdic[a][b][c][d] = defaultdict(dict)
                    resultsfinal[a][b][c][d] = defaultdict(dict)
                    for x in set_list:
                        reward_tempdic[a][b][c][d][x] = reward_results[a][x][b][c][d]

    for model in model_list:
        for b in table_list:
            for c in pos_list:
                for d in act_list:
                    resultsfinal[model][b][c][d] = round(sum(reward_tempdic[model][b][c][d].values()) / len(set_list), 2)

    dump(resultsfinal[model], 'model' + label + '.joblib')


def visualize_tables(model, resultsdic, label):
    def minisquare_values(reward_dic, board_pos):
        mini_dic = {}
        for i in range(-1, 2):
            for j in range(-1, 2):
                mini_dic[(i, j)] = round((reward_dic[board_pos][(i, j)]), 4)
        return mini_dic
    # reward_info_out = {"scenA": [resultsdic["A"], "AA_rewards", "AA"],
    #                    "scenB": [resultsdic["B"], "BB_rewards", "BB"],
    #                    "scenC": [resultsdic["C"], "CC_rewards", "CC"],
    #                    "scenD": [resultsdic["D"], "DD_rewards", "DD"],
    #                    "intA": [resultsdic["Aint"], "AAint_rewards", "AAint"],
    #                    "intB": [resultsdic["Bint"], "BBint_rewards", "BBint"],
    #                    "intC": [resultsdic["Cint"], "CCint_rewards", "CCint"],
    #                    "intD": [resultsdic["Dint"], "DDint_rewards", "DDint"]}

    reward_info_out = {"intA": [resultsdic["Aint"], "AAint_rewards", "AAint"],
                       "intB": [resultsdic["Bint"], "BBint_rewards", "BBint"],
                       "intC": [resultsdic["Cint"], "CCint_rewards", "CCint"],
                       "intD": [resultsdic["Dint"], "DDint_rewards", "DDint"]}

    for a in reward_info_out:
        dic_value_list = []
        for b in reward_info_out[a][0]:
            for c in reward_info_out[a][0][b]:
                dic_value_list.append(reward_info_out[a][0][b][c])
        min_val = min(dic_value_list)
        max_val = max(dic_value_list)
        zero_act_val = min_val
        dic00 = minisquare_values(reward_info_out[a][0], (0, 0))
        for key, val in dic00:
            if dic00[(key, val)] == -9.9:
                dic00[(key, val)] = zero_act_val
        v00_n1n1 = dic00[(-1, -1)]
        v00_n10 = dic00[(-1, 0)]
        v00_n11 = dic00[(-1, 1)]
        v00_0n1 = dic00[(0, -1)]
        v00_00 = dic00[(0, 0)]
        v00_01 = dic00[(0, 1)]
        v00_1n1 = dic00[(1, -1)]
        v00_10 = dic00[(1, 0)]
        v00_11 = dic00[(1, 1)]
        dic01 = minisquare_values(reward_info_out[a][0], (0, 1))
        for key, val in dic01:
            if dic01[(key, val)] == -9.9:
                dic01[(key, val)] = zero_act_val
        v01_n1n1 = dic01[(-1, -1)]
        v01_n10 = dic01[(-1, 0)]
        v01_n11 = dic01[(-1, 1)]
        v01_0n1 = dic01[(0, -1)]
        v01_00 = dic01[(0, 0)]
        v01_01 = dic01[(0, 1)]
        v01_1n1 = dic01[(1, -1)]
        v01_10 = dic01[(1, 0)]
        v01_11 = dic01[(1, 1)]
        dic02 = minisquare_values(reward_info_out[a][0], (0, 2))
        for key, val in dic02:
            if dic02[(key, val)] == -9.9:
                dic02[(key, val)] = zero_act_val
        v02_n1n1 = dic02[(-1, -1)]
        v02_n10 = dic02[(-1, 0)]
        v02_n11 = dic02[(-1, 1)]
        v02_0n1 = dic02[(0, -1)]
        v02_00 = dic02[(0, 0)]
        v02_01 = dic02[(0, 1)]
        v02_1n1 = dic02[(1, -1)]
        v02_10 = dic02[(1, 0)]
        v02_11 = dic02[(1, 1)]
        dic03 = minisquare_values(reward_info_out[a][0], (0, 3))
        for key, val in dic03:
            if dic03[(key, val)] == -9.9:
                dic03[(key, val)] = zero_act_val
        v03_n1n1 = dic03[(-1, -1)]
        v03_n10 = dic03[(-1, 0)]
        v03_n11 = dic03[(-1, 1)]
        v03_0n1 = dic03[(0, -1)]
        v03_00 = dic03[(0, 0)]
        v03_01 = dic03[(0, 1)]
        v03_1n1 = dic03[(1, -1)]
        v03_10 = dic03[(1, 0)]
        v03_11 = dic03[(1, 1)]
        dic04 = minisquare_values(reward_info_out[a][0], (0, 4))
        for key, val in dic04:
            if dic04[(key, val)] == -9.9:
                dic04[(key, val)] = zero_act_val
        v04_n1n1 = dic04[(-1, -1)]
        v04_n10 = dic04[(-1, 0)]
        v04_n11 = dic04[(-1, 1)]
        v04_0n1 = dic04[(0, -1)]
        v04_00 = dic04[(0, 0)]
        v04_01 = dic04[(0, 1)]
        v04_1n1 = dic04[(1, -1)]
        v04_10 = dic04[(1, 0)]
        v04_11 = dic04[(1, 1)]
        dic10 = minisquare_values(reward_info_out[a][0], (1, 0))
        for key, val in dic10:
            if dic10[(key, val)] == -9.9:
                dic10[(key, val)] = zero_act_val
        v10_n1n1 = dic10[(-1, -1)]
        v10_n10 = dic10[(-1, 0)]
        v10_n11 = dic10[(-1, 1)]
        v10_0n1 = dic10[(0, -1)]
        v10_00 = dic10[(0, 0)]
        v10_01 = dic10[(0, 1)]
        v10_1n1 = dic10[(1, -1)]
        v10_10 = dic10[(1, 0)]
        v10_11 = dic10[(1, 1)]
        dic11 = minisquare_values(reward_info_out[a][0], (1, 1))
        for key, val in dic11:
            if dic11[(key, val)] == -9.9:
                dic11[(key, val)] = zero_act_val
        v11_n1n1 = dic11[(-1, -1)]
        v11_n10 = dic11[(-1, 0)]
        v11_n11 = dic11[(-1, 1)]
        v11_0n1 = dic11[(0, -1)]
        v11_00 = dic11[(0, 0)]
        v11_01 = dic11[(0, 1)]
        v11_1n1 = dic11[(1, -1)]
        v11_10 = dic11[(1, 0)]
        v11_11 = dic11[(1, 1)]
        dic12 = minisquare_values(reward_info_out[a][0], (1, 2))
        for key, val in dic12:
            if dic12[(key, val)] == -9.9:
                dic12[(key, val)] = zero_act_val
        v12_n1n1 = dic12[(-1, -1)]
        v12_n10 = dic12[(-1, 0)]
        v12_n11 = dic12[(-1, 1)]
        v12_0n1 = dic12[(0, -1)]
        v12_00 = dic12[(0, 0)]
        v12_01 = dic12[(0, 1)]
        v12_1n1 = dic12[(1, -1)]
        v12_10 = dic12[(1, 0)]
        v12_11 = dic12[(1, 1)]
        dic13 = minisquare_values(reward_info_out[a][0], (1, 3))
        for key, val in dic13:
            if dic13[(key, val)] == -9.9:
                dic13[(key, val)] = zero_act_val
        v13_n1n1 = dic13[(-1, -1)]
        v13_n10 = dic13[(-1, 0)]
        v13_n11 = dic13[(-1, 1)]
        v13_0n1 = dic13[(0, -1)]
        v13_00 = dic13[(0, 0)]
        v13_01 = dic13[(0, 1)]
        v13_1n1 = dic13[(1, -1)]
        v13_10 = dic13[(1, 0)]
        v13_11 = dic13[(1, 1)]
        dic14 = minisquare_values(reward_info_out[a][0], (1, 4))
        for key, val in dic14:
            if dic14[(key, val)] == -9.9:
                dic14[(key, val)] = zero_act_val
        v14_n1n1 = dic14[(-1, -1)]
        v14_n10 = dic14[(-1, 0)]
        v14_n11 = dic14[(-1, 1)]
        v14_0n1 = dic14[(0, -1)]
        v14_00 = dic14[(0, 0)]
        v14_01 = dic14[(0, 1)]
        v14_1n1 = dic14[(1, -1)]
        v14_10 = dic14[(1, 0)]
        v14_11 = dic14[(1, 1)]
        dic20 = minisquare_values(reward_info_out[a][0], (2, 0))
        for key, val in dic20:
            if dic20[(key, val)] == -9.9:
                dic20[(key, val)] = zero_act_val
        v20_n1n1 = dic20[(-1, -1)]
        v20_n10 = dic20[(-1, 0)]
        v20_n11 = dic20[(-1, 1)]
        v20_0n1 = dic20[(0, -1)]
        v20_00 = dic20[(0, 0)]
        v20_01 = dic20[(0, 1)]
        v20_1n1 = dic20[(1, -1)]
        v20_10 = dic20[(1, 0)]
        v20_11 = dic20[(1, 1)]
        dic21 = minisquare_values(reward_info_out[a][0], (2, 1))
        for key, val in dic21:
            if dic21[(key, val)] == -9.9:
                dic21[(key, val)] = zero_act_val
        v21_n1n1 = dic21[(-1, -1)]
        v21_n10 = dic21[(-1, 0)]
        v21_n11 = dic21[(-1, 1)]
        v21_0n1 = dic21[(0, -1)]
        v21_00 = dic21[(0, 0)]
        v21_01 = dic21[(0, 1)]
        v21_1n1 = dic21[(1, -1)]
        v21_10 = dic21[(1, 0)]
        v21_11 = dic21[(1, 1)]
        dic22 = minisquare_values(reward_info_out[a][0], (2, 2))
        for key, val in dic22:
            if dic22[(key, val)] == -9.9:
                dic22[(key, val)] = zero_act_val
        v22_n1n1 = dic22[(-1, -1)]
        v22_n10 = dic22[(-1, 0)]
        v22_n11 = dic22[(-1, 1)]
        v22_0n1 = dic22[(0, -1)]
        v22_00 = dic22[(0, 0)]
        v22_01 = dic22[(0, 1)]
        v22_1n1 = dic22[(1, -1)]
        v22_10 = dic22[(1, 0)]
        v22_11 = dic22[(1, 1)]
        dic23 = minisquare_values(reward_info_out[a][0], (2, 3))
        for key, val in dic23:
            if dic23[(key, val)] == -9.9:
                dic23[(key, val)] = zero_act_val
        v23_n1n1 = dic23[(-1, -1)]
        v23_n10 = dic23[(-1, 0)]
        v23_n11 = dic23[(-1, 1)]
        v23_0n1 = dic23[(0, -1)]
        v23_00 = dic23[(0, 0)]
        v23_01 = dic23[(0, 1)]
        v23_1n1 = dic23[(1, -1)]
        v23_10 = dic23[(1, 0)]
        v23_11 = dic23[(1, 1)]
        dic24 = minisquare_values(reward_info_out[a][0], (2, 4))
        for key, val in dic24:
            if dic24[(key, val)] == -9.9:
                dic24[(key, val)] = zero_act_val
        v24_n1n1 = dic24[(-1, -1)]
        v24_n10 = dic24[(-1, 0)]
        v24_n11 = dic24[(-1, 1)]
        v24_0n1 = dic24[(0, -1)]
        v24_00 = dic24[(0, 0)]
        v24_01 = dic24[(0, 1)]
        v24_1n1 = dic24[(1, -1)]
        v24_10 = dic24[(1, 0)]
        v24_11 = dic24[(1, 1)]
        dic30 = minisquare_values(reward_info_out[a][0], (3, 0))
        for key, val in dic30:
            if dic30[(key, val)] == -9.9:
                dic30[(key, val)] = zero_act_val
        v30_n1n1 = dic30[(-1, -1)]
        v30_n10 = dic30[(-1, 0)]
        v30_n11 = dic30[(-1, 1)]
        v30_0n1 = dic30[(0, -1)]
        v30_00 = dic30[(0, 0)]
        v30_01 = dic30[(0, 1)]
        v30_1n1 = dic30[(1, -1)]
        v30_10 = dic30[(1, 0)]
        v30_11 = dic30[(1, 1)]
        dic31 = minisquare_values(reward_info_out[a][0], (3, 1))
        for key, val in dic31:
            if dic31[(key, val)] == -9.9:
                dic31[(key, val)] = zero_act_val
        v31_n1n1 = dic31[(-1, -1)]
        v31_n10 = dic31[(-1, 0)]
        v31_n11 = dic31[(-1, 1)]
        v31_0n1 = dic31[(0, -1)]
        v31_00 = dic31[(0, 0)]
        v31_01 = dic31[(0, 1)]
        v31_1n1 = dic31[(1, -1)]
        v31_10 = dic31[(1, 0)]
        v31_11 = dic31[(1, 1)]
        dic32 = minisquare_values(reward_info_out[a][0], (3, 2))
        for key, val in dic32:
            if dic32[(key, val)] == -9.9:
                dic32[(key, val)] = zero_act_val
        v32_n1n1 = dic32[(-1, -1)]
        v32_n10 = dic32[(-1, 0)]
        v32_n11 = dic32[(-1, 1)]
        v32_0n1 = dic32[(0, -1)]
        v32_00 = dic32[(0, 0)]
        v32_01 = dic32[(0, 1)]
        v32_1n1 = dic32[(1, -1)]
        v32_10 = dic32[(1, 0)]
        v32_11 = dic32[(1, 1)]
        dic33 = minisquare_values(reward_info_out[a][0], (3, 3))
        for key, val in dic33:
            if dic33[(key, val)] == -9.9:
                dic33[(key, val)] = zero_act_val
        v33_n1n1 = dic33[(-1, -1)]
        v33_n10 = dic33[(-1, 0)]
        v33_n11 = dic33[(-1, 1)]
        v33_0n1 = dic33[(0, -1)]
        v33_00 = dic33[(0, 0)]
        v33_01 = dic33[(0, 1)]
        v33_1n1 = dic33[(1, -1)]
        v33_10 = dic33[(1, 0)]
        v33_11 = dic33[(1, 1)]
        dic34 = minisquare_values(reward_info_out[a][0], (3, 4))
        for key, val in dic34:
            if dic34[(key, val)] == -9.9:
                dic34[(key, val)] = zero_act_val
        v34_n1n1 = dic34[(-1, -1)]
        v34_n10 = dic34[(-1, 0)]
        v34_n11 = dic34[(-1, 1)]
        v34_0n1 = dic34[(0, -1)]
        v34_00 = dic34[(0, 0)]
        v34_01 = dic34[(0, 1)]
        v34_1n1 = dic34[(1, -1)]
        v34_10 = dic34[(1, 0)]
        v34_11 = dic34[(1, 1)]
        dic40 = minisquare_values(reward_info_out[a][0], (4, 0))
        for key, val in dic40:
            if dic40[(key, val)] == -9.9:
                dic40[(key, val)] = zero_act_val
        v40_n1n1 = dic40[(-1, -1)]
        v40_n10 = dic40[(-1, 0)]
        v40_n11 = dic40[(-1, 1)]
        v40_0n1 = dic40[(0, -1)]
        v40_00 = dic40[(0, 0)]
        v40_01 = dic40[(0, 1)]
        v40_1n1 = dic40[(1, -1)]
        v40_10 = dic40[(1, 0)]
        v40_11 = dic40[(1, 1)]
        dic41 = minisquare_values(reward_info_out[a][0], (4, 1))
        for key, val in dic41:
            if dic41[(key, val)] == -9.9:
                dic41[(key, val)] = zero_act_val
        v41_n1n1 = dic41[(-1, -1)]
        v41_n10 = dic41[(-1, 0)]
        v41_n11 = dic41[(-1, 1)]
        v41_0n1 = dic41[(0, -1)]
        v41_00 = dic41[(0, 0)]
        v41_01 = dic41[(0, 1)]
        v41_1n1 = dic41[(1, -1)]
        v41_10 = dic41[(1, 0)]
        v41_11 = dic41[(1, 1)]
        dic42 = minisquare_values(reward_info_out[a][0], (4, 2))
        for key, val in dic42:
            if dic42[(key, val)] == -9.9:
                dic42[(key, val)] = zero_act_val
        v42_n1n1 = dic42[(-1, -1)]
        v42_n10 = dic42[(-1, 0)]
        v42_n11 = dic42[(-1, 1)]
        v42_0n1 = dic42[(0, -1)]
        v42_00 = dic42[(0, 0)]
        v42_01 = dic42[(0, 1)]
        v42_1n1 = dic42[(1, -1)]
        v42_10 = dic42[(1, 0)]
        v42_11 = dic42[(1, 1)]
        dic43 = minisquare_values(reward_info_out[a][0], (4, 3))
        for key, val in dic43:
            if dic43[(key, val)] == -9.9:
                dic43[(key, val)] = zero_act_val
        v43_n1n1 = dic43[(-1, -1)]
        v43_n10 = dic43[(-1, 0)]
        v43_n11 = dic43[(-1, 1)]
        v43_0n1 = dic43[(0, -1)]
        v43_00 = dic43[(0, 0)]
        v43_01 = dic43[(0, 1)]
        v43_1n1 = dic43[(1, -1)]
        v43_10 = dic43[(1, 0)]
        v43_11 = dic43[(1, 1)]
        dic44 = minisquare_values(reward_info_out[a][0], (4, 4))
        for key, val in dic44:
            if dic44[(key, val)] == -9.9:
                dic44[(key, val)] = zero_act_val
        v44_n1n1 = dic44[(-1, -1)]
        v44_n10 = dic44[(-1, 0)]
        v44_n11 = dic44[(-1, 1)]
        v44_0n1 = dic44[(0, -1)]
        v44_00 = dic44[(0, 0)]
        v44_01 = dic44[(0, 1)]
        v44_1n1 = dic44[(1, -1)]
        v44_10 = dic44[(1, 0)]
        v44_11 = dic44[(1, 1)]

        df00 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v00_n1n1, v00_n10, v00_n11, v00_0n1, v00_00, v00_01, v00_1n1, v00_10, v00_11]})
        df01 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v01_n1n1, v01_n10, v01_n11, v01_0n1, v01_00, v01_01, v01_1n1, v01_10, v01_11]})
        df02 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v02_n1n1, v02_n10, v02_n11, v02_0n1, v02_00, v02_01, v02_1n1, v02_10, v02_11]})
        df03 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v03_n1n1, v03_n10, v03_n11, v03_0n1, v03_00, v03_01, v03_1n1, v03_10, v03_11]})
        df04 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v04_n1n1, v04_n10, v04_n11, v04_0n1, v04_00, v04_01, v04_1n1, v04_10, v04_11]})
        df10 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v10_n1n1, v10_n10, v10_n11, v10_0n1, v10_00, v10_01, v10_1n1, v10_10, v10_11]})
        df11 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v11_n1n1, v11_n10, v11_n11, v11_0n1, v11_00, v11_01, v11_1n1, v11_10, v11_11]})
        df12 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v12_n1n1, v12_n10, v12_n11, v12_0n1, v12_00, v12_01, v12_1n1, v12_10, v12_11]})
        df13 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v13_n1n1, v13_n10, v13_n11, v13_0n1, v13_00, v13_01, v13_1n1, v13_10, v13_11]})
        df14 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v14_n1n1, v14_n10, v14_n11, v14_0n1, v14_00, v14_01, v14_1n1, v14_10, v14_11]})
        df20 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v20_n1n1, v20_n10, v20_n11, v20_0n1, v20_00, v20_01, v20_1n1, v20_10, v20_11]})
        df21 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v21_n1n1, v21_n10, v21_n11, v21_0n1, v21_00, v21_01, v21_1n1, v21_10, v21_11]})
        df22 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v22_n1n1, v22_n10, v22_n11, v22_0n1, v22_00, v22_01, v22_1n1, v22_10, v22_11]})
        df23 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v23_n1n1, v23_n10, v23_n11, v23_0n1, v23_00, v23_01, v23_1n1, v23_10, v23_11]})
        df24 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v24_n1n1, v24_n10, v24_n11, v24_0n1, v24_00, v24_01, v24_1n1, v24_10, v24_11]})
        df30 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v30_n1n1, v30_n10, v30_n11, v30_0n1, v30_00, v30_01, v30_1n1, v30_10, v30_11]})
        df31 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v31_n1n1, v31_n10, v31_n11, v31_0n1, v31_00, v31_01, v31_1n1, v31_10, v31_11]})
        df32 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v32_n1n1, v32_n10, v32_n11, v32_0n1, v32_00, v32_01, v32_1n1, v32_10, v32_11]})
        df33 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v33_n1n1, v33_n10, v33_n11, v33_0n1, v33_00, v33_01, v33_1n1, v33_10, v33_11]})
        df34 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v34_n1n1, v34_n10, v34_n11, v34_0n1, v34_00, v34_01, v34_1n1, v34_10, v34_11]})
        df40 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v40_n1n1, v40_n10, v40_n11, v40_0n1, v40_00, v40_01, v40_1n1, v40_10, v40_11]})
        df41 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v41_n1n1, v41_n10, v41_n11, v41_0n1, v41_00, v41_01, v41_1n1, v41_10, v41_11]})
        df42 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v42_n1n1, v42_n10, v42_n11, v42_0n1, v42_00, v42_01, v42_1n1, v42_10, v42_11]})
        df43 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v43_n1n1, v43_n10, v43_n11, v43_0n1, v43_00, v43_01, v43_1n1, v43_10, v43_11]})
        df44 = pd.DataFrame(
            {'X-direction': [-1, -1, -1, 0, 0, 0, 1, 1, 1], 'Y-direction': [-1, 0, 1, -1, 0, 1, -1, 0, 1],
             'test': ['value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:', 'value:'],
             'value': [v44_n1n1, v44_n10, v44_n11, v44_0n1, v44_00, v44_01, v44_1n1, v44_10, v44_11]})

        pivot00 = df00.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot01 = df01.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot02 = df02.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot03 = df03.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot04 = df04.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot10 = df10.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot11 = df11.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot12 = df12.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot13 = df13.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot14 = df14.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot20 = df20.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot21 = df21.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot22 = df22.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot23 = df23.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot24 = df24.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot30 = df30.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot31 = df31.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot32 = df32.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot33 = df33.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot34 = df34.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot40 = df40.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot41 = df41.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot42 = df42.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot43 = df43.pivot(index='X-direction', columns='Y-direction', values='value')
        pivot44 = df44.pivot(index='X-direction', columns='Y-direction', values='value')

        directory = 'static'
        parent_dir = '/Users/philcrawford/PycharmProjects/RLFlask/'
        path = os.path.join(parent_dir, directory)
        if not path:
            os.mkdir(path)

        fig, axn = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(10, 10))

        for ax in axn.flat:
            ax.set_axis_off()
            im = ax.imshow(np.random.random((16, 16)), cmap='vlag',
                           vmin=min_val, vmax=max_val)
        cb_ax = fig.add_axes([.91, .3, .03, .4])
        cb_ax = fig.colorbar(im, cax=cb_ax)
        # cbar.set_ticks([])

        ax = plt.subplot(5, 5, 1)
        sns.heatmap(pivot00, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 2)
        sns.heatmap(pivot01, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 3)
        sns.heatmap(pivot02, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 4)
        sns.heatmap(pivot03, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 5)
        sns.heatmap(pivot04, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 6)
        sns.heatmap(pivot10, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 7)
        sns.heatmap(pivot11, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 8)
        sns.heatmap(pivot12, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 9)
        sns.heatmap(pivot13, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 10)
        sns.heatmap(pivot14, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 11)
        sns.heatmap(pivot20, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 12)
        sns.heatmap(pivot21, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 13)
        sns.heatmap(pivot22, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 14)
        sns.heatmap(pivot23, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 15)
        sns.heatmap(pivot24, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 16)
        sns.heatmap(pivot30, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 17)
        sns.heatmap(pivot31, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 18)
        sns.heatmap(pivot32, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 19)
        sns.heatmap(pivot33, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 20)
        sns.heatmap(pivot34, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 21)
        sns.heatmap(pivot40, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 22)
        sns.heatmap(pivot41, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 23)
        sns.heatmap(pivot42, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 24)
        sns.heatmap(pivot43, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        ax = plt.subplot(5, 5, 25)
        sns.heatmap(pivot44, annot=True, fmt="g", cmap='vlag', vmin=min_val, vmax=max_val, cbar=False, ax=ax)
        ax.set_aspect('equal')

        fig.tight_layout(rect=[0, 0, .9, 1])
        fig.savefig(path + '/' + reward_info_out[a][2] + label + '.jpg')
        plt.close()
    return()

# run_rl('ABCDAB' * 8, .05, .2, -1)
# model_in = load('model.joblib')
# # visualize_tables(2, model_in, 'SimpleX')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)







