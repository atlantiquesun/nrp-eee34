import numpy as np
from scipy.special import comb
from itertools import count

def shift(list, times):
    return [0]*times + list[0:len(list)-times]

def set_sigma(list):
    return [len(i) for i in list]

def weight(sigma_state, q):  # Note, the state here is the sigma state.
    errors=len(sigma_state)-1
    return np.sum([sigma_state[i] * np.sum([comb(q, j, exact=True) for j in range(0, errors - i + 1)]) for i in
                    range(0, errors + 1)])

def character(sigma_state):
    return next(j for j in count(0, 1) if weight(sigma_state, j) <= 2 ** j)

def recursive_f(sigma_state):
    if (np.sum(sigma_state) <= 2):
        return character(sigma_state)
    else:
        return max(recursive_f(shift(sigma_state, 1)) + 3, character(sigma_state))

def gamma(sigma_state):
    errors=len(sigma_state)-1
    return [max(recursive_f(shift(sigma_state, errors - i)), recursive_f(sigma_state) - 3 * (errors - i))for i in range(0, errors + 1)]

def sigma_state_yes(sigma_state, t):
    # I'm interpreting Qi Yu's i-1 index where i=0 to be non-existent i.e. 0.
    errors=len(sigma_state)-1
    return [t[0]] + [sigma_state[i - 1] - t[i - 1] + t[i] for i in range(1, errors + 1)]

def sigma_state_no(sigma_state, t):
    # I'm interpreting Qi Yu's i-1 index where i=0 to be non-existent i.e. 0.
    errors=len(sigma_state)-1
    return [sigma_state[0] - t[0]] + [t[i - 1] + sigma_state[i] - t[i] for i in range(1, errors + 1)]

def condition_fulfilled(errors, i, ques_i, sigma_state, gamma_state):
    result_list = []
    state_i_sigma = shift(sigma_state, errors - i)
    for j in range(1, errors - i + 1):
        s_g_y_i = sigma_state_yes(state_i_sigma, ques_i)
        s_g_n_i = sigma_state_no(state_i_sigma, ques_i)
        alpha = [0] * (errors - i - j) + s_g_y_i[errors - i:] + [state_i_sigma[-1] - ques_i[-1]] + [0] * (j - 1)
        beta = [0] * (errors - i - j) + s_g_n_i[errors - i:] + [ques_i[-1]] + [0] * (j - 1)
        result_list.append(
            weight(alpha, gamma_state[i + j] - 1) and weight(beta, gamma_state[i + j] - 1) <= 2 ** (
                        gamma_state[i + j] - 1))
    return all(result_list)

def run_algorithm(sigma_state):
    errors=len(sigma_state)-1
    gamma_state = gamma(sigma_state)
    question = []
    if np.sum(sigma_state) == 1:
        print(["Check.", sigma_state])
    else:
        for i in range(0, errors + 1):
            potential_solutions = []
            state_i_sigma = shift(sigma_state, errors - i)
            for x in range(0, sigma_state[i] + 1):
                ques_i = [0] * (errors - i) + question + [x]
                s_state_i_yes = sigma_state_yes(state_i_sigma, ques_i)
                s_state_i_no = sigma_state_no(state_i_sigma, ques_i)
                fitness = abs(
                    weight(s_state_i_yes, gamma_state[i] - 1) - weight(s_state_i_no, gamma_state[i] - 1))
                potential_solutions.append([x, ques_i, fitness])
                potential_solutions.sort(key=lambda x: x[2])
            question.append(
                next(x[0] for x in potential_solutions if condition_fulfilled(errors, i, x[1], sigma_state, gamma_state)))
    return question

def generate_question(question_set_constraint):  # Yi-ran, could you help me with this :? #It should generate a question in the [1,2,3,"NOT","AND"] format
    return [None]

def nlp_generate_string(question_set):
    return "Test!" #This too!!

def process_yes(game_state, question_set):
    game_state[0] = list(set(game_state[0]) & set(question_set))
    for i in range(1, e + 1):
        game_state[i] = list(
            (set(game_state[i - 1]) - set(question_set)) + (set(game_state[i]) & set(question_set)))

def process_no(game_state, question_set):
    game_state[0] = list(set(game_state[0]) - set(question_set))
    for i in range(1, e + 1):
        game_state[i] = list(
            (set(game_state[i - 1]) & set(question_set)) + (set(game_state[i]) - set(question_set)))
