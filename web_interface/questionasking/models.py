import numpy as np
from scipy.special import comb
from itertools import count
from django.db import models
from ast import literal_eval

class Trait(models.Model):
    trait_name=models.CharField(max_length=50)
    def __str__(self):
        return self.trait_name

class Breed(models.Model):
    breed_name=models.TextField(null=True)
    traits=models.ManyToManyField(Trait)
    def __str__(self):
        return self.breed_name

class Image(models.Model):
    # Fields for the database.
    user_working_on_task = models.CharField(max_length=20, blank=True)
    number_of_times_served = models.IntegerField(default=0) #Not being used for anything rn, but i just left it in in case we wanna tweak stuff.
    errors = models.IntegerField(default=None, null=True) #Right now I'm assuming that they will only mess up a maximum of half the time. This corresponds to e in the paper.
    image = models.ImageField(default=None, null=True)  # URL linking to the image
    game_state = models.TextField(default=None, blank=True)  # Current state of the Ulam-Renyi game being played.
    sigma_game_state = models.TextField(default=None, blank=True)  # Current sigma state, included since it would be a waste of time to re-calculate sigma states over and over again.
    question_to_ask = models.TextField(default=None, blank=True) #What question to ask given the set T as dictated in the paper.
    breed = models.ForeignKey(Breed, default=None, on_delete=models.CASCADE, null=True) #Ultimately the breed that is decided.
    def __str__(self):
        return self.image.url
    
    @classmethod
    def create(cls, breed_no=Breed.objects.count(), trait_no=Trait.objects.count() , errors=Trait.objects.count()/2):
        return cls(errors=errors, 
                game_state=list(range(0,breed_no))+[]*errors,
                sigma_game_state=[breed_no]+[]*errors,
                question_to_ask=run_algorithm(),
                )

    def shift(self,list,times):
	    return [0]*times + list[0:len(list)-times]

    def set_sigma(self, list):
        return [len(i) for i in list]

    def weight(self, sigma_state, q):  # Note, the state here is the sigma state.
        return np.sum([sigma_state[i] * np.sum([comb(q, j, exact=True) for j in range(0, self.errors - i + 1)]) for i in
                       range(0, self.errors + 1)])

    def character(self, sigma_state):
        return next(j for j in count(0, 1) if self.weight(sigma_state, j) <= 2 ** j)

    def recursive_f(self, sigma_state):
        if (np.sum(sigma_state) <= 2):
            return self.character(sigma_state)
        else:
            return max(self.recursive_f(self.shift(sigma_state, 1)) + 3, self.character(sigma_state))

    def gamma(self, sigma_state):
        return [max(self.recursive_f(self.shift(sigma_state, self.errors - i)), self.recursive_f(sigma_state) - 3 * (self.errors - i))for i in range(0, self.errors + 1)]

    def sigma_game_state_yes(self, sigma_state, t):
        # I'm interpreting Qi Yu's i-1 index where i=0 to be non-existent i.e. 0.
        return [t[0]] + [sigma_state[i - 1] - t[i - 1] + t[i] for i in range(1, self.errors + 1)]

    def sigma_game_state_no(self, sigma_state, t):
        # I'm interpreting Qi Yu's i-1 index where i=0 to be non-existent i.e. 0.
        return [sigma_state[0] - t[0]] + [t[i - 1] + sigma_state[i] - t[i] for i in range(1, self.errors + 1)]

    def condition_fulfilled(self, i, ques_i, gamma_state):
        result_list = []
        state_i_sigma = self.shift(self.sigma_game_state, self.errors - i)
        for j in range(1, self.errors - i + 1):
            s_g_y_i = self.sigma_game_state_yes(state_i_sigma, ques_i)
            s_g_n_i = self.sigma_game_state_no(state_i_sigma, ques_i)
            alpha = [0] * (self.errors - i - j) + s_g_y_i[self.errors - i:] + [state_i_sigma[-1] - ques_i[-1]] + [0] * (j - 1)
            beta = [0] * (self.errors - i - j) + s_g_n_i[self.errors - i:] + [ques_i[-1]] + [0] * (j - 1)
            result_list.append(
                self.weight(alpha, gamma_state[i + j] - 1) and self.weight(beta, gamma_state[i + j] - 1) <= 2 ** (
                            gamma_state[i + j] - 1))
        return all(result_list)

    def run_algorithm(self):
        gamma_state = self.gamma(self.sigma_game_state)
        question = []
        if np.sum(self.sigma_game_state) == 1:
            print(["Check.", self.sigma_game_state])
        else:
            for i in range(0, self.errors + 1):
                potential_solutions = []
                state_i_sigma = self.shift(self.sigma_game_state, self.errors - i)
                for x in range(0, self.sigma_game_state[i] + 1):
                    ques_i = [0] * (self.errors - i) + question + [x]
                    s_state_i_yes = self.sigma_game_state_yes(state_i_sigma, ques_i)
                    s_state_i_no = self.sigma_game_state_no(state_i_sigma, ques_i)
                    fitness = abs(
                        self.weight(s_state_i_yes, gamma_state[i] - 1) - self.weight(s_state_i_no, gamma_state[i] - 1))
                    potential_solutions.append([x, ques_i, fitness])
                    potential_solutions.sort(key=lambda x: x[2])
                question.append(
                    next(x[0] for x in potential_solutions if self.condition_fulfilled(i, x[1], gamma_state)))
        return question

    def generate_question(self):  # Yi-ran, could you help me with this :? #It should generate a question in the [1,2,3,"NOT","AND"] format
        return "Idk!"
    
    def nlp_generate_string(self):
        return "Test!" #This too!!
"""
    def process_yes(self, question_set):
        self.game_state[0] = list(set(self.game_state[0]) & set(question_set))
        for i in range(1, e + 1):
            self.game_state[i] = list(
                (set(self.game_state[i - 1]) - set(question_set)) + (set(self.game_state[i]) & set(question_set)))
        self.sigma_game_state = self.set_sigma(self.game_state)

    def process_no(self, question_set):
        self.game_state[0] = list(set(self.game_state[0]) - set(question_set))
        for i in range(1, e + 1):
            self.game_state[i] = list(
                (set(self.game_state[i - 1]) & set(question_set)) + (set(self.game_state[i]) - set(question_set)))
        self.sigma_game_state = self.set_sigma(self.game_state)
"""

