import math
import random
import string
import numpy as np
import sys
import itertools

#nCr utility function
from scipy.special import comb

#Hey Yi Ran! This is the code for the basic program I've made.
#Images to be tested are placed in the ./doggos folder and are in jpg format and labelled numerically zero-indexed.
#I'm using PyQT for the GUI and I'm

#This function takes in a question written in Reverse Polish Notation and returns you the breeds for which you get a True/False answer!
#A question could be written as [0,1,"NOT","AND"] which is 0 AND (NOT 1) in infix notation.
#I hope this function is helpful to you if you're trying to do mass-tests!
#memoizer for the recursive functions.

#Utility functions
def sigma(lower_bound,upper_bound,function):
	sum=0
	while(lower_bound<upper_bound+1):
		sum+=function(lower_bound)
		lower_bound+=1
	return sum

#Shifts the list times- times to the right and inserts a zero in front.
def shift(list,times):
	return [0]*times + list[0:len(list)-times]

#Crude minimisation function that takes in a function and input list and two constraints.
""" Ignore this.
#Returns the number that yields the minimised objective function value and the objective function value.
def crappy_min(objective,input_range,constraint1,constraint2):
	answer_list=[]
	for i in input_range:
		if (constraint1(i) and constraint2(i) == True):
			answer_list.append([i,objective(i)])
	answer_list.sort(key=lambda answer: answer[1])
	return answer_list[0]
"""

def check_answers(trait_breed,question):
	stack = []
	for i in range(len(question)):
		if (question[i] == "NOT"):
			stack.append(np.invert(stack.pop()))
		elif (question[i] == "AND"):
			stack.append(np.logical_and(stack.pop(),stack.pop()))
		elif (question[i] == "OR"):
			stack.append(np.logical_or(stack.pop(),stack.pop()))
		else:
			stack.append(trait_breed[:,i])
	return stack

#This generates a random boolean breed-trait matrix for mass-testing!
def generate_breed_trait_matrix(trait_no,breed_no):
	trait_breed=[]
	for _ in range(breed_no):
		successful = False
		while (successful == False):
			rand_breed = (np.random.randint(2, size=trait_no, dtype=bool)).tolist()
			if (rand_breed not in trait_breed):
				successful = True
				trait_breed.append(rand_breed)
	trait_breed = np.asarray(trait_breed)

#Ulam-Renyi Class
class UlamRenyi(object):
	def set_sigma(self,list):
		return [len(i) for i in list]

	def __init__(self, breed_set,e):
		self.breed_set=breed_set
		self.e=e #Number of errors allowed.
		self.game_state=[self.breed_set]+[[]]*e #Initialising the gamestate as a list of lists.
		self.sigma_game_state=self.set_sigma(self.game_state) #This is the sigma(state).

	def weight(self,sigma_state,q): #Note, the state here the sigma state.
		return np.sum([sigma_state[i]*np.sum([comb(q,j,exact=True) for j in range(0,self.e-i+1)]) for i in range(0,self.e+1)])

	def character(self,sigma_state):
		return next(j for j in itertools.count(0,1) if self.weight(sigma_state,j)<=2**j)

	def recursive_f(self,sigma_state):
		if (np.sum(sigma_state)<=2):
		   return self.character(sigma_state)
		else:
			return max(self.recursive_f(shift(sigma_state,1))+3,self.character(sigma_state))


	def gamma(self,sigma_state):
		return [max(self.recursive_f(shift(sigma_state,self.e - i)), self.recursive_f(sigma_state) - 3*(self.e-i)) for i in range(0,self.e+1)]

	def sigma_game_state_yes(self,sigma_state,t):
		#I'm interpreting Qi Yu's i-1 index where i=0 to be non-existent i.e. 0.
		return [t[0]] + [sigma_state[i-1] - t[i-1] + t[i] for i in range(1,self.e+1)]

	def sigma_game_state_no(self,sigma_state,t):
		#I'm interpreting Qi Yu's i-1 index where i=0 to be non-existent i.e. 0.
		return [sigma_state[0]-t[0]] + [t[i-1] + sigma_state[i] - t[i] for i in range(1,self.e+1)]

	def condition_fulfilled(self, i, ques_i, gamma_state):
		result_list = []
		state_i_sigma=shift(self.sigma_game_state,self.e-i)
		for j in range(1, self.e - i + 1):
			s_g_y_i=self.sigma_game_state_yes(state_i_sigma,ques_i)
			s_g_n_i=self.sigma_game_state_no(state_i_sigma,ques_i)
			alpha = [0]*(self.e - i - j) + s_g_y_i[self.e-i:] + [state_i_sigma[-1] - ques_i[-1]] + [0] * (j - 1)
			beta = [0] * (self.e - i - j) + s_g_n_i[self.e-i:] + [ques_i[-1]] + [0] * (j - 1)
			result_list.append(self.weight(alpha, gamma_state[i+j] - 1) and self.weight(beta, gamma_state[i+j] - 1) <= 2 ** (gamma_state[i+j] - 1))
		return all(result_list)

	def run_algorithm(self):
		gamma_state=self.gamma(self.sigma_game_state)
		question=[]
		if np.sum(self.sigma_game_state)==1:
			print(["Check.",self.sigma_game_state])
		else:
			for i in range(0,self.e+1):
				potential_solutions=[]
				state_i_sigma=shift(self.sigma_game_state,self.e-i)
				for x in range(0,self.sigma_game_state[i]+1):
					ques_i=[0]*(self.e-i) + question + [x]
					s_state_i_yes=self.sigma_game_state_yes(state_i_sigma,ques_i)
					s_state_i_no=self.sigma_game_state_no(state_i_sigma,ques_i)
					fitness=abs(self.weight(s_state_i_yes,gamma_state[i]-1)-self.weight(s_state_i_no,gamma_state[i]-1))
					potential_solutions.append([x,ques_i,fitness])
					potential_solutions.sort(key=lambda x: x[2])
				question.append(next(x[0] for x in potential_solutions if self.condition_fulfilled(i,x[1],gamma_state)))
		return question

	def process_yes(self, question_set):
		self.game_state[0] = list(set(self.game_state[0]) & set(question_set))
		for i in range(1, e + 1):
			self.game_state[i] = list((set(self.game_state[i - 1]) - set(question_set)) + (set(self.game_state[i]) & set(question_set)))
		self.sigma_game_state=self.self.set_sigma(self.game_state)

	def process_no(self, question_set):
		self.game_state[0] = list(set(self.game_state[0]) - set(question_set))
		for i in range(1, e + 1):
			self.game_state[i] = list((set(self.game_state[i - 1]) & set(question_set)) + (set(self.game_state[i]) - set(question_set)))
		self.sigma_game_state = self.self.set_sigma(self.game_state)

	#Takes in the question and the trait_info list, a list of arrays of the trait number, the corresponding trait and the grammar classes it falls under.
	#For instance, [2, long legs, plural] or [5, a nice personality, singular]
	#Quite basic, feel free to edit if you want to!
	#Right now, all it does is just return the question typed in as a string. I don't know how it should be converted haha.
	def rpn_to_english(trait_info, question):
		string=""
		for i in question:
			string+=str(i)+" "
		return string

	def generate_question(self): #Yi-ran, could you help me with this :?
		return None
""" Ignore this.
	def alpha(self,i,j,t): #Alpha function, as defined by Qi Yu.
		return ([0]*(self.e-i-j) +
				self.sigma_game_state_yes(t)[:i+1] +
				[self.sigma_game_state[i]-t[i]] +
				[0]*(j-1))

	def beta(self,i,j,t): #Beta function, as defined by Qi Yu.
		return ([0]*(self.e-i-j) +
				self.sigma_game_state_no(t)[:i+1] +
				[t[i]] +
				[0]*(j-1))
"""

""" Ignore this.
	def question(state_sigma):
		num_error=len(state_sigma)-1
		question=np.zeros(1,num_error+1)
		if np.sum(state_sigma)== (1 or 0)
			print("Final state, check the answer. Yeet -Qi Yu our dank master PhD student")
		else
			for i in range(0,num_error)
				diff=np.zeros(1,state_sigma[i]+1)
				state_i_sigma = np.concatenate(np.zeros(1, num_error-i),state_sigma[0:i])
				for j in range(0,state_sigma[i])
					ques_i = np.concatenate(np.zeros(1, num_error-i), question[1:max(0, i)], j)
					[state_i_yes, state_i_no] = StateAfterQues(state_i_sigma, ques_i);
					diff(j + 1) = abs(Weight(state_i_yes, gamma(i + 1) - 1) - Weight(state_i_no, gamma(i + 1) - 1));
"""

trait_no=5
breed_no=16
bt_matrix=generate_breed_trait_matrix(trait_no,breed_no)
ug_game=UlamRenyi(list(range(0,breed_no)),9)
ug_game.sigma_game_state=[6,7,6,0,100,0,90,0,0,0]
print(ug_game.run_algorithm())







