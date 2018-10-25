import math
import random
import string
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

#nCr utility function
from scipy.special import comb

#Hey Yi Ran! This is the code for the basic program I've made.
#Images to be tested are placed in the ./doggos folder and are in jpg format and labelled numerically zero-indexed.
#I'm using PyQT for the GUI and I'm

#This function takes in a question written in Reverse Polish Notation and returns you the breeds for which you get a True/False answer!
#A question could be written as [0,1,"NOT","AND"] which is 0 AND (NOT 1) in infix notation.
#I hope this function is helpful to you if you're trying to do mass-tests!

#Utility functions
def sigma(lower_bound,upper_bound,function):
    sum=0
    while(lower_bound<upper_bound+1):
        sum+=function(lower_bound)
        lower_bound+=1
    return sum

def set_cardinality(list):
    return [len(i) for i in list]

#Shifts the list times- times to the right and inserts a zero in front.
def shift(list,times):
    if times==0:
        return list
    else:
        return [0]+shift(list,times-1)[0:-1]

#Crude minimisation function that takes in a function and input list and two constraints.
#Returns the number that yields the minimised objective function value and the objective function value.
def crappy_min(objective,input_range,constraint1,constraint2):
    answer_list=[]
    for i in input_range:
        if (constraint1(i) and constraint2(i) == True):
            answer_list.append([i,objective(i)])
    answer_list.sort(key=lambda answer: answer[1])
    return answer_list[0]

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

#Takes in the question and the trait_info list, a list of arrays of the trait number, the corresponding trait and the grammar classes it falls under.
#For instance, [2, long legs, plural] or [5, a nice personality, singular]
#Quite basic, feel free to edit if you want to!
#Right now, all it does is just return the question typed in as a string. I don't know how it should be converted haha.
def rpn_to_english(trait_info, question):
    string=""
    for i in question:
        string+=str(i)+" "
    return string

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

tb_matrix=generate_breed_trait_matrix(2,4)

#Ulam-Renyi Class
class UlamRenyi(object):
    def __init__(self, breed_set,e):
        self.breed_set=breed_set
        self.e=e
        self.game_state=[self.breed_set]+[[]]*e
        self.cardinal_game_state=set_cardinality(game_state)

    def cardinal_game_state_yes(self,t):
        #I'm interpreting Qi Yu's i-1 index where i=0 to be non-existent i.e. 0.
        return [t[0]]+[self.cardinal_game_state[i]-t[i-1]+t[i] for i in range(1,e+1)]

    def alpha(self,i,j,t):
        return [0]*(j-1) + self.cardinal_game_state_yes(t)[0:i+1] + self.cardinal_game_state[i]-t[i] + [0]*(j-1)

    def beta(self,i,j,t):
        return [0]*(j-1) + self.cardinal_game_state_no(t)[0:i+1] + self.cardinal_game_state[i]-t[i] + [0]*(j-1)

    def cardinal_game_state_no(self,t):
        return [self.cardinal_game_state[0]-t[0]]+[t[i-1]+self.cardinal_game_state[i]-t[i] for i in range(1,e+1)]

    def berkelamp_weight(self,state,q):
        cardinal_state=self.set_cardinality(state)
        return np.sum(self.cardinal_game_state[i]*(np.sum(comb(q,j,exact=True) for j in range(0,e-i+1))) for i in range(0,e+1))

    def character(self,state):
        j=0
        while (self.berkelamp_weight(state,j)>2**j)):
            j+=1
        return j

    def recursive_f(self,cardinal_state):
        if (np.sum(cardinal_state)<=2):
           return self.character(cardinal_state)
        else:
            return max(self.recursive_f(shift(cardinal_state,1))+3,self.character(cardinal_state))

    def gamma(self,cardinal_state):
        return [max(self.recursive_f(shift(cardinal_state, self.e - i)), self.recursive_f(cardinal_state) - 3 * (self.e - i)) for i in range(0, e + 1)]

    def run_algorithm(self):
        t=[0]*(self.e+1)
        if np.sum(self.cardinal_game_state)==1:
            print(self.game_state)
        else:
            gamma_state=gamma(self.game_state)
            for i in range(0,e+1):
                lowest=-1
                if self.berkelamp_weight(alpha(i,j,t),gamma[i+j]-1) <= 2**(gamma[i+j]-1) and self.berkelamp_weight(alpha(i,j,t),gamma[i+j]-1) <= 2**(gamma[i+j]-1):


    def process_yes(self,question_set):
        self.game_state[0]=list(set(self.game_state[0])&set(question_set))
        for i in range(1,e+1):
            self.game_state[i]=list((set(self.game_state[i-1])-set(question_set))+(set(self.game_state[i])&set(question_set)))

    def process_no(self,question_set):
        self.game_state[0]=list(set(self.game_state[0])-set(question_set))
        for i in range(1,e+1):
            self.game_state[i]=list((set(self.game_state[i-1])&set(question_set))+(set(self.game_state[i])-set(question_set)))

"""
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
    def compute()

class ResponseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 button - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 800
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        button = QPushButton('PyQt5 button', self)
        button.setToolTip('This is an example button')
        button.move(100, 70)
        button.clicked.connect(self.on_click)

        self.show()

    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ResponseApp()
    while(True):
        print("test")
    sys.exit(app.exec_())









