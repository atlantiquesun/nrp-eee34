import math
import random
import string
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

#Hey Yi Ran! This is the code for the basic program I've made.
#Images to be tested are placed in the ./doggos folder and are in jpg format and labelled numerically zero-indexed.
#I'm using PyQT for the GUI and I'm

#This function takes in a question written in Reverse Polish Notation and returns you the breeds for which you get a True/False answer!
#A question could be written as [0,1,"NOT","AND"] which is 0 AND (NOT 1) in infix notation.
#I hope this function is helpful to you if you're trying to do mass-tests!

#Utility functions

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
        self.game_state=[]
        self.game_state_yes=[]
        self.game_state_no=[]

    def set_cardinality(self, list):
        return [len(i) for i in list]

    def process_yes(game_state):

    def process_no(game_state):

    def berkelamp_weight

    def character:

    def recursive_f:

    def gamma:

    def process


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
    #Crude minimisation function that takes in a function and input list and two constraints.
    #Returns the number that yields the minimised objective function value.
    def crappy_min(objective,input_range,constraint1,constraint2):
        answer_list=[]
        for i in input_range:
            if (constraint1(i) and constraint2(i) == True):
                answer_list.append([i,objective(i)])
        answer_list.sort(key=lambda answer: answer[1])
        return answer_list[0][0]

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









