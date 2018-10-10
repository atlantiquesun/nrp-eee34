import math
import random
import string
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel

#Hey Yi Ran! This is the code for the basic program I've made.
#Images to be tested are placed in the ./doggos folder and are in jpg format and labelled numerically zero-indexed.
#I'm using PyQT for the GUI and I'm

#This function takes in a question written in Reverse Polish Notation and returns you the breeds for which you get a True/False answer!
#A question could be written as [0,1,"NOT","AND"] which is 0 AND (NOT 1) in infix notation.
#I hope this function is helpful to you if you're trying to do mass-tests!
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
#I've set it up to be able to handle one or two-word questions
#trait_info is a
def rpn_to_english(trait_info,question):
    print("This dog " + description + ".")
    description=""
    for i in range(len(question)):
        str1 = ""
        str2 = ""
        if (question[i] == "NOT"):
            stack.append(np.invert(stack.pop()))
        elif (question[i] == "AND"):
            stack.append(np.logical_and(stack.pop(),stack.pop()))
        elif (question[i] == "OR"):
            stack.append(np.logical_or(stack.pop(),stack.pop()))
        else:
            if str1==""
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

tb_matrix=generate_breed_trait_matrix(2,4)

app = QApplication([])
label = QLabel('Hello World!')
label.show()
app.exec_()










