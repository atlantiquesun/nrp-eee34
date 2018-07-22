import math
import numpy as np
import random
import string
import matplotlib.pyplot as mpl

class SequentialClassifier:
    def __init__(self, trait_no, breed_no, trait_breed, breed_set):
        self.trait_no=trait_no
        self.breed_no=breed_no
        self.trait_breed=trait_breed
        self.breed_set=breed_set

    class Question:
        def __init__(self, question_set):
            self.parent_question=Null
            self.question_set=question_set
            self.question=find_question()
            self.answer_set = find_answer_set()
            self.fpve_set = find_false_positive()
            self.fnve_set = find_false_negative()

        def find_answer_set(self,trait1,trait2,logic):
            
        def find_false_positive(self,trait1,trait2,logic):

        def find_false_negative(self,trait1,trait2,logic):
            
        def fitness(self,trait1,trait2,logic):
            #for logic 0 is or 1 is and
            if(logic==0):
            if(logic==1):

        def find_question(self):
            best_question_fitness=-1
            best_question=[]
            for i in range(trait_no):
                for j in range(trait_no):
                    if()
                    if()

test_classifier=SequentialClassifier([[1,4,7],[2,5,8],[3,6,9]])





    

            
        
        
