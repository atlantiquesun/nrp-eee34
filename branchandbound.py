import math
import random
import string
import numpy as np
import sys
import itertools

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
    return trait_breed

breed_no=6
trait_no=8

tb_matrix=generate_breed_trait_matrix(breed_no,trait_no)

def greedy(set1,set2,tb_matrix):
    answer=[]
    answer_found=False
    while answer_found==False:
        answer=1
    return answer

print(list(itertools.combinations([[1],[2,"NOT"]],1)))

