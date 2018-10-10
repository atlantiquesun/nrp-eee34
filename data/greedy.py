import math
import random
import string
import numpy as np

def check(question):
    stack = []
    for i in range(len(question)):
        if (question[i] == "AND"):
            stack.append(np.logical_and(stack.pop(),stack.pop()))
        elif (question[i] == "OR"):
            stack.append(np.logical_or(stack.pop(),stack.pop()))
        else:
            stack.append(trait_breed[:,i])
    return stack

trait_no = 6
breed_no = 10
trait_breed = []
for _ in range(breed_no):
    successful = False
    while (successful == False):
        rand_breed = np.random.randint(2, size=trait_no, dtype=bool)
        breed=(np.append(rand_breed,np.invert(rand_breed))).tolist()
        if (breed not in trait_breed):
            successful = True
            trait_breed.append(breed)
trait_breed=np.asarray(trait_breed)
print(trait_breed)
print("Done!")
print(check([0,1,"AND",2,3,"OR","AND"]))










