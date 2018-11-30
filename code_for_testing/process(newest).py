import math
from random import *
import string
import numpy as np
import sys
import itertools
from copy import deepcopy
import time

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




#helper functions for question generation function in class 'UlamRenyi'
def Distance(vector1,vector2):
        squaretotal = 0;
        for i in range(len(vector1)):
            squaretotal +=(vector2[i]-vector1[i])**2
        return round(squaretotal**(.5),3)

def sortkey(l):
    return l[2]

def sortkey2(l):
    return l[-1]

def veclen(l):
    new=[]
    for i in l:
        new.append(len(i))
    return new

def merge(l1,l2):
    result1=[]
    for i in l1:
        result1.extend(i)
    result=[list(filter(lambda x:x in result1,sublist))for sublist in l2]
    return result

def union(l1,l2):
    result=deepcopy(l2)
    for j in range(len(l2)):
        for i in l1[j]:
            if i not in l2[j]:
                result[j].append(i)
    return result
            
def mergeintlist(l1,l2):
    return [list(filter(lambda x:x in l1,l2))]
       
def fitness(l1,ideal,denomdis):
    if Distance(l1,ideal)==0:
        return -999 
    else:
        return Distance(l1,ideal)/(denomdis) #the smaller the better

def coverage(l1,ClassesNum):
    return np.sum(np.asarray(l1))/ClassesNum

def hammingdistance(l1,l2):
    l=np.asarray(l1)-np.asarray(l2)
    l=np.abs(l)
    dis = np.sum(l)
    return dis

def mergesol(l1,l2,uoi):
    '''
    l1: [[6,0]]
    l2: [[3,0]]
    uoi:'union' (or 'intersection')
    return: [[6,0],'+',[3,0]]
    '''
    l1=['(']+l1
    l2=l2+[')']
    if uoi=='union':
        l1.append('+')
    else:
        l1.append('-')
    l1.extend(l2)
    return l1

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

    def prepare_arguments(self,constraint,ClassesNum,TraitMatrix):
        '''
        prepare arguments for 'generate_question'
        assuming breeds are 1-indexed, breed 1,2,3,...
        states: 0,...,e (0-indexed)
        '''
        from copy import deepcopy
        l=list(np.zeros((self.e+1)))
        zerostates=deepcopy(l)
        for j in range(self.e+1):
            l[j]+= len(self.game_state[j])
        statecapacity=l
        denomdis=(Distance(statecapacity,constraint)+Distance(zerostates,constraint))/2
        classdc={}
        for i in range(self.e+1): 
            for j in self.game_state[i]:
                classdc[j]=i
        return(constraint,classdc,ClassesNum,TraitMatrix,denomdis)

    def generate_question(self,constraint,ClassesNum,TraitMatrix): 
        (constraint,classdc,ClassesNum,TraitMatrix,denomdis)=self.prepare_arguments(constraint,ClassesNum,TraitMatrix)
        print(classdc)
        generations=list(range(0,len(TraitMatrix[0]))) 
        profiles=[]
        for i in range(0,4): #four generation maximum
           #print("round:",i)
            if i==0:
                for k in generations:
                    temp1=[[]]
                    temp2=[[]]
                    for j in range(len(constraint)):
                        temp1[0].append([])
                        temp2[0].append([])
                    for q in range(ClassesNum):
                        if(TraitMatrix[q][k]==0):
                            temp1[0][classdc[q+1]].append(q+1)
                        else:
                            temp2[0][classdc[q+1]].append(q+1)
                    temp1.extend([veclen(temp1[0]),fitness(veclen(temp1[0]),constraint,denomdis),[[k+1,0]],coverage(veclen(temp1[0]),ClassesNum)])
                    temp2.extend([veclen(temp2[0]),fitness(veclen(temp2[0]),constraint,denomdis),[[k+1,1]],coverage(veclen(temp2[0]),ClassesNum)])
                    profiles.append(temp1)
                    profiles.append(temp2)
                profiles = sorted(profiles,key=sortkey)
                if profiles[0][2]<0.20: #a threshold, can be changed 
                    return (profiles[0][-2],profiles[0][0],profiles[0][1])#the solution
                    profiles=sorted(profiles,key=sortkey2)
                    del profiles[35:]
            else:
                print("more than one round")
                candidates=[]
                for t in profiles:
                    candidates.append([t[0],t[-2]]) #only the 'specific breeds' and 'dimension information'
                for k in candidates:
                    for q in candidates:
                        if(q!=k): 
                            tp = merge(k[0],q[0])
                            new=deepcopy(k[1])
                            #for y in q[1]: #add these two lines in when the algorithm supports more traits, don't forget the indentation
                               #if y not in new:
                                   #new.append(y)
                            new=mergesol(new,q[1],'intersection') 
                            profiles.append([tp,veclen(tp),fitness(veclen(tp),constraint,denomdis),new,coverage(veclen(tp),ClassesNum)])#jjjjjjjj
                #profiles=profiles[10:] delete so that we can take previous solutions into consideration
                for k in candidates:
                    for q in candidates:
                        if(q!=k): 
                            tp = union(k[0],q[0])
                            new=deepcopy(k[1])
                            #for y in q[1]: #add these two lines in when the algorithm supports more traits
                               #if y not in new:
                                   #new.append(y)
                            new=mergesol(new,q[1],'union')
                            profiles.append([tp,veclen(tp),fitness(veclen(tp),constraint,denomdis),new,coverage(veclen(tp),ClassesNum)])#jjjjjjjj
                profiles=sorted(profiles,key=sortkey)            
                #if profiles[0][2]<0.6-0.05*(4-i): #increasing threshold
                return (profiles[0][-2],profiles[0][0],profiles[0][1])
                profiles=sorted(profiles,key=sortkey2)
                del profiles[40:]
                if(i==3): #maximum # of rounds reached, yet solution is not found
                    return ([],[])

    def naturalquestion(chardic,constraint,ClassesNum,TraitMatrix):
        (l,breeds,_ )=self.generate_question(constraint,ClassesNum,TraitMatrix)  
        print(l)
        print(constraint)
        for k in TraitMatrix:
            print(k)
        if l!=[]:
            question=''
            if len(l)==1:
               if l[0][1]==1:
                   question+='The object in the photo has '+chardic[l[0][0]]
               else:
                   question+='The object in the photo does not have '+chardic[l[0][0]] 
               print(question)
            else:
                for i in range(len(l)):
                    if i==1:
                        if l[i][1]==1:
                            question+='The object in the photo has '+chardic[l[i][0]]
                        else:
                            question+='The object in the photo does not have '+chardic[l[i][0]]
                    else:
                        #if i%2==1:
                        if i==2:
                            if l[i]=='+': question+=' OR'
                            else: question+=' AND' 
                        elif i==3:
                            if l[i][1]==1:
                                question+=' has '+chardic[l[i][0]]
                            else:
                                question+=' does not have '+chardic[l[i][0]]
                print (question)
            a=[]
            for r in breeds:
                a.extend(r) #return the breeds characterised by the question
            return a
        else:
            print ("fail to generate a question")
            return None

'''
def __init__(self, breed_set=None,e,TraitMatrix,ClassesNum):
        self.breed_set=breed_set
        self.e=e #Number of errors allowed.
        self.game_state=[self.breed_set]+[[]]*e #Initialising the gamestate as a list of lists.
        self.sigma_game_state=self.set_sigma(self.game_state) #This is the sigma(state).
        self.TraitMatrix=TraitMatrix
        self.ClassesNum=ClassesNum #number of breeds in total
chardic,constraint
'''

def generate(ClassesNum,TraitsNum,e):
    TraitMatrix=[]
    for i in range(ClassesNum):
        TraitMatrix.append([])
        for j in range(TraitsNum):
            TraitMatrix[i].append(randint(0,1))
    question = [[],[]]
    game_state=[]
    for i in range(2):
        for j in range(e+1):
            question[i].append([])
            if i==0:
                game_state.append([])            
    for i in range(ClassesNum):
        k=randint(0,1)
        j=randint(0,e)
        question[k][j].append(i+1) #breeds are 1-indexed
        game_state[j].append(i+1)
    return (TraitMatrix,question,game_state)

def singletest(ClassesNum,TraitsNum,e):
    '''
    question: for e=3, [[[3,4,7],[1],[8]],[[2],[6,9],[5]]], breed 2(1-indexed) is in the second choice and it is currently in stage 1
    ideal: the constraint
    lenstates:the number of classes in each stage in current states
    denomdis: see the formula
    '''
    (TraitMatrix,question,states)=generate(ClassesNum,TraitsNum,e)
    testcase=UlamRenyi(e=e,breed_set=list(range(1,ClassesNum+1)))
    constraint=veclen(question[0])
    chardic={1:'long tail',2:'short hair',3:'sharp teeth',4:'black fur',5:'large body size',6:'short legs',7:'short tail'}
    testcase.game_state=states #set the current game state
    print(testcase.naturalquestion(chardic=chardic,constraint=constraint,ClassesNum=ClassesNum,TraitMatrix=TraitMatrix))    
    

       
    
    

# =============================================================================
# trait_no=5
# breed_no=16
# bt_matrix=generate_breed_trait_matrix(trait_no,breed_no)
# ug_game=UlamRenyi(list(range(0,breed_no)),9)
# ug_game.sigma_game_state=[6,7,6,0,100,0,90,0,0,0]
# print(ug_game.run_algorithm())
# 
# =============================================================================








