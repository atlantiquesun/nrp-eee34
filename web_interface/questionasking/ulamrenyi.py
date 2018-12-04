import math
from random import *
import string
import sys
import time
import numpy as np
from scipy.special import comb
from itertools import count
from copy import deepcopy

# Yi-ran's helper functions.
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


# Conrad's helper functions


def shift(list, times):
    return [0]*times + list[0:len(list)-times]


def set_sigma(list):
    return [len(i) for i in list]


def weight(sigma_state, q):  # Note, the state here is the sigma state.
    errors = len(sigma_state)-1
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
    errors = len(sigma_state)-1
    return [max(recursive_f(shift(sigma_state, errors - i)), recursive_f(sigma_state) - 3 * (errors - i))for i in range(0, errors + 1)]


def sigma_state_yes(sigma_state, t):
    # I'm interpreting Qi Yu's i-1 index where i=0 to be non-existent i.e. 0.
    errors = len(sigma_state)-1
    return [t[0]] + [sigma_state[i - 1] - t[i - 1] + t[i] for i in range(1, errors + 1)]


def sigma_state_no(sigma_state, t):
    # I'm interpreting Qi Yu's i-1 index where i=0 to be non-existent i.e. 0.
    errors = len(sigma_state)-1
    return [sigma_state[0] - t[0]] + [t[i - 1] + sigma_state[i] - t[i] for i in range(1, errors + 1)]


def condition_fulfilled(errors, i, ques_i, sigma_state, gamma_state):
    result_list = []
    state_i_sigma = shift(sigma_state, errors - i)
    for j in range(1, errors - i + 1):
        s_g_y_i = sigma_state_yes(state_i_sigma, ques_i)
        s_g_n_i = sigma_state_no(state_i_sigma, ques_i)
        alpha = [0] * (errors - i - j) + s_g_y_i[errors - i:] + \
            [state_i_sigma[-1] - ques_i[-1]] + [0] * (j - 1)
        beta = [0] * (errors - i - j) + s_g_n_i[errors - i:] + \
            [ques_i[-1]] + [0] * (j - 1)
        result_list.append(
            weight(alpha, gamma_state[i + j] - 1) and weight(beta, gamma_state[i + j] - 1) <= 2 ** (
                gamma_state[i + j] - 1))
    return all(result_list)


def run_algorithm(sigma_state):
    errors = len(sigma_state)-1
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
            question.append(next((x[0] for x in potential_solutions if condition_fulfilled(errors, i, x[1], sigma_state, gamma_state)),0))
    return question

def process_yes(game_state, question_set):
    e=len(game_state)
    evaluated_game_state=[[]]*e
    evaluated_game_state[0] = list(set(game_state[0]) & set(question_set))
    for i in range(1, e):
        evaluated_game_state[i] = list(
            (set(game_state[i - 1]) - set(question_set)) | (set(game_state[i]) & set(question_set)))
    return evaluated_game_state


def process_no(game_state, question_set):
    e=len(game_state)
    evaluated_game_state=[[]]*e
    evaluated_game_state[0] = list(set(game_state[0]) - set(question_set))
    for i in range(1, e):
        evaluated_game_state[i] = list(
            (set(game_state[i-1]) & set(question_set)) | (set(game_state[i]) - set(question_set)))
    return evaluated_game_state

def prepare_arguments(errors, game_state, constraint,ClassesNum,TraitMatrix):
        '''
        prepare arguments for 'generate_question'
        assuming breeds are 1-indexed, breed 1,2,3,...
        states: 0,...,e (0-indexed)
        '''
        from copy import deepcopy
        l=list(np.zeros((errors+1)))
        zerostates=deepcopy(l)
        for j in range(errors+1):
            l[j]+= len(game_state[j])
        statecapacity=l
        denomdis=(Distance(statecapacity,constraint)+Distance(zerostates,constraint))/2
        classdc={}
        for i in range(errors+1): 
            for j in game_state[i]:
                classdc[j]=i
        return(constraint,classdc,ClassesNum,TraitMatrix,denomdis)

def generate_question(errors, game_state, constraint, ClassesNum,TraitMatrix): 
    (constraint,classdc,ClassesNum,TraitMatrix,denomdis)=prepare_arguments(errors,game_state,constraint,ClassesNum,TraitMatrix)
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
                for q in classdc.keys():
                    if(TraitMatrix[q-1][k]==0):
                        temp1[0][classdc[q]].append(q)
                    else:
                        temp2[0][classdc[q]].append(q)
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

def naturalquestion(errors, game_state, chardic,constraint,ClassesNum,TraitMatrix):
    (l,breeds,_ )=generate_question(errors, game_state, constraint,ClassesNum,TraitMatrix)  
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
        return (a,question)
    else:
        print ("fail to generate a question")
        return None