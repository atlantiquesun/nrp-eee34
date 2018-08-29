import math
import random
import string
import sys
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter

#combine answer/falsepos/nega to one whole statement
#find some other way idk
#separate methods into different class
#fix to make it such that asked questions are only questions in parent string or something idk
class Wrapper:
    def __init__(self,trait_no,breed_no):
        Wrapper.trait_no=trait_no
        Wrapper.breed_no=breed_no
        Wrapper.breed_trait_matrix=[]
        for _ in range(Wrapper.breed_no):
            successful=False
            while(successful==False):
                random_breed = np.random.randint(2, size=Wrapper.trait_no, dtype=bool).tolist()
                if (random_breed not in Wrapper.breed_trait_matrix):
                    successful=True
                    Wrapper.breed_trait_matrix.append(random_breed)

class QuestionAsker:
    def __init__(self, question_set,askedQuestions=[]):
        if(question_set):
            self.askedQuestions=askedQuestions
            self.question_set = question_set
            self.question = self.find_question()
            self.answer_set,self.fpve_set,self.fnve_set = self.compute_all_responses(self.question)
            self.askedQuestions.append(self.question)
            self.fpve_question = QuestionAsker(self.fpve_set,self.askedQuestions)
            self.fnve_question = QuestionAsker(self.fnve_set,self.askedQuestions)

    def response(self,invert,breed,trait):
        return not(invert^Wrapper.breed_trait_matrix[breed][trait])

    def compute_all_responses(self, question):
        responses = [[], [], []]  # [[answer],[falsepositives],[falsenegatives]]
        if(question != []):
            invert1=bool(question[0])
            invert2=bool(question[2])
            for i in range(Wrapper.breed_no):
                response1=self.response(invert1,i,question[1])
                response2=self.response(invert2,i,question[3])
                if (question[4] == 0): #or
                    does_breed_fulfill=response1 or response2
                    if (does_breed_fulfill and i in self.question_set):
                        responses[0].append(i)
                    if (does_breed_fulfill and i not in self.question_set):
                        responses[1].append(i)
                    if (not(does_breed_fulfill) and i in self.question_set):
                        responses[2].append(i)
                if (question[4] == 1): #and
                    does_breed_fulfill=response1 and response2
                    if (does_breed_fulfill and i in self.question_set):
                        responses[0].append(i)
                    if (does_breed_fulfill and i not in self.question_set):
                        responses[1].append(i)
                    if (not(does_breed_fulfill) and i in self.question_set):
                        responses[2].append(i)
        return responses

    def fitness(self, question):
        # for logic 0 is or 1 is and
        counter = float('-inf')
        if (question!=None):
            answer,fpve,fnve = self.compute_all_responses(question)
            counter=len(answer)-len(fpve)-len(fnve)
        return counter

    def find_question(self):
        if (self.question_set != []):
            question_blacklist = []
            best_question_fitness = float('-inf')
            best_question = []
            for i in range(Wrapper.trait_no):
                for j in range(i,Wrapper.trait_no):
                    for or_and in [0,1]: #0 is OR, #1 is AND
                        for invertTrait1 in [0,1]: #0 is inversion #1 is no-inversion
                            for invertTrait2 in [0,1]:
                                if(i==j): #singular-trait questions
                                    or_and=1
                                    invertTrait1=invertTrait2
                                question_to_test=[invertTrait1, i, invertTrait2, j, or_and]
                                if (self.fitness(question_to_test) > best_question_fitness and self.compute_all_responses(question_to_test)[0] != False and (self.compute_all_responses(question_to_test)[1] and self.compute_all_responses(question_to_test)[2]) != self.question_set and (question_to_test not in self.askedQuestions)):
                                    best_question = question_to_test
                                    best_question_fitness=self.fitness(question_to_test)

            return best_question
        else:
            return None

sys.setrecursionlimit(2000)

workbook = xlsxwriter.Workbook('data.xlsx')
worksheet = workbook.add_worksheet()
timesToTest=300
x=[]
y=[]

for i in range(2,22):
    trait_no=i

    results = []
    x.append(i)
    for j in range(timesToTest):
        print(i,"attempt:", j)
        test1 = Wrapper(trait_no, i)
        questions = QuestionAsker(list(range(int(i/2))),[])
        results.append(len(questions.askedQuestions))
        worksheet.write(i,j,len(questions.askedQuestions))
    y.append(np.mean(results))
    worksheet.write(i, timesToTest, np.mean(results))

workbook.close()

plt.plot(x,y,'o')
plt.show()









