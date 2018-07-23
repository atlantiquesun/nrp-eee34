import math
import numpy as np
import random
import string
import matplotlib.pyplot as mpl


class QuestionAsker:
    def __init__(self, question_set):
        self.parent_question = None
        self.question_set = question_set
        if(question_set):
            self.question = self.find_question()
            self.answer_set = self.find_answer_set(self.question)
            self.fpve_set = self.find_false_positive(self.question)
            self.fnve_set = self.find_false_negative(self.question)
            self.fpve_question = QuestionAsker(self.fpve_set)
            ##elf.fpve_question.parent_question=self
            self.fnve_question = QuestionAsker(self.fnve_set)
            ##self.fpve_question.parent_question=self

    def find_answer_set(self, question):
        answer = []
        if(question != None):
            if (question[2] == 0):
                for i in range(breed_no):
                    if ((trait_breed[i][question[0]] or trait_breed[i][question[1]]) and i in self.question_set) == True:
                        answer.append(i)
            if (question[2] == 0):
                for i in range(breed_no):
                    if ((trait_breed[i][question[0]] and trait_breed[i][question[1]]) and i in self.question_set) == True:
                        answer.append(i)
            return answer

    def find_false_positive(self, question):
        if (question != None):
            answer = []
            if (question[2] == 0):
                for i in range(breed_no):
                    if ((trait_breed[i][question[0]] or trait_breed[i][
                        question[1]]) and i not in self.question_set) == True:
                        answer.append(i)
            if (question[2] == 1):
                for i in range(breed_no):
                    if ((trait_breed[i][question[0]] and trait_breed[i][
                        question[1]]) and i not in self.question_set) == True:
                        answer.append(i)
            return answer
        else:
            return None

    def find_false_negative(self, question):
        if (question != None):
            answer = []
            if (question[2] == 0):
                for i in range(breed_no):
                    if (not (trait_breed[i][question[0]] or trait_breed[i][
                        question[1]]) and i in self.question_set) == True:
                        answer.append(i)
            if (question[2] == 1):
                for i in range(breed_no):
                    if (not (trait_breed[i][question[0]] and trait_breed[i][
                        question[1]]) and i in self.question_set) == True:
                        answer.append(i)
            return answer
        else:
            return None

    def fitness(self, question):
        # for logic 0 is or 1 is and
        counter = 0
        if (question!=None):
            if (question[2] == 0):
                for i in range(breed_no):
                    if ((trait_breed[i][question[0]] or trait_breed[i][question[1]]) and i in self.question_set) == True:
                        counter += 1
                    if (not (
                            trait_breed[i][question[0]] or trait_breed[i][question[1]]) and i in self.question_set) == True:
                        counter -= 1
                    if ((trait_breed[i][question[0]] or trait_breed[i][
                        question[1]]) and i not in self.question_set) == True:
                        counter -= 1
            if (question[2] == 1):
                for i in range(breed_no):
                    if (not (trait_breed[i][question[0]] and trait_breed[i][
                        question[1]]) and i in self.question_set) == True:
                        counter -= 1
                    if ((trait_breed[i][question[0]] and trait_breed[i][
                        question[1]]) and i not in self.question_set) == True:
                        counter -= 1
            return counter

    def find_question(self):
        if (self.question_set != []):
            best_question_fitness = float('-inf')
            best_question = []
            for i in range(trait_no):
                for j in range(trait_no):
                    if (self.fitness([i, j, 0]) > best_question_fitness):
                        best_question = [i, j, 0]
                        best_question_fitness=self.fitness([i, j, 0])
                    if (self.fitness([i, j, 1]) > best_question_fitness):
                        best_question = [i, j, 1]
                        best_question_fitness=self.fitness([i, j, 1])
            return best_question
        else:
            return None


trait_no = 8
breed_no = 5
trait_breed = [[True, False, True, False, True, False, True, False],
               [True, False, True, False, True, True, True, False],
               [False, False, True, False, True, False, False, True],
               [True, True, True, False, True, False, True, True],
               [True, False, True, False, True, False, False, True]]
breed_set = [[0, 1, 2], [3, 4]]
grandcanyon = QuestionAsker([1, 2, 3])









