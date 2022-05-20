"""
Gil Diamant
Itamar Twersky
"""
# import matplotlib.pyplot as plt
# import pygame
# from tkinter import *
# from tkinter import messagebox
from os import stat
import random
import numpy as np
import sys

# global variables to be determind from file
matrix_size = []
pop_size = 100  # size of population
init_digits_coords = []  # list of tuples of coordinates and a value of it, look like this -> ((1,2),4)
signs_coords = []  # list of tuples of coordinates tuple pairs represent the sign location, look like this -> ((1,2),(1,5))
# hyper parameters
crossover_chance = 35
mutation_chance = 10
max_num_mutation =3

# get input from input files with details on the matrix and saved it on global variables
def openInputFile():
    # TODO edit path
    with open("input/input1.txt") as f:
        lines = f.readlines()
    lines = [lines[i].strip() for i in range(len(lines))]
    matrix_size.append(int(lines[0]))
    matrix_size.append(int(lines[0]))
    init_digits_num = int(lines[1])
    if init_digits_num > 0:
        for i in range(2, init_digits_num + 2):
            # remove one from any index because index starts from 0 and in the file it starts from 1
            new_given_num = ((int(lines[i][0]) - 1, int(lines[i][2]) - 1), int(lines[i][4]))
            init_digits_coords.append(new_given_num)
    current_index = 2 + init_digits_num  # the index that loop stopped at - indicate where we are in the 'lines' list
    signs_num = int(lines[current_index])
    if signs_num > 0:
        for i in range(current_index + 1, current_index + 1 + signs_num):
            # remove one from any index because index starts from 0 and in the file it starts from 1
            new_given_signs_location = ((int(lines[i][0]) - 1, int(lines[i][2]) - 1),
                                        (int(lines[i][4]) - 1, int(lines[i][6]) - 1))
            signs_coords.append(new_given_signs_location)


def get_random_sol(size):
    nums = list(range(1, (size[0] + 1)))
    sol = [np.random.permutation(nums) for i in range(size[0])]
    return np.asarray(sol)


# create 100 random solutions
def initial_random_sols():
    sols_array = []
    for i in range(pop_size):
        # new_sol = np.random.randint(1, matrix_size[0] + 1, size=matrix_size)
        # TODO check if better
        new_sol = get_random_sol(matrix_size)
        for i in range(len(init_digits_coords)):
            # add the initial values from the input file
            new_sol[init_digits_coords[i][0][0]][init_digits_coords[i][0][1]] = init_digits_coords[i][1]
        sols_array.append(new_sol)
    return sols_array


class Fitness_byPlace:
    def __init__(self, scores):
        self.calls = 0
        self.scores = scores
        the_range = range(1, len(scores) + 1)
        self.posabilities = [x / sum(the_range) for x in the_range]
        # will save indexes of scores in decreasing order - the best solution will be the last
        self.orderd_indexes = np.argsort(self.scores * -1)

    def get_fit(self):
        self.calls += 1
        # TODO maybe use by scoers instead 
        self.orderd_indexes = np.argsort(self.scores * -1)

    def get_newSol_idx(self):
        return np.random.choice(self.orderd_indexes, p=self.posabilities)


class GenericAlgo:
    def __init__(self, sols):
        self.sols = sols
        self.scores = np.array([-1] * len(self.sols))
        self.best_sol_idx = -10
        self.best_val = 10000
        self.prevBest_val = 10000
        self.fitness_f = Fitness_byPlace(self.scores)
        # TODO in Darvin
        # self.optimize_sols = sols.copy()
        # self.fitness_f = Fitness_byPlace(self.optimize_sols)

    # @classmethod
    def optimize(self):
        counter = 0
        numbers_opts = [i for i in range(1, matrix_size[0] + 1)]
        done = False
        for sol in self.sols:
            # check if first number is greater than the other which appear after the "bigger than" sign
            for i in range(len(signs_coords)):
                # first number before the sign - should be the bigger
                first_num = sol[signs_coords[i][0][0]][signs_coords[i][0][1]]
                # second number after the sign - should be the smaller
                second_num = sol[signs_coords[i][1][0]][signs_coords[i][1][1]]
                # if we have a mistake we replace between them
                if first_num <= second_num:
                    sol[signs_coords[i][0][0]][signs_coords[i][0][1]] = second_num
                    sol[signs_coords[i][1][0]][signs_coords[i][1][1]] = first_num
                    counter += 1
                if counter == matrix_size[0]:
                    done = True
                    break
            if done:
                continue
            # TODO remove duplicates from each line & row
            # for row in sol:
            #     if self.checkUnique(row) == 1:
            #         unused_num = list(set(numbers_opts) - set(row))
            #         unused_num = unused_num[0]

    # make crossover from two parents solutions
    def crossover(self, sol1, sol2):
        crossover_sol1 = []
        crossover_sol2 = []
        # get random rows number.
        random_row = np.random.randint(0, matrix_size[0] - 1)
        # take this numbers of rows from first solution matrix
        [crossover_sol1.append(sol1[i]) for i in range(0, random_row)]
        [crossover_sol2.append(sol2[i]) for i in range(0, random_row)]
        # take the rest numbers of rows from the second solution matrix
        [crossover_sol1.append(sol2[i]) for i in range(random_row, matrix_size[0])]
        [crossover_sol2.append(sol1[i]) for i in range(random_row, matrix_size[0])]
        crossover_sol1 = np.array(crossover_sol1)
        return crossover_sol1, crossover_sol2

    # create one mutation in random indexes -> replace the current value to another in the relevant range
    def create_mutation(self, sol):
        # get random indexes
        indexes = np.random.randint(0, matrix_size[0] - 1, 2)
        current_num = sol[indexes[0]][indexes[1]]
        stop = False
        # we don't want to create mutation on the permanent input values
        # we will do this loop until we get random value which not appear in the input values
        while not stop:
            for i in range(len(init_digits_coords)):
                if tuple(indexes) in init_digits_coords[i]:
                    stop = False
                    indexes = np.random.randint(0, matrix_size[0] - 1, 2)
                    break
                stop = True
        # get new number value for replacing the old value
        new_num = np.random.randint(1, matrix_size[0])
        # stop this loop when the numbers are different
        while current_num == new_num:
            new_num = np.random.randint(1, matrix_size[0])
        sol[indexes[0]][indexes[1]] = new_num

    def evaluation(self):
        for index, sol in enumerate(self.sols):
            # our score will be negative, we will add score for every mismatch.
            # if everything it's good the biggest score will be 0
            negative_score = 0
            # check if there are duplicate in every row
            for row in sol:
                negative_score += self.checkUnique(row)
            # check if there are duplicate in every column
            for col in range(matrix_size[0]):
                negative_score += self.checkUnique(sol[:, col])
            negative_score += self.checkSignsPlaces(sol)
            self.scores[index] = negative_score
            # minimum score is the best solution - add it to the sollutions
        self.prevBest_val = self.best_val
        self.best_sol_idx = np.argmin(self.scores)
        self.best_val = self.scores[self.best_sol_idx]

    # check if there are duplicates in every row or column - if there is it return one otherwise it return 0
    def checkUnique(self, row_or_col):
        if len(row_or_col) != len(np.unique(row_or_col)):
            return 1
        return 0

    # check according to "bigger than" sign locations on the board if it valid.
    # if it isn't, we add 1 to the negative score and return it
    def checkSignsPlaces(self, sol):
        signs_score = 0
        for i in range(len(signs_coords)):
            # first number before the sign
            first_num = sol[signs_coords[i][0][0]][signs_coords[i][0][1]]
            # second number after the sign
            second_num = sol[signs_coords[i][1][0]][signs_coords[i][1][1]]
            if first_num <= second_num:
                signs_score += 1
        return signs_score

    def create_next_sols(self):
        new_sols = []
        done = False
        while not done:
            # get create new solution fou to fittnes
            index = self.fitness_f.get_newSol_idx()
            sol = self.sols[index].copy()
            if random.randrange(0, 100) < crossover_chance:
                # choose one of the chosen sols te crossover
                if len(new_sols) > 1:
                    cross_index = np.random.randint(0, len(new_sols) - 1)
                    sol_cross = new_sols[cross_index]
                    sol, sol_cross = self.crossover(sol, sol_cross)
            # TODO only one mutation? maybe more
            for m in range(max_num_mutation):
                if random.randrange(0, 100) < mutation_chance:
                    self.create_mutation(sol)
            new_sols.append(sol)
            if (len(new_sols) + 5) == len(self.sols):
                done = True
        for t in range(5):
            new_sols.append(self.sols[self.best_sol_idx].copy())
        self.sols = new_sols

    def next_generation(self):
        self.evaluation()
        self.fitness_f.get_fit()
        self.create_next_sols()

    def solve_convergence(self):
        pass

    def run_algo(self):
        global mutation_chance
        global crossover_chance
        gen_counter = 0
        count = False
        for i in range(150000):
            if self.best_val != self.prevBest_val:
                print(self.best_val)
            if (i % 40) == 0 and i != 0:
                count = True
                print("gen: {} score:{}".format(i, self.best_val))
                print(self.sols[self.best_sol_idx])
            #     mutation_chance = 80
            #     crossover_chance = 80
            # if count:
            #     gen_counter += 1
            # if gen_counter == 10:
            #     print("gen: {} score:{}".format(i, self.best_val))
            #     mutation_chance = 35
            #     crossover_chance = 10
            #     count = False
            #     gen_counter = 0
            self.next_generation()
            if self.best_val == 0:
                print(self.best_val)
                break


class DarvinAlgo(GenericAlgo):
    def __init__(self, sols):
        super(DarvinAlgo, self).__init__(sols)

    def next_generation(self):
        self.evaluation()
        self.fitness_f.get_fit()
        self.create_next_sols()
        self.optimize()
        self.evaluation()


class LemarkAlgo(GenericAlgo):
    def __init__(self, sols):
        super(LemarkAlgo, self).__init__(sols)

    def next_generation(self):
        self.optimize()
        self.evaluation()
        self.fitness_f.get_fit()
        self.create_next_sols()


if __name__ == '__main__':
    openInputFile()
    random_sols = initial_random_sols()
    algo = LemarkAlgo(random_sols)
    algo.run_algo()
    # TODO לשים לבנת חבלה במעבדה של אונגר
