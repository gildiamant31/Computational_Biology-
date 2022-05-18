"""
Gil Diamant
Itamar Twersky
"""
# import matplotlib.pyplot as plt
# import pygame
from tkinter import *
from tkinter import messagebox
import random
import numpy as np

# these are all the global variables which define in the instructions of this exercise.
# they can be changed by the user while the input window is open

matrix_size = []
pop_size = 100
init_digits_num = 0
init_digits_coords = []  # list of tuples of coordinates and a value of it, look like this -> ((1,2),4)
signs_num = 0
signs_coords = []  # list of tuples of coordinates tuple pairs represent the sign location,


# look like this -> ((1,2),(1,5))


# get input from input files with details on the matrix and saved it on global variables
def openInputFile():
    with open('input/input1.txt') as f:
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
    current_index = 2 + init_digits_num  # the index that loop stopped at
    signs_num = int(lines[current_index])
    if signs_num > 0:
        for i in range(current_index + 1, current_index + 1 + signs_num):
            # remove one from any index because index starts from 0 and in the file it starts from 1
            new_given_signs_location = ((int(lines[i][0]) - 1, int(lines[i][2]) - 1),
                                        (int(lines[i][4]) - 1, int(lines[i][6]) - 1))
            signs_coords.append(new_given_signs_location)


# create 100 random solutions
def initial_random_sols():
    sols_array = []
    for i in range(pop_size):
        new_sol = np.random.randint(1, matrix_size[0] + 1, size=matrix_size)
        for i in range(len(init_digits_coords)):
            # add the initial values from the input file
            new_sol[init_digits_coords[i][0][0]][init_digits_coords[i][0][1]] = init_digits_coords[i][1]
        sols_array.append(new_sol)
    return sols_array


class GenericAlgo:
    def __init__(self, sols):
        self.sols = sols
        self.scores = [-1] * len(self.sols)
        self.greatest_sol_idx = -10

    # make crossover from two parents solutions
    def crossover(self, sol1, sol2):
        crossover_sol = []
        # get random rows number.
        random_row = np.random.randint(0, matrix_size[0] - 1)
        # take this numbers of rows from first solution matrix
        [crossover_sol.append(sol1[i]) for i in range(0, random_row)]
        # take the rest numbers of rows from the second solution matrix
        [crossover_sol.append(sol2[i]) for i in range(random_row, matrix_size[0])]
        crossover_sol = np.array(crossover_sol)
        return crossover_sol

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

    def next_generation(self):
        self.evaluation()
        # minimum score is the greatest solution
        self.greatest_sol_idx = self.scores.index(min(self.scores))
        new_sols = []
        new_sols.append(self.sols[self.greatest_sol_idx])
        done = False
        while not done:
            # TODO improve the randomly choose - create some priority for the better score (the lower in our case)
            index = np.random.randint(0, pop_size - 1)
            sol = self.sols[index]
            crossover_chance = 15
            if random.randrange(0, 100) < crossover_chance:
                cross_index = np.random.randint(0, pop_size - 1)
                sol_cross = self.sols[cross_index]
                sol = self.crossover(sol, sol_cross)
            mutation_chance = 15
            if random.randrange(0, 100) < mutation_chance:
                self.create_mutation(sol)
            new_sols.append(sol)
            if len(new_sols) == len(self.sols):
                done = True
        self.sols = new_sols

    def solve_convergence(self):
        pass

    def run_algo(self):
        for i in range(15000):
            print(self.scores[self.greatest_sol_idx])
            self.next_generation()
            if self.scores[self.greatest_sol_idx] == 0:
                print(self.sols[self.greatest_sol_idx])
                break

    # necessary ?
    def fitness(self):
        pass


if __name__ == '__main__':
    openInputFile()
    random_sols = initial_random_sols()
    algo = GenericAlgo(random_sols)
    algo.run_algo()
