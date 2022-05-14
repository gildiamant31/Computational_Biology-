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

    # make crossover from two solutions
    def crossover(self, sol1, sol2):
        crossover_sol = []
        random_row = np.random.randint(0, matrix_size[0] - 1)
        [crossover_sol.append(sol1[i]) for i in range(0, random_row)]
        [crossover_sol.append(sol2[i]) for i in range(random_row, matrix_size[0])]
        return crossover_sol

    # create one mutation in random indexes -> replace the current value to another in the relevant range
    def create_mutation(self, sol):
        indexes = np.random.randint(0, matrix_size[0] - 1, 2)
        current_num = sol[indexes[0]][indexes[1]]
        new_num = np.random.randint(1, matrix_size[0])
        while current_num == new_num:
            new_num = np.random.randint(1, matrix_size[0])
        sol[indexes[0]][indexes[1]] = new_num

    def evaluation(self, sol):
        pass

    def next_generation(self):
        pass

    def solve_convergence(self):
        pass

    def run_algo(self):
        pass

    # necessary ?
    def fitness(self):
        pass


if __name__ == '__main__':
    openInputFile()
    random_sols = initial_random_sols()
    algo = GenericAlgo(random_sols)
