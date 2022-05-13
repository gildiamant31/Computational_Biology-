"""
Gil Diamant
Itamar Twersky
"""

import pygame
from tkinter import *
from tkinter import messagebox
import random
import numpy as np

# import matplotlib.pyplot as plt

# these are all the global variables which define in the instructions of this exercise.
# they can be changed by the user while the input window is open

matrix_size = (5, 5)
pop_size = 100


# create 100 random solutions
def initial_random_sols():
    sols_array = []
    for i in range(pop_size):
        new_sol = np.random.randint(1, matrix_size[0] + 1, size=matrix_size)
        sols_array.append(new_sol)
    return sols_array

class GenericAlgo:
    def __init__(self,sols):
        self.sols = sols



if __name__ == '__main__':
    random_sols = initial_random_sols()
    GenericAlgo(random_sols)
