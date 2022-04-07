import pygame
import sys
import random
import numpy as np
import matplotlib.pyplot as plt


matrix_size = (200 , 200)
D = 0.5  # % of initial weak people
N = 10  # initial number of people in the module
M = 1/9  # possibility of moving to each near cell (or staying in place) in the matrix
R = 0.05  # % of faster people
P1 = 0.2  # possibility of infect close people
T = 20  # percentage of weak people threshold which after it P var is going down
P2 = 0.1  # possibility of infect close people when we pass the threshold (T var)
X = 5  # number of generation for being weak and infect other people.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(matrix_size)