import pygame
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

matrix_size = (200, 200)
D = 0.5  # % of initial sick people
N = 10  # initial number of people in the module
M = 1 / 9  # possibility of moving to each near cell (or staying in place) in the matrix
R = 0.05  # % of faster people
P1 = 0.2  # possibility of infect close people
T = 20  # percentage of sick people threshold which after it P var is going down
P2 = 0.1  # possibility of infect close people when we pass the threshold (T var)
X = 5  # number of generation for being weak and infect other people.


class Yetzur:
    def __init__(self, place, isInfected, isFast):
        self.place = place  # has to be two dimensions
        self.isInfected = isInfected  # true or false
        self.isFast = isFast  # true or false , tell us if this object can move 10 cells in one direction
        # per generation.

    def move(self):
        pass


class Cell:
    def __init__(self, newYetzur=None, isFull=False):
        self.isFull = isFull
        if self.isFull:
            self.newYetzur = newYetzur


# check if we pass the threshold and return the adjusted P
def passThreshold(sickPercentage):
    if sickPercentage > T:
        return P2
    else:
        return P1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # create matrix of cells:
    matrix = np.full(matrix_size, Cell())
    print(matrix)
