"""
Gil Diamant
Itamar Twersky
"""

import pandas as pd
import sys
import os
import pygame
from math import sin, cos, pi
from tkinter import *
from tkinter import messagebox
import random
import numpy as np

# these are all the global variables which define in the instructions of this exercise.
# TODO - IF you work with vscode change the path of the file:
csv_name = "Elec_24.csv"
df = pd.read_csv(csv_name)
clusters = []


# create one cluster with random values between the max val and min val in every column in the dataset
def create_cluster():
    cluster = []
    # get only the columns of features (from the third column)
    feature_cols = df.columns[2:]
    for col in feature_cols:
        # get the maximum & minimum values of this column
        max_val = df[col].max()
        min_val = df[col].min()
        cluster.append(np.random.randint(min_val, max_val))
    return np.asarray(cluster)


# create 2D list with rows with different length for each one - like on the hexagon grid
def array_of_clusters():
    for i in range(9):
        new_row = []
        # for getting different length for every row
        cells_num = 9 - abs(4 - i)
        for j in range(cells_num):
            new_row.append(create_cluster())
        clusters.append(new_row)


def convert_data():
    # convert data to numpy array and labels list
    # use only features columns -> start from the third one
    feature_cols = df.columns[2:]
    # convert all data to numpy array
    data_features = df[feature_cols].to_numpy()
    # Economic situation: Numbers between 1 to 10
    data_labels = marks_list = df["Economic Cluster"].tolist()
    return data_features, data_labels


# function to draw single hexagon
def draw_hexagon_cell(surface, color,
                      radius, position):
    n, r = 6, radius
    x, y = position
    pygame.draw.polygon(surface, color, [
        (x + r * cos(2 * pi * i / n),
         y + r * sin(2 * pi * i / n))
        for i in range(n)
    ])


# function to draw one board of big hexagon which contain 61 hexagons
def draw_board():
    # black background
    bg_color = (0, 0, 0)

    w, h = 600, 600

    pygame.init()
    root = pygame.display.set_mode((w, h))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        root.fill(bg_color)
        radius = min(w, h) / 50 + 10
        init_x = w / 2 - 170
        init_y = h / 2 - 150
        for i in range(9):
            # for getting different length for every row
            cells_num = 9 - abs(4 - i)
            for j in range(cells_num):
                extra_width = radius * (9 - cells_num)
                x = init_x + extra_width + j * 2 * radius
                y = init_y + 2 * i * radius
                # create random color
                color = tuple(np.random.randint(256, size=3))
                draw_hexagon_cell(root, color, radius, (x, y))

        pygame.display.flip()


if __name__ == '__main__':
    array_of_clusters()
    # train_set, train_labels = convert_data()
    # draw_board()
    # TODO should we add normalization?
