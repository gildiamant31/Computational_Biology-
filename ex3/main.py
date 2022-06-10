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
# they can be changed by the user while the input window is open
csv_name = "Elec_24.csv"
df = pd.read_csv(csv_name)


def convert_data():
    # convert data to numpy array and labels list
    feature_cols = df.columns[2:]
    data_features = df[feature_cols].to_numpy()
    # Economic situation: Numbers between 1 to 10
    data_labels = marks_list = df["Economic Cluster"].tolist()
    return data_features, data_labels

def draw_hexagon_cell(surface, color,
                         radius, position):
    n, r = 6, radius
    x, y = position
    pygame.draw.polygon(surface, color, [
        (x + r * cos(2 * pi * i / n),
         y + r * sin(2 * pi * i / n))
        for i in range(n)
    ])


def draw_board ():
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
            cells_num = 9 - abs(4 - i)
            for j in range(cells_num):
                extra_width = radius * (9 - cells_num)
                x = init_x + extra_width + j * 2 * radius
                y = init_y + 1.5 * i * radius
                color = tuple(np.random.randint(256, size=3))
                draw_hexagon_cell(root, color, radius, (x, y))


        pygame.display.flip()

# this function will run a simulation and
# create visualization for the using pygame
# def run_and_show_Simulation(simulation):
#     # Define some colors
#     WHITE = (255, 255, 255)  # empty cell
#     BLUE = (0, 0, 128)
#     GREEN = (0, 255, 0)  # healthy/regular people
#     RED = (255, 0, 0)  # sick person
#     # each cell's size
#     WIDTH = 3
#     HEIGHT = 2
#     MARGIN = 1
#
#     pygame.init()
#
#     # Set the width and height of the screen [width, height]
#     size = [1200, 800]
#     flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.FULLSCREEN
#     screen = pygame.display.set_mode(size, flags)
#     # set the tittle of the game
#     pygame.display.set_caption("WRAP_AROUND Covid-19 automate:")
#
#     # Loop until the user clicks the close button or Esc or no more sick organisms.
#     done = False
#     # Used to manage how fast tpython setup.py installhe screen updates
#     clock = pygame.time.Clock()
#     # define all texts in the simulation graphic
#     font_legend = pygame.font.SysFont('comicsans', 18)
#     font = pygame.font.SysFont('comicsans', 20)
#     leg_hea = font_legend.render('Healthy', True, GREEN, BLUE)
#     leg_infe = font_legend.render('Infected', True, RED, BLUE)
#     leg_emp = font_legend.render('Empty Cell', True, WHITE, BLUE)
#     text_num_crea = font.render('Number of Creatures: ' + str(simulation.board.num_residences), True, BLUE, WHITE)
#     text_exit = font.render('press Esc to exist', True, WHITE, BLUE)
#     leg_heaRect = leg_hea.get_rect()
#     leg_heaRect.center = (865, 80)
#     leg_infeRect = leg_infe.get_rect()
#     leg_infeRect.center = (865, 120)
#     leg_empRect = leg_emp.get_rect()
#     leg_empRect.center = (865, 160)
#     text_num_creaRect = text_num_crea.get_rect()
#     text_num_creaRect.center = (1000, 380)
#     text_exitRect = text_exit.get_rect()
#     text_exitRect.center = (600, 650)
#     # # -------- Main Program Loop -----------
#     # while not done:
#     # --- Main event loop
#     while not done:
#         for event in pygame.event.get():  # User did something
#             if event.type == pygame.QUIT:  # If user clicked close
#                 done = True  # Flag that we are done so we exit this loop
#             # use esc button to exit
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     done = True
#
#         # define the text which change on the main loop
#         text_num_infe = font.render('Number of infected: ' + str(simulation.board.num_sick), True, RED, WHITE)
#         text_num_gene = font.render('Number of Generation: ' + str(simulation.generation), True, BLUE, WHITE)
#         text_num_infeRec = text_num_infe.get_rect()
#         text_num_infeRec.center = (1000, 480)
#         text_num_geneRect = text_num_gene.get_rect()
#         text_num_geneRect.center = (1000, 580)
#         # draw the background and all the texts
#         screen.fill((40, 20, 128))
#         screen.blit(leg_hea, leg_heaRect)
#         screen.blit(leg_infe, leg_infeRect)
#         screen.blit(leg_emp, leg_empRect)
#         screen.blit(text_num_crea, text_num_creaRect)
#         screen.blit(text_num_infe, text_num_infeRec)
#         screen.blit(text_num_gene, text_num_geneRect)
#         screen.blit(text_exit, text_exitRect)
#         # Draw the matrix according to cell situation
#         for row in range(matrix_size[0]):
#             for column in range(matrix_size[1]):
#                 color = WHITE
#                 c = simulation.board.matrix[row][column]
#                 if c.isFull:
#                     if c.content.isSick:
#                         color = RED
#                     else:
#                         color = GREEN
#                 pygame.draw.rect(screen,
#                                  color,
#                                  [(MARGIN + WIDTH) * column + MARGIN,
#                                   (MARGIN + HEIGHT) * row + MARGIN,
#                                   WIDTH,
#                                   HEIGHT])
#
#         pygame.display.flip()
#
#         # --- Limit to 1 frame per second
#         clock.tick(1)
#         simulation.next_genartion()
#         if simulation.board.num_sick < 1:
#             done = True
#
#     # Close the window and quit.
#     pygame.quit()


# this function create an input window for simulation parameters with tkinter library.
# it fills the values by default the variables as they were defined on the top of this script.
# def getInput():
#     window = Tk()
#     window.title("Simulation Parameters")
#     main_lst = []
#     label1 = Label(window, text="D(fraction of sick): ", padx=20, pady=10)
#     label2 = Label(window, text="R(fraction of fast): ", padx=20, pady=10)
#     label3 = Label(window, text="N(num of creatures): ", padx=20, pady=10)
#     label4 = Label(window, text="P1 (Infection chance before T threshold(fraction)): ", padx=20, pady=10)
#     label5 = Label(window, text="T(threshold(percent of people (Integer))) ", padx=20, pady=10)
#     label6 = Label(window, text="P2 (Infection chance after threshold(fraction)): ", padx=20, pady=10)
#     label7 = Label(window, text="X(num of genertion until recoverd): ", padx=20, pady=10)
#     d = Entry(window, width=30, borderwidth=5)
#     d.insert(END, str(D))
#     r = Entry(window, width=30, borderwidth=5)
#     r.insert(END, str(R))
#     n = Entry(window, width=30, borderwidth=5)
#     n.insert(END, str(N))
#     p1 = Entry(window, width=30, borderwidth=5)
#     p1.insert(END, str(P1))
#     t = Entry(window, width=30, borderwidth=5)
#     t.insert(END, str(T))
#     p2 = Entry(window, width=30, borderwidth=5)
#     p2.insert(END, str(P2))
#     x = Entry(window, width=30, borderwidth=5)
#     x.insert(END, str(X))
#     Exit = Button(window, text="Start simulation", padx=20, pady=10, command=window.quit)
#     label1.grid(row=0, column=0)
#     label2.grid(row=1, column=0)
#     label3.grid(row=2, column=0)
#     label4.grid(row=3, column=0)
#     label5.grid(row=4, column=0)
#     label6.grid(row=5, column=0)
#     label7.grid(row=6, column=0)
#     d.grid(row=0, column=1)
#     r.grid(row=1, column=1)
#     n.grid(row=2, column=1)
#     p1.grid(row=3, column=1)
#     t.grid(row=4, column=1)
#     p2.grid(row=5, column=1)
#     x.grid(row=6, column=1)
#     Exit.grid(row=10, column=0, columnspan=2)
#     window.mainloop()
#     try:
#         # convert the input values from string to numeric values (int or float)
#         values = [float(d.get()), float(r.get()), int(n.get()), float(p1.get()), int(t.get()), float(p2.get()),
#                   int(x.get())]
#     except:  # if its faild to convert that means  Input not valid - show message and exit
#         messagebox.showwarning("WRAP_AROUND Covid-19 automate", "ERRRORRRR!!!!! Input not valid")
#         sys.exit(-1)
#     # check every fraction/float input, if its unvalid - show message and exit
#     for value in values:
#         if isinstance(value, float):
#             if not (0 <= value <= 1):
#                 messagebox.showwarning("WRAP_AROUND Covid-19 automate", "ERRRORRRR!!!!! Input not valid")
#                 sys.exit(-1)
#     # check the numeric values, if its unvalid - show message and exit
#     if values[2] < 0 or values[4] < 0 or values[6] < 0:
#         messagebox.showwarning("WRAP_AROUND Covid-19 automate", "ERRRORRRR!!!!! Input not valid")
#         sys.exit(-1)
#     #  if total number of creaturs is more then the matrix size - show message and exit
#     if values[2] > matrix_size[0] * matrix_size[1]:
#         messagebox.showwarning("WRAP_AROUND Covid-19 automate", "ERRRORRRR!!!!! too much creaturs")
#         sys.exit(-1)
#     window.destroy()
#     window.quit()
#     return values


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    train_set, train_labels = convert_data()
    draw_board()
    # TODO should we add normalization?
