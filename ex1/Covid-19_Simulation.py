"""
Gil Diamant
Itamar Twersky
"""
import sys
import os
import pygame
from tkinter import *
from tkinter import messagebox
import random
import numpy as np



# these are all the global variables which define in the instructions of this exercise.
# they can be changed by the user while the input window is open

matrix_size = (200, 200)
D = 0.01  # % of initial sick people
N = 32000  # initial number of people in the module - start with 70% of the automate size
R = 0.2  # % of faster people
P1 = 0.9  # possibility of infect close people
T = 10  # percentage of sick people threshold which after it P var is going down
P2 = 0.1  # possibility of infect close people when we pass the threshold (T var)
X = 3  # number of generation for being weak and infect other people.


# this is the class of the organism in the simulation - it can be sick,healthy or recovered.
class Yetzur:
    def __init__(self, location=((0, 0)), stats="H", isFast=False):
        assert stats in ["H", "S", "R"], str(
            stats) + "Yetzur can be is stats of H/S/R only"  # H/S/R for healthy/Sick/Recoverd
        self.isSick = self.isRecovered = False
        # self.stats = stats
        self.isHealthy = True
        # self.stats = stats
        if stats == "S":
            self.get_sick()
        if stats == "R":
            self.get_recovered()
        self.isFast = isFast  # true or false , tell us if this object can move 10 cells in one direction
        self.location = location
        self.sickTime = 0
        self.generation = 0  # necessary for the moving from one board to a new one

    # make the organism be infected in Covid-19
    def get_sick(self):
        self.isHealthy = self.isRecovered = False
        self.isSick = True
        # self.stats = "S"
        self.sickTime = 1

    # make the organism older in one generation and also increase his sickness time if it is sick
    def get_older(self):
        self.generation += 1
        if self.isSick:
            self.sickTime += 1
            if self.sickTime > X:
                self.sickTime = 0
                self.get_recovered()

    # convert the organism to recover mode - not going to be sick again ever
    def get_recovered(self):
        self.isHealthy = self.isSick = False
        self.isRecovered = True
        # self.stats = "R"

    # get randomly location for the next generation of this organism
    def next_location(self):
        if not self.isFast:
            choice = random.choice(self.get_neghibors_and_self_indexes())
        else:  # If it is fast-yetzur there are more indexes to choice from.
            vert_change = random.choice(range(-10, 11))
            new_vert = (self.location[0] + vert_change) % matrix_size[0]
            horiz_change = random.choice(range(-10, 11))
            new_horiz = (self.location[1] + horiz_change) % matrix_size[1]
            choice = tuple([new_vert, new_horiz])
        return choice

    # get all indexes near to this organism including its location
    def get_neghibors_and_self_indexes(self):
        neighbours = [tuple([(i + self.location[0]) % matrix_size[0], (j + self.location[1]) % matrix_size[1]]) for i in
                      range(-1, 2) for j in range(-1, 2)]
        return neighbours


# class of cell which can contain the organism and will be part of the simulation's board
class Cell:
    def __init__(self, isFull=False, content=None):
        self.isFull = isFull
        self.content = content

    # Add organism to this Cell only if it empty
    def add_content(self, new_yetzur, new_location):
        # throw assertion in case of addning to filled cell 
        assert not self.isFull, "tried to fill filled cell"
        self.content = new_yetzur
        self.content.location = new_location
        self.isFull = True

    # remove organism from this Cell only if it full
    def remove_content(self):
        # throw assertion in case of clearing empty cell
        assert self.isFull, "cant clear empty cell"
        self.isFull = False
        old_cont = self.content
        self.content = None
        return old_cont

    # define the way a cell is trarsformed to str
    def __str__(self):
        if not self.isFull:
            return "E"
        else:
            if self.content.isSick:
                return "S"
            else:
                return "H"


# class of board which contain a cell matrix and manage it.
# also keeps track of number of residences and sick residences
class Board:
    def __init__(self):
        # create a matrix of Cell object in size of the global variable (equal to 200 X 200 in exercise's definition)
        self.matrix = np.array([Cell() for i in range(matrix_size[0] * matrix_size[1])], dtype=object)
        self.matrix = self.matrix.reshape(matrix_size)
        # initialize counters
        self.num_residences = 0
        self.num_sick = 0

    # try add organism to specific location on the board. return true if it succeeded(the location was empty), and false otherwise
    def add_residence_to(self, residence, new_location):
        # raise assertion if the whole board is full
        assert self.matrix.size > self.num_residences, "tried to add residence to full matrix"
        # try add organism to specific location on board as described above
        if self.matrix[new_location[0]][new_location[1]].isFull:
            return False
        else:
            self.matrix[new_location[0]][new_location[1]].add_content(residence, (new_location[0], new_location[1]))
            self.num_residences += 1
            residence.location = new_location
            # update sick count
            if residence.isSick:
                self.num_sick += 1

            return True

    # will remove residence from specific location on the board. and return it
    def remove_residence_from(self, location):
        self.num_residences -= 1
        res = self.matrix[location[0], location[1]].remove_content()
        # update sick count
        if res.isSick:
            self.num_sick -= 1
        return res

    # add one organism to randomly location
    def add_residence_randomly(self, residence):
        # try to add to random place until we get True for successful adding
        while (not self.add_residence_to(residence, (
                random.choice(range(0, matrix_size[0])), random.choice(range(0, matrix_size[1]))))):
            pass

    # initilize 'N' of board's  residence according to global variables scpecification, and locate them randomaly
    def add_N_of_residences_randomly(self, N):
        # counters of how much residence from each types according to global variables scpecification
        counter_D = int(D * N)  # for sick
        counter_R = int(R * N)
        counter_DR = int(R * counter_D)  # for faster and sick
        counter_R = counter_R - counter_DR
        counter_D = counter_D - counter_DR
        counter_N = N - counter_D - counter_DR - counter_R # for healthy and slow
        # add residences to board
        [self.add_residence_randomly(Yetzur(isFast=True, stats="S")) for i in range(counter_DR)]
        [self.add_residence_randomly(Yetzur(stats="S")) for i in range(counter_D)]
        [self.add_residence_randomly(Yetzur(isFast=True)) for i in range(counter_R)]
        [self.add_residence_randomly(Yetzur()) for i in range(counter_N)]

    # for printing the board as string - mainly for debugging
    def __str__(self):
        as_a_str = "num of residences: {} \n".format(self.num_residences)
        as_a_str += "num of sicks: {} \n".format(self.num_sick)
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                as_a_str = as_a_str + str(self.matrix[i][j]) + "\t"
            as_a_str = as_a_str + "\n"
        return as_a_str


# class of simulation - contain the board of cells and run bio simulation on it.
class Simulation:

    def __init__(self, board):
        self.board = board
        self.generation = 0

    # main function of the simulation which update the simulation to the next generation of it.
    def next_genartion(self):
        # create new empty board for the next generation
        newBoard = Board()
        # for convinence
        self_matrix = self.board.matrix
        sick_chance = self.sick_chance()
        # make all the creaturs older for synchronization
        self.get_all_older()
        # move throu all the board and update the creaturs 
        for i in range(self_matrix.shape[0]):
            for j in range(self_matrix.shape[1]):
                c = self_matrix[i][j]
                # check if the cell full
                if c.isFull:
                    # remove the creature from the older board 
                    resid = c.remove_content()
                    # check if the residence is healthy - can become sick
                    if resid.isHealthy:
                        if self.search_sick_neghibors(resid):
                            # residence can be sick in possibility of P
                            if random.randrange(0, 100) < sick_chance * 100:
                                resid.get_sick()
                    # try to find a empty next place until we get 'False' for not-full location:
                    # !!!!
                    # For preventing of two creaturs in one cell, we crete a new empty board
                    #  and we only add to new location if the place in the new board is empty.
                    # For prevneting the case of creatur which all of its possible next location is full, we dont put new residences 
                    # where the place in the older board are full. 
                    # That make the possability for every creatur to remain in its place for the next generation
                    # !!!!
                    new_location_isFull = True
                    while new_location_isFull:
                        new_location = resid.next_location()
                        new_location_isFull = newBoard.matrix[new_location[0]][new_location[1]].isFull | \
                                              self_matrix[new_location[0]][new_location[1]].isFull
                    assert newBoard.add_residence_to(resid, new_location), "should be empty but is full"
        # update counter and board
        self.generation += 1
        self.board = newBoard

    # search for each organism neighbors if one of them is infected.
    # return True if there is, False otherwise
    def search_sick_neghibors(self, residence):
        neghibors = residence.get_neghibors_and_self_indexes()
        for ne in neghibors:
            if self.board.matrix[ne[0]][ne[1]].isFull:
                if self.board.matrix[ne[0]][ne[1]].content.isSick:
                    return True
        return False

    # check if we pass the threshold and return the adjusted P
    def sick_chance(self):
        if 100 * (self.board.num_sick / self.board.num_residences) > T:
            return P2
        else:
            return P1

    # make all the residences older in one generation
    def get_all_older(self):
        self_matrix = self.board.matrix
        for i in range(self_matrix.shape[0]):
            for j in range(self_matrix.shape[1]):
                if self_matrix[i][j].isFull:
                    self_matrix[i][j].content.get_older()


# this function will run a simulation and 
# create visualization for the using pygame
def run_and_show_Simulation(simulation):
    # Define some colors
    WHITE = (255, 255, 255)  # empty cell
    BLUE = (0, 0, 128)
    GREEN = (0, 255, 0)  # healthy/regular people
    RED = (255, 0, 0)  # sick person
    # each cell's size
    WIDTH = 3
    HEIGHT = 2
    MARGIN = 1

    pygame.init()

    # Set the width and height of the screen [width, height]
    size = [1200, 800]
    flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.FULLSCREEN
    screen = pygame.display.set_mode(size, flags)
    # set the tittle of the game
    pygame.display.set_caption("WRAP_AROUND Covid-19 automate:")

    # Loop until the user clicks the close button or Esc or no more sick organisms.
    done = False
    # Used to manage how fast tpython setup.py installhe screen updates
    clock = pygame.time.Clock()
    # define all texts in the simulation graphic
    font_legend = pygame.font.SysFont('comicsans', 18)
    font = pygame.font.SysFont('comicsans', 20)
    leg_hea = font_legend.render('Healthy', True, GREEN, BLUE)
    leg_infe = font_legend.render('Infected', True, RED, BLUE)
    leg_emp = font_legend.render('Empty Cell', True, WHITE, BLUE)
    text_num_crea = font.render('Number of Creatures: ' + str(simulation.board.num_residences), True, BLUE, WHITE)
    text_exit = font.render('press Esc to exist', True, WHITE, BLUE)
    leg_heaRect = leg_hea.get_rect()
    leg_heaRect.center = (865, 80)
    leg_infeRect = leg_infe.get_rect()
    leg_infeRect.center = (865, 120)
    leg_empRect = leg_emp.get_rect()
    leg_empRect.center = (865, 160)
    text_num_creaRect = text_num_crea.get_rect()
    text_num_creaRect.center = (1000, 380)
    text_exitRect = text_exit.get_rect()
    text_exitRect.center = (600, 650)
    # # -------- Main Program Loop -----------
    # while not done:
    # --- Main event loop
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            # use esc button to exit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

        # define the text which change on the main loop
        text_num_infe = font.render('Number of infected: ' + str(simulation.board.num_sick), True, RED, WHITE)
        text_num_gene = font.render('Number of Generation: ' + str(simulation.generation), True, BLUE, WHITE)
        text_num_infeRec = text_num_infe.get_rect()
        text_num_infeRec.center = (1000, 480)
        text_num_geneRect = text_num_gene.get_rect()
        text_num_geneRect.center = (1000, 580)
        # draw the background and all the texts
        screen.fill((40, 20, 128))
        screen.blit(leg_hea, leg_heaRect)
        screen.blit(leg_infe, leg_infeRect)
        screen.blit(leg_emp, leg_empRect)
        screen.blit(text_num_crea, text_num_creaRect)
        screen.blit(text_num_infe, text_num_infeRec)
        screen.blit(text_num_gene, text_num_geneRect)
        screen.blit(text_exit, text_exitRect)
        # Draw the matrix according to cell situation
        for row in range(matrix_size[0]):
            for column in range(matrix_size[1]):
                color = WHITE
                c = simulation.board.matrix[row][column]
                if c.isFull:
                    if c.content.isSick:
                        color = RED
                    else:
                        color = GREEN
                pygame.draw.rect(screen,
                                 color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN,
                                  WIDTH,
                                  HEIGHT])

        pygame.display.flip()

        # --- Limit to 1 frame per second
        clock.tick(1)
        simulation.next_genartion()
        if simulation.board.num_sick < 1:
            done = True

    # Close the window and quit.
    pygame.quit()


# this function create an input window for simulation parameters with tkinter library.
# it fills the values by default the variables as they were defined on the top of this script.
def getInput():
    window = Tk()
    window.title("Simulation Parameters")
    main_lst = []
    label1 = Label(window, text="D(fraction of sick): ", padx=20, pady=10)
    label2 = Label(window, text="R(fraction of fast): ", padx=20, pady=10)
    label3 = Label(window, text="N(num of creatures): ", padx=20, pady=10)
    label4 = Label(window, text="P1 (Infection chance before T threshold(fraction)): ", padx=20, pady=10)
    label5 = Label(window, text="T(threshold(percent of people (Integer))) ", padx=20, pady=10)
    label6 = Label(window, text="P2 (Infection chance after threshold(fraction)): ", padx=20, pady=10)
    label7 = Label(window, text="X(num of genertion until recoverd): ", padx=20, pady=10)
    d = Entry(window, width=30, borderwidth=5)
    d.insert(END, str(D))
    r = Entry(window, width=30, borderwidth=5)
    r.insert(END, str(R))
    n = Entry(window, width=30, borderwidth=5)
    n.insert(END, str(N))
    p1 = Entry(window, width=30, borderwidth=5)
    p1.insert(END, str(P1))
    t = Entry(window, width=30, borderwidth=5)
    t.insert(END, str(T))
    p2 = Entry(window, width=30, borderwidth=5)
    p2.insert(END, str(P2))
    x = Entry(window, width=30, borderwidth=5)
    x.insert(END, str(X))
    Exit = Button(window, text="Start simulation", padx=20, pady=10, command=window.quit)
    label1.grid(row=0, column=0)
    label2.grid(row=1, column=0)
    label3.grid(row=2, column=0)
    label4.grid(row=3, column=0)
    label5.grid(row=4, column=0)
    label6.grid(row=5, column=0)
    label7.grid(row=6, column=0)
    d.grid(row=0, column=1)
    r.grid(row=1, column=1)
    n.grid(row=2, column=1)
    p1.grid(row=3, column=1)
    t.grid(row=4, column=1)
    p2.grid(row=5, column=1)
    x.grid(row=6, column=1)
    Exit.grid(row=10, column=0, columnspan=2)
    window.mainloop()
    try:
        # convert the input values from string to numeric values (int or float)
        values = [float(d.get()), float(r.get()), int(n.get()), float(p1.get()), int(t.get()), float(p2.get()),
                int(x.get())]
    except: # if its faild to convert that means  Input not valid - show message and exit
        messagebox.showwarning("WRAP_AROUND Covid-19 automate", "ERRRORRRR!!!!! Input not valid")
        sys.exit(-1)
    # check every fraction/float input, if its unvalid - show message and exit
    for value in values: 
        if isinstance(value, float):
            if not (0 <= value <=1):
                messagebox.showwarning("WRAP_AROUND Covid-19 automate", "ERRRORRRR!!!!! Input not valid")
                sys.exit(-1)
    # check the numeric values, if its unvalid - show message and exit
    if values[2] < 0 or values[4] < 0 or values[6] < 0: 
        messagebox.showwarning("WRAP_AROUND Covid-19 automate", "ERRRORRRR!!!!! Input not valid")
        sys.exit(-1)
    #  if total number of creaturs is more then the matrix size - show message and exit
    if values[2] > matrix_size[0] * matrix_size[1]:
        messagebox.showwarning("WRAP_AROUND Covid-19 automate", "ERRRORRRR!!!!! too much creaturs")
        sys.exit(-1)        
    window.destroy()
    window.quit()
    return values


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # run loop for the simulation for multy simualtion in case the user wannt more then one 
    stop = False
    while not stop:
        # get hyper-parameters for the simulation
        inputInfo = getInput()
        # set the global variables as the values of user's input
        D = inputInfo[0]
        R = inputInfo[1]
        N = inputInfo[2]
        P1 = inputInfo[3]
        T = inputInfo[4]
        P2 = inputInfo[5]
        X = inputInfo[6]
        # show messages
        messagebox.showinfo("WRAP_AROUND Covid-19 automate", "The data was added successfully")
        messagebox.showwarning("WRAP_AROUND Covid-19 automate", "The simulation can take some time, Enjoy!!")
        newBoard = Board()
        newBoard.add_N_of_residences_randomly(int(N))
        # crate the Simulation
        simulation = Simulation(newBoard)
        # run the Simulation
        run_and_show_Simulation(simulation)
        ask_for_retry = messagebox.askyesno("WRAP_AROUND Covid-19 automate", "Simulation was over\n"
                                                                             "Do you want to start the simulation again?")
        if not ask_for_retry:
            stop = True

    # code for plots
    # import matplotlib.pyplot as plt
    # newBoard = Board()
    # newBoard.add_N_of_residences_randomly(int(N))
    # #     # crate the Simulation
    # simulation = Simulation(newBoard)
    # done = False
    # sick_history = []
    # while not done:
    #     print(simulation.generation)
    #     print(simulation.sick_chance())
    #     sick_history.append(simulation.board.num_sick)
    #     simulation.next_genartion()
    #     # print(simulation.board)
    #     # if simulation.generation%5 ==0:
    #     # plt.plot(range(simulation.generation), sick_history)
    #     # plt.show()
    #     if simulation.board.num_sick < 10:
    #         done = True
    
    # plt.plot(range(simulation.generation), sick_history)
    # plt.xlabel("Number of Generations")
    # plt.ylabel("Number of sick organisms")
    # plt.title(
    #     "N = " + str(N) + ", R = " + str(int(R * 100)) + "%, D = " + str(int(D * 100)) + "%, P1 = " + str(P1) + ", P2 = " +
    #     str(P2) + ", T = " + str(T) + "%, X = " + str(X))
    # plt.show()
