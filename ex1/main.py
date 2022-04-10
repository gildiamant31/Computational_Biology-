import pygame
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

matrix_size = (200, 200)
D = 0.5  # % of initial sick people
N = (200 * 200) * (70 / 100)  # initial number of people in the module - start with 70% of the automate size
M = 1 / 9  # possibility of moving to each near cell (or staying in place) in the matrix
R = 0.05  # % of faster people
P1 = 0.2  # possibility of infect close people
T = 20  # percentage of sick people threshold which after it P var is going down
P2 = 0.1  # possibility of infect close people when we pass the threshold (T var)
X = 5  # number of generation for being weak and infect other people.
directions = ['up', 'down', 'right', 'left', 'bootomright', 'bootomleft', 'upright', 'upleft', 'stay']


class Yetzur:
    def __init__(self, location, stats="H", isFast=False):
        assert stats in ["H", "S", "R"], str(
            stats) + "Yetzur can be is stats of H/S/R only"  # H/S/R for healthy/Sick/Recoverd
        self.isHealthy = self.isSick = self.isRecovered = False
        if stats == "H":
            self.isHealthy = True
        if stats == "S":
            self.isSick = True
        if stats == "R":
            self.isRecovered = True
        self.isFast = isFast  # true or false , tell us if this object can move 10 cells in one direction
        self.location = location

    def get_sick(self):
        self.isHealthy = self.isRecovered = False
        self.isSick = True

    def get_recovered(self):
        self.isHealthy = self.isSick = False
        self.isRecovered = True

    def next_location(self):
        if not self.isFast:
            return  random.choice(self.get_neghibors_and_self_indexes())
        else:
            vert_change = random.choice(range(-10, 11))
            new_vert = (self.location[0] + vert_change) % matrix_size[0]
            horiz_change = random.choice(range(-10, 11))
            new_horiz = (self.location[1] + horiz_change) % matrix_size[1]
            return tuple([new_vert, new_horiz])

    def get_neghibors_and_self_indexes(self):
        return [tuple([(i + self.location[0]) % matrix_size[0], (j + self.location[1]) % matrix_size[1]]) for i in
                range(-1, 2) for j in range(-1, 2)]


class Cell:
    def __init__(self, isFull=False, content=None):
        self.isFull = isFull
        self.content = content

    def add_content(self, new_yetzur):
        assert not self.isFull, "tried to fill filled cell"
        self.content = new_yetzur
        self.isFull = True

    def remove_content(self):
        assert self.isFull, "cant clear empty cell"
        self.isFull = False
        old_cont = self.content
        self.content = None
        return old_cont


class Board:
    def __init__(self, size=matrix_size):
        self.matrix = np.full(matrix_size, Cell())

    # TODO update residence location
    def add_residence_to(self,residence, new_location):
        if self.matrix[new_location[0],new_location[1]].isFull:
            return False
        else:
            self.matrix[new_location[0],new_location[1]].add_content(residence)
            return True

    def remove_residence_from(self,location):
        return self.matrix[location[0],location[1]].remove_content()

    def add_residence_randomly(self):
        while(not self.add_residence_to(((random.choice(range(0,matrix_size[0])),random.choice(range(0,matrix_size[1])))))):
            pass

    def add_N_of_residence_randomly(self, N):
        for i in range(N):
            self.add_residence_randomly()


# check if we pass the threshold and return the adjusted P
def passThreshold(sickPercentage):
    if sickPercentage > T:
        return P2
    else:
        return P1


# meanwhile this graphic doesn't related to the exercise - only played with it.
def show_Simulation():
    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)  # empty cell
    GREEN = (0, 255, 0)  # healthy/regular people
    RED = (255, 0, 0)  # sick person
    # each cell's size
    WIDTH = 2
    HEIGHT = 2
    MARGIN = 1
    # set the array
    grid = []
    for row in range(199):
        grid.append([])
        for column in range(199):
            grid[row].append(0)
    # Set row 1, cell 5 to one. (Remember rows and
    # column numbers start at zero.)
    grid[0][0] = 1

    pygame.init()

    # Set the width and height of the screen [width, height]
    size = [610, 700]
    screen = pygame.display.set_mode(size)
    # set the tittle of the game
    pygame.display.set_caption("WRAP_AROUND Covid-19 automate:")

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # -------- Main Program Loop -----------
    # while not done:
    # --- Main event loop
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # User clicks the mouse. Get the position
                pos = pygame.mouse.get_pos()
                # Change the x/y screen coordinates to grid coordinates
                column = pos[0] // (WIDTH + MARGIN)
                row = pos[1] // (HEIGHT + MARGIN)
                # Set that location to one
                grid[row][column] = 1
                print("Click ", pos, "Grid coordinates: ", row, column)

        # --- Game logic should go here

        # --- Screen-clearing code goes here

        # Here, we clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.

        # If you want a background image, replace this clear with blit'ing the
        # background image.
        screen.fill(BLACK)

        # --- Drawing code should go here
        # Draw the grid
        for row in range(199):
            for column in range(199):
                color = WHITE
                if grid[row][column] == 1:
                    color = GREEN
                pygame.draw.rect(screen,
                                 color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN,
                                  WIDTH,
                                  HEIGHT])

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # --- Limit to 60 frames per second
        clock.tick(60)

    # Close the window and quit.
    pygame.quit()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # get random initial indexes for start of simulation
    # np.random.seed(0)  # to get difference random index - mabey we can remove it
    i = np.random.choice(199, int(N))
    j = np.random.choice(199 - 1, int(N))
    j[j >= i] += 1
    print(np.any(i == j))
    # False
    ij = np.stack([i, j], axis=1)
    counter_fast_Y = 0
    counter_sick_Y = 0
    people_list = []
    for i in range(int(N)):
        newY = Yetzur(matrix, ij[i])
        # make some of them fast
        if counter_fast_Y < (R * N):
            newY.isFast = True
            counter_fast_Y += 1
        # make some of them infected in covid-19
        if counter_sick_Y < (D * N):
            newY.isInfected = True
            counter_sick_Y += 1
        people_list.append(newY)
    print(matrix[newY.location[0]][newY.location[1]])
    # show_Simulation()
