import random
import itertools
import numpy as np


def random_pairs(number_list):
    return [number_list[i] for i in random.sample(range(len(number_list)), 2)]


def generate_all_idx():
    numbers = [x for x in range(200)]
    pairs = []
    counter = N
    while counter > 0:
        pair = random_pairs(numbers)
        if pair not in pairs:
            pairs.append(pair)
            print("add new")
            counter -= 1
        else:
            print(str(pair) + " is already exist")
    print(pairs)


if __name__ == '__main__':
    lrr = [[1,4]]
    print(lrr)
    lrr =[[2,2]]
    print(lrr)
    # generate_all_idx()

# import pygame
# import sys
#
# SCREEN_WIDTH = 800
# SCREEN_HEIGHT = 800
# BLOCK_SIZE = 32
# WHITE = (255,255,255)
#
# pygame.init()
#
# frame = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption("PathFinder")
#
# # create list with all rects
# all_rects = []
# for y in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
#     row = []
#     for x in range(0, SCREEN_WIDTH, BLOCK_SIZE):
#         rect = pygame.Rect(x, y, BLOCK_SIZE-1, BLOCK_SIZE-1)
#         row.append([rect, (0, 255, 0)])
#     all_rects.append(row)
#
# while True:
#
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()
#         elif event.type == pygame.MOUSEBUTTONDOWN:
#             # check which rect was clicked and change its color on list
#             for row in all_rects:
#                 for item in row:
#                     rect, color = item
#                     if rect.collidepoint(event.pos):
#                         if color == (0, 255, 0):
#                             item[1] = (255, 0, 0)
#                         else:
#                             item[1] = (0, 255, 0)
#
#     # draw all in every loop
#
#     frame.fill(WHITE)
#
#     for row in all_rects:
#         for item in row:
#             rect, color = item
#             pygame.draw.rect(frame, color, rect)
#
#     pygame.display.flip()
