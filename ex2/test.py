from math import sin, cos, pi

import numpy as np
import pygame


def draw_regular_polygon(surface, color, vertex_count,
                         radius, position):
    n, r = vertex_count, radius
    x, y = position
    pygame.draw.polygon(surface, color, [
        (x + r * cos(2 * pi * i / n),
         y + r * sin(2 * pi * i / n))
        for i in range(n)
    ])


bg_color = (0, 0, 0)
fg_color = (0, 255, 255)

w, h = 800, 800
vertex_count = 6
width = 1

pygame.init()
root = pygame.display.set_mode((w, h))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        # Use UP / DOWN arrow keys to
        # increase / decrease the vertex count
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                vertex_count += 1
            elif event.key == pygame.K_DOWN:
                vertex_count = max(3, vertex_count - 1)

    root.fill(bg_color)
    row_len = 5
    decrease = False
    radius = min(w, h) / 100
    init_x = w / 2
    init_y = h / 2 +50
    for i in range(9):
        cells_num = 9 - abs(4 - i)
        for j in range(cells_num):
            extra_width = radius * (9 - cells_num)
            x = init_x + extra_width + j * 2 * radius
            y = init_y + 1.5 * i * radius
            color = tuple(np.random.randint(256, size=3))
            draw_regular_polygon(root, color, vertex_count, radius, (x, y))


    pygame.display.flip()