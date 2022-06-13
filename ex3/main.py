"""
Gil Diamant
Itamar Twersky
"""

from time import sleep
import pandas as pd
import sys
import os
import pygame
from math import sin, cos, pi, floor, ceil
from tkinter import *
from tkinter import messagebox
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# these are all the global variables which define in the instructions of this exercise:

# how many rows and columns of clusters
dim_size = 9
#
same_iterations_to_converge = 10
# max iterations to run if there is no converge
max_iterations = 100
num_of_trains = 10
alpha = 0.1
# how many neighbors rings around the sample's cluster to update
num_of_rings = 3

color_label = "Economic Cluster"


def fn_dist(dist):
    return 1 if dist == 0 else 1 / (dist * 10)


class Som_model:
    def __init__(self, csv_name):
        df = pd.read_csv(csv_name)
        # save names and feuters
        self.cities_names = df.iloc[:, 0].tolist()
        self.features = df.iloc[:, 1:]
        self.samples = self.features.to_numpy(dtype='float64')
        self.min_max = self.min_max()
        self.clusters = []
        self.map = [None] * self.samples.shape[0]

    def min_max(self):
        min_max = []
        for col in self.features:
            # get the maximum & minimum values of this column

            # TODO check median 
            col_med = self.features[col].median()
            c_min = max([0, self.features[col].min()])
            c_max = min([round(col_med + col_med), self.features[col].max()])
            min_max.append((c_min, c_max))
        return min_max

    def create_cluster(self):
        cluster = []
        for idx, col in enumerate(self.features.columns):
            cluster.append(float(np.random.randint(self.min_max[idx][0], self.min_max[idx][1])))
        return np.asarray(cluster)

    # create 2D list with rows with different length for each one - like on the hexagon grid
    def init_array_of_clusters(self):
        for i in range(9):
            new_row = []
            # for getting different length for every row
            cells_num = dim_size - abs(4 - i)
            for j in range(cells_num):
                new_row.append(self.create_cluster())
            self.clusters.append(new_row)

    # create 2D list with rows with different length for each one - like on the hexagon grid
    '''def init_array_of_clusters(self):
        self.clusters = []
        for i in range(9):
            new_row = []
            # for getting different length for every row
            cells_num = 9 - abs(4 - i)
            for j in range(cells_num):
                new_row.append(np.random.uniform(size=self.features.shape[1]))
            self.clusters.append(new_row)'''

    def train(self):
        self.normalize()
        same_map_iter = 0
        # run 10000 times
        for i in range(max_iterations):
            # print(i)
            prev_map = self.map.copy()
            self.map_sampels()
            # if the map didnt change for several times - its converge and stop train
            if prev_map == self.map:
                # print("AAA")
                same_map_iter += 1
                if same_map_iter == same_iterations_to_converge:
                    # print("BBB")
                    # print(len(set(self.map)))
                    break

    def normalize(self):
        min_max_s = MinMaxScaler()
        self.samples = min_max_s.fit_transform(self.samples)
        for i in range(len(self.clusters)):
            for j in range(len(self.clusters[i])):
                a = min_max_s.transform(self.clusters[i][j].reshape(1, -1))
                self.clusters[i][j] = a
                # a = a.reshape(-1)

    # will map sampels to clusters
    def map_sampels(self, update_wights=True):
        for idx in range(self.samples.shape[0]):
            sample = self.samples[idx,]
            # get closest clusters and update map
            clusters_idx, best_dist = self.get_2_closeset_clusters(sample)
            self.map[idx] = clusters_idx[0]
            # update wights
            if update_wights:
                self.update_clusters_wights(sample, clusters_idx[0])

    # will get 2 most close clusters and the best distance
    def get_2_closeset_clusters(self, sample):
        best_dist = np.inf
        # will save the two 2d indexes
        best_idxes = [None, None]
        # calculate distance for every cluster, if its new best distance - save itand indexes
        for i in range(len(self.clusters)):
            for j in range(len(self.clusters[i])):
                dist = np.linalg.norm(sample - self.clusters[i][j])
                if dist < best_dist:
                    best_dist = dist
                    best_idxes[1] = best_idxes[0]
                    best_idxes[0] = tuple([i, j])

        return best_idxes, best_dist

    # will update clusters wights according to maped sample
    def update_clusters_wights(self, sample, cluster_idx):
        # the 0 ring is the best cluste
        neighbors_rings = [[cluster_idx]]
        # add the other neighboors rings
        neighbors_rings.extend(self.get_neighbors(cluster_idx))
        # for every ring update clustr according to formula and distance
        for dist, ring in enumerate(neighbors_rings):
            for clusIdx in ring:
                cluster = self.clusters[clusIdx[0]][clusIdx[1]]
                cluster += alpha * fn_dist(dist) * (sample - cluster)

    # will the neighbors of cluster
    # will return as list of rings
    # every ring is a list of neighbors with the same distace from the cluster
    # number of rings will determind by global variable
    def get_neighbors(self, cluster_idx):
        # every ring is a set of nighbors
        rings_list = []
        # for preventing of neighbor to apear more then once
        all_neighbors = set()
        all_neighbors.add(cluster_idx)
        new_ring = [cluster_idx]
        # add neighbors of neighbors recursivly without duplictions
        for ring_n in range(num_of_rings):
            temp = set()
            [temp.update(self.get_6_neighbors(clus)) for clus in new_ring]
            new_ring = [neig for neig in temp if neig not in all_neighbors]
            all_neighbors.update(new_ring)
            rings_list.append(new_ring)
        return rings_list

    # will get list of indexes  of the 6 cluster neighbors
    def get_6_neighbors(self, cluster_idx):
        neighb = []
        middle_line = floor(len(self.clusters) / 2)
        # add left and right neighbors
        neighb.extend([tuple([cluster_idx[0], cluster_idx[1] + 1]), tuple([cluster_idx[0], cluster_idx[1] - 1])])
        # take care of dfferent indexes situation 
        if cluster_idx[0] == middle_line:
            # add upper neighbors 
            neighb.extend(
                [tuple([cluster_idx[0] - 1, cluster_idx[1] - 1]), tuple([cluster_idx[0] - 1, cluster_idx[1]])])
            # add lower neighbors
            neighb.extend(
                [tuple([cluster_idx[0] + 1, cluster_idx[1] - 1]), tuple([cluster_idx[0] + 1, cluster_idx[1]])])
        elif cluster_idx[0] < middle_line:
            # add upper neighbors 
            neighb.extend(
                [tuple([cluster_idx[0] - 1, cluster_idx[1] - 1]), tuple([cluster_idx[0] - 1, cluster_idx[1]])])
            # add lower neighbors
            neighb.extend(
                [tuple([cluster_idx[0] + 1, cluster_idx[1]]), tuple([cluster_idx[0] + 1, cluster_idx[1] + 1])])

        else:  # cluster_idx[0] > middle_line
            # add upper neighbors 
            neighb.extend(
                [tuple([cluster_idx[0] - 1, cluster_idx[1]]), tuple([cluster_idx[0] - 1, cluster_idx[1] + 1])])
            # add lower neighbors
            neighb.extend(
                [tuple([cluster_idx[0] + 1, cluster_idx[1] - 1]), tuple([cluster_idx[0] + 1, cluster_idx[1]])])
        neighb = self.remove_unvalid(neighb)
        return neighb

    # will remove unvalid indexes from list of clusters indexes
    def remove_unvalid(self, clusters_idx_list):
        valid_list = []
        for idxes in clusters_idx_list:
            # if not row or coll is out of bund
            if not (idxes[0] < 0 or idxes[0] >= len(self.clusters) or idxes[1] < 0 or idxes[1] >= len(
                    self.clusters[idxes[0]])):
                valid_list.append(idxes)
        return valid_list

    def create_color(self, range_num, all_max):
        range_num = int(range_num)
        if range_num == 0:
            color = (255, 255, 255)
            return color
        range_list = [i for i in range(1, 10)]
        tmp_divide = (range_num / all_max)
        if range_num in range_list[:3]:
            color = (255, ceil(20 * (tmp_divide)), ceil(120 * (tmp_divide)))
        elif range_num in range_list[3:6]:
            color = (ceil(20 * (tmp_divide)), 255, ceil(120 * (tmp_divide)))
        elif range_num in range_list[6:9]:
            color = (ceil(20 * (tmp_divide)), ceil(120 * (tmp_divide)), 255)
        return color

    def initial_colors(self):
        tmp_economic_per_cluster = []
        counter = []
        labels = self.features[color_label].tolist()
        for i in range(9):
            zero_row = []
            for i in range(len(self.clusters[i])):
                zero_row.append([0, 0])
            tmp_economic_per_cluster.append(zero_row)
        for idx, cluster_idx in enumerate(self.map):
            r_idx = cluster_idx[0]
            c_idx = cluster_idx[1]
            test = tmp_economic_per_cluster[r_idx][c_idx]
            tmp_economic_per_cluster[r_idx][c_idx][0] += labels[idx]
            # count the num of times that we add value
            tmp_economic_per_cluster[r_idx][c_idx][1] += 1
        avg_economic = []
        all_avg = []
        for row in tmp_economic_per_cluster:
            tmp_row = []
            for col in row:
                if col[0] == 0:
                    tmp_row.append(0)
                    continue
                tmp_row.append(col[0] / col[1])
                all_avg.append(col[0] / col[1])
            avg_economic.append(tmp_row)
        # all_max = max(labels)
        all_max = max(all_avg)
        colors = []
        for row in avg_economic:
            tmp_row = []
            for col in row:
                tmp_row.append(self.create_color(col, all_max))
            colors.append(tmp_row)
        return colors


# function to draw single hexagon
def draw_hexagon_cell(surface, color,
                      radius, position):
    n, r = 6, radius
    x, y = position
    pygame.draw.polygon(surface, color, [
        (x + r * sin(2 * pi * i / n),
         y + r * cos(2 * pi * i / n))
        for i in range(n)
    ])


# function to draw one board of big hexagon which contain 61 hexagons
def draw_board(model):
    # black background
    bg_color = (0, 0, 0)

    w, h = 600, 600
    colors = model.initial_colors()

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
                y = init_y + 1.6 * i * radius
                # create random color
                color = colors[i][j]
                draw_hexagon_cell(root, color, radius, (x, y))

        pygame.display.flip()
        sleep(1)


if __name__ == '__main__':
    our_model = Som_model("Elec_24.csv")
    # create random clusters
    our_model.init_array_of_clusters()
    # train model
    our_model.train()
    print(len(set(our_model.map)))
    # print(our_model.initial_colors())
    draw_board(our_model)
    # our_model.train_n_times(10)
    # draw_board(our_model)

    # TODO  add option to select color by label (color should be the avarge value of residents(why not 
    # the inner vector value))
    # TODO add option to do multiple training and choose the best(distance and topological),
    #  give normaliztion/wight to every method.
    # method implentation:
    # distance:
    # for every sample calculate distance between it and its cluter's inner vector
    # sum all distances
    # topological:
    # for every sample calculate two closest cluster, check distance between them
    # sum distances
