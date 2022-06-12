"""
Gil Diamant
Itamar Twersky
"""

from hashlib import new
from re import I
from time import sleep
import pandas as pd
import sys
import os
import pygame
from math import sin, cos, pi
from tkinter import *
from tkinter import messagebox
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math


# these are all the global variables which define in the instructions of this exercise.
same_itrations_to_converge = 10
num_of_trains = 10
alpha = 0.1
# how many neighbors rings around the sample's cluster to update
num_of_rings = 3

def fn_dist(dist):
    return 1 if dist == 0 else 1/dist*10

class Som_model:
    def __init__(self, csv_name):
        df = pd.read_csv(csv_name)
        # save names and feuters
        self.cities_names = df.iloc[:,1].tolist()
        self.features = df.iloc[:,1:]
        self.samples = self.features.to_numpy()
        # normilize sampels to prevent scale impact on the results
        min_max_scaler = MinMaxScaler()
        self.samples = min_max_scaler.fit_transform(self.samples)
        # round numbers to reduce running time
        # TODO check impact on running time
        # self.samples = np.around(self.samples, 6)
        self.clusters = []
        self.map = []

    # def min_max(self):
    #     min_max = []
    #     for col in self.features:
    #         # get the maximum & minimum values of this column
    #         min_max.append((self.features[col].min(), self.features[col].max()))
    #     return min_max

    # create 2D list with rows with different length for each one - like on the hexagon grid
    def init_array_of_clusters(self):
        self.clusters = []
        for i in range(9):
            new_row = []
            # for getting different length for every row
            cells_num = 9 - abs(4 - i)
            for j in range(cells_num):
                new_row.append(np.random.uniform(size=self.features.shape[1]))
            self.clusters.append(new_row)


    
    def train(self):
        same_map_iter = 0 
        # run 10000 times
        for i in range(10000):
            prev_map = self.map.copy()
            self.map_sampels()
            # if the map didnt change for several times - its converge and stop train
            if prev_map == self.map:
                same_map_iter += 1
                if same_map_iter == same_itrations_to_converge:
                    break
   
    # will map sampels to clusters
    def map_sampels(self, update_wights=True):
        for idx, sample in np.ndenumerate(self.samples):
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
                for j in range(len(self.clusters[i]))
                    dist = np.linalg.norm(sample - self.clusters[i][j])
                    if dist < best_dist:
                        best_dist = dist
                        best_idxes[1] = best_idxes[0]
                        best_idxes[0] = tuple([i, j])

            return best_idxes ,best_dist

        

    # will update clusters wights according to maped sample
    def update_clusters_wights(self, sample, cluster_idx):
        # the 0 ring is the best cluste
        neighbors_rings = [cluster_idx]
        # add the other neighboors rings
        neighbors_rings.extend(self.get_neighbors(cluster_idx))
        # for every ring update clustr according to formula and distance
        for dist, ring in enumerate(neighbors_rings):
            for clusIdx in ring:
                cluster = self.clusters[clusIdx[0]][clusIdx[1]]
                cluster += alpha*fn_dist(dist)*(sample - cluster)

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
        new_ring = cluster_idx
        # add neighbors of neighbors recursivly without duplictions
        for ring_n in range(num_of_rings):
            temp = []
            [temp.extend(self.get_6_neighbors(clus)) for clus in new_ring]
            new_ring = [neig for neig in temp if neig not in all_neighbors]
            all_neighbors.update(new_ring)
            rings_list.append(new_ring)
            

    # will get list of indexes  of the 6 cluster neighbors
    def get_6_neighbors(self, cluster_idx):
        neighb = []
        middle_line = math.ceil(len(self.clusters)/2)
        # add left and right neighbors
        neighb.extend([tuple([cluster_idx[0],cluster_idx[1]+1]), tuple([cluster_idx[0],cluster_idx[1]-1])])
        # take care of dfferent indexes situation 
        if cluster_idx[0] == middle_line:
            # add upper neighbors 
            neighb.extend([tuple([cluster_idx[0]-1,cluster_idx[1]-1]),tuple([cluster_idx[0]-1,cluster_idx[1]])])
            # add lower neighbors
            neighb.extend([tuple([cluster_idx[0]+1,cluster_idx[1]-1]),tuple([cluster_idx[0]+1,cluster_idx[1]])])
        elif cluster_idx[0] < middle_line:
            # add upper neighbors 
            neighb.extend([tuple([cluster_idx[0]-1,cluster_idx[1]]),tuple([cluster_idx[0]-1,cluster_idx[1]+1])])
            # add lower neighbors
            neighb.extend([tuple([cluster_idx[0]+1,cluster_idx[1]-1]),tuple([cluster_idx[0]+1,cluster_idx[1]])])
        
        else: #  cluster_idx[0] > middle_line
            # add upper neighbors 
            neighb.extend([tuple([cluster_idx[0]-1,cluster_idx[1]-1]),tuple([cluster_idx[0]-1,cluster_idx[1]])])
            # add lower neighbors
            neighb.extend([tuple([cluster_idx[0]+1,cluster_idx[1]]),tuple([cluster_idx[0]+1,cluster_idx[1]+1])])
        neighb = self.remove_unvalid(neighb)
        return neighb
            
    # will remove unvalid indexes from list of clusters indexes
    def remove_unvalid(self, clusters_idx_list):
        # will save which indexes unvalid
        to_del = []
        for i, idxes in enumerate(clusters_idx_list):
            # row or coll is out of bund
            if idxes[0] < 0 or idxes[0] >= len (self.clusters) or idxes[1] < 0 or idxes[1] > len (self.clusters[idxes[0]]):
                to_del.append(i)
        for unval_idx in to_del:
            del clusters_idx_list[unval_idx]
        return clusters_idx_list



            


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
                y = init_y + 1.6 * i * radius
                # create random color
                color = tuple(np.random.randint(256, size=3))
                draw_hexagon_cell(root, color, radius, (x, y))

        pygame.display.flip()
        sleep(1)


if __name__ == '__main__':
    our_model = Som_model("ex3/Elec_24.csv")
    # create random clusters
    our_model.init_array_of_clusters()
    # train model
    our_model.train()

    # our_model.train_n_times(10)
    # our_model.draw_board()

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