"""
Gil Diamant
Itamar Twersky
"""
import os
from time import sleep
import pandas as pd
import pygame
from math import sin, cos, pi, floor, ceil
from tkinter import *
from tkinter import messagebox
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# these are all the global variables which define in the instructions of this exercise:

# how many rows and columns of clusters
dim_size = 9

same_iterations_to_converge = 10
# max iterations to run if there is no converge
max_iterations = 100
alpha = 0.4
# how many neighbors rings around the sample's cluster to update
num_of_rings = 3
# how much models do we want to check
num_of_models = 10

# default labels
color_label = "Economic Cluster"


def fn_dist(dist):
    return 1 if dist == 0 else 1 / (dist * 5)


class Som_model:
    def __init__(self, csv_name):
        df = pd.read_csv(csv_name)
        # save names and feuters
        self.cities_names = df.iloc[:, 0].tolist()
        self.features = df.iloc[:, 1:]
        self.samples = self.features.to_numpy(dtype='float64')
        self.min_max = self.min_max()
        self.clusters = []
        # will save the cluster index for every sample which mapped to it
        self.map = [None] * self.samples.shape[0]

    def min_max(self):
        min_max = []
        for col in self.features:
            # get the maximum & minimum values of this column
            col_med = self.features[col].median()
            c_min = max([0, self.features[col].min()])
            c_max = min([round(col_med + col_med), self.features[col].max()])
            min_max.append((c_min, c_max))
        return min_max

    # create single cluster which contain one vector with random values using min and max dataset ranges
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

    # "train" the mosel
    def train(self):
        self.normalize()
        same_map_iter = 0
        # run 10000 times
        for i in range(max_iterations):
            prev_map = self.map.copy()
            self.map_sampels()
            # if the map didnt change for several times - its converge and stop train
            if prev_map == self.map:
                same_map_iter += 1
                if same_map_iter == same_iterations_to_converge:
                    break

    # normalization function using built in funnction from sklearn
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
        sec_dist = best_dist - 1
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
                elif dist < sec_dist and best_idxes[1] is None:
                    best_idxes[1] = tuple([i, j])

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

    # create one RGB color
    def create_color(self, range_num, all_max):
        range_num = int(range_num)
        if range_num == 0:
            color = (255, 255, 255)
            return color
        range_list = [i for i in range(1, 10)]
        tmp_divide = (range_num / all_max)
        # separate it to 3 main colors - red, blue & green.
        if range_num in range_list[:3]:
            color = (255, ceil(20 * (tmp_divide)), ceil(120 * (tmp_divide)))
        elif range_num in range_list[3:6]:
            color = (ceil(20 * (tmp_divide)), 255, ceil(120 * (tmp_divide)))
        elif range_num in range_list[6:9]:
            color = (ceil(20 * (tmp_divide)), ceil(120 * (tmp_divide)), 255)
        # for labels which aren't the economic labels:
        else:
            color = (ceil(250 * (tmp_divide)), ceil(100 * (tmp_divide)), ceil(250 * (tmp_divide)))
        return color

    # create list of rgb colors
    def initial_colors(self):
        tmp_economic_per_cluster = []
        labels = self.features[color_label].tolist()
        # initial zero matrix
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
        # create list of average value of each cluster
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
        # generate the colors list
        for row in avg_economic:
            tmp_row = []
            for col in row:
                tmp_row.append(self.create_color(col, all_max))
            colors.append(tmp_row)
        return colors


# function to draw single hexagon with pygame
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
    if color_label == "Economic Cluster":
        # black background
        bg_color = (0, 0, 0)
    else:
        bg_color = (255, 255, 0)
    w, h = 600, 600
    colors = model.initial_colors()

    pygame.init()
    root = pygame.display.set_mode((w, h))
    pygame.display.set_caption("SOM map colored by: " + color_label)

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


# calculate all topological(physical) mistakes for all tested models
def get_topological_distance(models):
    distances = []
    for model in models:
        distance_sum = 0
        for sample in model.samples:
            best_idxes, best_dist = model.get_2_closeset_clusters(sample)
            horizontal_dist = abs(best_idxes[0][0] - best_idxes[1][0])
            vertical_dist = abs(best_idxes[0][1] - best_idxes[1][1])
            distance_sum += (horizontal_dist + vertical_dist)
        distances.append(distance_sum)
    return distances


# calculate all distance(math) mistakes for all tested models
def calculate_distances(models):
    distances = []
    for model in models:
        distance_sum = 0
        for idx, cluster_idx in enumerate(model.map):
            r_idx = cluster_idx[0]
            c_idx = cluster_idx[1]
            sample = model.samples[idx]
            distance_sum += np.linalg.norm(sample - model.clusters[r_idx][c_idx])
        distances.append(distance_sum)
    return distances


def choose_best_model(num_of_models, file_name):
    models = []
    for i in range(num_of_models):
        new_model = Som_model(file_name)
        new_model.init_array_of_clusters()
        new_model.train()
        models.append(new_model)
    # distance mistake
    distances = calculate_distances(models)
    # for topological(physical) mistake
    topological = get_topological_distance(models)
    # we want to weigh the tow kinds of distances by the average of each one
    avg_dist = sum(distances) / len(distances)
    avg_top = sum(topological) / len(topological)
    # we take the fraction of each mistake(both kinds) and then sum them together
    divide_dists = [dist / avg_dist for dist in distances]
    divide_tops = [dist / avg_top for dist in topological]
    merged = [dist + top for dist, top in zip(divide_dists, divide_tops)]
    # we take the minimum sum from above & and the index should match the index of the suit model
    best_model_idx = merged.index(min(merged))
    return models[best_model_idx]


# open window to insert the path to input file.txt with tkinter package
def get_file_path():
    done = False
    while (not done):
        # default path
        default = "Elec_24.csv"
        window = Tk()
        window.title("SOM")
        window.eval('tk::PlaceWindow . center')
        frame = Frame(window)
        frame.pack()
        label1 = Label(frame, text="Please insert the path of your file: ", padx=20, pady=10)
        d = Entry(frame, width=30, borderwidth=5)
        d.insert(END, default)
        exit = Button(frame, text="OK", padx=20, pady=10, command=window.quit)
        label1.grid(row=0, column=0)
        d.grid(row=0, column=1)
        exit.grid(row=5, column=0, columnspan=2)
        window.mainloop()
        new_path = d.get()
        window.destroy()
        window.quit()
        done = True
        # if the path isn't correct
        if not os.path.exists(new_path):
            print("file dosent exist")
            done = False
    return new_path


# this function create an input window for simulation parameters with tkinter library.
# it fills the values by default the variables as they were defined on the top of this script.
def getInput():
    # call all global vars to change them
    global same_iterations_to_converge, max_iterations, \
        alpha, num_of_rings, color_label, num_of_models

    window = Tk()
    window.title("SOM Parameters")
    window.eval('tk::PlaceWindow . center')
    main_lst = []
    label1 = Label(window, text="Number of models to train and choose the best: ", padx=20, pady=10)
    label2 = Label(window, text="Iterations for coverage: ", padx=20, pady=10)
    label3 = Label(window, text="Max iterations(if no coverage): ", padx=20, pady=10)
    label5 = Label(window, text="Alpha (learning rate): ", padx=20, pady=10)
    label6 = Label(window, text="Number of neighbors to update (0-6): ", padx=20, pady=10)
    label7 = Label(window, text="Choose label which will reflect in the hexagons color (must choose): ", padx=20,
                   pady=10)
    d = Entry(window, width=30, borderwidth=5)
    d.insert(END, str(num_of_models))
    r = Entry(window, width=30, borderwidth=5)
    r.insert(END, str(same_iterations_to_converge))
    n = Entry(window, width=30, borderwidth=5)
    n.insert(END, str(max_iterations))
    t = Entry(window, width=30, borderwidth=5)
    t.insert(END, str(alpha))
    p2 = Entry(window, width=30, borderwidth=5)
    p2.insert(END, str(num_of_rings))
    variable = StringVar(window)
    variable.set(color_label)
    option_menu = OptionMenu(window, variable, color_label, "Total Votes", "Labour", "Yamina", "Yahadot Hatora", "The Joint Party", "Zionut Datit", "Kachul Lavan", "Israel Betinu", "Licod", "Merez","Raam",  "Yesh Atid", "Shas", "Tikva Hadasha")
    x = Entry(window, width=30, borderwidth=5)
    x.insert(END, str(X))
    Exit = Button(window, text="Start simulation", padx=20, pady=10, command=window.quit)
    label1.grid(row=0, column=0)
    label2.grid(row=1, column=0)
    label3.grid(row=2, column=0)
    label5.grid(row=3, column=0)
    label6.grid(row=4, column=0)
    label7.grid(row=5, column=0)
    d.grid(row=0, column=1)
    r.grid(row=1, column=1)
    n.grid(row=2, column=1)
    t.grid(row=3, column=1)
    p2.grid(row=4, column=1)
    option_menu.grid(row=5, column=1)
    Exit.grid(row=10, column=0, columnspan=2)
    window.mainloop()
    try:
        # convert the input values from string to numeric values (int or float)
        num_of_models = int(d.get())
        same_iterations_to_converge = int(r.get())
        max_iterations = int(n.get())
        alpha = float(t.get())
        num_of_rings = int(p2.get())
        color_label = str(variable.get())
    except:  # if its faild to convert that means  Input not valid - show message and exit
        messagebox.showwarning("WRAP_AROUND Covid-19 automate", "ERRRORRRR!!!!! Input not valid")
        sys.exit(-1)
    # check every fraction/float input, if its unvalid - show message and exit
    window.destroy()
    window.quit()


if __name__ == '__main__':
    # inset to the function the number of models
    file_path = get_file_path()
    getInput()
    our_model = choose_best_model(num_of_models, file_path)
    # print the cluster index of every city - index look like this "(row, column)"
    for city, cluster_idx in zip(our_model.cities_names, our_model.map):
        print("Cluster index of - " + city + " is: " + str(cluster_idx))
    draw_board(our_model)
