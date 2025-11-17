import cv2
import time
import numpy as np
import numpy.random as npr


def generate_maze(width, height, weight = 0.2):
    # generate a random maze of size width x height
    # the probability of a cell being 0 is weight, and 1 otherwise
    maze = np.random.choice([0, 1], size = (width, height), p = [weight, 1 - weight])
    return maze

def check_num_labels(landscape, width, height):
    landscape = np.array(landscape, dtype = np.uint8).reshape(width, height)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(landscape,connectivity=4)
    num_freespace = np.count_nonzero(landscape)
    # replace 1 in landscape with 255
    landscape[landscape == 1] = 255
    return num_labels, labels, stats, centroids, num_freespace, landscape

def check_repeat(maze_pool, maze):
    # check if the maze is already in the pool
    # if so, return true
    # otherwise, return false
    for i in range(len(maze_pool)):
        if np.array_equal(maze_pool[i], maze):
            return True
    return False

def generate_maze_pool(num_mazes = 20, width = 10, height = 10, lb = 0.01, ub = 0.99, weight = -1):
    maze_pool = []
    manual_weight = True
    if weight == -1:
        manual_weight = False
    while len(maze_pool) < num_mazes:
        if manual_weight == False:
            weight = np.random.uniform(lb, ub)
        landscape = generate_maze(width, height, weight)
        num_labels, labels, stats, centroids, num_freespace, landscape_img = check_num_labels(landscape, width, height)
        non_zeros = np.count_nonzero(landscape)
        if num_labels == 2 and non_zeros >= 5:
            repeated = check_repeat(maze_pool, landscape)
            if not repeated:
                maze_pool.append(landscape)
    return maze_pool
    

# start = time.time()
# maze_pool = generate_maze_pool(num_mazes = 500)
# print("generate_maze_pool took : ", time.time() - start)

# # print(len(maze_pool))
# for i in range(len(maze_pool)):
#     flat_maze = maze_pool[i].flatten()
#     str_maze = "[" + ",".join([str(cell) for cell in flat_maze]) + "]"
#     print(str_maze, ",")