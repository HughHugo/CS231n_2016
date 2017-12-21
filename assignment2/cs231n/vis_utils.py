from math import sqrt, ceil
import numpy as np


def visualize_grid(xs, ubound = 255.0, padding = 1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.
    
    Inputs:
    - xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: THe number of blank pixels between elements of the grid
    """
    (N, H, W, C) = xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0: y1, x0: x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


def vis_grid(xs):
    """
    Visualize a grid of images.
    """
    (N, H, W, C) = xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * H + A, A * W + A, C), xs.dtype)
    G *= np.min(xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y * H + y: (y + 1) * H + y, x * W + x: (x + 1) * W + x, :] = xs[n, :, :, :]
                n += 1
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def vis_nn(rows):
    """
    Visualize array of arrays of images.
    """
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    xs = rows[0][0]
    G = np.ones((N * H + H, D * W + D, C), xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y * H + y: (y + 1) * H + y, x * W + x: (x + 1) * W + x, :] = rows[y][x]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    