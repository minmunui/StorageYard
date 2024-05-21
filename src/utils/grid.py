from collections import deque

import numpy as np


def is_reachable(grid, src, dst):
    """
    Check if it is possible to move from src to dst
    :param grid: 2D array
    :param src: tuple
    :param dst: tuple
    :return: bool
    """
    if type(grid) == list:
        grid = np.array(grid)
    n_row, n_col = grid.shape
    visited = np.zeros((n_row, n_col))
    q = deque([src])
    visited[src] = 1
    while q:
        r, c = q.popleft()
        if (r, c) == dst:
            return True
        for dr, dc in zip([0, 0, 1, -1], [1, -1, 0, 0]):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_row and 0 <= nc < n_col and not visited[nr, nc] and (not grid[nr, nc] or grid[nr, nc] == -1):
                q.append((nr, nc))
                visited[nr, nc] = 1
    return False


def generate_random_map(n_row, n_col, stock_prob):
    """
    Generate random map
    :param n_row: int
    :param n_col: int
    :param stock_prob: float
    :return: 2D array
    """
    while True:
        grid = np.zeros((n_row, n_col))
        for r in range(n_row):
            for c in range(n_col):
                if np.random.rand() < stock_prob:
                    grid[r, c] = 1
        grid[0, 0] = 0
        grid[n_row - 1, n_col - 1] = 0
        if is_reachable(grid, (0, 0), (n_row - 1, n_col - 1)):
            return grid
