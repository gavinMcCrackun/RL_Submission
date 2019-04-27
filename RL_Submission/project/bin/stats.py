import math
import time

from numba import jit

from project.mmove import MovementFilter, Dataset
import numpy as np
import argparse


@jit(nopython=True)
def test_iter_check(data: np.ndarray):
    total = 0
    pts, n, _ = data.shape
    for i in range(pts):
        for j in range(n - 1):
            total += math.sqrt(data[i, j, 0] + data[i, j + 1, 0])

    return total


def stats(filename: str):
    max_steps = 500
    min_target_dist = 150
    max_target_dist = 250
    num_interp_points = 40

    movement_filter = MovementFilter(
        num_points_bounds=(5, max_steps),
        distance_bounds=(min_target_dist, max_target_dist),
        time_duration_bounds=(0, 1.5)
    )

    data = Dataset(
        filename=filename,
        interpolation_pts=num_interp_points,
        movement_filter=movement_filter,
    )

    print(data.interpolated.shape)
    print(test_iter_check(data.interpolated))

    start = time.time()
    sum = 0
    for _ in range(200):
        sum += test_iter_check(data.interpolated)
    print("per second:", 200 / (time.time() - start))
    print(sum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run some stats.')
    parser.add_argument('input', type=str, help='input datafile in npz format')
    args = parser.parse_args()

    stats(args.input)
