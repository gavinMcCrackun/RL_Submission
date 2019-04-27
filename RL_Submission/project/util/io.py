from project.mmove import Movement
import numpy as np


# data can be downloaded from:
#   https://drive.google.com/open?id=1k9IV_hbmlBvjLW9KkddpgaBRsLMKjv6-
def load_movements(filename: str):
    print("loading '{}'...".format(filename), end="")
    data = np.load(filename)
    movement_data = data['movements']
    length_data = data['lengths']
    # slices are references, etc
    movements = []
    for a, b in zip(length_data[:-1], length_data[1:]):
        movements.append(Movement(movement_data[a:b]))

    print("done")
    return movements


def load_interpolated(filename: str, num_points: int):
    data = load_movements(filename)
    print("interpolating movements...", end="")
    interpolated = np.array([m.lerp_n(num_points) for m in data])
    print("done")
    return interpolated[:, :, 1:]
