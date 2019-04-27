import math
import typing as tp
from dataclasses import dataclass

import numpy as np


class Movement:
    def __init__(self, data: np.ndarray):
        self.data = data

    @property
    def time(self) -> np.array:
        return self.data['time']

    @property
    def x(self) -> np.array:
        return self.data['x']

    @property
    def y(self) -> np.array:
        return self.data['y']

    def __len__(self):
        return len(self.data)

    def lerp_xy(self, num_pts):
        # time starts at 0
        time_end = self.time[-1]
        interp_time = np.arange(0.0, 1.0, 1.0 / num_pts) * time_end

        interp_x = np.interp(
            interp_time,
            np.hstack((0.0, self.time)),
            np.hstack((0.0, self.x))
        )

        interp_y = np.interp(
            interp_time,
            np.hstack((0.0, self.time)),
            np.hstack((0.0, self.y))
        )

        return np.column_stack((interp_time, interp_x, interp_y))


@dataclass
class MovementFilter:
    num_points_bounds: (int, int)
    distance_bounds: (float, float)
    time_duration_bounds: (float, float)

    def filter(self, movements: tp.List[Movement]) -> tp.List[Movement]:
        min_pts, max_pts = self.num_points_bounds
        min_dist, max_dist = self.distance_bounds
        min_time, max_time = self.time_duration_bounds

        return [
            m for m in movements
            if min_pts <= len(m) <= max_pts
            and min_time <= m.time[-1] <= max_time
            and min_dist <= math.hypot(m.x[-1], m.y[-1]) <= max_dist
        ]


# note: data can be downloaded from:
#   https://drive.google.com/open?id=1k9IV_hbmlBvjLW9KkddpgaBRsLMKjv6-
class Dataset:
    def __init__(
            self,
            filename: str,
            interpolation_pts: int = 40,
            movement_filter: tp.Optional[MovementFilter] = None,
    ):
        self.filename = filename

        self.movements = Dataset.load(filename)
        if movement_filter is not None:
            print("filtering {} movements...".format(len(self.movements)), end="")
            self.movements = movement_filter.filter(self.movements)
            print("done")

        self.interpolated = Dataset.interpolate_movements(self.movements, interpolation_pts)

    @staticmethod
    def interpolate_movements(movements: tp.List[Movement], num_pts: int):
        print("interpolating {} movements...".format(len(movements)), end="")
        interpolation_times = np.linspace(start=0.0, stop=1.0, num=num_pts, endpoint=True)
        # array of [# of movements] arrays of [num_pts] (x, y) coordinates
        interpolated = np.ndarray((len(movements), num_pts, 2), dtype=float)
        for i, m in enumerate(movements):
            m_times = np.insert(m.time, 0, 0.0) / m.time[-1]
            interpolated[i, :, 0] = np.interp(interpolation_times, m_times, np.insert(m.x, 0, 0.0))
            interpolated[i, :, 1] = np.interp(interpolation_times, m_times, np.insert(m.y, 0, 0.0))

        print("done")
        return interpolated

    @staticmethod
    def load(filename: str):
        print("loading '{}'...".format(filename), end="")
        data = np.load(filename)
        movement_data = data['movements']
        length_data = data['lengths']
        # movements only store references to the actual data
        movements = [
            Movement(movement_data[a:b])
            for a, b in zip(length_data[:-1], length_data[1:])
        ]
        print("done")
        return movements

