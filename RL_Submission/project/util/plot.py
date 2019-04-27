import matplotlib.pyplot as plt
import numpy as np


def ctx_black():
    plt.rc_context({
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'figure.facecolor': 'black',
        'figure.edgecolor': 'black',
        'savefig.facecolor': 'black',
        'savefig.edgecolor': 'black',
    })


class ColorMapper:
    def __init__(self, cmap_name: str = "viridis"):
        cmap = plt.get_cmap(cmap_name)
        self.t_vals = np.linspace(0, 1, cmap.N)
        self.colors = np.array(cmap.colors).T

    def color_for(self, t):
        return np.column_stack((
            np.interp(t, self.t_vals, self.colors[0]),
            np.interp(t, self.t_vals, self.colors[1]),
            np.interp(t, self.t_vals, self.colors[2])
        ))
