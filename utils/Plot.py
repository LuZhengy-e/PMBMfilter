import numpy as np
from matplotlib import pyplot as plt


def plot_track(Track):
    color = ['red', 'black', 'blue', 'green', 'yellow', 'orange']
    i = 0
    x = []
    y = []
    for obj in Track:
        x = obj.x[0]
        y = obj.x[1]
        ID = obj.id
        plt.scatter(x, y, c=color[ID])

    # plt.show()
