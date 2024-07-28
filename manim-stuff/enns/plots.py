from manim import *
from manim_slides import Slide
import re
from manim import Square
import matplotlib.pyplot as plt
import numpy as np
import math

def generate_reproducible_2d_data_points_2_clusters(
    ax=Axes, seed: int = 0, num_data_points: int = 10
):
    # Set the seed for reproducibility
    np.random.seed(0)

    # Generate 10 data points for cluster 1
    x1 = np.random.normal(5, 3, 10)
    y1 = np.random.normal(5, 3, 10)

    # Generate 10 data points for cluster 2
    x2 = np.random.normal(12, 4, 10)
    y2 = np.random.normal(12, 4, 10)

    class1_dots = VGroup()
    class2_dots = VGroup()

    for x_value, y_value in zip(x1, y1):
        dot = Dot(ax.c2p(x_value, y_value), color=GRAY)
        class1_dots.add(dot)

    for x_value, y_value in zip(x2, y2):
        dot = Dot(ax.c2p(x_value, y_value), color=GRAY)
        class2_dots.add(dot)

    return class1_dots, class2_dots


def create_scatter_plot(x_range: list[int], y_range: list[int], scale: float = 0.5):
    ax = Axes(x_range=x_range, y_range=y_range).scale(scale)
    return ax
