from manim import *
from manim_slides import Slide
import re
from manim import Square
import matplotlib.pyplot as plt
import numpy as np
import math
from plots import *
from constants import *

class CircleWithText(Mobject):
    def __init__(self, scene: Scene, text: str, font_size: int, radius: float = 1, color: str = WHITE):
        self.scene = scene
        self.text = text
        self.font_size = font_size
        self.radius = radius
        self.color = color

    def create(self):
        circle = Circle(radius=self.radius, color=self.color)
        text = Tex(self.text, font_size=self.font_size).scale(0.5)
        text.move_to(circle)
        self.scene.play(Create(circle), run_time=SPEEDUP_TIME)
        self.scene.play(FadeIn(text), run_time=SPEEDUP_TIME)

class SquareWithText(Mobject):
    def __init__(self, scene: Scene, text: str, font_size: int, side_length: float = 1, color: str = WHITE):
        self.scene = scene
        self.text = text
        self.font_size = font_size
        self.side_length = side_length
        self.color = color

    def create(self):
        square = Square(side_length=self.side_length, color=self.color)
        text = Tex(self.text, font_size=self.font_size).scale(0.5)
        text.move_to(square)
        self.scene.play(Create(square), run_time=SPEEDUP_TIME)
        self.scene.play(FadeIn(text), run_time=SPEEDUP_TIME)


class PriorityFun():

    def __init__(self, scene: Scene, position: np.array = ORIGIN, name: str = r"g_{\theta}"):
        self.scene = scene

        self.arrow_g_theta = MathTex(fr"\xrightarrow{{{name}}}")
        self.box_arrow = SurroundingRectangle(self.arrow_g_theta, color=WHITE)

        self.formula_x = (
            MathTex(r"x")
            .scale(0.7)
            .next_to(self.arrow_g_theta, direction=LEFT, buff=0.7)
        )

        self.formula_g_theta_x = (
            MathTex(r"g  (\theta, x) \in \mathbb{R}")
            .scale(0.7)
            .next_to(self.arrow_g_theta, direction=RIGHT, buff=0.7)
        )

    def create(self):
        self.scene.play(Create(self.arrow_g_theta), run_time=SPEEDUP_TIME)
        self.scene.play(Create(self.box_arrow), run_time=SPEEDUP_TIME)
        self.scene.play(FadeIn(self.formula_x, shift=DOWN, run_time=SPEEDUP_TIME))
        self.scene.play(FadeIn(self.formula_g_theta_x), run_time=SPEEDUP_TIME)


class NormalDistributionPlot(VMobject):

    def __init__(self, position=ORIGIN, scale_factor=1, show_labels=False, **kwargs):
        super().__init__(**kwargs)
        self.position = (position,)
        self.scale_factor = (scale_factor,)
        self.show_labels = show_labels

        self.create_plot()

    def create_plot(self):
        # Set up the axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            axis_config={"color": BLUE},
            x_axis_config={"include_tip": False},
            y_axis_config={"include_tip": False},
        )
        axes.move_to(self.position)

        if self.show_labels:
            # Labels for the axes
            x_label = axes.get_x_axis_label(Tex("x"))
            y_label = axes.get_y_axis_label(Tex("Probability Density"))
            curve_label = axes.get_graph_label(
                normal_curve, label="Standard Normal Distribution"
            )

        # Plot the standard normal distribution
        normal_curve = axes.plot(
            lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2),
            color=RED,
        )
        if self.show_labels:
            self.plot = VGroup(axes, x_label, y_label, normal_curve, curve_label)
        else:
            self.plot = VGroup(axes, normal_curve)

        # Scale the entire plot
        self.plot.scale(self.scale_factor)
        self.add(self.plot)


class NeuralNetworkVisualization(VMobject):
    def __init__(
        self,
        input_layer_size,
        hidden_layer_size,
        output_layer_size,
        position=ORIGIN,
        scale_factor=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.position = position
        self.scale_factor = scale_factor

        self.create_network()

    def create_network(self):
        # Define layer positions
        input_layer = self.create_layer(self.input_layer_size)
        hidden_layer = self.create_layer(self.hidden_layer_size)
        output_layer = self.create_layer(self.output_layer_size)

        # Center the layers vertically around the common center
        self.center_layer(input_layer, self.position + LEFT * 4)
        self.center_layer(hidden_layer, self.position)
        self.center_layer(output_layer, self.position + RIGHT * 4)

        # Create a VGroup to hold the entire network
        self.network = VGroup(input_layer, hidden_layer, output_layer)

        # Connect layers and add lines to the network VGroup
        self.connect_layers(input_layer, hidden_layer)
        self.connect_layers(hidden_layer, output_layer)

        # Scale the entire network
        self.scale_network(self.network, self.scale_factor)

        # Add the network to the VMobject
        self.add(self.network)

    def create_layer(self, num_neurons):
        # Create a VGroup to hold the neurons
        layer = VGroup()
        # Create each neuron as a Circle
        for i in range(num_neurons):
            neuron = Circle(radius=0.2, color=BLUE)
            layer.add(neuron)
        return layer

    def center_layer(self, layer, position):
        # Calculate the vertical offset to center the layer around the position
        num_neurons = len(layer)
        layer_height = (num_neurons - 1) * 1.2
        for i, neuron in enumerate(layer):
            neuron.move_to(position + UP * (layer_height / 2 - i * 1.2))

    def connect_layers(self, layer1, layer2):
        # Connect every neuron in layer1 to every neuron in layer2
        for neuron1 in layer1:
            for neuron2 in layer2:
                line = Line(
                    neuron1.get_center(),
                    neuron2.get_center(),
                    color=GRAY,
                    stroke_width=1,
                )
                self.network.add(line)

    def scale_network(self, network, scale_factor):
        # Scale the entire network
        network.scale(scale_factor)


class UniformDistributionPlot(VMobject):

    def __init__(self, position=ORIGIN, scale_factor=1, show_labels=False, **kwargs):
        super().__init__(**kwargs)
        self.position = (position,)
        self.scale_factor = (scale_factor,)
        self.show_labels = show_labels

        self.create_plot()

    def create_plot(self):
        # Set up the axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1, 0.1],  # Adjusted for uniform distribution
            axis_config={"color": BLUE},
            x_axis_config={"include_tip": False},
            y_axis_config={"include_tip": False},
        )
        axes.move_to(self.position)

        if self.show_labels:
            # Labels for the axes
            x_label = axes.get_x_axis_label(Tex("x"))
            y_label = axes.get_y_axis_label(Tex("Probability Density"))
            curve_label = axes.get_graph_label(
                uniform_curve, label="Uniform Distribution"
            )

        # Plot the uniform distribution
        uniform_curve = axes.plot(
            lambda x: 1 if -1 <= x <= 1 else 0,  # Uniform distribution function
            color=RED,
        )
        if self.show_labels:
            self.plot = VGroup(axes, x_label, y_label, uniform_curve, curve_label)
        else:
            self.plot = VGroup(axes, uniform_curve)

        # Scale the entire plot
        self.plot.scale(self.scale_factor)
        self.add(self.plot)
