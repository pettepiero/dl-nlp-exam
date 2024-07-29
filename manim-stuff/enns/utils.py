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
        self.position = position
        self.scale_factor = scale_factor
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

        # Plot the standard normal distribution
        normal_curve = axes.plot(
            lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2),
            color=RED,
        )

        if self.show_labels:
            # Labels for the axes
            x_label = axes.get_x_axis_label(Tex("x"))
            y_label = axes.get_y_axis_label(Tex("Probability Density"))
            curve_label = axes.get_graph_label(
                normal_curve, label="Standard Normal Distribution"
            )
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


class BertExplanation(Slide):
    def create_LM_figure(self, first_text, second_text, color=BLUE):
        text = Text(first_text, font_size=20)
        square = SurroundingRectangle(text, color=color, fill_opacity=0)
        llm = VGroup(square, text)

        rect = (
            Rectangle(height=4, width=2, color=color, fill_color=BLACK)
            .next_to(llm, direction=RIGHT * 2)
            .add_background_rectangle(color=BLACK, opacity=1)
        )
        rect.set_z_index(1)
        rect.background_rectangle.set_z_index(rect.get_z_index())
        fine_tun = Text(second_text, font_size=20).move_to(rect.get_center())
        fine_tun.set_z_index(rect.get_z_index() + 1)
        group = VGroup(llm, rect, fine_tun)

        return group

    def train_LM_animation(self, model, num_data_points, speedup_factor=5, show_data_file=False):
        data_points = []
        # setting the level of the model
        model[2].set_z_index(1)

        # Setting up generic loss function plot
        ax = Axes(x_range=[0, 3, 3], y_range=[0, 1.3, 3]).scale(0.5).next_to(model[2], direction=RIGHT).shift(RIGHT)
        vt = ValueTracker(0)
        loss_f = always_redraw(lambda: ax.plot(lambda x: np.exp(-x), color=RED, x_range=[0, vt.get_value()]))
        f_dot = always_redraw(
            lambda: Dot(
                point=ax.c2p(vt.get_value(), loss_f.underlying_function(vt.get_value())),
                color=RED,
            )
        )

        for i in range(num_data_points):
            data_point = (
                Square(side_length=0.15 * model.get_height(), color=WHITE)
                .add_background_rectangle(color=LIGHT_GRAY, opacity=1)
            )
            text_data = Text(f"data", font_size=20, color=BLACK).move_to(data_point.get_center())
            data_point.add(text_data)

            if i == 0:
                data_point.next_to(model[2].get_top(), buff=0.5, direction=UP)
            else:
                data_point.next_to(data_points[-1].get_top(), buff=0.1, direction=UP)
            data_point.set_z_index(model[2].get_z_index() - 1)
            data_point.background_rectangle.set_z_index(data_point.get_z_index() - 1)
            data_points.append(data_point)
            # setting the level of the data points behind the model
            data_point.set_z_index(model[2].get_z_index() - 1)
            data_point.background_rectangle.set_z_index(data_point.get_z_index() - 1)

        # Setting target positions
        for i in reversed(range(num_data_points)):
            if i == num_data_points - 1:
                data_points[i].target = (
                    data_points[i]
                    .copy()
                    .move_to(model[2].get_bottom() + DOWN * data_points[i].side_length)
                )
            else:
                data_points[i].target = (
                    data_points[i]
                    .copy()
                    .move_to(
                        data_points[i + 1].target.get_center()
                        + DOWN * (0.1 + data_points[i].side_length)
                    )
                )

        # Setting up animations
        animations = []
        for i in range(num_data_points):
            animations.append(FadeIn(data_points[i]))

        self.play(Write(ax), run_time=SPEEDUP_TIME)
        self.add(loss_f, f_dot)
        # self.play(vt.animate.set_value(20), run_time=speedup_factor*SPEEDUP_TIME)

        if(show_data_file):
            data_file=(
                Square(side_length=0.3 * model.get_height(), color=WHITE)
                .add_background_rectangle(color=LIGHT_GRAY, opacity=1)
                .set_z_index(model[2].get_z_index() +2)
                .move_to(model[2].get_center())
            )
            text_data_file = (
                Text(f"Specific\nknowledge\ndata file", font_size=17, color=BLACK)
                .move_to(data_file.get_center())
                .set_z_index(data_file.get_z_index() + 1)
            )
            data_file.add(text_data_file)

            self.play(FadeIn(data_file), run_time=SPEEDUP_TIME)
            self.wait(1)
            animations.append(ReplacementTransform(data_file, data_points[2]))

        # Playing animations
        self.play(
            AnimationGroup(*animations, lag_ratio=0), run_time=SPEEDUP_TIME
        )
        self.wait(0.5)

        animations = []
        for i in range(num_data_points):
            animations.append(MoveToTarget(data_points[i]))

        animations.append(vt.animate.set_value(3))

        self.play(
            AnimationGroup(*animations, lag_ratio=0),
            run_time=speedup_factor * SPEEDUP_TIME,
        )
        self.wait(1)

        animations = []
        for i in range(num_data_points):
            animations.append(FadeOut(data_points[i]))
        animations.append(FadeOut(ax))
        animations.append(FadeOut(f_dot))
        animations.append(FadeOut(loss_f))

        self.play(
            AnimationGroup(*animations, lag_ratio=0), run_time=SPEEDUP_TIME
        )

    def no_ft_bullet_point_list(self, model):
        bullet_point_list = VGroup()
        for i, point in enumerate(
            ["Token Embeddings", "Hidden States", "Final Layer Embeddings", "Attention weights", "..."]
        ):
            dot = Dot().scale(0.75)
            text = Text(point, font_size=20)
            dot.next_to(text, direction=LEFT)
            line = VGroup(dot, text)
            if i != 0:
                line.next_to(bullet_point_list, direction=DOWN, aligned_edge=LEFT)
            bullet_point_list.add(line)
        bullet_point_list.next_to(model[0], direction=RIGHT)

        self.play(Create(bullet_point_list), run_time=SPEEDUP_TIME)
        self.wait(0.2)
        self.next_slide()
        self.play(FadeOut(bullet_point_list), run_time=SPEEDUP_TIME)
        self.wait(0.2)

    def ft_bullet_point_list(self, model):
        bullet_point_list = VGroup()
        for i, point in enumerate(
            [
                "Sentiment analysis",
                "Text generation",
                "Question answering",
                "Summarizing text",
                "...",
            ]
        ):
            dot = Dot().scale(0.75)
            text = Text(point, font_size=20)
            dot.next_to(text, direction=LEFT)
            line = VGroup(dot, text)
            if i != 0:
                line.next_to(bullet_point_list, direction=DOWN, aligned_edge=LEFT)
            bullet_point_list.add(line)
        bullet_point_list.next_to(model, direction=RIGHT)

        self.play(Create(bullet_point_list), run_time=SPEEDUP_TIME)
        self.wait(0.2)
        return bullet_point_list
