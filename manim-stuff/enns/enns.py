from manim import *
from manim_slides import Slide
import re
# Presentation overview:
# - Introduction slide
# - what the slides are about
# - what is the problem of normal neural networks and uncertainty (premise)
# - example from paper on joint predictions
# - what is an Epistemic Neural Network
# - a bit of formulas
# - architecture diagram
# - ENNs for LLMs fine tuning
# - Results
# - Conclusion (mindmap of topics)
# - References

# Introduction
class Introduction(Slide):
    def add_spaces(self, s: str):
        return re.sub(r"(?<=\w)([A-Z])", r" \1", s)

    def send_title_to_top(self, title: Text):
        top_text = Text(self.add_spaces(title.text), font_size=40)
        top_text.to_corner(UP + LEFT)
        self.wait(1)
        self.play(Transform(title, top_text), run_time=0.5)

    def construct(self):
        # Title slide
        center_text = Text("Epistemic\nNeural Networks", font_size=110)
        self.add(center_text)
        self.next_slide()
        self.send_title_to_top(center_text)

        # What are the slides about
        bullet_point_list = VGroup()
        for i, point in enumerate(["NNs and Uncertainty", "ENNs", "Formulas", "Architecture Diagram", "ENNs for LLMs", "Results", "Conclusion", "References"]):
            dot = Dot().scale(0.75)
            text = Text(point, font_size=20)
            dot.next_to(text, direction=LEFT)
            line = VGroup(dot, text)
            if i != 0:
                line.next_to(bullet_point_list, direction=DOWN, aligned_edge=LEFT)
            bullet_point_list.add(line)
        bullet_point_list.to_edge(UP+LEFT)
        bullet_point_list.shift(DOWN + RIGHT)

        self.play(Create(bullet_point_list))
        self.next_slide()


class NNsAndUncertainty(Slide):
    def construct(self):
        title = Text("Neural Networks and Uncertainty", font_size=40).to_corner(UP + LEFT)
        problem = Text(
            (
                "Conventional Neural Networks don't have the ability to distinguish between \n"
                "uncertainty due to genuine ambiguity and uncertainty due to insufficiency of data"
            ),
            font_size=20,
            slant=ITALIC,
        ).move_to((0,2,0))
        box = SurroundingRectangle(problem, buff=0.1, color=BLUE)
        self.play(*[Write(obj) for obj in [title, problem, box]])
        self.next_slide()
        self.wait(1)

        # Example
        example_title = Text("Example", font_size=25).next_to(title, DOWN, buff=2, aligned_edge=LEFT)

        # Show image
        img = ImageMobject("./media/images/fig1.png").next_to(example_title, buff=1).scale(0.8)
        img.center()
        img.shift(UP*0.2)

        source1_text = Text("Source [1] (Epistemic Neural Networks paper)", font_size=15, slant=ITALIC).next_to(img, DOWN, buff=1)
        self.next_slide()
        self.play(FadeIn(img))
        self.play(Write(source1_text))
        self.next_slide()

        self.play(FadeOut(img), FadeOut(source1_text), FadeOut(problem))

        text2 = Text(
            "It can be critical for decision making systems to know what they don't know.",
            font_size=20)
        #.move_to(problem)
        self.play(FadeIn(problem))
        self.wait(2)
        self.play(FadeIn(text2))
        self.next_slide()

        cite_paper = Text(
            (
                "From Predictions to Decisions: The Importance of Joint Predictive Distributions\n"
                "Wen Z. et al. (2022)"
            ),
            font_size=20,
            slant=ITALIC,
        )

        cite_paper.align_to(text2, LEFT).shift(DOWN*1)
        self.play(Write(cite_paper))
        self.wait(2)
        self.next_slide()


class JointPredictions(Slide):
    def construct(self):
        title = Text("Joint Predictions in a Combinatorial decision problem", font_size=40).to_corner(UP + LEFT)
        


class WhatAreENNs(Slide):
    def construct(self):
        enns_specs_1 = (
            Text("A parametrized function class f", font_size=20)
            .to_edge(LEFT + UP).shift(DOWN)
        )
        enns_specs_2 = Text(
            "A reference distribution Pz",
            font_size=20,
        ).next_to(enns_specs_1, DOWN, aligned_edge=LEFT)

        # Create normal distribution plot
        gauss_plot = NormalDistributionPlot(
            position=enns_specs_2.get_right() + RIGHT * 3 + UP * 1, scale_factor=0.1
        )
        unif_plot = UniformDistributionPlot(
            position=enns_specs_2.get_right() + RIGHT * 3 + DOWN * 1, scale_factor=0.1
        )
        dists_arrows = VGroup()
        dists_arrows.add(Arrow(enns_specs_2.get_right(), gauss_plot.get_left()))
        dists_arrows.add(Arrow(enns_specs_2.get_right(), unif_plot.get_left()))

        specs_group = Group(enns_specs_1, enns_specs_2)
        enns_specs_brace = Brace(specs_group, direction=LEFT, buff=0.2)
        self.next_slide()
        self.play(GrowFromCenter(enns_specs_brace))
        self.next_slide()
        self.play(FadeIn(specs_group))
        self.next_slide()
        self.play(FadeIn(gauss_plot, unif_plot, dists_arrows))
        self.next_slide()

        self.play(Transform(unif_plot, gauss_plot))

        # Introducing the index
        f_theta_x_z = MathTex(r"f_{\theta}(x, z)").next_to(
            enns_specs_2, DOWN, buff=1, aligned_edge=LEFT
        )
        r_arrow = Arrow(f_theta_x_z.get_right(), f_theta_x_z.get_right() + RIGHT)
        epis_idx = Text("Epistemic Index z", font_size=20, slant=ITALIC).next_to(
            r_arrow, RIGHT
        )
        self.play(FadeIn(f_theta_x_z))
        self.play(GrowArrow(r_arrow))
        self.play(FadeIn(epis_idx))
        self.wait(2)


# Shows formulas (1) and (2) in the paper
class ENNs2(Scene):
    def construct(self):
        inputs = MathTex(r"x_1, x_2, ..., x_{\tau}")
        nn_prob = MathTex(
            r"\hat{P}_{1:\tau}^{NN} (y_{1:\tau}) = \prod_{t=1}^{\tau} softmax(f_\theta (x_t))_{y_t}"
        ).next_to(inputs, DOWN, buff=1, aligned_edge=LEFT)
        enn_prob = MathTex(
            r"\hat{P}_{1:\tau}^{ENN} (y_{1:\tau}) = \int_{z} P_z(dz) \prod_{t=1}^{\tau} softmax(f_\theta (x_t, z))_{y_t}"
        ).next_to(nn_prob, DOWN, buff=1, aligned_edge=LEFT)

    
        # Create a neural network visualization object
        neural_network = NeuralNetworkVisualization(
            input_layer_size=3,
            hidden_layer_size=4,
            output_layer_size=2,
            position=nn_prob.get_right() + RIGHT * 3,
            scale_factor=0.5,
        )

        self.add(inputs, nn_prob, neural_network)
        self.wait(2)
        self.play(FadeIn(enn_prob))


class NormalDistributionPlot(VMobject):

    def __init__(self, position=ORIGIN, scale_factor=1, show_labels=False, **kwargs):
        super().__init__(**kwargs)
        self.position = position,
        self.scale_factor = scale_factor,
        self.show_labels = show_labels

        self.create_plot()

    def create_plot(self):
        # Set up the axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            axis_config={
                "color": BLUE
            },
            x_axis_config={
                "include_tip": False
            },
            y_axis_config={
                "include_tip": False
            }
        )
        axes.move_to(self.position)

        if self.show_labels:
            # Labels for the axes
            x_label = axes.get_x_axis_label(Tex("x"))
            y_label = axes.get_y_axis_label(Tex("Probability Density"))
            curve_label = axes.get_graph_label(normal_curve, label="Standard Normal Distribution")

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
                line = Line(neuron1.get_center(), neuron2.get_center(), color=GRAY)
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

class Premise(Slide):
    def construct(self):
        premise_title = Text("Premise", font_size=40).to_corner(UP + LEFT)
        premise_subtitle = Text(("Underlying idea"), font_size=25
                                ).next_to(premise_title, DOWN, buff=0.5, aligned_edge=LEFT)
        premise = Text(
            (   "Joint predictions allow the distinction between "
                "uncertainty due to genuine ambiguty \nand insufficiency of data"
            ),
            font_size=20,
            slant=ITALIC,
        ).next_to(premise_subtitle, DOWN, buff=0.5, aligned_edge=LEFT)
        box = SurroundingRectangle(premise, buff=0.1, color=BLUE)
        self.play(*[Write(obj) for obj in [premise_title, premise_subtitle, premise, box]])
        self.next_slide()

        # Show image
        img = ImageMobject("./media/images/fig1-slides.jpg").to_edge(DOWN, buff=0.8)
        source1_text = Text('"Epistemic Neural Networks",', font_size=15, slant=ITALIC)
        source1_link = Text(
            "Osband et al. (2023) - https://arxiv.org/abs/2107.08924", font_size=15
        ).next_to(source1_text, DOWN, buff=0.1, aligned_edge=LEFT)
        source = VGroup(source1_text, source1_link).next_to(img, DOWN)
        self.next_slide()
        self.play(FadeIn(img))
        self.play(Write(source))
        self.next_slide()

class WhyNotBNNs(Slide):
    def construct(self):
        title = Text("Why not Bayesian Neural Networks?", font_size=40).to_corner(UP + LEFT)
        img = ImageMobject("./media/images/uncert-compute-graph.png").to_edge(DOWN, buff=0.8).scale(0.8)
        source1_text = Text('"Introduction to Uncertainty in Deep Learning",', font_size=15, slant=ITALIC)
        source1_link = Text("Balaji Lakshminarayanan \nhttps://www.gatsby.ucl.ac.uk/~balaji/balaji-uncertainty-talk-cifar-dlrl.pdf", font_size=15).next_to(source1_text, DOWN, buff=0.1, aligned_edge=LEFT)    
        source = VGroup(source1_text, source1_link).next_to(img, DOWN)

        self.play(Write(title))
        self.next_slide()
        self.play(FadeIn(img))
        self.play(Write(source))
        self.next_slide()


class References(Slide):
    def construct(self):
        title = Text("References", font_size=40).to_corner(UP + LEFT)

        refs = VGroup()

        reference1 = (
            Text(
                (
                    "[1] Osband et al. (2023). Epistemic Neural Networks.\n\t"
                    "arXiv:2107.08924."
                ),
                font_size=20,
            )
            .next_to(title, DOWN, buff=0.5)
            .align_to(title, LEFT)
        )

        reference2 = (
            Text(
                ("[2] Balaji Lakshminarayanan. Introduction to Uncertainty in Deep Learning.\n\t"
                "https://www.gatsby.ucl.ac.uk/~balaji/balaji-uncertainty-talk-cifar-dlrl.pdf"),
                font_size=20,
            )
            .next_to(reference1, DOWN, buff=0.5)
            .align_to(reference1, LEFT)
        )

        reference3 = (
            Text(
                (
                    "[3] Wen et al. (2022). From Predictions to Decisions: The Importance of Joint Predictive Distributions.\n\t"
                    "arXiv:2107.09224"
                ),
                font_size=20,
            )
            .next_to(reference2, DOWN, buff=0.5)
            .align_to(reference2, LEFT)
        )

        refs.add(reference1, reference2, reference3)

        self.play(Write(title))
        self.play(Write(refs))
        self.wait(2)
