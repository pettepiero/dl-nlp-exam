from manim import *
from manim_slides import Slide
import re
from constants import *
from utils import *
from plots import *

# Presentation overview:
# - Introduction slide
# - What the slides are about
# - What is the problem of normal neural networks and uncertainty (premise)
# - Example from paper on joint predictions
# - What is an Epistemic Neural Network
# - A bit of formulas
# - Architecture diagram
# - ENNs for LLMs fine tuning
# - Results
# - Conclusion (mindmap of topics)
# - References

# Introduction
class Introduction(Slide):
    def construct(self):
        # Title slide
        self.next_slide()
        center_text = Tex(r"Epistemic\\Neural Networks", font_size=110)
        self.play(Write(center_text), run_time=1)
        self.next_slide()
        small_title = Tex(r"Epistemic Neural Networks", font_size=50, color=BLUE).to_corner(UP + LEFT)
        self.play(Transform(center_text, small_title), run_time=SPEEDUP_TIME)

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
        self.wait(0.5)
        self.next_slide()
        self.play(FadeOut(bullet_point_list), FadeOut(center_text), run_time=SPEEDUP_TIME)


class NNsAndUncertainty(Slide):
    def construct(self):
        title = Tex(r"Neural Networks and Uncertainty", font_size=50, color=BLUE).to_corner(UP + LEFT)
        problem = Tex(
            (
                r"\textit{Conventional Neural Networks don't have the ability to distinguish between }"
                r"\textit{uncertainty due to genuine ambiguity and uncertainty due to insufficiency of data}"
            ),
            font_size=30,
        ).move_to(ORIGIN)    
        box = SurroundingRectangle(problem, buff=0.1, color=BLUE)
        self.play(*[FadeIn(obj) for obj in [title, problem]], run_time=SPEEDUP_TIME)
        self.play(Create(box), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(
            problem.animate.move_to((0, 2, 0)),
            box.animate.move_to((0, 2, 0)),
            run_time=SPEEDUP_TIME,
        )

        # Show image
        img = ImageMobject("./media/images/fig1.png").next_to(title, buff=1).scale(0.8)
        img.center()

        source1_text = Tex(r"\textit{Source [1] (Epistemic Neural Networks paper)}", font_size=20).next_to(img, DOWN, buff=1)
        self.play(FadeIn(img), run_time=SPEEDUP_TIME)
        self.play(Write(source1_text), run_time=SPEEDUP_TIME)
        self.next_slide()

        self.play(FadeOut(img), FadeOut(source1_text), run_time=SPEEDUP_TIME)

        text2 = Tex(
            r"It can be critical for decision making systems to know what they don't know.",
            font_size=35)
        self.play(FadeIn(text2), run_time=SPEEDUP_TIME)
        cite_paper = Tex(
            (
                r"\textit{From Predictions to Decisions: The Importance of Joint Predictive Distributions}\\"
                r"Wen Z. et al. (2022)"
            ),
            font_size=25,
        )

        cite_paper.move_to(ORIGIN).shift(DOWN*1)
        self.play(Write(cite_paper), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeOut(title), FadeOut(problem), FadeOut(box), FadeOut(text2), FadeOut(cite_paper), run_time=SPEEDUP_TIME)

class JointPredDec(Slide):
    def construct(self):
        title = Tex(r"Joint Predictions in decision making", font_size=50, color=BLUE).to_corner(UP + LEFT)

        premise = Tex(
            (
                r"Joint predictions allow the distinction between "
                r"uncertainty due to genuine ambiguty and insufficiency of data"
            ),
            font_size=30,
        ).move_to((0, 2, 0))    
        box = SurroundingRectangle(premise, buff=0.1, color=BLUE)
        self.play(*[Write(obj) for obj in [title, premise, box]], run_time=SPEEDUP_TIME)

        # Show image
        img = ImageMobject("./media/images/fig1-slides.jpg").next_to(box, DOWN, buff=1).scale(0.9)
        source1_text = Tex(r'\textit{Epistemic Neural Networks},', font_size=25,)
        source1_link = Tex(
            r"Osband et al. (2023) - https://arxiv.org/abs/2107.08924", font_size=25
        ).next_to(source1_text, DOWN, buff=0.1, aligned_edge=LEFT)
        source = VGroup(source1_text, source1_link).next_to(img, DOWN)
        
        self.next_slide()
        self.play(FadeIn(img), run_time=SPEEDUP_TIME)
        self.play(Write(source), run_time=SPEEDUP_TIME)
        self.next_slide()


class JointPredictions(Slide):
    def construct(self):
        title = Tex(r"Joint Predictions in a Combinatorial decision problem", font_size=50, color=BLUE).to_corner(UP + LEFT)
        text_example = Tex(r"Example: recommendation system", font_size=30).next_to(title, DOWN, buff=0.8, aligned_edge=LEFT)
        customer_circle = Circle(radius=0.5, color=GREEN_C).shift(LEFT*2)
        system_square = Square(side_length=1, color=YELLOW).shift(RIGHT*2)

        inventory_rectangle = (
            Rectangle(height=3, width=1, color=BLUE)
            .next_to(system_square, RIGHT, buff=1)
        )

        all_mobjects = VGroup(inventory_rectangle, system_square, customer_circle)

        all_mobjects.arrange_in_grid(1, 3, buff=2, row_alignments="c")
        all_mobjects.move_to(ORIGIN).shift(DOWN*0.5)

        inventory_dots = VGroup()
        for i in range(21):
            dot = Dot(radius=0.1, color=BLUE).shift(DOWN * i)
            inventory_dots.add(dot)
        inventory_dots.arrange_in_grid(7, 3, buff=0.1)
        inventory_dots.move_to(inventory_rectangle.get_center())
        all_mobjects.add(inventory_dots)

        system_text = Tex(r"Recommendation\\System", font_size=30).next_to(system_square, DOWN)
        customer_text = Tex(r"Customer", font_size=30).next_to(customer_circle, DOWN)
        inventory_text = Tex(r"Inventory\\N elements", font_size=30).next_to(inventory_rectangle, DOWN)

        # example mobjects
        formula_customer_examples = (
            MathTex(r"\phi_1 = (1,0)\\", 
                    r"\phi_2 = (0,1)",
                    font_size=30, 
                    color=GREEN_C)
            .next_to(customer_circle, RIGHT, buff=1)
        )
        example_elements = (
            VGroup(
                MathTex(r"X_1 = (10, -10)", font_size=30, color=BLUE),
                MathTex(r"X_2 = (-10, 10)", font_size=30, color=BLUE),
                MathTex(r"X_3 = (1, 0)", font_size=30, color=BLUE),
                MathTex(r"X_4 = (0, 1)", font_size=30, color=BLUE),
            )
            .arrange(DOWN, buff=0.5)
            .next_to(inventory_rectangle, LEFT, buff=1)
        )

        all_text = VGroup(system_text, customer_text, inventory_text)

        formula_prob_usr_enjoys = MathTex(
            r"Y_i \sim logit(", r"\phi_{*}^{T}", r"X_i)", font_size=35
        ).next_to(all_mobjects[1], UP, buff=1)
        formula_prob_usr_enjoys[1].set_color(GREEN_C)

        # formula_prob_usr_enjoys = (
        #     MathTex(r"Y_i \sim logit(\phi_{*}^{T}X_i)", font_size=30)
        #     .next_to(all_mobjects[1], UP, buff=1)
        # )

        # Arrows
        arrow1 = DoubleArrow(
            system_square.get_right(),
            customer_circle.get_left(),
            stroke_width=1,
            tip_length=0.25,
        )
        arrow2 = DoubleArrow(
            inventory_rectangle.get_right(),
            system_square.get_left(),
            stroke_width=1,
            tip_length=0.25,
        )
        arrows = VGroup(arrow1, arrow2)

        # Animations

        self.play(FadeIn(title), run_time=SPEEDUP_TIME)
        self.play(Write(text_example), run_time=SPEEDUP_TIME)
        self.play(
            Create(customer_circle),
            Create(system_square),
            Write(system_text),
            Write(customer_text),
            Create(inventory_rectangle),
            Write(inventory_text),
            run_time=SPEEDUP_TIME,
        )
        self.play(Create(inventory_dots), run_time=SPEEDUP_TIME)

        self.play(GrowArrow(arrows[0]), GrowArrow(arrows[1]), run_time=SPEEDUP_TIME)
        self.play(Write(formula_prob_usr_enjoys), run_time=SPEEDUP_TIME)

        self.next_slide()

        # highlight some dots
        dots_to_highlight = VGroup(inventory_dots[4], inventory_dots[5], inventory_dots[7], inventory_dots[11], inventory_dots[17])
        text_k_elements = Tex(r"Selection of K elements", font_size=30, color=YELLOW).next_to(inventory_rectangle, LEFT)
        self.play(*[dot.animate.set_color(YELLOW) for dot in dots_to_highlight], run_time=SPEEDUP_TIME)
        self.play(Write(text_k_elements), run_time=SPEEDUP_TIME)
        self.next_slide()

        # move highlighted dots to system square
        dots_to_highlight_copy = VGroup(dots_to_highlight.copy())
        self.play(FadeOut(dots_to_highlight_copy, shift=RIGHT), dots_to_highlight.animate.set_color(BLUE), run_time=SPEEDUP_TIME)
        self.play(FadeOut(text_k_elements), run_time=SPEEDUP_TIME)
        self.next_slide()
        all_mobjects.save_state()

        # EXAMPLE SLIDES
        self.play(Write(formula_customer_examples), run_time=SPEEDUP_TIME)
        self.play(*[Write(item) for item in example_elements], run_time=SPEEDUP_TIME)
        self.next_slide()

        self.play(FadeOut(formula_customer_examples), FadeOut(example_elements), run_time=SPEEDUP_TIME)
        self.play(FadeOut(all_mobjects), FadeOut(all_text), FadeOut(arrows), run_time=SPEEDUP_TIME)

        # Create table
        table1_data = np.array([[1, 0, 0.73, 0.5], [0, 1, 0.5, 0.73], [0.5, 0.5, 0.62, 0.62]])
        table1_data_str = [[str(item) for item in row] for row in table1_data]

        table1 = Table(
            table1_data_str,
            row_labels=[
                MathTex(r"\phi_1 = (1,0)", color=GREEN_C, font_size=70),
                MathTex(r"\phi_2 = (0,1)", color=GREEN_C, font_size=70),
                MathTex(r"\phi \sim Unif(\phi_1, \phi_2)", color=GREEN_C, font_size=70),
            ],
            col_labels=[
                MathTex(r"X_1 = (10, -10)", color=GREEN_C, font_size=70),
                MathTex(r"X_2 = (-10, 10)", color=GREEN_C, font_size=70),
                MathTex(r"X_3 = (1, 0)", color=GREEN_C, font_size=70),
                MathTex(r"X_4 = (0, 1)", color=GREEN_C, font_size=70),
            ],
        ).scale(0.4)

        table1.move_to(ORIGIN).shift(DOWN * 0.5)
        self.play(Create(table1), run_time=SPEEDUP_TIME)
        self.wait(0.5)
        self.next_slide()


class WhatAreENNs(Slide):
    def construct(self):
        title = Tex(r"What are Epistemic Neural Networks (ENNs)?", font_size=50, color=BLUE).to_corner(UP + LEFT)
        ##############################################################################
        text_conventional_nn = (
            Tex(r"Conventional neural networks are specified by", font_size=30)
            .next_to(title, DOWN, aligned_edge=LEFT)
        )
        blist_conventional_nn = BulletedList(
            r"Parameters $\theta$",
            r"A parameterized function class $f$ $\xrightarrow{produces}$ vector valued output $f_\theta(x)$",
            font_size=30,
        ).next_to(text_conventional_nn, DOWN, buff=0.6, aligned_edge=LEFT)

        text1 = Tex(r"Multiclass classification problem", font_size=35, color=BLUE).next_to(blist_conventional_nn, DOWN, buff=1, aligned_edge=LEFT)

        text_output_to_prob = Tex(
            r"\textbf{Output to probability}", font_size=30
        ).next_to(text1, DOWN, buff=0.5, aligned_edge=LEFT)

        formula_output_to_prob_conv = MathTex(r"\hat{P}(y) = softmax(f_{\theta}(x))_y",
            font_size=30
        ).next_to(text_output_to_prob, DOWN, buff=0.7, aligned_edge=LEFT)

        text_joint_predictions = (
            Tex(r"\textbf{Joint predictions}", font_size=30)
            .next_to(text_output_to_prob, RIGHT, buff=2)
        )
        formula_conv_nn_joint = MathTex(
            r"\hat{P}_{1:\tau}^{NN} (y_{1:\tau}) =",
            r"\prod_{t=1}^{\tau} softmax",
            r"(f_\theta (x_t))_{y_t}",
            font_size=30,
        ).next_to(text_joint_predictions, DOWN, buff=0.5, aligned_edge=LEFT)

        formula_output_to_prob_enn = MathTex(
            r"\hat{P}(y) = softmax(f_{\theta}(x, z))_y", font_size=30
        ).next_to(text_output_to_prob, DOWN, buff=0.5, aligned_edge=LEFT)

        formula_enn_joint = MathTex(
            r"\hat{P}_{1:\tau}^{ENN} (y_{1:\tau}) =",
            r"\int_z P_Z(dz)\prod_{t=1}^{\tau} softmax",
            r"(f_\theta (x_t, z))_{y_t}",
            font_size=30,
        ).next_to(text_joint_predictions, DOWN, buff=0.5, aligned_edge=LEFT)

        text_enns_intro = (
            Tex(r"Epistemic Neural Networks are specified by", font_size=30)
            .next_to(title, DOWN, aligned_edge=LEFT)
        )
        blist_enns = BulletedList(
            r"A reference distribution  $P_Z$",
            r"\textit{Epistemic index} $z \sim P_Z$",
            r"A parameterized function class $f$ $\xrightarrow{produces}$ vector valued output $f_\theta(x, z)$",
            font_size=30,
        ).next_to(text_enns_intro, DOWN, buff=0.6, aligned_edge=LEFT)

        self.play(FadeIn(title), FadeIn(text_conventional_nn), FadeIn(blist_conventional_nn), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeIn(text1), run_time=SPEEDUP_TIME)
        self.play(FadeIn(text_output_to_prob), FadeIn(formula_output_to_prob_conv), run_time=SPEEDUP_TIME)
        self.play(FadeIn(text_joint_predictions), FadeIn(formula_conv_nn_joint), run_time=SPEEDUP_TIME)

        self.next_slide()
        self.play(
            Transform(text_conventional_nn, text_enns_intro),
            Transform(blist_conventional_nn, blist_enns),
            Transform(formula_output_to_prob_conv, formula_output_to_prob_enn),
            Transform(formula_conv_nn_joint, formula_enn_joint),
            
            run_time=SPEEDUP_TIME,
        )

        self.next_slide()
        self.play(
            FadeOut(text_output_to_prob),
            FadeOut(text_joint_predictions),
            FadeOut(formula_output_to_prob_conv),
            FadeOut(formula_conv_nn_joint),
            FadeOut(blist_conventional_nn),
            FadeOut(text_conventional_nn),
            FadeOut(text1),
            run_time=SPEEDUP_TIME,
        )

#################################################################################################

class WhyNotBNNs(Slide):
    def construct(self):
        title = Tex(r"Why not Bayesian NNs or Ensemble methods?", font_size=50, color=BLUE).to_corner(UP + LEFT)
        img = ImageMobject("./media/images/uncert-compute-graph.png").to_edge(DOWN, buff=0.8).scale(0.8)
        source1_text = Text('"Introduction to Uncertainty in Deep Learning",', font_size=15, slant=ITALIC)
        source1_link = Text("Balaji Lakshminarayanan \nhttps://www.gatsby.ucl.ac.uk/~balaji/balaji-uncertainty-talk-cifar-dlrl.pdf", font_size=15).next_to(source1_text, DOWN, buff=0.1, aligned_edge=LEFT)    
        source = VGroup(source1_text, source1_link).next_to(img, DOWN)

        self.play(Write(title), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeIn(img), run_time=SPEEDUP_TIME)
        self.play(Write(source), run_time=SPEEDUP_TIME)
        self.next_slide()



class Epinet(Slide):
    def construct(self):
        title = Tex(r"The epinet", font_size=50, color=BLUE).to_corner(UP + LEFT)

        formula_4 = MathTex(r"f_\theta (x,z) = ", r"\mu_{\zeta}(x) + \sigma_{\eta}(sg[\phi_\zeta (x)],z)", font_size=30).to_edge(LEFT, buff=0.1)

        img = ImageMobject("./media/images/enns/fig4.png").to_edge(RIGHT, buff=0.1).scale(0.7)

        self.play(Write(title), run_time=SPEEDUP_TIME)
        self.play(Write(formula_4), run_time=SPEEDUP_TIME)
        self.play(FadeIn(img), run_time=SPEEDUP_TIME)
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

        reference4 = (
            Text(
                (
                    "[4] Osband et al. (2022). Fine-Tuning Language Models via Epistemic Neural Networks\n\t"
                    "arXiv:2211.01568"
                ),
                font_size=20,
            )
            .next_to(reference3, DOWN, buff=0.5, aligned_edge=LEFT)
        )

        refs.add(reference1, reference2, reference3, reference4)

        self.play(Write(title), run_time=SPEEDUP_TIME)
        self.play(Write(refs), run_time=SPEEDUP_TIME)
        self.wait(2)
