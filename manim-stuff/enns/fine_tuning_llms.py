from manim import *
from manim_slides import Slide
import re
from manim import Square
import matplotlib.pyplot as plt
import numpy as np
from enns import NeuralNetworkVisualization
import math
from plots import *
from utils import *
from constants import *

# Fine tuning LLMs

class FineTuningLLMs(Slide):
    def construct(self):

        # 1. Title slide
        title = Tex(
            (r"Fine-Tuning Language Models \\ via Epistemic Neural Networks"),
            color=BLUE,
            font_size=50,
        )
        title_authors = Tex(
            r"Osband, Asghari, Van Roy, McAleese, Aslanides and Irving",
            font_size=30,
        ).next_to(title, DOWN, buff=0.5)

        small_title = Tex(
            (r"Fine-Tuning Language Models via Epistemic Neural Networks"),
            color=BLUE,
            font_size=45,
        ).to_edge(UP)

        line = Line(title.get_left(), title.get_right(), color=WHITE, stroke_width=1).shift(DOWN*0.8)

        self.play(Write(title), run_time=SPEEDUP_TIME)
        self.play(Write(line), run_time=SPEEDUP_TIME)
        self.play(Write(title_authors), run_time=SPEEDUP_TIME)

        # 2. Contents
        self.next_slide()
        self.play(FadeOut(line), run_time=SPEEDUP_TIME) 
        self.play(FadeOut(title_authors), run_time=SPEEDUP_TIME)
        self.play(Transform(title, small_title), run_time=SPEEDUP_TIME)

        text_contents = Tex(r"\textbf{Contents}", font_size=30).next_to(
            small_title, DOWN * 2, buff=0.5, aligned_edge=LEFT
        )

        text_bullet_points = BulletedList(
            r"Fine tuning LLMs",
            r"Active learning framework",
            r"Statement of the paper",
            r"Priority functions",
            r"Training algorithm and loss function",
            r"Comparison of active learning agents for LLMs",
            r"Conclusion",
            font_size=25).next_to(text_contents, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(FadeIn(text_contents), run_time=SPEEDUP_TIME)
        self.play(FadeIn(text_bullet_points), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeOut(*[text for text in (text_contents, text_bullet_points)]), run_time=SPEEDUP_TIME)

        # 3. BERT before fine tuning
        model = BertExplanation.create_LM_figure(
            self=self,
            first_text="BERT\nLLM",
            second_text="Fine\ntuning", 
            color=BLUE
        ).shift(LEFT * 3 + DOWN * 0.2)
        text_ideally = Text("Ideally", font_size=20).next_to(model, direction=LEFT)
        self.play(Write(text_ideally), run_time=SPEEDUP_TIME)
        self.play(FadeIn(model[0]), run_time=SPEEDUP_TIME)
        BertExplanation.no_ft_bullet_point_list(self, model)
        self.play([Write(model[1]), Write(model[2])], run_time=SPEEDUP_TIME)

        # 4. BERT Language model tasks after fine tuning
        bullet_point_list = BertExplanation.ft_bullet_point_list(self, model)
        self.next_slide()
        self.play(FadeOut(bullet_point_list), run_time=SPEEDUP_TIME)

        # 5. Training animation
        BertExplanation.train_LM_animation(self, model, num_data_points=20, speedup_factor=5, show_data_file=True)
        fine_tuned_llm = BertExplanation.create_LM_figure(
            self=self,
            first_text="Fine-tuned\nBERT LLM",
            second_text="Fine\ntuning", 
            color=BLUE
        ).next_to(model, direction=RIGHT*2)  
        self.play(FadeIn(fine_tuned_llm[0]), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeOut(fine_tuned_llm[0]), run_time=SPEEDUP_TIME)

        # 5. Active learning
        text_in_practice = Text("In practice", font_size=20).next_to(model, direction=LEFT)
        self.play(ReplacementTransform(text_ideally, text_in_practice), run_time=SPEEDUP_TIME)
        BertExplanation.train_LM_animation(self, model, num_data_points=500, speedup_factor=10)
        # delete all mobjects to free memory
        self.play(FadeOut(text_in_practice), run_time=SPEEDUP_TIME)
        
        self.next_slide()
        # remove training animation stuff

        

        ft_techniques = VGroup()
        ft_text = Text("Fine\nTuning", font_size=20).to_edge(LEFT)
        box = SurroundingRectangle(ft_text, buff=0.1, color=BLUE)

        ft_techniques.add(ft_text, box)

        self.play(Transform(model, ft_techniques))
        self.next_slide()

        # 6. Fine tuning with ENN


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
                Text(f"Specific\nknowledge\ndata file", font_size=18, color=BLACK)
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
        self.next_slide()
        self.wait(1)

        self.play(FadeOut(bullet_point_list), run_time=SPEEDUP_TIME)

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


# Active learning framework
class ActiveLearningFramework(Slide):
    def al_scatter_plot_animation(self, shift = None, fade_out_circles = False):
        ax = create_scatter_plot([0, 20, 5], [0, 20, 5]).shift(shift)
        class1_dots, class2_dots = generate_reproducible_2d_data_points_2_clusters(ax=ax)

        self.play(Write(ax), run_time=SPEEDUP_TIME)
        self.play(
            LaggedStart(*[Write(dot) for dot in class1_dots], lag_ratio=0.1),
            run_time=SPEEDUP_TIME,
        )
        self.play(
            LaggedStart(*[Write(dot) for dot in class2_dots], lag_ratio=0.1),
            run_time=SPEEDUP_TIME,
        )

        svm_line = ax.plot(lambda x: 15 - 0.5 * x, color=RED)
        self.play(Write(svm_line), run_time=SPEEDUP_TIME)
        self.wait(1)
        selected_dots = VGroup(
            class1_dots[1], class1_dots[6], class2_dots[5], class2_dots[2]
        )

        class1_indices = [
            i for i, dot in enumerate(class1_dots) if dot in selected_dots
        ]
        class2_indices = [
            i for i, dot in enumerate(class2_dots) if dot in selected_dots
        ]

        circles = VGroup(*[Circle(radius=0.18, color=RED_A).move_to(dot.get_center()) for dot in selected_dots])

        # Animation for selecting the data points and highlighting them
        self.play(
            LaggedStart(
                *[Write(circle) for circle in circles],
                lag_ratio=0.1,
            ),
            *[ApplyMethod(class1_dots[i].set_color, BLUE) for i in class1_indices],
            *[ApplyMethod(class2_dots[i].set_color, GREEN) for i in class2_indices],
        )
        self.wait(1.5)
        if(fade_out_circles):
            self.play(FadeOut(circles), run_time=SPEEDUP_TIME)
            self.play(*[ApplyMethod(dot.set_color, GRAY) for dot in selected_dots], run_time=SPEEDUP_TIME)

        scatter_plot = VGroup(ax, class1_dots, class2_dots, selected_dots, circles, svm_line)
        return scatter_plot

    def time_step_animation(self, scatter_plot, time_step):
        ax, class1_dots, selected_dots, svm_line = (
            scatter_plot[0],
            scatter_plot[1],
            scatter_plot[3],
            scatter_plot[4],
        )
        selected_dot = selected_dots[time_step]
        if selected_dot in class1_dots:
            color = BLUE
        else:
            color = GREEN

        if time_step == 0:
            new_svm_line = ax.plot(lambda x: 15.5 - 0.5 * x, color=RED)
        elif time_step ==1:
            new_svm_line = ax.plot(lambda x: 16 - 0.5 * x, color=RED)
        elif time_step == 2:
            new_svm_line = ax.plot(lambda x: 13 - 0.2 * x, color=RED)
        elif time_step == 3:
            new_svm_line = ax.plot(lambda x: 15 - 0.5 * x, color=RED)

        circle = Circle(radius=0.18, color=RED_A).move_to(selected_dot.get_center())
        self.play(Write(circle), run_time=0.5*SPEEDUP_TIME)
        self.play(ApplyMethod(selected_dot.set_color, color), run_time=SPEEDUP_TIME)
        self.play(ReplacementTransform(svm_line, new_svm_line), run_time=SPEEDUP_TIME)
        self.play(FadeOut(circle), run_time=SPEEDUP_TIME)

    def modify_text_time_step(self, full_notation_text, time_step):

        formula_time_step = full_notation_text[4]
        formula_visible_data = full_notation_text[6]
        formula_class_label_t = full_notation_text[9]
        formula_obtain = full_notation_text[11]
        al_text2 = full_notation_text[0]
        text_original_dataset = full_notation_text[1]
        n_dataset = full_notation_text[2]
        text_time_step = full_notation_text[3]
        text_pick = full_notation_text[8]
        text_obtain = full_notation_text[10]
        text_agent_sees = full_notation_text[5]
        t_box = full_notation_text[7]
        formula_model_params = full_notation_text[13]
        text_model_params = full_notation_text[12]

        if time_step == 1:
            new_formula_time_step = (
                MathTex(r"t = 1").scale(0.7).move_to(formula_time_step.get_center())
            )
            new_formula_visible_data = (
                MathTex(r"D_{1} = D_{X} \cup \{y_{a_{s}}\}_{s=0}^{1}")
                .scale(0.8)
                .move_to(formula_visible_data.get_center())
            )
            new_formula_class_label_t = (
                MathTex(r"a_{1} \in \{1, ..., N\}")
                .scale(0.7)
                .move_to(formula_class_label_t.get_center())
            )
            new_formula_obtain = (
                MathTex(r"y_{a_{1}}")
                .scale(0.7)
                .move_to(formula_obtain.get_center())
            )
            new_formula_update_params = (
                MathTex(r"\theta_{1}")
                .scale(0.7)
                .move_to(formula_model_params.get_center())
            )

        elif time_step == 2:
            new_formula_time_step = (
                MathTex(r"t = 2").scale(0.7).move_to(formula_time_step.get_center())
            )
            new_formula_visible_data = (
                MathTex(r"D_{2} = D_{1} \cup \{y_{a_{s}}\}_{s=0}^{2}")
                .scale(0.8)
                .move_to(formula_visible_data.get_center())
            )
            new_formula_class_label_t = (
                MathTex(r"a_{2} \in \{1, ..., N\}")
                .scale(0.7)
                .move_to(formula_class_label_t.get_center())
            )
            new_formula_obtain = (
                MathTex(r"y_{a_{2}}")
                .scale(0.7)
                .move_to(formula_obtain.get_center())
            )
            new_formula_update_params = (
                MathTex(r"\theta_{2}")
                .scale(0.7)
                .move_to(formula_model_params.get_center())
            )

        elif time_step == 3:
            new_formula_time_step = (
                MathTex(r"t = 3").scale(0.7).move_to(formula_time_step.get_center())
            )
            new_formula_visible_data = (
                MathTex(r"D_{3} = D_{2} \cup \{y_{a_{s}}\}_{s=0}^{3}")
                .scale(0.8)
                .move_to(formula_visible_data.get_center())
            )
            new_formula_class_label_t = (
                MathTex(r"a_{3} \in \{1, ..., N\}")
                .scale(0.7)
                .move_to(formula_class_label_t.get_center())
            )
            new_formula_obtain = (
                MathTex(r"y_{a_{3}}")
                .scale(0.7)
                .move_to(formula_obtain.get_center())
            )
            new_formula_update_params = (
                MathTex(r"\theta_{3}")
                .scale(0.7)
                .move_to(formula_model_params.get_center())
            )

        self.play(
            ReplacementTransform(formula_time_step, new_formula_time_step),
            ReplacementTransform(formula_visible_data, new_formula_visible_data),
            ReplacementTransform(formula_class_label_t, new_formula_class_label_t),
            ReplacementTransform(formula_obtain, new_formula_obtain),
            ReplacementTransform(formula_model_params, new_formula_update_params),
            run_time=SPEEDUP_TIME)

        new_full_notation_text = VGroup(
            al_text2,
            text_original_dataset,
            n_dataset,
            text_time_step,
            new_formula_time_step,
            text_agent_sees,
            new_formula_visible_data,
            t_box,
            text_pick,
            new_formula_class_label_t,
            text_obtain,
            new_formula_obtain,
            text_model_params,
            new_formula_update_params,
        )

        return new_full_notation_text

    def create_full_notation_text(self, title):
        full_notation_text = VGroup()

        al_text2 = Tex(
            (
                r"Active learning implies that a learning agent is able to prioritize "
                r"training examples in order to improve performance "
                r"on held out data."
            ),
            font_size=30,
        ).next_to(title, direction=DOWN, aligned_edge=LEFT)

        text_original_dataset = Text("Original dataset", font_size=20).next_to(
            al_text2, direction=DOWN, aligned_edge=LEFT
        )
        n_dataset = (
            MathTex(r"D = \{(x_{i}, y_{i}, i)\}_{i=1}^{N}")
            .scale(0.7)
            .next_to(text_original_dataset, direction=RIGHT, buff=1)
        )

        text_time_step = Text("Time step", font_size=20).next_to(
            text_original_dataset, direction=DOWN, aligned_edge=LEFT, buff=0.35
        )
        formula_time_step = (
            MathTex(r"t = 0")
            .scale(0.7)
            .next_to(n_dataset, direction=DOWN, aligned_edge=LEFT)
        )
        t_box = SurroundingRectangle(formula_time_step, color=WHITE)

        text_pick = Text(f"Pick index", font_size=20).next_to(
            text_time_step, direction=DOWN, aligned_edge=LEFT, buff=0.35
        )

        text_obtain = Text("Obtains", font_size=20).next_to(
            text_pick, direction=DOWN, aligned_edge=LEFT, buff=0.35
        )

        formula_class_label_t = (
            MathTex(r"a_{t} \in \{1, ..., N\}")
            .scale(0.7)
            .next_to(formula_time_step, direction=DOWN, aligned_edge=LEFT)
        )

        formula_obtain = (
            MathTex(r"y_{a_{t}}")
            .scale(0.7)
            .next_to(formula_class_label_t, direction=DOWN, aligned_edge=LEFT)
        )

        text_agent_sees = Text("See", font_size=20).next_to(
            text_obtain, direction=DOWN, aligned_edge=LEFT, buff=0.35
        )

        formula_visible_data = (
            MathTex(r"D_{X} = \{(x_i, i)\}_{i=1}^{N}")
            .scale(0.8)
            .next_to(formula_obtain, direction=DOWN, aligned_edge=LEFT)
        )

        text_update_params = Text("Update model \nparameters", font_size=20).next_to(
            text_agent_sees, direction=DOWN, buff=0.35, aligned_edge=LEFT
        )

        formula_model_params = (
            MathTex(r"\theta_{t}")
            .scale(0.7)
            .next_to(formula_visible_data, direction=DOWN, aligned_edge=LEFT)
        )

        full_notation_text.add(
            al_text2,
            text_original_dataset,
            n_dataset,
            text_time_step,
            formula_time_step,
            text_agent_sees,
            formula_visible_data,
            t_box,
            text_pick,
            formula_class_label_t,
            text_obtain,
            formula_obtain,
            text_update_params,
            formula_model_params,
        )

        return full_notation_text

    def construct(self):
        title = Tex("Active Learning Framework", font_size=50, color=BLUE).to_edge(UP+LEFT)
        self.play(FadeIn(title), run_time=SPEEDUP_TIME)
        al_text = Tex(("Active learning is about "
                        "choosing specific examples during learning."), font_size=40).next_to(title, direction=DOWN, aligned_edge=LEFT)
        self.play(Write(al_text))

        # LLM_text = Text("LLM", font_size=20, color=WHITE).to_edge(LEFT)
        # LLM_box = SurroundingRectangle(
        #     LLM_text, color=BLUE_B)
        # LLM_figure = VGroup(LLM_box, LLM_text)
        # self.add(LLM_figure)

        scatter_plot = (
            self.al_scatter_plot_animation(shift=RIGHT*3, fade_out_circles=False)
        )

        # Pros and cons of Active Learning
        blist = BulletedList(
            r"Efficiency in labeling",
            r"Improved model performance",
            r"Better handling of imbalanced data",
            r"General adaptability of the \\model to new domains",
        ).next_to(scatter_plot, direction=LEFT)
        blist.font_size = 40

        self.play(Write(blist), run_time=SPEEDUP_TIME)
        self.next_slide()
        actions = [
            FadeOut(blist), 
            FadeOut(scatter_plot[4]), 
            *[ApplyMethod(dot.set_color, GRAY) for dot in scatter_plot[3]]
            ]
        self.play(AnimationGroup(*actions, lag_ratio=0), run_time=SPEEDUP_TIME)
        self.wait(0.5)
        scatter_plot.remove(scatter_plot[4])

        scatter_plot_small = scatter_plot.copy().to_edge(RIGHT).scale(0.7)
        self.play(ReplacementTransform(scatter_plot, scatter_plot_small), run_time=SPEEDUP_TIME)        
        self.play(
            # *[FadeOut(obj) for obj in [LLM_figure, al_text]], run_time=SPEEDUP_TIME
            FadeOut(al_text), run_time=SPEEDUP_TIME
        )

        full_notation_text = self.create_full_notation_text(title)

        self.play(Write(full_notation_text, run_time=SPEEDUP_TIME))

        self.next_slide()

        full_notation_text = self.modify_text_time_step(full_notation_text, 1)
        self.next_slide()
        self.time_step_animation(scatter_plot_small, 0)
        self.next_slide()

        full_notation_text = self.modify_text_time_step(full_notation_text, 2)
        self.next_slide()
        self.time_step_animation(scatter_plot_small, 1)
        self.next_slide()

        full_notation_text = self.modify_text_time_step(full_notation_text, 3)
        self.next_slide()
        self.time_step_animation(scatter_plot_small, 2)
        self.next_slide()

        # Clean plot, introduce priority functions
        self.play(FadeOut(full_notation_text), run_time=SPEEDUP_TIME)
        self.play(FadeOut(scatter_plot_small[4]), run_time=SPEEDUP_TIME)
        self.play(*[ApplyMethod(dot.set_color, GRAY) for dot in scatter_plot[3]])
        self.play(FadeOut(title), run_time=SPEEDUP_TIME)
        self.play(FadeOut(scatter_plot_small), run_time=SPEEDUP_TIME)
        self.next_slide()


class PriorityFunctions(Slide):
    def create_classification_squares(self, group_g_arrow):
        squares_classes = VGroup()
        for i in range(1, 5):
            square = VGroup()
            box = Square(side_length=0.5, color=BLUE)
            if i == 3:
                LLM_text = Text(f"Class ...", font_size=20).next_to(
                    box, direction=RIGHT
                )
            elif i == 4:
                LLM_text = Text(f"Class C", font_size=20).next_to(box, direction=RIGHT)
            else:
                LLM_text = Text(f"Class {i}", font_size=20).next_to(
                    box, direction=RIGHT
                )
            square.add(box, LLM_text)
            squares_classes.add(square)
        squares_classes.arrange(direction=DOWN, buff=0.2).next_to(
            group_g_arrow, direction=RIGHT, buff=2
        )
        text_classification_c = Tex("Classification c", font_size=30).next_to(
            squares_classes, direction=UP, buff=0.5
        )

        self.play(
            FadeIn(
                *[obj for obj in (squares_classes, text_classification_c)], shift=RIGHT
            ),
            run_time=SPEEDUP_TIME,
        )
        return squares_classes, text_classification_c

    def construct(self):
        title_priority_functions_small = Tex(r"Priority Functions", font_size=50, color=BLUE).to_edge(UP+LEFT)
        self.play(Write(title_priority_functions_small), run_time=SPEEDUP_TIME)

        priority_fun = PriorityFun(scene=self)
        priority_fun.create()

        group_g_arrow = VGroup(
            priority_fun.arrow_g_theta,
            priority_fun.formula_x,
            priority_fun.formula_g_theta_x,
            priority_fun.box_arrow,
        )

        squares_classes, text_classification_c = self.create_classification_squares(group_g_arrow)

        formula_p_class_with_z = MathTex(r"p(c|\theta, x, z) = softmax(f_{\theta}(x, z))_c").scale(0.7).next_to(group_g_arrow, direction=DOWN, buff=0.7, aligned_edge=LEFT)
        formula_p_class = MathTex(r"\int_z P_Z(dz)p(c|\theta, x, z)").scale(0.7).next_to(formula_p_class_with_z, direction=DOWN, buff=0.7, aligned_edge=LEFT)
        self.play(*[FadeIn(formula, shift=DOWN) for formula in [formula_p_class_with_z, formula_p_class]], run_time=SPEEDUP_TIME)
        self.next_slide()

        group_g_arrow.add(squares_classes, text_classification_c, formula_p_class_with_z, formula_p_class)

        self.play(group_g_arrow.animate.to_edge(LEFT), run_time=SPEEDUP_TIME)

        self.play(
            FadeOut(
                *[obj for obj in (squares_classes, text_classification_c)], shift=RIGHT
            ),
            run_time=SPEEDUP_TIME,
        )
        self.play(*[FadeOut(obj) for obj in [formula_p_class_with_z, formula_p_class]], run_time=SPEEDUP_TIME)

        text_uniform_prioritization = Tex(
            r"Uniform prioritization",
            font_size=40
            )

        # Marginal priority functions
        text_marginal_priority_functions = Tex(
            r"Marginal priority functions", font_size=40,
            )
        text_entropy = (Tex(
                r"\textbf{Entropy}", 
                font_size=20)
                .next_to(
                text_marginal_priority_functions, direction=DOWN, aligned_edge=LEFT
            )
            .shift(RIGHT * 0.5)
        )
        formula_entropy = MathTex(r"g^{entropy}(\theta, x) = H[p(\cdot |\theta, x)]").scale(0.7).next_to(text_entropy, direction=DOWN, aligned_edge=LEFT)
        entropy = VGroup(text_entropy, formula_entropy)

        text_margin = Tex(
            r"\textbf{Margin}", 
            font_size=20,
        ).next_to(
            text_entropy, direction=DOWN, buff=1.2, aligned_edge=LEFT
        )
        formula_margin = MathTex(r"g^{margin}(\theta, x) = p(c_2|\theta, x) - p(c_1|\theta, x)").scale(0.7).next_to(text_margin, direction=DOWN, aligned_edge=LEFT)
        margin = VGroup(text_margin, formula_margin)
        marginal_priority_functions = VGroup(text_marginal_priority_functions, entropy, margin)

        # ENNs priority functions
        text_enns_priority_functions = Tex(
            r"Epistemic priority functions", 
            font_size=40
        )
        text_bald_ = (
            Tex(r"\textbf{Bald}", font_size=20)
            .next_to(text_enns_priority_functions, direction=DOWN, aligned_edge=LEFT)
            .shift(RIGHT * 0.5)
        )
        formula_bald = (
            MathTex(r"g^{bald}(\theta, x) = \mathbb{H}[p(\cdot | \theta, x)] - \int_z P_z (dz)\mathbb{H}[p(\cdot | \theta, x,z)]")
            .scale(0.7)
            .next_to(text_bald_, direction=DOWN, aligned_edge=LEFT)
        )
        bald = VGroup(text_bald_, formula_bald)

        text_variance = Tex(r"\textbf{Variance}", font_size=20).next_to(
            text_bald_, direction=DOWN, buff=1.2, aligned_edge=LEFT
        )
        formula_variance = (
            MathTex(r"g^{variance}(\theta, x) = \sum_c \int_z P_z(dz)(p(c|\theta, x, z) - p(c|\theta,x))^2")
            .scale(0.7)
            .next_to(text_variance, direction=DOWN, aligned_edge=LEFT)
        )
        variance = VGroup(text_variance, formula_variance)

        enn_priority_functions = VGroup(text_enns_priority_functions, bald, variance)

        all_priority_functions_text = VGroup(
            text_uniform_prioritization,
            marginal_priority_functions,
            enn_priority_functions,
        )
        all_priority_functions_text.arrange(DOWN, aligned_edge=LEFT, buff=1).to_corner(UP + RIGHT)

        box_uniform_prioritization = (
            Rectangle(height=1.3, width=9.1, color=WHITE)
            .move_to(text_uniform_prioritization
            .get_center()).align_to(text_uniform_prioritization, LEFT)
            .shift(LEFT*0.1)
        )
        box_marginal_priority_functions = (
            Rectangle(height=3.8, width=9.1, color=WHITE)
            .move_to(marginal_priority_functions
            .get_center())
            .align_to(marginal_priority_functions, LEFT)
            .shift(LEFT*0.1)
        )
        box_enn_priority_functions = (
            Rectangle(height=4, width=9.1, color=WHITE)
            .move_to(enn_priority_functions
            .get_center())
            .align_to(enn_priority_functions, LEFT)
            .shift(LEFT*0.1)
        )

        full_vgroup = VGroup(
            all_priority_functions_text,
            box_uniform_prioritization,
            box_marginal_priority_functions,
            box_enn_priority_functions,
        )

        self.play(*[Create(box) for box in [box_uniform_prioritization, box_marginal_priority_functions, box_enn_priority_functions
                                            ]], run_time=SPEEDUP_TIME)
        self.play(Write(all_priority_functions_text), run_time=SPEEDUP_TIME)

        self.next_slide()
        self.play(full_vgroup.animate.shift(UP * 3), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeOut(full_vgroup), run_time=SPEEDUP_TIME)
        self.play(
            FadeOut(
                *[
                    obj
                    for obj in (
                        priority_fun.arrow_g_theta,
                        priority_fun.formula_x,
                        priority_fun.formula_g_theta_x,
                        priority_fun.box_arrow,
                    )
                ]
            )
        )

class TrainingAlgorithm(Slide):
    def construct(self):
        title = Tex(r"Training Algorithm and Loss Function", font_size=50, color=BLUE).to_edge(UP+LEFT)
        self.play(FadeIn(title), run_time=SPEEDUP_TIME)
        self.next_slide()
        img = ImageMobject("./media/images/fine-tuning-llm-enns/algorithm.png").to_edge(
            UP, buff=1
        )
        source1_text = Text('"Epistemic Neural Networks",', font_size=15, slant=ITALIC)
        source1_link = Text(
            "Osband et al. (2023) - https://arxiv.org/abs/2107.08924", font_size=15
        ).next_to(source1_text, RIGHT, buff=0.1)
        training_algo = Group(source1_text, source1_link).next_to(img, DOWN)
        training_algo.add(img)
        self.play(FadeIn(training_algo))
        self.next_slide()
        text_loss_function = Tex(r"Cross-entropy loss \\ with regularization: ", font_size=20).next_to(training_algo, DOWN, buff=0.5).to_edge(LEFT)
        formula_loss = (
            MathTex(
                r"\mathcal{L}_{\lambda}^{XENT}(\theta, z, x_i, y_i, i) = - \ln(\text{softmax}(f_{\theta}(x_i, z))_{y_i}) + \lambda \left|\left| \theta \right|\right|_2^2"
            )
            .scale(0.7)
            .next_to(text_loss_function, RIGHT, buff=0.5)
        )
        self.play(Write(text_loss_function), run_time=SPEEDUP_TIME)
        self.play(Write(formula_loss), run_time=SPEEDUP_TIME)

class ComparisonActiveLearningAgents(Slide):
    def construct(self):
        title = Tex(
            r"\textbf{Comparison of Active Learning Agents}",
            font_size=50,
            color=BLUE,
        ).to_edge(UP + LEFT)
        self.play(FadeIn(title), run_time=SPEEDUP_TIME)

        self.next_slide()
        text_task = Tex(
            r"Task: \textbf{Neural Testbed}, an open-source active learning benchmark",
            font_size=30,
        ).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
        text_neural_testbed = Tex(
            r"""More in detail: a collection of neural-network-based, synthetic classification problems
            that evaluate the quality of an agent’s predictive distributions.
            """,
            font_size=30,
        ).next_to(text_task, DOWN, buff=0.4, aligned_edge=LEFT)

        self.play(Write(text_task, run_time=SPEEDUP_TIME))
        self.play(Write(text_neural_testbed, run_time=SPEEDUP_TIME))
        
        self.next_slide()
        ###############################################################

        text_random_MLP = Tex(r"Random MLP", font_size=25).to_edge(LEFT)
        neural_network = (
            NeuralNetworkVisualization(
                input_layer_size=4,
                hidden_layer_size=9,
                output_layer_size=2,
                position=text_random_MLP.get_bottom() + DOWN,
                scale_factor=0.5,
            )
            .shift(DOWN)
            .scale(0.5)
        )
        self.play(FadeIn(text_random_MLP), run_time=SPEEDUP_TIME)
        self.play(Write(neural_network), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeOut(neural_network), run_time=SPEEDUP_TIME)
        self.play(FadeOut(text_random_MLP), run_time=SPEEDUP_TIME)
        ###############################################################
        self.play(FadeOut(text_neural_testbed), run_time=SPEEDUP_TIME)
        text_comparison = Tex(
            r"Comparison of active learning agents with respect to a baseline agent",
            font_size=30,
        ).next_to(text_task, DOWN, buff=0.4, aligned_edge=LEFT)
        text_baseline_agent = Tex(
            r"Baseline agent:",
            font_size=25 ,
        ).next_to(text_comparison, DOWN, buff=0.4, aligned_edge=LEFT)

        item_list_baseline_agent = BulletedList(
            r"Doesn't use active learning",
            r"Uses standard supervised learning",
            r"""Trained on a fraction $\psi \in \{0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.\}$ \\
                of the training data""",
            r"For each subset, sweeps over batch size $\in \{4, 16, 64\}$",
            r"Learning rate $\in \{1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4\}$",
            r"$L_2$ regularization $\in \{0, 1e-3, 1e-2, 1e-1, 1\}$",
            r"For each setting, the results are averaged over three random seeds and the best parameters are selected",
            font_size=25,
        ).next_to(text_baseline_agent, DOWN, buff=0.5, aligned_edge=LEFT)
        baseline_agent = VGroup(text_baseline_agent, item_list_baseline_agent)
        self.play(Write(text_comparison), run_time=SPEEDUP_TIME)
        self.play(FadeIn(baseline_agent), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.wait(1)

        self.play(FadeOut(baseline_agent, dir=LEFT), run_time=SPEEDUP_TIME)
        img = ImageMobject(
            "./media/images/fine-tuning-llm-enns/fig2.png"
            ).move_to(ORIGIN).shift(DOWN)
        self.play(FadeIn(img), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.wait(1)
        self.play(FadeOut(img), run_time=SPEEDUP_TIME)
        self.play(FadeOut(text_comparison), run_time=SPEEDUP_TIME)
        self.next_slide()
        ###############################################################
        text_other_agents = Tex(
            r"""
            A comparison is performed against other methods 
            that do not use epistemic uncertainty:""",
            font_size=30,
        ).next_to(text_task, DOWN, buff=0.4, aligned_edge=LEFT)
        self.play(Write(text_other_agents), run_time=SPEEDUP_TIME)
        item_list = Tex(
            r"""
            \begin{itemize}
            \item MLP with margin prioritization
            \item MLP with entropy prioritization
            \item EPINET with margin prioritization
            \item EPINET with entropy prioritization
            \item EPINET with variance prioritization
            \end{itemize}
            """,
            font_size=25,
        ).next_to(text_other_agents, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(FadeIn(item_list), run_time=SPEEDUP_TIME)
        self.wait(1)
        self.next_slide()
        img2 = (
            ImageMobject("./media/images/fine-tuning-llm-enns/fig3a.png")
            .move_to(ORIGIN)
            .shift(DOWN)
        )
        self.play(FadeOut(item_list), run_time=SPEEDUP_TIME)
        self.play(FadeIn(img2), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.wait(1)
        self.play(FadeOut(img2), run_time=SPEEDUP_TIME)
        self.play(FadeOut(text_other_agents), run_time=SPEEDUP_TIME)

        ####################################################à
        text_bayesian_comparison = Tex(
            r"""
            A comparison is performed against approximate Bayesian approaches:
            """,
            font_size=30,
        ).next_to(text_task, DOWN, buff=0.4, aligned_edge=LEFT)
        self.play(Write(text_bayesian_comparison), run_time=SPEEDUP_TIME)
        self.wait(1)
        self.next_slide()

        img3 = (
            ImageMobject("./media/images/fine-tuning-llm-enns/fig3b.png")
            .move_to(ORIGIN)
            .shift(DOWN)
        )
        self.play(FadeIn(img3), run_time=SPEEDUP_TIME)
        self.wait(1)
        self.next_slide()


class LanguageModels(Slide):
    def create_GLUE_tasks(self, words):
        squares = VGroup()
        square_size = 1

        for word in words:
            square = Square(side_length=square_size)
            text = Text(word, font_size=20, color=WHITE).scale(0.5)
            square.add(text)
            squares.add(square)
        squares.arrange_in_grid(2, math.ceil(len(words)/2), buff=0.2)
        return squares

    def create_GLUE_slide(self, title):
        text_GLUE = Tex(
            r"""
            \textbf{General Language Understanding Evaluation (GLUE) Benchmark}""",
            font_size=30,
        ).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(Write(text_GLUE), run_time=SPEEDUP_TIME)

        words = [
            "CoLA",
            "SST-2",
            "MRPC",
            "DM",
            "WNLI",
            "MNLI\nmatched",
            "MNLI\nmismatched",
            "QNLI",
            "RTE",
            "QQP",
            "STS-B",
        ]
        squares = self.create_GLUE_tasks(words)

        self.play(*[Create(square) for square in squares], run_time=SPEEDUP_TIME)
        self.wait(1)

        squares_to_fade = VGroup()
        for square in squares:
            word = square[1].get_text()
            if (word == "WNLI") or (word == "STS-B") or (word == "DM"):
                squares_to_fade.add(square)

        self.play(
            *[square[0].animate.set_fill(GRAY, 0.5) for square in squares_to_fade] ,
            run_time=SPEEDUP_TIME,
        )
        self.play(
            *[FadeOut(square, shift=UP) for square in squares_to_fade], run_time=SPEEDUP_TIME
        )
        squares.remove(*[square for square in squares_to_fade])
        print(f"DEBUG - len(squares) = {len(squares)}\n\n\n")

        self.play(squares.animate.arrange_in_grid(2, math.ceil(len(squares)/2), buff=0.2), run_time=SPEEDUP_TIME)
        self.wait(1)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=SPEEDUP_TIME)
        self.next_slide()
        self.wait(1)

    def create_language_model_slide(self, title):
        text_bert = Tex(
            r"\textbf{BERT}", font_size=30
        ).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
        list_bert = BulletedList(
            r"Bidirectional Encoder Representations from Transformers",
            r"Encoder only architecture",
            r"Pretrained 100M parameters",
            r"Open-source implementations (also for GLUE)",
            r"Needs fine-tuning for specific tasks",
            font_size=25,
        )
        list_bert.next_to(text_bert, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(FadeIn(text_bert, shift=RIGHT), run_time=SPEEDUP_TIME)
        self.play(FadeIn(list_bert, shift=RIGHT), run_time=SPEEDUP_TIME)
        self.wait(1)
        words = [
            "CoLA",
            "SST-2",
            "MRPC",
            "MNLI\nmatched",
            "MNLI\nmismatched",
            "QNLI",
            "RTE",
            "QQP",
        ]

        squares = self.create_GLUE_tasks(words)
        squares.to_corner(DOWN + LEFT).scale(0.8)
        bert_box = BertExplanation.create_LM_figure(self, first_text="Pretrained\nBERT\nLLM", second_text="Task\nSpecific\nFine\ntuning", color=BLUE).to_edge(RIGHT).shift(LEFT)
        self.play(FadeIn(bert_box, shift=RIGHT), run_time=SPEEDUP_TIME)
        self.play(FadeIn(squares, shift=RIGHT), run_time=SPEEDUP_TIME)
        self.wait(1)
        BertExplanation.train_LM_animation(self, bert_box, num_data_points=20)
        self.wait(1)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=SPEEDUP_TIME)
        self.next_slide()
        self.wait(1)

    def create_baseline_slide(self, title):
        text_baseline = Tex(
            r"\textbf{Baseline Agent}", font_size=30
        ).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
        text_baseline_list = BulletedList(
            r"Trained by selecting a fixed and random subset of the training data",
            r"Sweeps over batch size $\in \{4, 16, 64\}$",
            r"Learning rate $\in \{1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4\}$",
            r"For each setting, perform 10 epochs of SGD training",
            font_size=25,
        ).next_to(text_baseline, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(Write(text_baseline), run_time=SPEEDUP_TIME)
        self.play(Write(text_baseline_list), run_time=SPEEDUP_TIME)
        self.wait(1)
        self.next_slide()
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=SPEEDUP_TIME)
        self.wait(1)
    
    def fine_tuning_slide(self, title):
        text_fine_tuning = Tex(
            r"\textbf{Fine-tuning}", font_size=30
        ).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
        

    #############################################
    def construct(self):
        text_title = Tex(
            r"\textbf{Language Models}", font_size=40
        ).to_edge(UP + LEFT)

        self.play(Write(text_title), run_time=SPEEDUP_TIME)
        self.create_GLUE_slide(text_title)
        self.next_slide()
        self.create_baseline_slide(text_title)
        self.create_language_model_slide(text_title)
