from manim import *
from manim_slides import Slide
import re
from manim import Square
import matplotlib.pyplot as plt
import numpy as np


SPEEDUP_TIME = 0.5

# Fine tuning LLMs

class FineTuningLLMs(Slide):
    def construct(self):    
        square = Square(side_length=1, color=BLUE)
        LLM_text = Text("LLM", font_size=20).move_to(square.get_center())
        self.add(square, LLM_text)
        self.wait(2)


class BertExplanation(Slide):
    def create_LM_figure(self, text, color=BLUE):
        square = Square(side_length=1, color=color).add_background_rectangle(color=BLACK, opacity=1)
        LLM_text = Text(text, font_size=20, color=WHITE).move_to(square.get_center()).set_z_index(square.get_z_index() + 1)

        rect = (
                Rectangle(height=4, width=2, color=color, fill_color=BLACK)
                .next_to(square, direction=RIGHT * 2)
                .add_background_rectangle(color=BLACK, opacity=1)
            )
        rect.set_z_index(1)
        rect.background_rectangle.set_z_index(rect.get_z_index())
        fine_tun = Text("Fine\ntuning", font_size=20).move_to(rect.get_center())
        fine_tun.set_z_index(rect.get_z_index() + 1)
        group = VGroup(square, LLM_text, rect, fine_tun)

        return group

    def train_LM_animation(self, model, num_data_points):
        data_points = []
        for i in range(num_data_points):
            data_point = (
                Square(side_length=0.15 * model.get_height(), color=WHITE)
                .add_background_rectangle(color=BLACK, opacity=1)
            )
            if i == 0:
                data_point.next_to(model[2].get_top(), buff=0.5, direction=UP)
            else:
                data_point.next_to(data_points[-1].get_top(), buff=0.1, direction=UP)
            data_point.set_z_index(model[2].get_z_index() - 1)
            data_point.background_rectangle.set_z_index(data_point.get_z_index() - 1)
            data_points.append(data_point)

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

        animations = []
        for i in range(num_data_points):
            animations.append(FadeIn(data_points[i]))

        self.play(
            AnimationGroup(*animations, lag_ratio=0), run_time=SPEEDUP_TIME
        )
        self.wait(0.5)

        animations = []
        for i in range(num_data_points):
            animations.append(MoveToTarget(data_points[i]))

        self.play(
                AnimationGroup(*animations, lag_ratio=0), run_time=5*SPEEDUP_TIME
            )

        animations = []
        for i in range(num_data_points):
            animations.append(FadeOut(data_points[i]))

        self.play(
            AnimationGroup(*animations, lag_ratio=0), run_time=SPEEDUP_TIME
        )
        self.wait(1)

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
        self.next_slide()
        self.wait(1)
        self.play(FadeOut(bullet_point_list), run_time=SPEEDUP_TIME)

    ##########################################################################
    def construct(self):

        #1. Title slide
        title = Text(
            ("Fine-Tuning Language Models\n" "via Epistemic Neural Networks"),
            color=BLUE,
            font_size=50,
        )

        small_title = Text(
            ("Fine-Tuning Language Models via Epistemic Neural Networks"),
            color=BLUE,
            font_size=30,
        ).to_edge(UP)

        self.play(
            Write(title), run_time=SPEEDUP_TIME
        )

        #2. BERT Language model tasks before fine tuning
        self.next_slide()
        self.wait(1)
        self.play(Transform(title, small_title), run_time=SPEEDUP_TIME)

        model = self.create_LM_figure(text="BERT\nLLM", color=BLUE).shift(LEFT * 3 + DOWN*0.2)
        self.add(model[0])
        self.add(model[1])

        #3. BERT Language model tasks after fine tuning
        self.next_slide()
        self.wait(2)

        self.no_ft_bullet_point_list(model)
        self.play([Write(model[2]), Write(model[3])], run_time=SPEEDUP_TIME)
        self.wait(1)
        self.ft_bullet_point_list(model)

        #4. Training animation
        self.train_LM_animation(model, num_data_points=20)

        #5. Active learning
        self.wait(1)
        self.next_slide()
        #remove training animation stuff

        self.clear()

        # ft_techniques = VGroup()
        # ft_text = Text("Fine\nTuning", font_size=20).to_edge(LEFT)
        # box = SurroundingRectangle(ft_text, buff=0.1, color=BLUE)

        # ft_techniques.add(ft_text, box)

        # self.play(Transform(self.model, ft_techniques))
        

        #6. Fine tuning with ENN


# Active learning framework
class ActiveLearningFramework(Slide):
    def al_scatter_plot_animation(self):
        ax = Axes(x_range=[0, 20, 5], y_range=[0, 20, 5]).scale(0.5)
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
        self.next_slide()

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

        # self.play(
        #     LaggedStart(
        #         *[Write(circle) for circle in circles],
        #         lag_ratio=0.1,
        #     ),
        #     *[ApplyMethod(class1_dots[i].set_color, BLUE) for i in class1_indices],
        #     *[ApplyMethod(class2_dots[i].set_color, GREEN) for i in class2_indices],
        # )
        # self.next_slide()
        # self.wait(1)

        # self.play(FadeOut(circles), run_time=SPEEDUP_TIME)
        # self.play(*[ApplyMethod(dot.set_color, GRAY) for dot in selected_dots], run_time=SPEEDUP_TIME)

        scatter_plot = VGroup(ax, class1_dots, class2_dots, selected_dots, svm_line)
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
        self.play(Write(circle), run_time=SPEEDUP_TIME)
        self.play(ApplyMethod(selected_dot.set_color, color), run_time=SPEEDUP_TIME)
        self.play(ReplacementTransform(svm_line, new_svm_line), run_time=SPEEDUP_TIME)
        self.play(FadeOut(circle), run_time=SPEEDUP_TIME)

    def modify_text_time_step(self, full_notation_text, time_step):

        print("************************************")
        print(f"Full notation text: {full_notation_text}")
        print(f"Length of full notation text: {len(full_notation_text)}")
        print("************************************")

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

        self.play(
            ReplacementTransform(formula_time_step, new_formula_time_step),
            ReplacementTransform(formula_visible_data, new_formula_visible_data),
            ReplacementTransform(formula_class_label_t, new_formula_class_label_t),
            ReplacementTransform(formula_obtain, new_formula_obtain),
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
        )

        print("************************************")
        print(f"Type of new_full_notation_text: {type(new_full_notation_text)}")
        print("************************************")

        return new_full_notation_text

    def create_full_notation_text(self, title):
        full_notation_text = VGroup()

        al_text2 = Text(
            (
                "Active learning implies that a learning agent is able to prioritize "
                "training examples \nin order to improve performance "
                "on held out data."
            ),
            font_size=20,
        ).next_to(title, direction=DOWN, aligned_edge=LEFT)
        self.play(Write(al_text2), run_time=SPEEDUP_TIME)

        text_original_dataset = Text("Original dataset", font_size=20).next_to(
            al_text2, direction=DOWN, aligned_edge=LEFT
        )
        n_dataset = (
            MathTex(r"D = \{(x_{i}, y_{i}, i)\}_{i=1}^{N}")
            .scale(0.7)
            .next_to(text_original_dataset, direction=RIGHT, buff=1)
        )

        text_time_step = Text("Time step", font_size=20).next_to(
            text_original_dataset, direction=DOWN, aligned_edge=LEFT, buff=0.3
        )
        formula_time_step = (
            MathTex(r"t = 0")
            .scale(0.7)
            .next_to(n_dataset, direction=DOWN, aligned_edge=LEFT)
        )
        t_box = SurroundingRectangle(text_time_step, color=WHITE)

        text_pick = Text(f"Agent picks index", font_size=20).next_to(
            text_time_step, direction=DOWN, aligned_edge=LEFT, buff=0.3
        )

        text_obtain = Text("Agent obtains", font_size=20).next_to(
            text_pick, direction=DOWN, aligned_edge=LEFT, buff=0.3
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

        text_agent_sees = Text("The agent sees", font_size=20).next_to(
            text_time_step, direction=DOWN, aligned_edge=LEFT, buff=2
        )

        formula_visible_data = (
            MathTex(r"D_{X} = \{(x_i, i)\}_{i=1}^{N}")
            .scale(0.8)
            .next_to(text_agent_sees, direction=DOWN, aligned_edge=LEFT)
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
        )

        return full_notation_text

    def construct(self):
        title = Text("Active Learning Framework", font_size=30, color=BLUE).to_edge(UP+LEFT)
        self.play(Write(title), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.wait(1)

        problem = Text(("Fine-tuning a 'foundation model' can be much"
                        "more sample efficient than \nre-training from scratch, but"
                        "may still require substantial amounts of data"), font_size=20).next_to(title, direction=DOWN, aligned_edge=LEFT)

        self.add(problem)
        al_text = Text(("Active machine learning is about "
                        "choosing specific examples during learning."), font_size=20).move_to(problem)
        self.play(ReplacementTransform(problem, al_text))

        LLM_text = Text("LLM", font_size=20, color=WHITE).to_edge(LEFT)
        LLM_box = SurroundingRectangle(
            LLM_text, color=BLUE_B)
        LLM_figure = VGroup(LLM_box, LLM_text)
        self.add(LLM_figure)

        scatter_plot = (
            self.al_scatter_plot_animation()
        )

        scatter_plot_small = scatter_plot.copy().to_edge(RIGHT).scale(0.7)

        self.next_slide()

        self.play(ReplacementTransform(scatter_plot, scatter_plot_small), run_time=SPEEDUP_TIME)        
        self.play(
            *[FadeOut(obj) for obj in [LLM_figure, al_text]], run_time=SPEEDUP_TIME
        )

        full_notation_text = self.create_full_notation_text(title)

        print("************************************")
        print("Just after creating full notation text")
        print(f"Full notation text: {full_notation_text}")
        print(f"Length of full notation text: {len(full_notation_text)}")
        print("************************************")

        self.next_slide()
        self.wait(1)

        self.play(Write(full_notation_text, run_time=SPEEDUP_TIME))

        self.next_slide()
        self.wait(1)
        full_notation_text = self.modify_text_time_step(full_notation_text, 1)
        self.time_step_animation(scatter_plot_small, 0)
        print("************************************")
        print("Just after modifying full notation text")
        print(f"Full notation text: {full_notation_text}")
        print(f"Length of full notation text: {len(full_notation_text)}")
        print("************************************")

        self.wait(1)
        full_notation_text = self.modify_text_time_step(full_notation_text, 2)
        self.time_step_animation(scatter_plot_small, 1)
        self.wait(1)
        full_notation_text = self.modify_text_time_step(full_notation_text, 3)
        self.time_step_animation(scatter_plot_small, 2)
        self.wait(1)
