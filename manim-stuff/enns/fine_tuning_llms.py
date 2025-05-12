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
            font_size=70,
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

        line = Line(title.get_left(), title.get_right(), color=WHITE, stroke_width=1).shift(DOWN)

        self.play(Write(title), run_time=SPEEDUP_TIME)
        self.play(Write(line), run_time=SPEEDUP_TIME)
        self.play(Write(title_authors), run_time=SPEEDUP_TIME)

        # 2. Contents
        self.next_slide()
        self.play(FadeOut(line), run_time=SPEEDUP_TIME) 
        self.play(FadeOut(title_authors), run_time=SPEEDUP_TIME)
        self.play(Transform(title, small_title), run_time=SPEEDUP_TIME)

        # text_contents = Tex(r"\textbf{Contents}", font_size=30).next_to(
        #     small_title, DOWN * 2, buff=0.5, aligned_edge=LEFT
        # )

        # text_bullet_points = BulletedList(
        #     r"Fine tuning LLMs",
        #     r"Active learning framework",
        #     r"Statement of the paper",
        #     r"Priority functions",
        #     r"Training algorithm and loss function",
        #     r"Comparison of active learning agents for LLMs",
        #     r"Conclusion",
        #     font_size=25).next_to(text_contents, DOWN, buff=0.5, aligned_edge=LEFT)

        # self.play(FadeIn(text_contents), run_time=SPEEDUP_TIME)
        # self.play(FadeIn(text_bullet_points), run_time=SPEEDUP_TIME)
        # self.next_slide()
        # self.play(FadeOut(*[text for text in (text_contents, text_bullet_points)]), run_time=SPEEDUP_TIME)

        # 3. BERT before fine tuning
        model = BertExplanation.create_LM_figure(
            self=self,
            first_text=r"BERT\\LLM",
            second_text=r"Fine\\tuning", 
            color=BLUE
        ).shift(LEFT * 3 + DOWN * 0.2)
        # text_ideally = Text("Ideally", font_size=20).next_to(model, direction=LEFT)
        # self.play(Write(text_ideally), run_time=SPEEDUP_TIME)
        self.play(FadeIn(model[0]), run_time=SPEEDUP_TIME)
        BertExplanation.no_ft_bullet_point_list(self, model)
        self.play([Write(model[1]), GrowFromCenter(model[2])], run_time=SPEEDUP_TIME)

        # 4. BERT Language model tasks after fine tuning
        bplist, brace = BertExplanation.ft_bullet_point_list(self, model, False)
        self.play(Transform(model[2], Tex(r"ENN", font_size=30).move_to(model[2].get_center())), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(
            FadeOut(bplist), 
            FadeOut(brace),
            FadeOut(title),
            FadeOut(model), 
            run_time=SPEEDUP_TIME)
        
        # 5. Training animation
        # BertExplanation.train_LM_animation(self, model, num_data_points=20, speedup_factor=5, show_data_file=True)
        # fine_tuned_llm = BertExplanation.create_LM_figure(
        #     self=self,
        #     first_text="Fine-tuned\nBERT LLM",
        #     second_text="Fine\ntuning", 
        #     color=BLUE
        # ).next_to(model, direction=RIGHT*2)  
        # self.play(FadeIn(fine_tuned_llm[0]), run_time=SPEEDUP_TIME)
        # self.next_slide()
        # self.play(FadeOut(fine_tuned_llm[0]), run_time=SPEEDUP_TIME)

        # 5. Active learning
        # text_in_practice = Text("In practice", font_size=20).next_to(model, direction=LEFT)
        # self.play(ReplacementTransform(text_ideally, text_in_practice), run_time=SPEEDUP_TIME)
        # BertExplanation.train_LM_animation(self, model, num_data_points=500, speedup_factor=10)
        # # delete all mobjects to free memory
        # self.play(FadeOut(text_in_practice), run_time=SPEEDUP_TIME)
        
        # self.next_slide()
        # remove training animation stuff

        # ft_techniques = VGroup()
        # ft_text = Text("Fine\nTuning", font_size=20).to_edge(LEFT)
        # box = SurroundingRectangle(ft_text, buff=0.1, color=BLUE)

        # ft_techniques.add(ft_text, box)

        # self.play(Transform(model, ft_techniques))
        # self.next_slide()

        # 6. Fine tuning with ENN

# Active learning framework
class ActiveLearningFramework(Slide):
    def al_scatter_plot_animation(self, shift = None, fade_out_circles = False, faster=False, scale_factor: float = 1, show_line=True, highlight_dots=True):
        ax = create_scatter_plot([0, 20, 5], [0, 20, 5]).shift(shift).scale(scale_factor)
        class1_dots, class2_dots = generate_reproducible_2d_data_points_2_clusters(ax=ax)

        all_dots = class1_dots + class2_dots
        if show_line:
            svm_line = ax.plot(lambda x: 12 - 0.5 * x, color=RED).scale(scale_factor)
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

        if not faster:
            if show_line:
                self.play(Write(ax), run_time=SPEEDUP_TIME)
                self.play(
                    LaggedStart(*[Write(dot) for dot in all_dots], lag_ratio=0.1),
                    Write(svm_line),
                    run_time=SPEEDUP_TIME,
                )
            else:
                self.play(Write(ax), run_time=SPEEDUP_TIME)
                self.play(
                    LaggedStart(*[Write(dot) for dot in all_dots], lag_ratio=0.1),
                    run_time=SPEEDUP_TIME,
                )

            self.wait(1)

            if highlight_dots:
                self.play(
                    LaggedStart(
                        *[Write(circle) for circle in circles],
                        lag_ratio=0.1,
                    ),
                    *[ApplyMethod(class1_dots[i].set_color, BLUE) for i in class1_indices],
                    *[ApplyMethod(class2_dots[i].set_color, GREEN) for i in class2_indices],
                )
        elif faster:
            if show_line:
                self.play(
                    FadeIn(ax),
                    LaggedStart(*[FadeIn(dot) for dot in all_dots], lag_ratio=0.1),
                    FadeIn(svm_line),
                    run_time=SPEEDUP_TIME,
                )
            else:
                self.play(
                    FadeIn(ax),
                    LaggedStart(*[FadeIn(dot) for dot in all_dots], lag_ratio=0.1),
                    run_time=SPEEDUP_TIME,
                )

            
            if highlight_dots:
                self.play(
                    LaggedStart(
                        *[Write(circle) for circle in circles],
                        lag_ratio=0.1,
                    ),
                    *[ApplyMethod(class1_dots[i].set_color, BLUE) for i in class1_indices],
                    *[ApplyMethod(class2_dots[i].set_color, GREEN) for i in class2_indices],
                )

        if(fade_out_circles):
            if highlight_dots:
                self.play(FadeOut(circles), run_time=SPEEDUP_TIME)
                self.play(*[ApplyMethod(dot.set_color, GRAY) for dot in selected_dots], run_time=SPEEDUP_TIME)

        if show_line:
            scatter_plot = VGroup(ax, class1_dots, class2_dots, selected_dots, circles, svm_line)
        else:
            scatter_plot = VGroup(ax, class1_dots, class2_dots, selected_dots, circles)
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
        self.play(ApplyMethod(selected_dot.set_color, color),
                ReplacementTransform(svm_line, new_svm_line),
                run_time=SPEEDUP_TIME)
        # self.play(ReplacementTransform(svm_line, new_svm_line), run_time=SPEEDUP_TIME)
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
                MathTex(r"y_{a_{1}}", color=BLUE)
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
                MathTex(r"y_{a_{2}}", color=BLUE)
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
                MathTex(r"y_{a_{3}}", color=GREEN)
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
                r"A learning agent is able to prioritize "
                r"training examples in order to improve performance "
                r"on held out data."
            ),
            font_size=30,
        ).next_to(title, direction=DOWN, aligned_edge=LEFT)

        text_original_dataset = Tex(r"Original dataset", font_size=30).next_to(
            al_text2, direction=DOWN, aligned_edge=LEFT
        )
        n_dataset = (
            MathTex(r"D = \{(x_{i}, y_{i}, i)\}_{i=1}^{N}")
            .scale(0.7)
            .next_to(text_original_dataset, direction=RIGHT, buff=1)
        )

        text_time_step = Tex(r"Time step", font_size=30).next_to(
            text_original_dataset, direction=DOWN, aligned_edge=LEFT, buff=0.35
        )
        formula_time_step = (
            MathTex(r"t = 0")
            .scale(0.7)
            .next_to(n_dataset, direction=DOWN, aligned_edge=LEFT)
        )
        t_box = SurroundingRectangle(formula_time_step, color=WHITE)

        text_pick = Tex(r"Pick index", font_size=30).next_to(
            text_time_step, direction=DOWN, aligned_edge=LEFT, buff=0.35
        )

        text_obtain = Tex(r"Obtain", font_size=30).next_to(
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

        text_agent_sees = Tex(r"See", font_size=30).next_to(
            text_obtain, direction=DOWN, aligned_edge=LEFT, buff=0.35
        )

        formula_visible_data = (
            MathTex(r"D_{X} = \{(x_i, i)\}_{i=1}^{N}")
            .scale(0.7)
            .next_to(formula_obtain, direction=DOWN, aligned_edge=LEFT)
        )

        text_update_params = Tex(r"Update model \\parameters", font_size=30).next_to(
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
        al_text = (
            Tex((r"\textit{Active learning is about }"
                r"\textit{choosing specific examples during learning.}"
                ), font_size=40
            ).move_to(ORIGIN))
        box = SurroundingRectangle(al_text, color=BLUE, buff=0.12)

        al_text_box = VGroup(al_text, box)
        self.play(FadeIn(title), FadeIn(al_text), Create(box), run_time=SPEEDUP_TIME)
        self.next_slide()

        self.play(al_text_box.animate.next_to(title, direction=DOWN, buff=0.5, aligned_edge=LEFT), run_time=SPEEDUP_TIME)

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
            FadeOut(al_text_box), run_time=SPEEDUP_TIME
        )

        full_notation_text = self.create_full_notation_text(title)

        self.play(Write(full_notation_text, run_time=SPEEDUP_TIME))

        self.next_slide()

        full_notation_text = self.modify_text_time_step(full_notation_text, 1)
        self.wait(0.2)
        self.time_step_animation(scatter_plot_small, 0)
        self.wait(0.2)
        full_notation_text = self.modify_text_time_step(full_notation_text, 2)
        self.wait(0.2)
        self.time_step_animation(scatter_plot_small, 1)
        self.wait(0.2)

        full_notation_text = self.modify_text_time_step(full_notation_text, 3)
        self.wait(0.2)
        self.time_step_animation(scatter_plot_small, 2)
        self.next_slide()

        pick_index_text = full_notation_text[8]
        # move the pick index text to center and fade out the rest
        self.play(
            pick_index_text.animate.move_to(ORIGIN),
            *[FadeOut(obj) for obj in full_notation_text if obj != pick_index_text],
            FadeOut(scatter_plot_small),
            run_time=SPEEDUP_TIME,
        )

        self.next_slide()
        self.play(FadeOut(pick_index_text), run_time=SPEEDUP_TIME)
        # self.play(FadeOut(scatter_plot_small[4]), run_time=SPEEDUP_TIME)
        # self.play(*[ApplyMethod(dot.set_color, GRAY) for dot in scatter_plot[3]])
        self.play(
            FadeOut(title),
            run_time=SPEEDUP_TIME)
        self.next_slide()

class PriorityFunctions(Slide):
    def create_classification_squares(self, group_g_arrow, shift: np.ndarray=ORIGIN):
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
        ).shift(shift)
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

        priority_fun = PriorityFun(scene=self, position=ORIGIN+UP*1.7)
        priority_fun.create()
        scatter_plot = (
            ActiveLearningFramework.al_scatter_plot_animation(self, shift=LEFT*4.5, fade_out_circles=False, faster=True, scale_factor=0.7, show_line=False, highlight_dots=False)
        )

        group_g_arrow = VGroup(
            priority_fun.arrow_g_theta,
            priority_fun.formula_x,
            priority_fun.formula_g_theta_x,
            priority_fun.box_arrow,
        )

        squares_classes, text_classification_c = self.create_classification_squares(group_g_arrow, shift=DOWN*1.5)

        formula_p_class_with_z = MathTex(r"p(c|\theta, x, z) = softmax(f_{\theta}(x, z))_c").scale(0.7).next_to(group_g_arrow, direction=DOWN, buff=0.7, aligned_edge=LEFT)
        formula_p_class = (
            MathTex(r"p(c|\theta, x) = \int_z P_Z(dz)p(c|\theta, x, z)")
            .scale(0.7)
            .next_to(
                formula_p_class_with_z, direction=DOWN, buff=0.7, aligned_edge=LEFT
            )
        )

        self.play(FadeIn(title_priority_functions_small), run_time=SPEEDUP_TIME)
        self.play(*[FadeIn(formula, shift=DOWN) for formula in [formula_p_class_with_z, formula_p_class]], run_time=SPEEDUP_TIME)
        self.next_slide()

        self.play(FadeOut(scatter_plot), run_time=SPEEDUP_TIME)
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
        formula_entropy = (
            MathTex(r"g^{entropy}(\theta, x) = \mathbb{H}[p(\cdot |\theta, x)]")
            .scale(0.7)
            .next_to(text_entropy, direction=DOWN, aligned_edge=LEFT)
        )
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

        text_concept = Tex(
            r"Prefer examples with \\ high variability w.r.t z",
            font_size=30,
        ).next_to(box_enn_priority_functions, direction = LEFT, buff=0.5)
        self.play(
            Write(text_concept),
            run_time=SPEEDUP_TIME
        )
        self.next_slide()

        self.play(
            FadeOut(title_priority_functions_small),
            FadeOut(full_vgroup),
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
            ),
            FadeOut(text_concept),
            run_time=SPEEDUP_TIME,
        )

class FTTrainingAlgorithm(Slide):
    def construct(self):
        title = Tex(r"Training Algorithm and Loss Function", font_size=50, color=BLUE).to_edge(UP+LEFT)
        self.play(FadeIn(title), run_time=SPEEDUP_TIME)
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
        self.next_slide()
        self.play(
            FadeOut(title),
            FadeOut(training_algo),
            FadeOut(text_loss_function),
            FadeOut(formula_loss),
            run_time=SPEEDUP_TIME)

class ComparisonActiveLearningAgents(Slide):
    def construct(self):
        title = Tex(
            r"Comparison of Active Learning Agents",
            font_size=50,
            color=BLUE,
        ).to_edge(UP + LEFT)
        self.play(FadeIn(title), run_time=SPEEDUP_TIME)

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
                position=text_random_MLP.get_bottom() + DOWN + RIGHT,
                scale_factor=0.5,
            )
            .shift(DOWN)
            .scale(0.5)
        )
        formula5 = MathTex(r"P(y_t = y | \theta^*) = softmax(h_{\theta^*}(x_t)/\rho)_y",
                           font_size=30)
        formula5.next_to(neural_network, RIGHT)

        self.play(FadeIn(text_random_MLP), run_time=SPEEDUP_TIME)
        self.play(Write(neural_network), Write(formula5), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeOut(neural_network), FadeOut(text_random_MLP), FadeOut(formula5), run_time=SPEEDUP_TIME)
        ###############################################################
        self.play(FadeOut(text_neural_testbed), run_time=SPEEDUP_TIME)
        text_baseline_agent = Tex(
            r"Baseline agent:",
            font_size=25 ,
        ).next_to(text_task, DOWN, buff=0.4, aligned_edge=LEFT)

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
        self.play(FadeIn(baseline_agent), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.wait(1)

        self.play(FadeOut(baseline_agent, dir=LEFT), run_time=SPEEDUP_TIME)
        img = ImageMobject(
            "./media/images/fine-tuning-llm-enns/fig2.png"
            ).move_to(ORIGIN).shift(DOWN).scale(1.2)
        self.play(FadeIn(img), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeOut(img), run_time=SPEEDUP_TIME)
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

class LMExperiment(Slide):
    def create_GLUE_tasks(self, words):
        squares = VGroup()
        square_size = 1

        for word in words:
            square = Square(side_length=square_size)
            text = Text(word, font_size=30, color=WHITE).scale(0.5)
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
            "MNLI\nmat.",
            "MNLI\nmis.",
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

        self.next_slide()

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
        self.next_slide()
        self.play(
            FadeOut(text_GLUE),
            FadeOut(squares),
            run_time=SPEEDUP_TIME)


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
        BertExplanation.train_LM_animation(self, bert_box, num_data_points=20)
        self.next_slide()
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=SPEEDUP_TIME)
        self.next_slide()

    def create_baseline_slide(self, title):
        text_baseline = Tex(
            r"\textbf{Baseline Agent}", font_size=30
        ).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
        text_baseline_list = BulletedList(
            r"Does not perform active learning",
            r"Trained by selecting a fixed and random subset of the training data",
            r"Sweeps over batch size $\in \{4, 16, 64\}$",
            r"Learning rate $\in \{1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4\}$",
            r"For each setting, perform 10 epochs of SGD training",
            font_size=25,
        ).next_to(text_baseline, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(FadeIn(text_baseline, shift=RIGHT), run_time=SPEEDUP_TIME)
        self.play(FadeIn(text_baseline_list, shift=RIGHT), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=SPEEDUP_TIME)
    
    def fine_tuning_slide(self, title):
        text_fine_tuning = Tex(
            r"\textbf{Fine-tuning}", font_size=30
        ).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
        

    #############################################
    def construct(self):
        text_title = Tex(
            r"Language Models", font_size=50, color=BLUE
           ).to_edge(UP + LEFT)

        self.play(Write(text_title), run_time=SPEEDUP_TIME)
        self.create_GLUE_slide(text_title)
        # self.create_baseline_slide(text_title)
        # self.create_language_model_slide(text_title)
        

        agents_img = ImageMobject(
            "./media/images/fine-tuning-llm-enns/llm-agents.png"
        ).move_to(ORIGIN)
        formula_epinet = MathTex(
            r"\sigma_\eta (\tilde x, z) = (h_\eta(concat(\tilde x, z))+\lambda h^P (concat(\tilde x, z)))^T z",
            font_size=30
        ).next_to(agents_img, DOWN, buff=0.5)

        # self.play(FadeIn(agents_img), FadeIn(formula_epinet), run_time=SPEEDUP_TIME)
        # self.next_slide()
        self.play(
            # FadeOut(agents_img), 
            # FadeOut(formula_epinet), 
            FadeOut(text_title),
            run_time=SPEEDUP_TIME)
        self.next_slide()

class Results(Slide):
    def construct(self):
        title = Tex(
            r"Results", font_size=50, color=BLUE
        ).to_edge(UP + LEFT)
        fig4a = ImageMobject("./media/images/fine-tuning-llm-enns/fig4a.png").move_to(ORIGIN).shift(DOWN)
        fig4b = ImageMobject("./media/images/fine-tuning-llm-enns/fig4b.png").move_to(ORIGIN).shift(DOWN)
        fig5 = ImageMobject("./media/images/fine-tuning-llm-enns/fig5.png").move_to(ORIGIN).shift(DOWN)
        text_fig4a = Tex(r"Epinet prioritized by variance vs other methods that ",
                         r"do not use epistemic uncertainty (MNLI Dataset)", font_size=30).next_to(fig4a, UP, buff=0.5)
        text_fig4b = Tex(
            r"Agent performance when prioritizing by variance, ",
            r"changing the ENN architecture (MNLI Dataset)",
            font_size=30,
        ).next_to(fig4b, UP, buff=0.5)
        text_fig5 = Tex(
            r"Fine-tuning BERT models across GLUE tasks", font_size=30
        ).next_to(fig5, UP, buff=0.5)

        self.play(FadeIn(title), run_time=SPEEDUP_TIME)
        self.play(FadeIn(fig4a, shift=RIGHT), Write(text_fig4a), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeOut(fig4a, shift=RIGHT), run_time=SPEEDUP_TIME)
        self.wait(0.2)
        self.play(ReplacementTransform(text_fig4a, text_fig4b), run_time=SPEEDUP_TIME)
        self.play(FadeIn(fig4b, shift=RIGHT), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(FadeOut(fig4b, shift=RIGHT), run_time=SPEEDUP_TIME)
        self.wait(0.2)
        self.play(ReplacementTransform(text_fig4b, text_fig5), run_time=SPEEDUP_TIME)
        self.play(FadeIn(fig5, shift=RIGHT), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(
            FadeOut(fig5), FadeOut(title), FadeOut(text_fig5), run_time=SPEEDUP_TIME
        )

class Project(Slide):
    def flow_chart(self):
        text_1 = Tex(
            r"1. Load pretrained BERT model",
            font_size=30,
        ).move_to([-1, 1.4, 0])
        text_2 = Tex(
            r"2. Prepare STS dataset",
            font_size=30,
        ).move_to([0, 0.8, 0]).align_to(text_1, LEFT)
        text_3 = Tex(
            r"3. Create regression head for STS",
            font_size=30,
        ).move_to([0, 0.2, 0]).align_to(text_1, LEFT)
        text_4 = Tex(
            r"4. Define loss function for regression",
            font_size=30,
        ).move_to([0, -0.4, 0]).align_to(text_1, LEFT)
        text_5 = Tex(
            r"5. Define active learning priority functions",
            font_size=30,
        ).move_to([0, -1.0, 0]).align_to(text_1, LEFT)
        text_6 = (
            Tex(
                r"6. Fine-tune BERT with STS dataset",
                font_size=30,
            )
            .move_to([0, -1.6, 0])
            .align_to(text_1, LEFT)
        )

        flow_chart = VGroup(text_1, text_2, text_3, text_4, text_5, text_6)
        return flow_chart

    def first_flow_chart(self):
        text_1 = Tex(
            r"1. Get BERT pretrained model",
            font_size=30,
        ).move_to([-1, 1.4, 0])
        text_2 = Tex(
            r"2. Create a regression head",
            font_size=30,
        ).move_to([0, 0.8, 0]).align_to(text_1, LEFT)
        text_3 = Tex(
            r"3. Prepare dataset",
            font_size=30,
        ).move_to([0, 0.2, 0]).align_to(text_1, LEFT)
        text_4 = Tex(
            r"4. Fine tune model",
            font_size=30,
        ).move_to([0, -0.4, 0]).align_to(text_1, LEFT)

        flow_chart = VGroup(text_1, text_2, text_3, text_4)
        return flow_chart

    def second_flow_chart(self):
        text_1 = Tex(
            r"1. Get BERT-base pretrained model \\ from HuggingFace (PyTorch or TensorFlow)",
            font_size=30,
        ).move_to([1, 1.4, 0])
        text_2 = Tex(
            r"2. Modify BERT Haiku implementation to create classification head \\ (base model without ENNs)",
            font_size=30,
        ).move_to([0, 0.2, 0]).align_to(text_1, LEFT)
        text_3 = (
            Tex(
                r"3. Modify ENN library to add epinet architecture to classifier BERT base model.",
                font_size=30,
            )
            .move_to([0, -0.8, 0])
            .align_to(text_1, LEFT)
        )
        text_4 = Tex(
            r"4. Fine tune model",
            font_size=30,
        ).move_to([0, -1.2, 0]).align_to(text_1, LEFT)

        flow_chart = VGroup(text_1, text_2, text_3, text_4)
        return flow_chart

    def sst2_task(self):
        description_text = Tex(
            r'"\textit{hilariously inept \\ and ridiculous.}"',
            font_size=30,
        ).move_to([-5, 0, 0])
        bert_text = Tex(
            r"Fine-tuned \\ BERT",
            font_size=30,
        ).next_to(description_text, RIGHT, buff=0.5)
        bert_box = SurroundingRectangle(
            bert_text, color=WHITE, buff=0.1
        ).set_fill(BLUE, opacity=0.5)
        classification_head_text = Tex(
            r"Classification \\ head",
            font_size=30,
        ).next_to(bert_text, RIGHT, buff=0.5)
        head_box = SurroundingRectangle(
            classification_head_text, color=WHITE, buff=0.1
        ).set_fill(BLUE, opacity=0.5)
        text_positive = Tex(
            r"Positive",
            font_size=30,
        ).move_to([5, 1, 0])
        text_negative = Tex(
            r"Negative",
            font_size=30,
        ).move_to([5, -1, 0])
        arrow1 = Arrow(
            classification_head_text.get_right(),
            text_positive.get_left(),
            buff=0.1,
            color=WHITE,
        )
        arrow2 = Arrow(
            classification_head_text.get_right(),
            text_negative.get_left(),
            buff=0.1,
            color=WHITE,
        )

        sst2_task = VGroup(
            description_text,
            bert_text,
            bert_box,
            classification_head_text,
            head_box,
            text_positive,
            text_negative,
            arrow1,
            arrow2,
        )

        return sst2_task

    def construct(self):
        title = Tex(
            r"Project", font_size=50, color=BLUE
        ).to_edge(UP + LEFT)
        text_project = Tex(
            r"Reproducing Osband's results Stanford Sentiment Treebank (SST2) task",
            font_size=30,
        ).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)

        # SST2 slide
        first_scheme_group = self.sst2_task()

        self.play(
            FadeIn(title),
            FadeIn(text_project),
            *[Create(obj) for obj in first_scheme_group[:-2]],
            run_time=SPEEDUP_TIME,
        )
        self.wait(0.1)
        self.play(
            Write(first_scheme_group[-2]),
            Write(first_scheme_group[-1]),
            run_time=SPEEDUP_TIME,
        )
        self.next_slide()
        self.play(
            FadeOut(first_scheme_group),
            run_time=SPEEDUP_TIME
        )
        self.wait(0.2)

        first_flow_chart = self.first_flow_chart()

        flow_chart = self.second_flow_chart()
        flow_chart.shift(RIGHT*4)
        self.play(
            Write(first_flow_chart),
            run_time=SPEEDUP_TIME
        )
        self.next_slide()
        self.play(
            first_flow_chart.animate.shift(LEFT*2),
            run_time=SPEEDUP_TIME
        )
        self.wait(0.2)
        self.play(
            Write(flow_chart),
            run_time=SPEEDUP_TIME
        )
        self.next_slide()

        self.play(
            FadeOut(first_flow_chart),
            FadeOut(flow_chart),
            FadeOut(title),
            FadeOut(text_project),
            run_time=SPEEDUP_TIME
        )

class EpinetNetworkDiagram(Slide):

    def construct(self):
        # c = NumberPlane().add_coordinates()
        # self.play(Write(c))
        # Standard font size
        fs_main = 25
        fs_labels = 22

        # Colors
        base_color = BLUE
        feature_color = PURPLE
        epinet_color = RED
        param_color = BLUE_A
        input_color = YELLOW
        index_color = ORANGE
        reference_color = GREEN
        output_color = PINK

        # Base Net
        base_net_box = DashedVMobject(
            Rectangle(color=base_color, width=3.5, height=2.2).move_to(
                [-3, 2, 0]
            ), num_dashes=50
        )
        base_network = (
            Rectangle(width=2, height=0.6)
            .set_color(WHITE)
            .move_to(base_net_box.get_center() + DOWN * 0.4)
        )
        base_text = MathTex(r"\text{Base network } \mu", font_size=fs_main).move_to(
            base_network.get_center()
        )
        param_zeta = (
            Rectangle(width=1.6, height=0.6)
            .set_color(param_color)
            .next_to(base_network, UP, buff=0.3)
        )
        param_text = MathTex(r"\text{Parameters } \zeta", font_size=fs_labels).move_to(
            param_zeta.get_center()
        )
        base_net_label = Tex(
            r"\textbf{Base net}", font_size=fs_labels, color=base_color
        ).next_to(base_net_box, UP, buff=0.1)

        # Input
        input_box = Rectangle(color=input_color, height=0.8, width=1.5)
        input_text = MathTex(r"\text{Input } \mathbf{x}", font_size=fs_main).move_to(
            input_box.get_center()
        )
        input_group = VGroup(input_box, input_text).next_to(
            base_network.get_left(), LEFT, buff=1
        )

        # Epinet
        epinet_box = DashedVMobject(
            Rectangle(color=epinet_color, width=5, height=4)
            .move_to([3.5, -1, 0]),
            num_dashes=70
        )
        epinet_label = Tex(
            r"\textbf{Epinet}", font_size=fs_main, color=epinet_color
        ).next_to(epinet_box, UP, buff=0.1)

        # Prior network
        prior_box = (
            Rectangle(width=2.2, height=0.6)
            .set_color(GREY)
            .move_to(epinet_box.get_top() + DOWN * 0.8 + LEFT * 1.1)
        )
        prior_text = MathTex(
            r"\text{Prior network } \sigma^P", font_size=fs_labels
        ).move_to(prior_box.get_center())

        # Learnable
        learn_box = (
            Rectangle(width=2.2, height=0.6)
            .set_color(GREY)
            .next_to(prior_box, DOWN, buff=0.6)
        )
        learn_text = MathTex(
            r"\text{Learnable } \sigma^L", font_size=fs_labels
        ).move_to(learn_box.get_center())

        # Features
        features_box = Rectangle(color=feature_color, width=2.2, height=0.6).move_to(
            [base_net_box.get_center()[0], prior_box.get_center()[1], 0]
        )
        features_text = MathTex(
            r"\text{Features } \tilde{\mathbf{x}}", font_size=fs_main
        ).move_to(features_box.get_center())

        # Index z
        index_box = Rectangle(color=index_color, width=1.5, height=0.6).move_to(
            [features_box.get_center()[0], learn_box.get_center()[1], 0]
        )
        index_text = MathTex(r"\text{Index } \mathbf{z}", font_size=fs_main).move_to(
            index_box.get_center()
        )

        # Reference Z
        reference_box = Rectangle(color=reference_color, width=2.2, height=0.6).move_to(
            epinet_box.get_bottom() + UP*0.8 + LEFT * 1.1
        )
        reference_text = MathTex(
            r"\text{Reference } \mathbf{Z} \sim P_z", font_size=fs_labels
        ).move_to(reference_box.get_center())

        # Parameters η
        param_eta = (
            Rectangle(width=1.6, height=0.6)
            .set_color(param_color)
            .next_to(reference_box, RIGHT, buff=0.6)
        )
        param_eta_text = MathTex(
            r"\text{Parameters } \eta", font_size=fs_labels
        ).move_to(param_eta.get_center())

        # Output
        output_box = Rectangle(color=output_color, width=2.5, height=0.6).move_to(
            [param_eta.get_center()[0], base_net_box.get_center()[1], 0]
        )
        output_text = MathTex(
            r"\text{Output } f_\theta(\mathbf{x}, \mathbf{z})", font_size=fs_labels
        ).move_to(output_box.get_center())

        # Lines for arrows
        lines = [
            Line(  # 0
                reference_box.get_bottom(),
                reference_box.get_bottom() + DOWN * 0.25,
                color=WHITE,
            ),
            Line(  # 1
                reference_box.get_bottom() + DOWN * 0.25,
                [
                    index_box.get_center()[0],
                    (reference_box.get_bottom() + DOWN * 0.25)[1],
                    0,
                ],
                color=WHITE,
            ),
            Line(  # 2
                prior_box.get_right(),
                [
                    4.5,
                    prior_box.get_right()[1],
                    0,
                ],
                color=WHITE,
            ),
            Line(  # 3
                learn_box.get_right(),
                [
                    5.5,
                    learn_box.get_right()[1],
                    0,
                ],
                color=WHITE,
            ),
            Line(  # 4
                param_eta.get_top(),
                param_eta.get_top() + UP * 0.3,
                color=WHITE,
            ),
            Line(  # 5
                param_eta.get_top() + UP * 0.3,
                [
                    learn_box.get_center()[0],
                    (param_eta.get_top() + UP * 0.3)[1],
                    0,
                ],
                color=WHITE,
            ),
            Line(  # 6
                base_network.get_right(),
                [
                    -1,
                    base_network.get_right()[1],
                    0,
                ],
                color=WHITE,
            ),
            Line(  # 7
                [
                    -1,
                    base_network.get_right()[1],
                    0,
                ],
                [
                    -1,
                    output_box.get_left()[1],
                    0,
                ],
                color=WHITE,
            ),
        ]

        # Arrows tips
        arrows = [
            Arrow( #0
                input_group.get_right(),
                base_network.get_left(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.1,
                buff=0,
            ),
            Arrow( #1
                base_network.get_bottom(),
                features_box.get_top(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.15,
                buff=0,
            ),
            Arrow( #2
                lines[1].get_end(),
                index_box.get_bottom(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.1,
                buff=0,
            ),
            Arrow( #3
                features_box.get_right(),
                prior_box.get_left(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.05,
                buff=0,
            ),
            Arrow( #4
                features_box.get_right(),
                learn_box.get_left(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.05,
                buff=0,
            ),
            Arrow( #5
                index_box.get_right(),
                prior_box.get_left(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.05,
                buff=0,
            ),
            Arrow( #6
                index_box.get_right(),
                learn_box.get_left(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.05,
                buff=0,
            ),
            Arrow( #7
                lines[2].get_end(),
                [4.5, output_box.get_bottom()[1], 0],
                stroke_width=4,
                max_tip_length_to_length_ratio=0.1,
                buff=0,
            ),
            Arrow( #8
                lines[3].get_end(),
                [5.5, output_box.get_bottom()[1], 0],
                stroke_width=4,
                max_tip_length_to_length_ratio=0.08,
                buff=0,
            ),
            Arrow( #9
                param_zeta.get_bottom(), 
                base_network.get_top(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.5,
                buff=0,
            ),
            Arrow( #10
                lines[5].get_end(), 
                learn_box.get_bottom(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.5,
                buff=0,
            ),
            Arrow( #11
                lines[7].get_end(),
                output_box.get_left(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.1,
                buff=0,
            )
        ]

        # Add everything
        self.play(
            FadeIn(input_group),
            FadeIn(
                base_net_box,
                base_network,
                base_text,
                param_zeta,
                param_text,
                base_net_label,
            ),
            FadeIn(features_box, features_text),
            FadeIn(index_box, index_text),
            FadeIn(reference_box, reference_text),
            FadeIn(epinet_box, epinet_label),
            FadeIn(prior_box, prior_text),
            FadeIn(learn_box, learn_text),
            FadeIn(output_box, output_text),
            FadeIn(param_eta, param_eta_text),
            FadeIn(*arrows),
            FadeIn(*lines),
        )
        self.wait(0.3)
        self.next_slide()

        bert_network = (
            Rectangle(width=2, height=0.6)
            .set_color(WHITE)
            .move_to(base_net_box.get_center() + DOWN * 0.4)
        )
        bert_text = MathTex(r"\text{Bert-base } \mu", font_size=fs_main).move_to(
            base_network.get_center()
        )
        bert_net_label = Tex(
            r"\textbf{BERT}", font_size=fs_labels, color=base_color
        ).next_to(base_net_box, UP, buff=0.1)

        hdlayer_text = MathTex(
            r"\text{Final hidden layer}", font_size=fs_main
        ).move_to(features_box.get_center())

        class_head_box = (
            Rectangle(width=2.2, height=0.6)
            .set_color(GREY)
            .move_to([1, output_box.get_center()[1], 0])
        )
        class_head_text = Tex(
            r"\text{Classification head }", font_size=fs_labels
        ).move_to(class_head_box.get_center())
        new_arrow_11 = Arrow(
            class_head_box.get_right(),
            output_box.get_left(),
            stroke_width=4,
            max_tip_length_to_length_ratio=0.1,
            buff=0,
        )
        new_line_11 = Line(
            lines[7].get_end(),
            class_head_box.get_left(),
            color=WHITE,
        )

        self.play(
            ReplacementTransform(base_network, bert_network),
            run_time=SPEEDUP_TIME
        )
        self.play(ReplacementTransform(base_text, bert_text), run_time=SPEEDUP_TIME)
        self.play(ReplacementTransform(base_net_label, bert_net_label), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(ReplacementTransform(features_text, hdlayer_text), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.play(ReplacementTransform(arrows[11], new_arrow_11), run_time=SPEEDUP_TIME)
        self.play(Create(new_line_11), run_time=SPEEDUP_TIME)
        self.play(Create(class_head_box), run_time=SPEEDUP_TIME)
        self.play(Write(class_head_text), run_time=SPEEDUP_TIME)
        self.next_slide()


class Load_pre(Slide):
    def construct(self):
        title = Tex(
            r"Load pretrained BERT model", font_size=50, color=BLUE
        ).to_edge(UP + LEFT)




        # text_library = Tex(
        #     r"Haiku",
        #     font_size=30,
        # ).next_to(flow_chart[0], 3*RIGHT)
        # text_opt1 = Tex(
        #     r"Find Haiku pretrained BERT",
        #     font_size=20,
        # ).move_to([3, 2, 0])
        # text_opt2 = Tex(
        #     r"Do the pre training",
        #     font_size=20,
        # ).move_to([3, 0, 0  ])
        # arrow1 = Arrow(text_library.get_right(), text_opt1.get_left())
        # arrow2 = Arrow(text_library.get_right(), text_opt2.get_left())
        # group1 = VGroup(text_library, text_opt1, text_opt2, arrow1, arrow2)

        # self.play(
        #     Write(group1),
        #     run_time=SPEEDUP_TIME
        # )
        # self.next_slide()

class Conclusion(Slide):
    def construct(self):
        title = Tex(
            r"Conclusion", font_size=50, color=BLUE
        ).to_edge(UP + LEFT)
        subtitle_pros = Tex(
            r"Pros", font_size=30, color=GREEN
        ).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
        pros = BulletedList(
            r"Uncertainty in NNs is important for decision-making systems and active learning",
            r"ENNs are an efficient alternative to classical Bayesian methods",
            r"Language model fine tuning is possible in a more computationally tractable manner (OpenAI)",
            font_size=30
        ).next_to(subtitle_pros, DOWN, buff=0.5, aligned_edge=LEFT)
        subtitle_cons = Tex(
            r"Cons", font_size=30, color=RED
        ).next_to(pros, DOWN, buff=0.5, aligned_edge=LEFT)
        cons = BulletedList(
            r"Choice of reference distribution is arbitrary",
            r"Difficult to measure the uncertainty",
            r"Open source code in Haiku and JAX",
            r"Unclear how to train with epinet", 
            font_size=30,
        ).next_to(subtitle_cons, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(FadeIn(title), run_time=SPEEDUP_TIME)
        self.play(
            Write(subtitle_pros),
            Write(pros),
            Write(subtitle_cons),
            Write(cons),
            run_time=SPEEDUP_TIME,
        )
        self.next_slide()
        self.play(
            FadeOut(title),
            FadeOut(subtitle_pros), 
            FadeOut(pros),
            FadeOut(subtitle_cons),
            FadeOut(cons), 
            run_time=SPEEDUP_TIME)
        self.next_slide()

class References(Slide):
    def construct(self):
        title = Tex("References", font_size=50, color=BLUE).to_corner(UP + LEFT)

        refs = VGroup()

        refs_blist = BulletedList(
            # r"Balaji Lakshminarayanan. \textit{Introduction to Uncertainty in Deep Learning}. https://www.gatsby.ucl.ac.uk/~balaji/balaji-uncertainty-talk-cifar-dlrl.pdf",
            r"Osband et al. (2021). \textit{Epistemic Neural Networks}. arXiv:2107.08924.",
            r"Wen et al. (2021). \textit{From Predictions to Decisions: The Importance of Joint Predictive Distributions}. arXiv:2107.09224",
            r"Osband et al. (2022). \textit{Fine-Tuning Language Models via Epistemic Neural Networks}. arXiv:2211.01568",
            r"GLUE Benchmark https://gluebenchmark.com/",
            r"textit{TalkRL: The Reinforcement Learning Podcast} - Ian Osband episode https://www.talkrl.com/episodes/ian-osband",
            r"Stanford RL Forum - \textit{Epistemic Neural Networks} talk https://www.youtube.com/@stanfordrlforum6601",
            font_size=25
        ).next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)

        reference1 = (
            Tex(
                (
                    r"[1] Osband et al. (2023). \textit{Epistemic Neural Networks}.\\"
                    r"arXiv:2107.08924."
                ),
                font_size=30,
            )
            .next_to(title, DOWN, buff=0.5)
            .align_to(title, LEFT)
        )

        reference2 = (
            Tex(
                (
                    r"[2] Balaji Lakshminarayanan. \textit{Introduction to Uncertainty in Deep Learning}.\\"
                    r"https://www.gatsby.ucl.ac.uk/~balaji/balaji-uncertainty-talk-cifar-dlrl.pdf"
                ),
                font_size=30,
            )
            .next_to(reference1, DOWN, buff=0.5)
            .align_to(reference1, LEFT)
        )

        reference3 = (
            Tex(
                (
                    r"[3] Wen et al. (2022). From Predictions to Decisions: The Importance of Joint Predictive Distributions.\\"
                    r"arXiv:2107.09224"
                ),
                font_size=30,
            )
            .next_to(reference2, DOWN, buff=0.5)
            .align_to(reference2, LEFT)
        )

        reference4 = Tex(
            (
                r"[4] Osband et al. (2022). Fine-Tuning Language Models via Epistemic Neural Networks\\"
                r"arXiv:2211.01568"
            ),
            font_size=30,
        ).next_to(reference3, DOWN, buff=0.5, aligned_edge=LEFT)

        reference5 = Tex(
            (
                r"[5] Osband et al. Epistemic Neural Network slides\\"
                r"https://docs.google.com/presentation/d/1jCY9-\_vGkUV1wFcHxp07lWNF6XMITMZIiYdYnYT6IHs/edit?resourcekey=0-WceWVLKaJMiJ0VLXoPXANw\#slide=id.gad757c9405\_4\_449"
            ),
            font_size=30,
        ).next_to(reference4, DOWN, buff=0.5, aligned_edge=LEFT)

        reference6 = Tex(
            (
                r"[6] GLUE Benchmark\\"
                r"https://gluebenchmark.com/"
            ),
            font_size=30,
        ).next_to(reference5, DOWN, buff=0.5, aligned_edge=LEFT)

        reference7 = Tex(
            (
                r"[7] TalkRL: The Reinforcement Learning Podcast - Ian Osband episode\\"
                r"https://www.talkrl.com/episodes/ian-osband"
            ),
            font_size=30,
        )

        reference8 = Tex(
            (
                r"[8] Stanford RL Forum - Epistemic Neural Networks talk\\"
                r"https://www.youtube.com/@stanfordrlforum6601"
            ),
            font_size=30,
        )

        refs.add(reference1, reference2, reference3, reference4, reference5, reference6, reference7, reference8)
        refs.arrange_in_grid(8, 1, buff=0.5).to_corner(LEFT + DOWN)

        self.play(FadeIn(title), run_time=SPEEDUP_TIME)
        self.play(Write(refs_blist), run_time=SPEEDUP_TIME)
        self.next_slide()
