from manim import *
from manim_slides import Slide
import re
from manim import Square

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
        )  # Wrap the Text object in Write
        self.next_slide()
        self.wait(1)
        self.play(Transform(title, small_title), run_time=SPEEDUP_TIME)

        model = self.create_LM_figure(text="BERT\nLLM", color=BLUE).shift(LEFT * 3 + DOWN*0.2)

        print(f"***********************************\n\nmodel[0:1] = {model[0:1]}\n\n************************")

        self.add(model[0])
        self.add(model[1])
        self.wait(2)

        self.no_ft_bullet_point_list(model)
        self.play([Write(model[2]), Write(model[3])], run_time=SPEEDUP_TIME)
        self.wait(1)
        self.ft_bullet_point_list(model)

        self.train_LM_animation(model, num_data_points=20)

# Active learning framework
class ActiveLearningFramework(Slide):
    def construct(self):
        title = Text("Active Learning Framework", font_size=30, color=BLUE).to_edge(UP+LEFT)
        self.play(Write(title), run_time=SPEEDUP_TIME)
        self.next_slide()
        self.wait(1)

        problem = Text(("Fine-tuning a 'foundation model' can be much"
                        "more sample efficient than \nre-training from scratch, but"
                        "may still require substantial amounts of data"), font_size=20).next_to(title, direction=DOWN, aligned_edge=LEFT)

        self.add(problem)
        self.wait(2)
