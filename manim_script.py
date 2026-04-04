from manim import *
import numpy as np
import random

class RadialGridlock(Scene):
    def construct(self):
        # 1. The Radial Network
        hub = Circle(radius=0.8, color=BLUE, fill_opacity=0.2)
        spokes = VGroup()
        
        for i in range(8):
            angle = i * (360 / 8) * DEGREES
            spoke = Line(ORIGIN, [4, 0, 0], color=WHITE)
            spoke.shift(RIGHT * 0.8)
            spoke.rotate(angle, about_point=ORIGIN)
            spokes.add(spoke)
        
        self.play(Create(hub), Create(spokes))
        
        # 2. Setup the Cars (Static dots first)
        cars = VGroup()
        for spoke in spokes:
            for p in [0.8, 0.6, 0.4]:
                car = Dot(radius=0.08, color=YELLOW).move_to(spoke.point_from_proportion(p))
                cars.add(car)
        
        self.play(FadeIn(cars))

        # 3. The Clump (Shockwave) Logic
        # We prepare a list of animations rather than adding them to a VGroup
        clump_animations = []
        
        for i, car in enumerate(cars):
            if i < 3: # First 3 cars are on the "jammed" spoke (spokes[0])
                target_pos = spokes[0].point_from_proportion(0.1 + (i * 0.05))
                clump_animations.append(car.animate.move_to(target_pos).set_color(RED))
            else:
                # Other cars just slow down slightly (move forward 10%)
                clump_animations.append(car.animate.set_color(YELLOW_E))

        # We play all prepared animations at once
        self.play(*clump_animations, run_time=2)
        
        # 4. Final Shockwave
        shockwave = Circle(radius=0.1, color=RED, stroke_width=10).move_to(ORIGIN)
        self.play(
            shockwave.animate.scale(60).set_opacity(0),
            hub.animate.set_color(RED),
            spokes.animate.set_color(RED_E),
            run_time=1.5
        )
        
        self.wait(2)

class BlindClock(Scene):
    def construct(self):
        # 2. Classical Solutions
        # Roads
        road_h = Line(LEFT * 4, RIGHT * 4, stroke_width=10, color=GRAY)
        road_v = Line(DOWN * 3, UP * 3, stroke_width=10, color=GRAY)
        self.add(road_h, road_v)

        # Traffic Light & Analog Clock
        light = Circle(radius=0.4, color=RED, fill_opacity=1).move_to(UP * 2 + LEFT * 2)
        clock_face = Circle(radius=0.6, color=WHITE).move_to(UP * 2 + LEFT * 3.5)
        clock_hand = Line(clock_face.get_center(), clock_face.get_top(), color=YELLOW)
        
        self.play(Create(clock_face), Create(clock_hand), FadeIn(light))

        # Massive pile of cars on right (blocked), zero on bottom
        blocked_cars = VGroup(*[Dot(color=YELLOW).move_to(RIGHT * (0.5 + i*0.2)) for i in range(15)])
        self.play(FadeIn(blocked_cars))

        # Clock ticks, light turns green for the EMPTY bottom road
        self.play(
            Rotate(clock_hand, angle=-PI, about_point=clock_face.get_center(), run_time=3),
            light.animate.set_color(GREEN)
        )
        
        text = Text("30 Seconds Empty Green", font_size=24, color=RED).next_to(road_v, DOWN)
        self.play(Write(text))
        self.wait(1)

class Awakening(Scene):
    def construct(self):
        # 3. Our Solution
        # Start with previous scene's setup
        clock = VGroup(Circle(radius=0.6, color=WHITE), Line(ORIGIN, UP, color=YELLOW)).move_to(UP*2 + LEFT*3.5)
        light = Circle(radius=0.4, color=RED, fill_opacity=1).move_to(UP * 2 + LEFT * 2)
        road_h = Line(LEFT * 4, RIGHT * 4, stroke_width=10, color=GRAY)
        blocked_cars = VGroup(*[Dot(color=YELLOW).move_to(RIGHT * (0.5 + i*0.2)) for i in range(15)])
        
        self.add(clock, light, road_h, blocked_cars)

        # Shatter the clock (represented by shrinking/fading quickly)
        self.play(clock.animate.scale(0.1).set_opacity(0), run_time=0.5)

        # Replace with Neural Network Mesh
        nodes = VGroup(*[Dot(radius=0.05, color=BLUE).move_to(UP*2 + LEFT*3.5 + np.array([random.uniform(-0.5,0.5), random.uniform(-0.5,0.5), 0])) for _ in range(10)])
        edges = VGroup(*[Line(nodes[i].get_center(), nodes[j].get_center(), stroke_width=1, color=BLUE_C) for i in range(10) for j in range(i+1, 10) if random.random() > 0.6])
        nn_mesh = VGroup(nodes, edges)
        
        self.play(Create(nn_mesh))

        # Visual Cone (Radar)
        radar = Polygon(light.get_center(), RIGHT*4 + UP*1, RIGHT*4 + DOWN*1, color=YELLOW, fill_opacity=0.3, stroke_width=0)
        self.play(FadeIn(radar))

        # Detection & Reaction
        self.play(Indicate(nn_mesh, color=YELLOW))
        self.play(light.animate.set_color(GREEN))
        
        # Clear cars
        self.play(blocked_cars.animate.shift(LEFT * 6), run_time=2)
        self.wait(1)

class CurveFit(Scene):
    def construct(self):
        # 4A. What is Machine Learning?
        axes = Axes(x_range=[0, 10], y_range=[0, 10], axis_config={"color": BLUE})
        
        # Messy Data
        data_points = [(1, 2), (2, 3), (3, 2.5), (4, 5), (5, 4), (6, 7), (7, 6.5), (8, 9), (9, 8)]
        dots = VGroup(*[Dot(axes.c2p(x, y), color=YELLOW) for x, y in data_points])
        
        self.play(Create(axes), FadeIn(dots))

        # Rigid Line (Classical) fails
        rigid_line = axes.plot(lambda x: 4, color=RED)
        self.play(Create(rigid_line))
        self.play(rigid_line.animate.shift(UP*2).rotate(0.2)) # Tries to fit but fails
        self.play(FadeOut(rigid_line))

        # Dynamic Line (ML) learns
        def ml_curve(x, iteration):
            # A mock function that gets closer to the dots over iterations
            target_poly = np.poly1d(np.polyfit([p[0] for p in data_points], [p[1] for p in data_points], 5))
            return (target_poly(x) * iteration) + (5 * (1 - iteration))

        dynamic_line = axes.plot(lambda x: ml_curve(x, 0), color=GREEN)
        self.play(Create(dynamic_line))
        
        # Animate the curve bending to fit the points
        for i in np.linspace(0.2, 1.0, 5):
            new_line = axes.plot(lambda x: ml_curve(x, i), color=GREEN)
            self.play(Transform(dynamic_line, new_line), run_time=0.4)
        
        self.wait(1)

class Scoreboard(Scene):
    def construct(self):
        # 1. SETUP: The Traffic Light & Environment
        light = Circle(radius=0.8, color=RED, fill_opacity=1).move_to(LEFT * 2)
        self.add(light)

        # --- PHASE A: FAILURE (Penalty) ---
        # The AI makes a bad choice (Turning Yellow too early/late)
        self.play(light.animate.set_color(YELLOW))
        
        # FIX: Replacing 'DropIn' with 'FadeIn' + 'shift'
        crash_text = Text("-50 Penalty", color=RED, font_size=40).move_to(RIGHT * 2)
        
        self.play(
            FadeIn(crash_text, shift=DOWN),
            # Start the "Indicate" slightly after the fade begins
            Indicate(crash_text, color=RED, run_time=1.5),
            Flash(light, color=RED),
            run_time=2
        )
        
        # AI "learns" from the mistake
        self.play(Wiggle(light))
        self.play(FadeOut(crash_text))

        # --- PHASE B: SUCCESS (Reward) ---
        # The AI optimizes the timing
        self.play(light.animate.set_color(GREEN))
        reward_text = Text("+100 Reward", color=GOLD, font_size=40).move_to(RIGHT * 2)
        
        # 2. Intelligence Bar Setup
        bar_bg = Rectangle(height=4, width=1, color=WHITE).move_to(RIGHT * 5)
        # Initialize at near-zero height, anchored to the bottom edge
        bar_fill = Rectangle(height=0.05, width=0.9, color=GREEN, fill_opacity=0.8)
        bar_fill.align_to(bar_bg, DOWN).shift(UP * 0.1)
        
        self.play(Create(bar_bg), FadeIn(bar_fill))
        
        # 3. TRANSFORMATION: Reward becomes Progress
        # We transform the reward text directly into the bar filling up
        self.play(
            ReplacementTransform(reward_text, bar_fill),
            bar_fill.animate.stretch_to_fit_height(3.8, about_edge=DOWN),
            run_time=2
        )
        
        # Final success indicator
        success_label = Text("OPTIMIZED", font_size=24, color=GREEN).next_to(bar_bg, UP)
        self.play(Write(success_label))
        self.wait(2)

class SelfishAgents(Scene):
    def construct(self):
        # 4C. Multi-Agent RL
        agent_a = Circle(radius=0.5, color=RED, fill_opacity=1).move_to(LEFT * 4)
        agent_b = Circle(radius=0.5, color=RED, fill_opacity=1).move_to(RIGHT * 4)
        
        score_a = Integer(0).next_to(agent_a, UP)
        score_b = Integer(0).next_to(agent_b, UP)
        
        self.play(FadeIn(agent_a, agent_b, score_a, score_b))

        # Cars moving from A
        cars = VGroup(*[Dot(color=YELLOW).move_to(LEFT * 4 + RIGHT * (i*0.2)) for i in range(10)])
        
        # Agent A acts selfishly
        self.play(agent_a.animate.set_color(GREEN))
        self.play(
            cars.animate.shift(RIGHT * 7),
            score_a.animate.set_value(1000), run_time=1.5
        )

        # Cars hit B, fiery gridlock (red flash)
        flash = Flash(RIGHT * 3, color=RED, line_length=1, num_lines=12)
        self.play(flash)
        self.play(score_b.animate.set_value(-5000), agent_b.animate.set_color(RED))
        self.wait(1)

class QMIX(Scene):
    def construct(self):
        # 4D. The Master Conductor
        # 8 glowing nodes at bottom
        nodes = VGroup(*[Dot(radius=0.2, color=BLUE).move_to(DOWN * 3 + LEFT * 3.5 + RIGHT * i) for i in range(8)])
        self.play(FadeIn(nodes))

        # Floating geometric brain (Matrix)
        brain = Polygon(LEFT*2+UP*3, RIGHT*2+UP*3, RIGHT*3+UP*1, LEFT*3+UP*1, color=PURPLE, fill_opacity=0.2)
        matrix_text = MathTex(r"W \ge 0").move_to(brain.get_center())
        
        self.play(Create(brain), Write(matrix_text))

        # Shoot beams up
        beams_up = VGroup(*[Line(node.get_center(), brain.get_bottom(), color=YELLOW) for node in nodes])
        self.play(*[ShowPassingFlash(beam.copy().set_color(YELLOW), time_width=0.5) for beam in beams_up])
        
        # Math resolves (pulse)
        self.play(Indicate(matrix_text, color=GREEN, scale_factor=1.5))

        # Shoot beams down (Synchronized Green Wave)
        beams_down = VGroup(*[Line(brain.get_bottom(), node.get_center(), color=GREEN) for node in nodes])
        self.play(*[ShowPassingFlash(beam.copy().set_color(GREEN), time_width=0.5) for beam in beams_down])
        
        self.play(*[node.animate.set_color(GREEN) for node in nodes])
        self.wait(1)

class MemoryTape(Scene):
    def construct(self):
        # 4E. Our Specific Model (V2 GRU)

        # 1. The GRU "Memory Box"
        gru_box = Rectangle(height=2.2, width=3.2, color=ORANGE, fill_opacity=0.2, stroke_width=2)
        gru_label = Text("GRU Memory", font_size=28, weight=BOLD).move_to(gru_box.get_top() + DOWN * 0.4)
        
        # 2. CREATE A MECHANICAL GEAR FROM SCRATCH
        # (This is a more professional approach using nested shapes)
        
        # The main body of the gear
        gear_body = Circle(radius=0.6, color=GRAY, fill_opacity=1, stroke_width=0)
        
        # Create the 12 teeth
        teeth = VGroup()
        for i in range(12):
            # Each tooth is a small rectangle, rotated and positioned radially
            tooth = Rectangle(height=0.25, width=0.15, color=GRAY, fill_opacity=1, stroke_width=0)
            angle = i * (360 / 12) * DEGREES
            tooth.rotate(angle)
            # Position it at the edge of the gear body
            tooth.move_to(gear_body.get_center() + 0.6 * np.array([np.cos(angle), np.sin(angle), 0]))
            teeth.add(tooth)
            
        # The center cutout hole
        gear_hole = Circle(radius=0.2, color=BLACK, fill_opacity=1, stroke_width=0)
        
        # Combine everything into one "Gear" object
        full_gear = VGroup(gear_body, teeth, gear_hole).scale(0.7).move_to(gru_box.get_center() + DOWN * 0.4)

        # Add the box, label, and gear to the scene
        self.play(FadeIn(gru_box), Write(gru_label), Create(full_gear))
        self.wait(0.5)

        # 3. CREATE THE FILM STRIP (THE CHRONOLOGICAL TAPE)
        
        # The main strip body
        strip_height = 1.0
        strip_width = 8.0 # Make it long
        film_strip_bg = Rectangle(height=strip_height, width=strip_width, color=WHITE, fill_opacity=1, stroke_width=2)
        
        # Create the top and bottom sprocket holes (a series of small squares)
        sprocket_holes = VGroup()
        num_holes = 20 # Add many holes for the scrolling effect
        hole_size = 0.12
        
        for i in range(num_holes):
            # Top hole
            top_hole = Square(side_length=hole_size, color=BLACK, fill_opacity=1, stroke_width=0)
            top_hole.move_to(film_strip_bg.get_top() + LEFT * (strip_width/2 - 0.2) + RIGHT * i * 0.4 + DOWN * 0.15)
            sprocket_holes.add(top_hole)
            
            # Bottom hole
            bottom_hole = Square(side_length=hole_size, color=BLACK, fill_opacity=1, stroke_width=0)
            bottom_hole.move_to(film_strip_bg.get_bottom() + LEFT * (strip_width/2 - 0.2) + RIGHT * i * 0.4 + UP * 0.15)
            sprocket_holes.add(bottom_hole)
            
        # Combine strip and holes
        film_strip = VGroup(film_strip_bg, sprocket_holes)
        
        # Text label for the tape
        tape_text = Text("Traffic History Tape (T-5m → Now)", font_size=20, color=BLACK).move_to(film_strip.get_center())
        
        # Position the complete tape to the left, ready to scroll
        complete_tape = VGroup(film_strip, tape_text).move_to(LEFT * 7 + UP * 1.5)
        
        # Add the tape and its text to the scene
        self.play(FadeIn(complete_tape))
        self.wait(0.5)

        # 4. ANIMATE: TAPE SCROLLS IN, GEAR TURNS
        # We use a single 'self.play' to synchronize both animations.
        self.play(
            # 1. The tape scrolls from the left into the GRU box
            complete_tape.animate.move_to(gru_box.get_center()).scale(0.05).set_opacity(0),
            
            # 2. The gear turns (we combine 'Rotate' with 'animate' for sync)
            Rotate(full_gear, angle=-2*PI, about_point=full_gear.get_center()),
            
            run_time=3, 
            rate_func=linear # Use linear for a steady, mechanical feel
        )
        self.wait(1)

        # 5. RESULT: THE "FUTURE PREDICTION"
        # A simple road and a ghost-car outline to represent the AI's prediction
        road = Line(DOWN*2.5 + LEFT*4, DOWN*2.5 + RIGHT*4, stroke_width=4, color=GRAY)
        self.play(Create(road))
        
        # The 'ghost' prediction: a semi-transparent car
        ghost_car = Dot(color=YELLOW, fill_opacity=0.3, radius=0.2).move_to(DOWN*2.5 + LEFT*3)
        ghost_label = Text("AI Predicted Traffic Wave", font_size=18, color=YELLOW).next_to(ghost_car, UP, buff=0.2)
        
        self.play(FadeIn(ghost_car), Write(ghost_label))
        # Animate the prediction moving along the road
        self.play(ghost_car.animate.shift(RIGHT * 6), run_time=2.5, rate_func=smooth)
        self.wait(2)

class TransitionScenes(Scene):
    def construct(self):
        # 5 & 6. Transitions for Screen Recordings
        text_5 = Text("Actual V2 Model Execution | SUMO Physics Engine", font_size=32, color=BLUE)
        self.play(Write(text_5))
        self.wait(2)
        self.play(FadeOut(text_5))
        
        text_6 = Text("Production Dashboard | Live Metrics", font_size=32, color=GREEN)
        self.play(Write(text_6))
        self.wait(2)
        self.play(FadeOut(text_6))
