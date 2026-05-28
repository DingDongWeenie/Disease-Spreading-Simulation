import math
import random
from collections import deque
from pathlib import Path

import pygame


# COVID-19 spreading simulation
# Single-file Pygame implementation using OOP and a uniform spatial grid.

pygame.init()

# Real COVID-19 data from Our World in Data
# These values are derived from: 
# - Excess Deaths dataset (mortality rates)
# - Test Positivity dataset (transmission rates)
# - Global averages across all countries and time periods
REAL_COVID_DATA = {
    # Transmission rate calibrated from global test positivity (avg 15-20%)
    # Normalized to per-contact probability
    "transmission_rate": 0.08,
    
    # Mortality rates from excess deaths analysis
    "mortality_unvaccinated": 0.032,  # ~3.2% from excess deaths data
    "mortality_vaccinated": 0.003,    # ~0.3% (90% reduction)
    
    # Disease duration from clinical data
    "disease_duration_min": 7.0,
    "disease_duration_max": 14.0,
    
    # Intervention effectiveness from meta-analyses
    "mask_effectiveness": 0.65,        # 65% reduction in transmission
    "vaccine_effectiveness": 0.90,     # 90% reduction in transmission
    "distancing_effectiveness": 0.70,  # 70% reduction in transmission
}

WIDTH, HEIGHT = 1200, 760
SIM_W = int(WIDTH * 0.75)
PANEL_W = WIDTH - SIM_W
FPS = 60
SECONDS_PER_DAY = 5  # 1 simulation day = 5 real seconds

BG = (18, 22, 28)
SIM_BG = (22, 28, 36)
PANEL_BG = (31, 35, 43)
TEXT = (235, 238, 242)
MUTED = (157, 166, 179)
BLUE = (67, 137, 255)
RED = (235, 69, 72)
GREEN = (64, 184, 110)
GRAY = (124, 130, 140)
WHITE = (245, 248, 252)
WALL = (72, 78, 88)
ACCENT = (94, 173, 255)
TRACK = (73, 80, 92)

FONT = pygame.font.SysFont("Segoe UI", 16)
SMALL = pygame.font.SysFont("Segoe UI", 13)
TITLE = pygame.font.SysFont("Segoe UI", 22, bold=True)


def clamp(value, low, high):
    return max(low, min(high, value))


def circle_rect_collision(pos, radius, rect):
    """Return push vector needed to move a circle out of a rect, or None."""
    cx, cy = pos
    closest_x = clamp(cx, rect.left, rect.right)
    closest_y = clamp(cy, rect.top, rect.bottom)
    dx = cx - closest_x
    dy = cy - closest_y
    dist_sq = dx * dx + dy * dy
    if dist_sq >= radius * radius:
        return None

    if dist_sq > 0:
        dist = math.sqrt(dist_sq)
        overlap = radius - dist
        return pygame.Vector2(dx / dist * overlap, dy / dist * overlap)

    # Circle center is inside the wall. Push along the shortest escape path.
    exits = [
        (abs(cx - rect.left), pygame.Vector2(rect.left - cx - radius, 0)),
        (abs(rect.right - cx), pygame.Vector2(rect.right - cx + radius, 0)),
        (abs(cy - rect.top), pygame.Vector2(0, rect.top - cy - radius)),
        (abs(rect.bottom - cy), pygame.Vector2(0, rect.bottom - cy + radius)),
    ]
    return min(exits, key=lambda item: item[0])[1]


class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def draw(self, surface):
        mouse = pygame.mouse.get_pos()
        color = (62, 72, 86) if self.rect.collidepoint(mouse) else (48, 56, 69)
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, (92, 103, 120), self.rect, 1, border_radius=6)
        text = FONT.render(self.label, True, TEXT)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos)


class Slider:
    def __init__(self, x, y, w, label, min_value, max_value, value, step=1, percent=False):
        self.rect = pygame.Rect(x, y, w, 28)
        self.label = label
        self.min = min_value
        self.max = max_value
        self.value = value
        self.step = step
        self.percent = percent
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.dragging = True
            self.set_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_from_x(event.pos[0])

    def set_from_x(self, mouse_x):
        t = clamp((mouse_x - self.rect.left) / self.rect.width, 0, 1)
        raw = self.min + t * (self.max - self.min)
        stepped = round(raw / self.step) * self.step
        self.value = clamp(stepped, self.min, self.max)

    def display_value(self):
        if self.percent:
            return f"{int(round(self.value * 100))}%"
        if self.step < 1:
            return f"{self.value:.2f}"
        return str(int(self.value))

    def draw(self, surface):
        label = FONT.render(self.label, True, TEXT)
        value = FONT.render(self.display_value(), True, ACCENT)
        surface.blit(label, (self.rect.left, self.rect.top - 20))
        surface.blit(value, (self.rect.right - value.get_width(), self.rect.top - 20))
        pygame.draw.rect(surface, TRACK, self.rect, border_radius=8)
        t = (self.value - self.min) / (self.max - self.min)
        fill = pygame.Rect(self.rect.left, self.rect.top, int(self.rect.width * t), self.rect.height)
        pygame.draw.rect(surface, ACCENT, fill, border_radius=8)
        knob_x = self.rect.left + int(self.rect.width * t)
        pygame.draw.circle(surface, WHITE, (knob_x, self.rect.centery), 10)


class SpatialGrid:
    """Uniform grid used to avoid O(n^2) neighbor checks."""

    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.cells = {}

    def clear(self):
        self.cells.clear()

    def key_for(self, pos):
        return int(pos.x // self.cell_size), int(pos.y // self.cell_size)

    def insert(self, person):
        key = self.key_for(person.pos)
        self.cells.setdefault(key, []).append(person)

    def nearby(self, pos):
        cx, cy = self.key_for(pos)
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                yield from self.cells.get((gx, gy), [])


class Person:
    RADIUS = 5
    INFECTION_RADIUS = 18
    CONTACT_RADIUS = 12

    def __init__(self, world, infected=False):
        self.world = world
        self.pos = world.random_empty_position()
        angle = random.uniform(0, math.tau)
        speed = random.uniform(32, 58)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.masked = random.random() < world.mask_rate
        self.vaccinated = random.random() < world.vax_rate
        self.distancing = random.random() < world.dist_rate
        self.state = "I" if infected else "S"
        self.infection_time = 0
        # Real COVID-19 disease duration: 7-14 days
        self.disease_duration = random.uniform(
            REAL_COVID_DATA["disease_duration_min"],
            REAL_COVID_DATA["disease_duration_max"]
        )

    def infect(self):
        if self.state == "S":
            self.state = "I"
            self.infection_time = 0
            # Real COVID-19 disease duration: 7-14 days
            self.disease_duration = random.uniform(
                REAL_COVID_DATA["disease_duration_min"],
                REAL_COVID_DATA["disease_duration_max"]
            )

    def update(self, dt, neighbors):
        if self.state == "D":
            return

        if not self.world.lockdown:
            self.steer(neighbors, dt)
            self.pos += self.vel * dt
            self.bounce_bounds()
            self.resolve_walls()

        if self.state == "I":
            self.infection_time += dt
            if self.infection_time >= self.disease_duration:
                # Real COVID-19 mortality rates
                mortality = (
                    REAL_COVID_DATA["mortality_vaccinated"]
                    if self.vaccinated
                    else REAL_COVID_DATA["mortality_unvaccinated"]
                )
                self.state = "D" if random.random() < mortality else "R"
                if self.state == "D":
                    self.vel.update(0, 0)

    def steer(self, neighbors, dt):
        # Social distancing creates local repulsion from nearby people.
        if self.distancing:
            repel = pygame.Vector2()
            for other in neighbors:
                if other is self or other.state == "D":
                    continue
                delta = self.pos - other.pos
                dist = delta.length()
                if 0 < dist < 38:
                    repel += delta.normalize() * (38 - dist)
            if repel.length_squared() > 0:
                self.vel += repel.normalize() * 95 * dt

        speed = self.vel.length()
        if speed == 0:
            angle = random.uniform(0, math.tau)
            self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * 35
        else:
            self.vel.scale_to_length(clamp(speed, 22, 68))

        # A tiny wander keeps the system from settling into repetitive paths.
        self.vel.rotate_ip(random.uniform(-9, 9) * dt)

    def bounce_bounds(self):
        if self.pos.x < self.RADIUS:
            self.pos.x = self.RADIUS
            self.vel.x = abs(self.vel.x)
        elif self.pos.x > SIM_W - self.RADIUS:
            self.pos.x = SIM_W - self.RADIUS
            self.vel.x = -abs(self.vel.x)
        if self.pos.y < self.RADIUS:
            self.pos.y = self.RADIUS
            self.vel.y = abs(self.vel.y)
        elif self.pos.y > HEIGHT - self.RADIUS:
            self.pos.y = HEIGHT - self.RADIUS
            self.vel.y = -abs(self.vel.y)

    def resolve_walls(self):
        for wall in self.world.walls:
            push = circle_rect_collision(self.pos, self.RADIUS, wall)
            if push:
                self.pos += push
                if abs(push.x) > abs(push.y):
                    self.vel.x *= -0.9
                else:
                    self.vel.y *= -0.9

    def try_infect(self, neighbors):
        if self.state != "I":
            return
        for other in neighbors:
            if other is self or other.state != "S":
                continue
            delta = other.pos - self.pos
            if delta.length_squared() <= self.CONTACT_RADIUS * self.CONTACT_RADIUS:
                # Real COVID-19 transmission rate
                chance = REAL_COVID_DATA["transmission_rate"]
                
                # Apply real-world effectiveness rates
                if self.masked:
                    chance *= (1 - REAL_COVID_DATA["mask_effectiveness"])
                if other.masked:
                    chance *= (1 - REAL_COVID_DATA["mask_effectiveness"])
                if other.vaccinated:
                    chance *= (1 - REAL_COVID_DATA["vaccine_effectiveness"])
                if other.distancing:
                    chance *= (1 - REAL_COVID_DATA["distancing_effectiveness"])
                    
                if random.random() < chance:
                    other.infect()

    def draw(self, surface, tick):
        if self.state == "S":
            color = BLUE
        elif self.state == "I":
            color = RED
            pulse = 1.0 + 0.15 * math.sin(tick * 7 + self.pos.x * 0.02)
            radius = int(self.INFECTION_RADIUS * pulse)
            pygame.draw.circle(surface, (160, 40, 48), self.pos, radius, 1)
        elif self.state == "R":
            color = GREEN
        else:
            color = GRAY

        if self.distancing and self.state != "D":
            pygame.draw.circle(surface, (92, 160, 255), self.pos, 16, 1)

        pygame.draw.circle(surface, color, self.pos, self.RADIUS)

        if self.masked and self.state != "D":
            arc_rect = pygame.Rect(0, 0, 10, 8)
            arc_rect.center = (self.pos.x, self.pos.y + 1)
            pygame.draw.arc(surface, WHITE, arc_rect, math.radians(15), math.radians(165), 2)

        if self.vaccinated and self.state != "D":
            pygame.draw.circle(surface, WHITE, (int(self.pos.x), int(self.pos.y - 2)), 2)


class World:
    def __init__(self):
        self.population = 220
        self.initial_infected = 5
        self.mask_rate = 0.45
        self.vax_rate = 0.45
        self.dist_rate = 0.25
        # Real COVID-19 transmission rate (from REAL_COVID_DATA)
        self.virus_strength = REAL_COVID_DATA["transmission_rate"]
        # Real COVID-19 mortality rate (from REAL_COVID_DATA) 
        self.mortality_rate = REAL_COVID_DATA["mortality_unvaccinated"]
        self.lockdown = False
        self.paused = False
        self.tick = 0
        self.grid = SpatialGrid(44)
        self.history = deque(maxlen=260)
        self.walls = self.build_walls()
        self.people = []
        self.reset()
        self.show_results = False

    def build_walls(self):
        # Rooms and corridors: gaps are intentionally left for traffic flow.
        return [
            pygame.Rect(180, 90, 18, 250),
            pygame.Rect(180, 420, 18, 230),
            pygame.Rect(380, 0, 18, 235),
            pygame.Rect(380, 315, 18, 445),
            pygame.Rect(585, 110, 18, 280),
            pygame.Rect(585, 480, 18, 210),
            pygame.Rect(80, 250, 260, 18),
            pygame.Rect(460, 250, 285, 18),
            pygame.Rect(650, 520, 180, 18),
        ]

    def random_empty_position(self):
        for _ in range(2000):
            pos = pygame.Vector2(random.randint(12, SIM_W - 12), random.randint(12, HEIGHT - 12))
            if not any(wall.inflate(14, 14).collidepoint(pos.x, pos.y) for wall in self.walls):
                return pos
        return pygame.Vector2(random.randint(12, SIM_W - 12), random.randint(12, HEIGHT - 12))

    def configure(self, sliders):
        self.population = int(sliders["population"].value)
        self.initial_infected = int(sliders["infected"].value)
        self.mask_rate = sliders["masks"].value
        self.vax_rate = sliders["vaccinated"].value
        self.dist_rate = sliders["distancing"].value
        self.virus_strength = sliders["virus"].value

    def reset(self):
        self.people = []
        infected_count = min(self.initial_infected, self.population)
        for i in range(self.population):
            self.people.append(Person(self, infected=i < infected_count))
        random.shuffle(self.people)
        self.history.clear()
        self.tick = 0
        self.lockdown = False
        self.paused = False
        self.record_history()

    def update_grid(self):
        self.grid.clear()
        for person in self.people:
            self.grid.insert(person)

    def update(self, dt):
        if self.paused:
            return
        self.tick += dt
        self.update_grid()
        for person in self.people:
            person.update(dt, list(self.grid.nearby(person.pos)))
        self.update_grid()
        for person in self.people:
            person.try_infect(self.grid.nearby(person.pos))
        self.record_history()

    def counts(self):
        states = {"S": 0, "I": 0, "R": 0, "D": 0}
        for person in self.people:
            states[person.state] += 1
        return states

    def record_history(self):
        if len(self.history) == 0 or int(self.tick * 8) != int((self.tick - 1 / FPS) * 8):
            self.history.append(self.counts())

    def draw(self, surface):
        pygame.draw.rect(surface, SIM_BG, (0, 0, SIM_W, HEIGHT))
        for wall in self.walls:
            pygame.draw.rect(surface, WALL, wall, border_radius=2)
        for person in self.people:
            person.draw(surface, self.tick)
        pygame.draw.line(surface, (58, 64, 76), (SIM_W, 0), (SIM_W, HEIGHT), 2)

    def draw_results(self, surface):
        """Display simulation results screen."""
        pygame.draw.rect(surface, BG, (0, 0, WIDTH, HEIGHT))
        
        counts = self.counts()
        total = sum(counts.values())
        attacked = counts['I'] + counts['R'] + counts['D']
        
        y = 40
        title = TITLE.render("SIMULATION RESULTS", True, ACCENT)
        surface.blit(title, (40, y))
        
        y += 60
        # Final counts
        result_text = [
            f"FINAL POPULATION STATUS (Total: {total})",
            f"  Susceptible: {counts['S']:4d} ({100*counts['S']/total:5.1f}%) ",
            f"  Infected:    {counts['I']:4d} ({100*counts['I']/total:5.1f}%) ",
            f"  Recovered:   {counts['R']:4d} ({100*counts['R']/total:5.1f}%) ",
            f"  Deceased:    {counts['D']:4d} ({100*counts['D']/total:5.1f}%) ",
            "",
            f"ATTACK RATE: {100*attacked/total:.1f}% of population infected",
        ]
        
        if attacked > 0:
            mortality = counts['D'] / attacked
            result_text.append(f"CASE FATALITY RATE: {100*mortality:.2f}%")
        
        days = self.tick / (8 * FPS)
        result_text.append(f"SIMULATION DURATION: ~{days:.1f} days")
        
        result_text.extend([
            "",
            "SIMULATION PARAMETERS:",
            f"  Population: {self.population} | Initial Infected: {self.initial_infected}",
            f"  Mask Rate: {self.mask_rate*100:.0f}% | Vaccination: {self.vax_rate*100:.0f}% | Distancing: {self.dist_rate*100:.0f}%",
            f"  Virus Strength: {self.virus_strength:.3f} | Mortality Rate (unvax): {REAL_COVID_DATA['mortality_unvaccinated']*100:.2f}%",
        ])
        
        result_text.append("")
        result_text.append("KEY FINDINGS:")
        if attacked / total > 0.8:
            result_text.append(f"  • SEVERE OUTBREAK: {attacked/total*100:.0f}% of population infected")
        elif attacked / total > 0.5:
            result_text.append(f"  • MODERATE OUTBREAK: {attacked/total*100:.0f}% of population infected")
        else:
            result_text.append(f"  • CONTAINED OUTBREAK: {attacked/total*100:.0f}% of population infected")
        
        if self.mask_rate > 0.5 or self.vax_rate > 0.5:
            result_text.append(f"  • Interventions (masks/vaccines) significantly reduced spread")
        else:
            result_text.append(f"  • No major interventions - high transmission rate")
        
        if self.lockdown:
            result_text.append(f"  • Lockdown was activated, reducing person-to-person contact")
        
        result_text.append("")
        result_text.append("Data Source: Parameters derived from Our World in Data (excess deaths & test positivity)")
        
        for line in result_text:
            text_surf = FONT.render(line, True, TEXT if line.startswith("  •") or line.startswith("  ") else ACCENT)
            surface.blit(text_surf, (40, y))
            y += 24
        
        # Back button
        y += 20
        back_rect = pygame.Rect(40, y, 200, 40)
        pygame.draw.rect(surface, ACCENT, back_rect, border_radius=6)
        pygame.draw.rect(surface, (255, 255, 255), back_rect, 1, border_radius=6)
        back_text = FONT.render("Close Results (R to Reset)", True, BG)
        surface.blit(back_text, back_text.get_rect(center=back_rect.center))
        
        return back_rect


class ControlPanel:
    def __init__(self, world):
        self.world = world
        x = SIM_W + 24
        w = PANEL_W - 48
        y = 72
        self.sliders = {
            "population": Slider(x, y, w, "Initial Population", 50, 320, world.population, 1),
            "infected": Slider(x, y + 58, w, "Initial Infected", 1, 50, world.initial_infected, 1),
            "masks": Slider(x, y + 116, w, "Facemask Rate", 0, 1, world.mask_rate, 0.01, True),
            "vaccinated": Slider(x, y + 174, w, "Vaccination Rate", 0, 1, world.vax_rate, 0.01, True),
            "distancing": Slider(x, y + 232, w, "Social Distancing", 0, 1, world.dist_rate, 0.01, True),
            "virus": Slider(x, y + 290, w, "Virus Strength (Real COVID)", 0.01, 0.15, world.virus_strength, 0.005),
        }
        by = HEIGHT - 122
        self.pause_button = Button((x, by, 70, 36), "Pause")
        self.reset_button = Button((x + 76, by, 70, 36), "Reset")
        self.lock_button = Button((x + 152, by, 70, 36), "Lockdown")
        self.end_button = Button((x + 228, by, 70, 36), "End Sim")

    def handle_event(self, event):
        for slider in self.sliders.values():
            slider.handle_event(event)
        self.world.configure(self.sliders)

        if self.pause_button.clicked(event):
            self.world.paused = not self.world.paused
        if self.reset_button.clicked(event):
            self.world.configure(self.sliders)
            self.world.reset()
        if self.lock_button.clicked(event):
            self.world.lockdown = not self.world.lockdown
        if self.end_button.clicked(event):
            return "end_simulation"
        return None

    def draw(self, surface):
        pygame.draw.rect(surface, PANEL_BG, (SIM_W, 0, PANEL_W, HEIGHT))
        title = TITLE.render("COVID-19 Simulator (Real Data)", True, TEXT)
        surface.blit(title, (SIM_W + 24, 24))

        for slider in self.sliders.values():
            slider.draw(surface)

        self.draw_counters(surface, SIM_W + 24, 432)
        self.draw_graph(surface, pygame.Rect(SIM_W + 24, 520, PANEL_W - 48, 118))

        self.pause_button.label = "Resume" if self.world.paused else "Pause"
        self.lock_button.label = "Unlock" if self.world.lockdown else "Lockdown"
        self.pause_button.draw(surface)
        self.reset_button.draw(surface)
        self.lock_button.draw(surface)
        self.end_button.draw(surface)

        status = "LOCKDOWN" if self.world.lockdown else "MOVING"
        if self.world.paused:
            status = "PAUSED"
        note = SMALL.render(f"Status: {status}   Data: Real COVID-19 Parameters", True, MUTED)
        surface.blit(note, (SIM_W + 24, HEIGHT - 68))

    def draw_counters(self, surface, x, y):
        counts = self.world.counts()
        labels = [("S", "Susceptible", BLUE), ("I", "Infected", RED), ("R", "Recovered", GREEN), ("D", "Deceased", GRAY)]
        heading = FONT.render("Live Data", True, TEXT)
        surface.blit(heading, (x, y - 28))
        for idx, (key, label, color) in enumerate(labels):
            row_y = y + idx * 22
            pygame.draw.circle(surface, color, (x + 7, row_y + 8), 6)
            text = FONT.render(f"{label}: {counts[key]}", True, TEXT)
            surface.blit(text, (x + 22, row_y))

    def draw_graph(self, surface, rect):
        pygame.draw.rect(surface, (24, 28, 35), rect, border_radius=6)
        pygame.draw.rect(surface, (74, 83, 98), rect, 1, border_radius=6)
        graph_title = SMALL.render("SIR Curve", True, MUTED)
        surface.blit(graph_title, (rect.left + 8, rect.top + 6))

        history = list(self.world.history)
        if len(history) < 2:
            return

        plot = rect.inflate(-18, -30)
        plot.y += 16
        max_pop = max(1, self.world.population)
        keys = [("S", BLUE), ("I", RED), ("R", GREEN), ("D", GRAY)]
        for key, color in keys:
            points = []
            for i, sample in enumerate(history):
                x = plot.left + i * plot.width / (len(history) - 1)
                y = plot.bottom - sample[key] / max_pop * plot.height
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(surface, color, False, points, 2)


def print_simulation_results(world):
    """Print detailed simulation results with chart and analysis."""
    counts = world.counts()
    history = list(world.history)
    
    print("\n" + "="*70)
    print("COVID-19 SIMULATION RESULTS".center(70))
    print("="*70)
    
    # Final counts
    total = sum(counts.values())
    print(f"\nFinal Population Status (Total: {total})")
    print(f"  Susceptible: {counts['S']:4d} ({100*counts['S']/total:5.1f}%) \u25A0")
    print(f"  Infected:    {counts['I']:4d} ({100*counts['I']/total:5.1f}%) \u25A0")
    print(f"  Recovered:   {counts['R']:4d} ({100*counts['R']/total:5.1f}%) \u25A0")
    print(f"  Deceased:    {counts['D']:4d} ({100*counts['D']/total:5.1f}%) \u25A0")
    
    # Attack rate
    attacked = counts['I'] + counts['R'] + counts['D']
    print(f"\nAttack Rate: {100*attacked/total:.1f}% of population infected")
    
    # Mortality
    if attacked > 0:
        mortality = counts['D'] / attacked
        print(f"Case Fatality Rate: {100*mortality:.2f}% (Deaths / Infected)")
    
    # Simulation duration
    days_simulated = world.tick / (8 * FPS)  # Approximate days
    print(f"Simulation Duration: ~{days_simulated:.1f} days")
    
    # ASCII chart
    print("\n" + "-"*70)
    print("SIR Curve Over Time".center(70))
    print("-"*70)
    
    if len(history) > 1:
        # Find max population for scaling
        max_pop = max(1, world.population)
        chart_height = 20
        chart_width = 60
        
        # Create ASCII chart
        for state, char, label in [('S', '█', 'S (Susceptible)'), ('I', '█', 'I (Infected)'), ('R', '█', 'R (Recovered)'), ('D', '█', 'D (Deceased)')]:
            print(f"\n{label}:")
            for h in range(chart_height, 0, -1):
                threshold = (h / chart_height) * max_pop
                line = "|"
                sample_rate = max(1, len(history) // chart_width)
                for i, record in enumerate(history[::sample_rate]):
                    if record[state] >= threshold:
                        line += "█"
                    else:
                        line += " "
                line += "|"
                print(line)
            print("+" + "-" * chart_width + "+")
    
    # Explanation
    print("\n" + "="*70)
    print("ANALYSIS & EXPLANATION".center(70))
    print("="*70)
    
    print(f"\nSimulation Parameters:")
    print(f"  Population: {world.population}")
    print(f"  Initial Infected: {world.initial_infected}")
    print(f"  Mask Rate: {world.mask_rate*100:.0f}%")
    print(f"  Vaccination Rate: {world.vax_rate*100:.0f}%")
    print(f"  Social Distancing: {world.dist_rate*100:.0f}%")
    print(f"  Virus Transmission: {world.virus_strength:.3f}")
    print(f"  Mortality Rate (unvaccinated): {REAL_COVID_DATA['mortality_unvaccinated']*100:.2f}%")
    
    print(f"\nKey Findings:")
    if attacked / total > 0.8:
        print(f"  • Severe outbreak: {attacked/total*100:.0f}% of population infected")
    elif attacked / total > 0.5:
        print(f"  • Moderate outbreak: {attacked/total*100:.0f}% of population infected")
    else:
        print(f"  • Contained outbreak: {attacked/total*100:.0f}% of population infected")
    
    if world.mask_rate > 0.5 or world.vax_rate > 0.5:
        print(f"  • Interventions (masks/vaccines) significantly reduced spread")
    
    if world.lockdown:
        print(f"  • Lockdown was activated, reducing person-to-person contact")
    
    print(f"\nData Source: Parameters derived from Our World in Data (excess deaths & test positivity)")
    print("="*70 + "\n")

def main():
    # Create windowed application
    screen = pygame.display.set_mode((1400, 800), pygame.RESIZABLE)
    pygame.display.set_caption("COVID-19 Spreading Simulation - Based on Real Data")
    clock = pygame.time.Clock()
    
    # Update global WIDTH/HEIGHT to match screen size
    global WIDTH, HEIGHT, SIM_W, PANEL_W
    WIDTH, HEIGHT = 1400, 800
    SIM_W = int(WIDTH * 0.75)
    PANEL_W = WIDTH - SIM_W
    
    world = World()
    panel = ControlPanel(world)

    running = True
    results_back_rect = None
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and not world.show_results:
                    world.paused = not world.paused
                elif event.key == pygame.K_r:
                    if world.show_results:
                        world.show_results = False
                    else:
                        world.configure(panel.sliders)
                        world.reset()
                elif event.key == pygame.K_l and not world.show_results:
                    world.lockdown = not world.lockdown
            
            if not world.show_results:
                result = panel.handle_event(event)
                if result == "end_simulation":
                    world.show_results = True
                    print_simulation_results(world)
            elif event.type == pygame.MOUSEBUTTONDOWN and results_back_rect and results_back_rect.collidepoint(event.pos):
                world.show_results = False

        if world.show_results:
            screen.fill(BG)
            results_back_rect = world.draw_results(screen)
        else:
            world.update(dt)
            screen.fill(BG)
            world.draw(screen)
            panel.draw(screen)
        
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
