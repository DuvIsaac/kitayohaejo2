import pygame
import random
import math
import time
import numpy as np
from optimization import calculate_optimal_routes

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1550, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mini Metro Game & Optimized Routes")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
COLORS = [RED, GREEN, BLUE]

# Game variables
STATION_RADIUS = 12
PASSENGER_RADIUS = 5
TRAIN_RADIUS = 10
MAX_PASSENGERS = 6
TRAIN_SPEED = 2
MAX_CONNECTIONS = 40
WINNING_SCORE = 150

# Debounce time for clicks (in seconds)
DEBOUNCE_TIME = 0.2

class Station:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.passengers = []
        self.connections = {}  # {color: [connected_stations]}
        self.label = None
        self.font = pygame.font.Font(None, 24)

    def draw(self, screen, offset_x=0):
        x = self.x + offset_x
        if self.shape == "circle":
            pygame.draw.circle(screen, BLACK, (x, self.y), STATION_RADIUS, 2)
        elif self.shape == "square":
            pygame.draw.rect(screen, BLACK, (x - STATION_RADIUS, self.y - STATION_RADIUS, 
                                             STATION_RADIUS * 2, STATION_RADIUS * 2), 2)
        elif self.shape == "triangle":
            points = [
                (x, self.y - STATION_RADIUS),
                (x - STATION_RADIUS, self.y + STATION_RADIUS),
                (x + STATION_RADIUS, self.y + STATION_RADIUS)
            ]
            pygame.draw.polygon(screen, BLACK, points, 2)

        for i, passenger in enumerate(self.passengers):
            self.draw_passenger(screen, x + 20 + i * 10, self.y, passenger)

        if self.label:
            label_text = self.font.render(self.label, True, BLACK)
            screen.blit(label_text, (x - 20, self.y - 40))

    def draw_passenger(self, screen, x, y, shape):
        if shape == "circle":
            pygame.draw.circle(screen, BLACK, (x, y), PASSENGER_RADIUS)
        elif shape == "square":
            pygame.draw.rect(screen, BLACK, (x - PASSENGER_RADIUS, y - PASSENGER_RADIUS, 
                                             PASSENGER_RADIUS * 2, PASSENGER_RADIUS * 2))
        elif shape == "triangle":
            points = [
                (x, y - PASSENGER_RADIUS),
                (x - PASSENGER_RADIUS, y + PASSENGER_RADIUS),
                (x + PASSENGER_RADIUS, y + PASSENGER_RADIUS)
            ]
            pygame.draw.polygon(screen, BLACK, points)

    def set_label(self, label):
        self.label = label

    def clear_label(self):
        self.label = None

class Train:
    def __init__(self, line, start_station):
        self.line = line
        self.current_station = start_station
        self.next_station = None
        self.progress = 0
        self.passengers = []
        self.x, self.y = self.current_station.x, self.current_station.y
        self.direction = 1  # 1 for forward, -1 for backward
        self.wait_time = 1  # 역에서 대기하는 시간 (초)
        self.wait_start = None  # 대기 시작 시간

    def move(self):
        if self.wait_start is not None:
            if time.time() - self.wait_start >= self.wait_time:
                self.wait_start = None
                self.choose_next_station()
            return 0

        if not self.next_station:
            self.choose_next_station()
            if not self.next_station:
                return 0

        dx = self.next_station.x - self.current_station.x
        dy = self.next_station.y - self.current_station.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance == 0:
            return 0

        self.progress += TRAIN_SPEED / distance
        if self.progress >= 1:
            self.current_station = self.next_station
            self.next_station = None
            self.progress = 0
            delivered = self.load_unload()
            self.x, self.y = self.current_station.x, self.current_station.y
            self.wait_start = time.time()  # 대기 시작
            return delivered

        self.x = self.current_station.x + dx * self.progress
        self.y = self.current_station.y + dy * self.progress
        return 0

    def choose_next_station(self):
        if not self.line.stations:
            return

        current_index = self.line.stations.index(self.current_station)

        # Check for loop condition
        if current_index == len(self.line.stations) - 1 and self.direction == 1:
            self.next_station = self.line.stations[0]
        elif current_index == 0 and self.direction == -1:
            self.next_station = self.line.stations[-1]
        else:
            self.next_station = self.line.stations[current_index + self.direction]

    def load_unload(self):
        delivered = sum(1 for p in self.passengers if p == self.current_station.shape)
        self.passengers = [p for p in self.passengers if p != self.current_station.shape]
        
        available_shapes = self.line.get_available_shapes()
        remaining_passengers = []
        for passenger in self.current_station.passengers:
            if len(self.passengers) < MAX_PASSENGERS and passenger in available_shapes:
                self.passengers.append(passenger)
            else:
                remaining_passengers.append(passenger)
        self.current_station.passengers = remaining_passengers

        return delivered

    def draw(self, screen, offset_x=0):
        pygame.draw.circle(screen, self.line.color, (int(self.x + offset_x), int(self.y)), TRAIN_RADIUS)
        for i, passenger in enumerate(self.passengers):
            self.draw_passenger(screen, int(self.x + offset_x) + 15 + i * 8, int(self.y) - 15, passenger)

    def draw_passenger(self, screen, x, y, shape):
        if shape == "circle":
            pygame.draw.circle(screen, BLACK, (x, y), PASSENGER_RADIUS - 2)
        elif shape == "square":
            pygame.draw.rect(screen, BLACK, (x - PASSENGER_RADIUS + 2, y - PASSENGER_RADIUS + 2, 
                                             (PASSENGER_RADIUS - 2) * 2, (PASSENGER_RADIUS - 2) * 2))
        elif shape == "triangle":
            points = [
                (x, y - PASSENGER_RADIUS + 2),
                (x - PASSENGER_RADIUS + 2, y + PASSENGER_RADIUS - 2),
                (x + PASSENGER_RADIUS - 2, y + PASSENGER_RADIUS - 2)
            ]
            pygame.draw.polygon(screen, BLACK, points)

class Line:
    def __init__(self, color):
        self.stations = []
        self.color = color
        self.train = None

    def add_station(self, station, game):
        if station not in self.stations:
            if self.stations:
                last_station = self.stations[-1]
                if station not in last_station.connections.get(self.color, []):
                    last_station.connections.setdefault(self.color, []).append(station)
                    station.connections.setdefault(self.color, []).append(last_station)
                    game.available_connections -= 1
            self.stations.append(station)
            if not self.train and len(self.stations) > 1:
                self.add_train(self.stations[0])
            return True
        return False

    def remove_station(self, station):
        if station in self.stations:
            index = self.stations.index(station)
            self.stations.remove(station)
            if index > 0:
                prev_station = self.stations[index - 1]
                prev_station.connections[self.color].remove(station)
                station.connections[self.color].remove(prev_station)
            if index < len(self.stations):
                next_station = self.stations[index]
                next_station.connections[self.color].remove(station)
                station.connections[self.color].remove(next_station)
            if self.train and self.train.current_station == station:
                self.train.current_station = self.stations[0] if self.stations else None
                self.train.next_station = None

    def draw(self, screen, offset_x=0):
        if len(self.stations) > 1:
            points = [(station.x + offset_x, station.y) for station in self.stations]
            pygame.draw.lines(screen, self.color, False, points, 4)

            # Draw line between first and last stations with the line's color
            first_station = self.stations[0]
            last_station = self.stations[-1]
            pygame.draw.line(screen, self.color, 
                             (first_station.x + offset_x, first_station.y), 
                             (last_station.x + offset_x, last_station.y), 2)

        if self.train:
            self.train.draw(screen, offset_x)

    def add_train(self, start_station):
        if not self.train and start_station in self.stations:
            self.train = Train(self, start_station)

    def get_available_shapes(self):
        return set(station.shape for station in self.stations)

class Game:
    def __init__(self):
        self.stations = []
        self.lines = {RED: Line(RED), GREEN: Line(GREEN), BLUE: Line(BLUE)}
        self.optimized_lines = {RED: Line(RED), GREEN: Line(GREEN), BLUE: Line(BLUE)}
        self.available_connections = MAX_CONNECTIONS
        self.score = 0
        self.optimized_score = 0
        self.font = pygame.font.Font(None, 36)
        self.last_click_time = 0
        self.current_line = None
        self.start_station = None
        self.selected_color = RED  # Default selected color
        self.last_double_click_time = 0
        self.double_click_interval = 0.3  # 더블클릭 간격 (초)
        self.reset_button = pygame.Rect(340, 10, 100, 40)
        self.optimal_routes = None
        self.game_over = False
        self.winner = None
        self.game_over_time = None
        self.game_started = False

    def generate_station(self):
        shapes = ["circle", "square", "triangle"]
        x = random.randint(50, WIDTH//2 - 50)
        y = random.randint(100, HEIGHT - 50)
        shape = random.choice(shapes)
        new_station = Station(x, y, shape)
        self.stations.append(new_station)

    def generate_passenger(self):
        for station in self.stations:
            if len(station.passengers) < MAX_PASSENGERS and random.random() < 0.001:
                shapes = ["circle", "square", "triangle"]
                shapes.remove(station.shape)
                station.passengers.append(random.choice(shapes))

    def draw_buttons(self, screen):
        for color, rect in COLOR_BUTTONS.items():
            pygame.draw.rect(screen, color, rect)
            if color == self.selected_color:
                pygame.draw.rect(screen, BLACK, rect, 2)

        pygame.draw.rect(screen, BLACK, self.reset_button)
        reset_text = self.font.render("Start" if not self.game_started else "Reset", True, WHITE)
        text_rect = reset_text.get_rect(center=self.reset_button.center)
        screen.blit(reset_text, text_rect)

    def draw_game(self, screen):
        screen.fill(WHITE)
        pygame.draw.line(screen, BLACK, (WIDTH//2, 0), (WIDTH//2, HEIGHT), 2)
        
        for line in self.lines.values():
            line.draw(screen)
        for station in self.stations:
            station.draw(screen)

        self.draw_buttons(screen)

        lines_text = self.font.render(f"Available Connections: {self.available_connections}", True, BLACK)
        screen.blit(lines_text, (10, 60))

    def draw_optimization(self, screen):
        for line in self.optimized_lines.values():
            line.draw(screen, WIDTH//2)

        for station in self.stations:
            station.draw(screen, WIDTH//2)

    def draw_scores(self, screen):
        score_text = self.font.render(f"{self.score} : {self.optimized_score}", True, BLACK)
        text_rect = score_text.get_rect(center=(WIDTH // 2, 30))
        screen.blit(score_text, text_rect)

    def draw_winner(self, screen):
        if self.winner:
            winner_text = self.font.render(f"{self.winner} Wins!", True, BLACK)
            text_rect = winner_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(winner_text, text_rect)

    def is_point_on_line(self, x, y, line):
        for i in range(len(line.stations) - 1):
            start = line.stations[i]
            end = line.stations[i + 1]
            
            line_length = math.hypot(end.x - start.x, end.y - start.y)
            
            d1 = math.hypot(x - start.x, y - start.y)
            d2 = math.hypot(x - end.x, y - end.y)
            
            buffer = 5
            
            if abs(d1 + d2 - line_length) <= buffer:
                return True, i
        return False, -1

    def remove_connection(self, line, index):
        if 0 <= index < len(line.stations) - 1:
            station1 = line.stations[index]
            station2 = line.stations[index + 1]
            
            station1.connections[line.color].remove(station2)
            station2.connections[line.color].remove(station1)
            
            line.stations.pop(index + 1)

            self.available_connections += 1

    def reset_lines(self):
        for line in self.lines.values():
            for station in line.stations:
                station.connections[line.color] = []
            line.stations = []
            line.train = None
        self.available_connections = MAX_CONNECTIONS
        self.current_line = None
        self.start_station = None
        self.clear_all_labels()
        self.game_started = True
        self.score = 0
        self.optimized_score = 0
        self.game_over = False
        self.winner = None
        self.game_over_time = None

    def clear_all_labels(self):
        for station in self.stations:
            station.clear_label()

    def calculate_optimal_routes(self):
        station_positions = [(station.x, station.y) for station in self.stations]
        station_shapes = [station.shape for station in self.stations]
        number_of_lines = len(self.lines)
        
        optimal_routes, _ = calculate_optimal_routes(station_positions, station_shapes, number_of_lines)
        
        # Convert optimal routes to Line objects
        self.optimized_lines = {}
        for i, route in enumerate(optimal_routes):
            color = COLORS[i % len(COLORS)]
            line = Line(color)
            for x, y in route:
                station = next((s for s in self.stations if s.x == x and s.y == y), None)
                if station:
                    line.add_station(station, self)
            self.optimized_lines[color] = line
            
            # Add train to the optimized line
            if line.stations:
                line.add_train(line.stations[0])

    def check_win_condition(self):
        if self.score >= WINNING_SCORE:
            self.game_over = True
            self.winner = 'player'
            self.game_over_time = time.time()
        elif self.optimized_score >= WINNING_SCORE:
            self.game_over = True
            self.winner = 'COM'
            self.game_over_time = time.time()

    def run(self):
        clock = pygame.time.Clock()
        running = True

        for _ in range(45):  # Generate 40 stations initially
            self.generate_station()

        self.calculate_optimal_routes()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    current_time = time.time()
                    x, y = event.pos

                    if self.reset_button.collidepoint(x, y):
                        self.reset_lines()
                    elif self.game_started and not self.game_over and x < WIDTH // 2:
                        if event.button == 1:  # 왼쪽 클릭
                            if current_time - self.last_double_click_time < self.double_click_interval:
                                # 더블클릭 감지
                                for color, line in self.lines.items():
                                    on_line, index = self.is_point_on_line(x, y, line)
                                    if on_line:
                                        self.remove_connection(line, index)
                                        break
                            else:
                                # 단일 클릭 처리
                                if current_time - self.last_click_time < DEBOUNCE_TIME:
                                    continue  # Skip this event to debounce
                                self.last_click_time = current_time

                                # Check if a color button was clicked
                                for color, rect in COLOR_BUTTONS.items():
                                    if rect.collidepoint(x, y):
                                        self.selected_color = color
                                        self.current_line = None
                                        self.start_station = None
                                        self.clear_all_labels()
                                        break
                                else:
                                    # If no button was clicked, proceed with station logic
                                    clicked_station = None
                                    for station in self.stations:
                                        if math.hypot(station.x - x, station.y - y) < STATION_RADIUS:
                                            clicked_station = station
                                            break
                                    if clicked_station:
                                        if self.available_connections > 0:
                                            if not self.current_line:
                                                self.current_line = self.lines[self.selected_color]
                                                self.clear_all_labels()
                                                self.start_station = clicked_station
                                                self.current_line.add_station(clicked_station, self)
                                            else:
                                                if self.current_line.add_station(clicked_station, self):
                                                    if clicked_station == self.start_station:
                                                        self.current_line = None
                                                        self.start_station = None

                            self.last_double_click_time = current_time

                        elif event.button == 3:  # 오른쪽 클릭
                            clicked_station = None
                            for station in self.stations:
                                if math.hypot(station.x - x, station.y - y) < STATION_RADIUS:
                                    clicked_station = station
                                    break
                            if clicked_station:
                                for line in self.lines.values():
                                    if clicked_station in line.stations:
                                        line.remove_station(clicked_station)
                                        self.available_connections += 1
                                clicked_station.clear_label()
                                self.current_line = None
                                self.start_station = None

            if self.game_started and not self.game_over:
                self.generate_passenger()

                for line in self.lines.values():
                    if line.train:
                        self.score += line.train.move()

                for line in self.optimized_lines.values():
                    if line.train:
                        self.optimized_score += line.train.move()

                self.check_win_condition()

            screen.fill(WHITE)
            self.draw_game(screen)
            self.draw_optimization(screen)
            self.draw_scores(screen)

            if not self.game_started:
                start_text = self.font.render("Press 'Start' to begin the game", True, BLACK)
                text_rect = start_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                screen.blit(start_text, text_rect)

            if self.game_over:
                self.draw_winner(screen)
                if time.time() - self.game_over_time > 10:  # Wait for 10 seconds after game over
                    running = False

            pygame.display.flip()

            clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    # Button dimensions and positions
    BUTTON_WIDTH = 100
    BUTTON_HEIGHT = 40
    BUTTON_SPACING = 20
    BUTTON_Y = 10
    COLOR_BUTTONS = {
        RED: pygame.Rect(10, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT),
        GREEN: pygame.Rect(120, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT),
        BLUE: pygame.Rect(230, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT),
    }

    game = Game()
    game.run()