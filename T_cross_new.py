#%%
import pygame
import numpy as np
from scipy.stats import geninvgauss, lognorm
import csv  # Přidáno pro ukládání dat do CSV
import sys   # Přidáno pro ukončení skriptu

# --- Helper Function ---
def apply_zoom(x, y, zoom_factor, center_x, center_y):
    """
    Applies zoom to a point (x, y) centered at (center_x, center_y).
    """
    x -= center_x
    y -= center_y
    x *= zoom_factor
    y *= zoom_factor
    x += center_x
    y += center_y
    return int(x), int(y)

# --- Config Class ---
class Config:
    # Visualization settings
    VISUALIZE = False  # Set to True to visualize the simulation

    # Window dimensions
    WIDTH, HEIGHT = 900, 600  # 900x600 pixels (800 for simulation, 100 for slider)

    # Scale (conversion from meters to pixels)
    SCALE_MAIN = 1         # 1 pixel = 1 meter for main road
    SCALE_SECONDARY = 1.33 # 1 pixel = 1.33 meters for secondary road

    # Maximum speed (speed limiter) in m/s
    MAX_SPEED = 50.0  # ~180 km/h upper bound

    # Maximum deceleration in m/s² (positive value)
    MAX_DECELERATION = 4.0

    # Road parameters
    ROAD_LENGTH_MAIN = 600      # in meters
    ROAD_LENGTH_SECONDARY = 300 # in meters

    # IDM parameters for the main road
    MAIN_IDM_PARAMETERS = {
        'min_gap': 1.625,
        'acceleration': 2.0,
        'comfortable_deceleration': 4.0,
        'react_time': 0.3,
        'desired_speed': 19.44,  # ~50 km/h
        'delta': 2,
    }

    # IDM parameters for the secondary road
    SECONDARY_IDM_PARAMETERS = {
        'min_gap': 1.625,
        'acceleration': 4.0,
        'comfortable_deceleration': 4.0,
        'react_time': 0.3,
        'desired_speed': 13.88,  # ~50 km/h
        'delta': 1,
    }

    # Start slowing down 50m before intersection, target speed is 10 km/h
    SEC_SLOWDOWN_START_DISTANCE = 50.0
    SEC_SLOWDOWN_TARGET_SPEED = 10.0 / 3.6  # 10 km/h -> 2.78 m/s

    # GIG distribution parameters for spawn intervals (now in seconds)
    MAIN_GIG_PARAMETERS = {
        'lambda_': 2.71,
        'beta': 1.03
    }
    SECONDARY_GIG_PARAMETERS = {
        'lambda_': 11.5,
        'beta': 4.7,
    }

    # Lognormal parameters for initial speeds
    MAIN_LOGNORM_PARAMETERS = {
        'mu_log': np.log(13.89) - (0.107**2)/2,
        'sigma_log': 0.107,
    }
    SECONDARY_LOGNORM_PARAMETERS = {
        'mu_log': np.log(11.11) - (0.107**2)/2,
        'sigma_log': 0.107,
    }

    # Merging behavior
    LINEUP = True   # Whether merging is allowed from secondary
    MERGE_CHECK_DISTANCE = 50.0
    MERGE_DISTANCE_THRESHOLD = 1.0  # If the secondary car is this close, attempt final merge
    MERGING_SPEED = 10 / 3.6       # 10 km/h in m/s

    # Road positions
    MAIN_ROAD_X = 400        # 400 pixels (left side of simulation area)
    SECONDARY_ROAD_Y = 300   # 300 pixels (middle vertically)

    # Intersection position on the main road (in meters)
    INTERSECTION_POSITION_MAIN_ROAD = 300.0  # Exactly at 300 meters

    # Colors
    COLORS = {
        'WHITE': (255, 255, 255),
        'BLACK': (0, 0, 0),
        'BLUE': (0, 0, 255),
        'RED': (255, 0, 0),
        'GRAY': (200, 200, 200),
        'GREEN': (0, 255, 0),  # Added for button and slider
    }

    # Maximum total spawned cars on main road
    MAX_TOTAL_SPAWNED_CARS_MAIN = 20000  # Nastavte podle potřeby

# --- IDM Class ---
class IDM:
    def __init__(self, config, road_type='main'):
        if road_type == 'main':
            params = config.MAIN_IDM_PARAMETERS
        else:
            params = config.SECONDARY_IDM_PARAMETERS

        self.v0 = params['desired_speed']
        self.a = params['acceleration']
        self.b = config.MAX_DECELERATION
        self.delta = params['delta']
        self.s0 = params['min_gap']
        self.t0 = params['react_time']

    def compute_acceleration(self, v, delta_v, s):
        # If desired speed is almost zero, force deceleration
        if self.v0 < 1e-3:
            return -self.b

        if s <= 0:
            s = 0.1  # avoid division by zero

        s_star = self.s0 + max(0, v * self.t0 + (v * delta_v) / (2 * np.sqrt(self.a * self.b)))
        accel = self.a * (1 - (v / self.v0) ** self.delta - (s_star / s) ** 2)
        return max(accel, -self.b)

    def update_parameters(self, params):
        self.v0 = params.get('desired_speed', self.v0)
        self.a = params.get('acceleration', self.a)
        self.b = params.get('comfortable_deceleration', self.b)
        self.delta = params.get('delta', self.delta)
        self.s0 = params.get('min_gap', self.s0)
        self.t0 = params.get('react_time', self.t0)

# --- Car Class ---
class Car:
    WIDTH = 10
    HEIGHT = 5  # Upraveno na menší výšku pro vizuální přehlednost

    def __init__(self, position, road_type, config, initial_speed=0.0):
        self.position = position  # in meters
        self.road_type = road_type
        self.length = 5.0  # in meters
        self.v = initial_speed  # in m/s
        self.a = 0.0  # in m/s²
        self.config = config
        self.idm = IDM(config, road_type)
        self.color = config.COLORS['BLUE'] if road_type == 'main' else config.COLORS['RED']
        self.has_passed_intersection = False  # Přidáno pro sledování průjezdu

    def update(self, dt, lead_car=None, simulation_time=None, simulation=None):
        if lead_car:
            s = lead_car.position - self.position - lead_car.length
            delta_v = self.v - lead_car.v
            self.a = self.idm.compute_acceleration(self.v, delta_v, s)
        else:
            # Free-flow scenario
            if self.idm.v0 < 1e-3:
                self.a = -self.config.MAX_DECELERATION
            else:
                self.a = self.idm.a * (1 - (self.v / self.idm.v0) ** self.idm.delta)

        # Clamp deceleration
        self.a = max(self.a, -self.config.MAX_DECELERATION)

        # Update speed
        self.v += self.a * dt
        self.v = max(min(self.v, self.config.MAX_SPEED), 0)

        # Update position
        self.position += self.v * dt + 0.5 * self.a * dt**2

        # Check if main road car has passed the intersection
        if self.road_type == 'main' and self.color == self.config.COLORS['BLUE'] and not self.has_passed_intersection:
            if self.position >= self.config.INTERSECTION_POSITION_MAIN_ROAD:
                self.has_passed_intersection = True
                if simulation:
                    simulation.record_main_pass(simulation_time)

    def draw(self, window, font, zoom_factor):
        if self.road_type == 'main':
            x = self.config.MAIN_ROAD_X
            y = self.config.HEIGHT - self.position * self.config.SCALE_MAIN
        else:
            x = self.position * self.config.SCALE_SECONDARY
            y = self.config.SECONDARY_ROAD_Y

        # Apply zoom centered at intersection
        x_zoomed, y_zoomed = apply_zoom(x, y, zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)

        # Scale car size based on zoom_factor
        scaled_width = max(1, int(self.WIDTH * zoom_factor))
        scaled_height = max(1, int(self.HEIGHT * zoom_factor))

        rect = pygame.Rect(x_zoomed - scaled_width // 2, y_zoomed - scaled_height // 2, scaled_width, scaled_height)
        pygame.draw.rect(window, self.color, rect)

        speed_kmh = self.v * 3.6
        text_surface = font.render(f"{speed_kmh:.1f} km/h", True, self.config.COLORS['BLACK'])
        text_rect = text_surface.get_rect(center=(rect.centerx, rect.centery - 15 * zoom_factor))
        window.blit(text_surface, text_rect)

    def get_position(self):
        # Returns the car's pixel position for possible collision-drawing, etc.
        if self.road_type == 'main':
            x = self.config.MAIN_ROAD_X
            y = self.config.HEIGHT - self.position * self.config.SCALE_MAIN
        else:
            x = self.position * self.config.SCALE_SECONDARY
            y = self.config.SECONDARY_ROAD_Y
        return x, y

    def update_idm_parameters(self, params):
        self.idm.update_parameters(params)

# --- GIG Distribution Class ---
class GIGDistribution:
    def __init__(self, lambda_, beta):
        self.lambda_ = lambda_
        self.beta = beta
        self.geninvgauss_dist = geninvgauss(self.lambda_, self.beta)

    def sample(self, num_samples=1, max_value=None):
        results = []
        while len(results) < num_samples:
            val = self.geninvgauss_dist.rvs()
            if val > 0 and (max_value is None or val <= max_value):
                results.append(val)
        return results

# --- Road Class ---
class Road:
    def __init__(self, config, road_type):
        self.config = config
        self.road_type = road_type
        self.cars = []

        # Set up spawn distributions
        if road_type == 'main':
            gig_params = config.MAIN_GIG_PARAMETERS
            log_params = config.MAIN_LOGNORM_PARAMETERS
            self.scale = config.SCALE_MAIN
            self.road_length = config.ROAD_LENGTH_MAIN
            self.max_total_spawned_cars_main = config.MAX_TOTAL_SPAWNED_CARS_MAIN  # Celkový počet spawnovaných aut
        else:
            gig_params = config.SECONDARY_GIG_PARAMETERS
            log_params = config.SECONDARY_LOGNORM_PARAMETERS
            self.scale = config.SCALE_SECONDARY
            self.road_length = config.ROAD_LENGTH_SECONDARY
            self.max_total_spawned_cars_main = None  # Neplatné pro sekundární silnici

        self.gig_distribution = GIGDistribution(gig_params['lambda_'], gig_params['beta'])
        self.lognorm_mu_log = log_params['mu_log']
        self.lognorm_sigma_log = log_params['sigma_log']

        self.next_spawn_time = self.gig_distribution.sample()[0]  # Čas do dalšího spawnu v sekundách
        self.time_since_last_spawn = 0.0  # Čas od posledního spawnu

        # Počítadlo spawnovaných aut
        self.total_spawned_cars = 0

        # Intersection stopping point for secondary
        if self.road_type == 'secondary':
            self.stopping_point = self.config.INTERSECTION_POSITION_MAIN_ROAD  # 300.0 meters
        else:
            self.stopping_point = None

    def spawn_car(self):
        # Kontrola maximálního počtu aut na hlavní silnici
        if self.road_type == 'main':
            if self.max_total_spawned_cars_main is not None and self.total_spawned_cars >= self.max_total_spawned_cars_main:
                simulation.save_data_to_csv()  # Zavolání uložení dat
                if self.config.VISUALIZE:
                    pygame.quit()  # Zavření Pygame okna
                sys.exit()

        speed_sample = lognorm.rvs(s=self.lognorm_sigma_log, scale=np.exp(self.lognorm_mu_log))
        speed_sample = max(speed_sample, 0.0)

        new_car = Car(
            position=0.0,
            road_type=self.road_type,
            config=self.config,
            initial_speed=speed_sample
        )
        self.cars.append(new_car)
        self.time_since_last_spawn = 0.0  # Resetování času od posledního spawnu

        # Inkrementace počítadla spawnovaných aut
        if self.road_type == 'main' and new_car.color == self.config.COLORS['BLUE']:
            self.total_spawned_cars += 1

        # Nyní budeme generovat čas do dalšího spawnu
        self.next_spawn_time = self.gig_distribution.sample()[0]

        # Pokud bylo dosaženo maximálního celkového počtu aut na hlavní silnici, indikujte to
        if self.road_type == 'main' and self.total_spawned_cars >= self.max_total_spawned_cars_main:
            return True  # Indikuje, že bylo spawnováno poslední auto

        return False

    def get_linear_desired_speed(self, distance):
        """
        Return a linearly interpolated desired speed for the secondary road car,
        from original ~50 km/h down to 10 km/h, based on distance to intersection.
        """
        original_speed = self.config.SECONDARY_IDM_PARAMETERS['desired_speed']  # ~50 km/h
        target_speed = self.config.SEC_SLOWDOWN_TARGET_SPEED   # 10 km/h
        start_dist = self.config.SEC_SLOWDOWN_START_DISTANCE

        if distance >= start_dist:
            return original_speed
        if distance <= 0:
            return target_speed

        ratio = distance / start_dist
        return target_speed + ratio * (original_speed - target_speed)

    def recheck_merge(self, car, main_road):
        """
        Helper method to check merge feasibility each frame:
        If the main road is empty or if there's sufficient gap, we can merge.
        """
        if main_road and len(main_road.cars) == 0:
            return True

        intersection = self.config.INTERSECTION_POSITION_MAIN_ROAD
        lead_car, lag_car = None, None
        for main_car in main_road.cars:
            if main_car.position > intersection:
                # Potential lead
                if not lead_car or main_car.position < lead_car.position:
                    lead_car = main_car
            else:
                # Potential lag
                if not lag_car or main_car.position > lag_car.position:
                    lag_car = main_car

        gap_ahead = float('inf') if not lead_car else (lead_car.position - intersection - lead_car.length)
        gap_behind = float('inf') if not lag_car else (intersection - lag_car.position - car.length - 2*(lag_car.v - car.v))
        min_gap = car.idm.s0

        if gap_ahead >= 0 and gap_behind >= min_gap:
            return True

        return False

    def update_cars(self, dt, simulation_time, simulation, main_road=None):
        if not self.cars:
            return False  # Žádná auta k aktualizaci

        # Sort from front to back
        self.cars.sort(key=lambda c: c.position, reverse=True)

        i = 0
        last_car_spawned = False
        while i < len(self.cars):
            car = self.cars[i]

            if self.road_type == 'secondary' and i == 0:
                # The first secondary car is the one that tries to merge
                distance_to_int = self.stopping_point - car.position

                # Adjust its desired speed
                new_speed = self.get_linear_desired_speed(distance_to_int)
                updated_params = self.config.SECONDARY_IDM_PARAMETERS.copy()
                updated_params['desired_speed'] = new_speed
                car.update_idm_parameters(updated_params)

                # Create a virtual car at intersection
                virtual_car = Car(
                    position=self.stopping_point,
                    road_type='secondary',
                    config=self.config,
                    initial_speed=0.0
                )
                virtual_car.v = 0.0
                virtual_car.length = 0.0

                # Check if we are within MERGE_CHECK_DISTANCE
                if distance_to_int <= self.config.MERGE_CHECK_DISTANCE:
                    can_merge = False
                    if main_road:
                        can_merge = self.recheck_merge(car, main_road)

                    if can_merge:
                        # Merge with speed no higher than MERGING_SPEED
                        desired_speed = self.config.MERGING_SPEED
                        merge_params = self.config.SECONDARY_IDM_PARAMETERS.copy()
                        merge_params['desired_speed'] = desired_speed
                        car.update_idm_parameters(merge_params)
                        lead_car = None
                    else:
                        lead_car = virtual_car
                else:
                    # If not close enough, just follow the virtual car
                    lead_car = virtual_car

                # Update the car
                car.update(dt, lead_car=lead_car, simulation_time=simulation_time, simulation=simulation)

                # Final check: if we reached intersection, attempt to merge again
                if (self.config.LINEUP
                    and main_road
                    and abs(car.position - self.stopping_point) <= self.config.MERGE_DISTANCE_THRESHOLD):
                    can_merge_final = False
                    if main_road:
                        can_merge_final = self.recheck_merge(car, main_road)

                    if can_merge_final:
                        # Merge onto main road
                        self.cars.pop(i)
                        car.road_type = 'main'
                        car.position = self.config.INTERSECTION_POSITION_MAIN_ROAD
                        car.v = min(car.v, self.config.MERGING_SPEED)
                        car.idm = IDM(self.config, road_type='main')
                        main_road.cars.append(car)
                        print("Car from secondary road merged onto main road.")
                        # Record the merge time
                        simulation.record_secondary_merge(simulation_time)
                        continue  # we removed this car from secondary
            else:
                # For main road cars, or non-first cars on secondary
                if i == 0:
                    # The first car on the main road
                    car.update(dt, lead_car=None, simulation_time=simulation_time, simulation=simulation)
                else:
                    lead_car = self.cars[i - 1]
                    car.update(dt, lead_car=lead_car, simulation_time=simulation_time, simulation=simulation)

            i += 1

        # Aktualizace času od posledního spawnu
        self.time_since_last_spawn += dt

        # Spawn new car if needed
        if self.time_since_last_spawn >= self.next_spawn_time:
            was_last_spawn = self.spawn_car()
            if was_last_spawn:
                return True  # Indikuje, že bylo spawnováno poslední auto
        return False

    def draw(self, window, font, zoom_factor):
        if self.road_type == 'main':
            start_px = (self.config.MAIN_ROAD_X, self.config.HEIGHT)
            end_px = (self.config.MAIN_ROAD_X, self.config.HEIGHT - int(self.config.ROAD_LENGTH_MAIN * self.config.SCALE_MAIN))
            start_px_zoomed = apply_zoom(start_px[0], start_px[1], zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)
            end_px_zoomed = apply_zoom(end_px[0], end_px[1], zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)
            pygame.draw.line(window, self.config.COLORS['BLACK'], start_px_zoomed, end_px_zoomed, max(1, int(5 * zoom_factor)))
        else:
            start_px = (0, self.config.SECONDARY_ROAD_Y)
            end_px = (int(self.config.ROAD_LENGTH_SECONDARY * self.config.SCALE_SECONDARY), self.config.SECONDARY_ROAD_Y)
            start_px_zoomed = apply_zoom(start_px[0], start_px[1], zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)
            end_px_zoomed = apply_zoom(end_px[0], end_px[1], zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)
            pygame.draw.line(window, self.config.COLORS['BLACK'], start_px_zoomed, end_px_zoomed, max(1, int(5 * zoom_factor)))

        # Draw road markings
        if self.road_type == 'main':
            max_dist = self.config.ROAD_LENGTH_MAIN
            for pos in range(0, int(max_dist) + 1, 50):
                y = self.config.HEIGHT - pos * self.config.SCALE_MAIN
                x_start = self.config.MAIN_ROAD_X - 20
                x_end = self.config.MAIN_ROAD_X + 20

                # Apply zoom to the endpoints of the marking
                y_zoomed = apply_zoom(self.config.MAIN_ROAD_X, y, zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)[1]
                x_start_zoomed = apply_zoom(x_start, y, zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)[0]
                x_end_zoomed = apply_zoom(x_end, y, zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)[0]

                pygame.draw.line(
                    window,
                    self.config.COLORS['GRAY'],
                    (x_start_zoomed, y_zoomed),
                    (x_end_zoomed, y_zoomed),
                    max(1, int(2 * zoom_factor))
                )
                label = font.render(f"{pos} m", True, self.config.COLORS['BLACK'])
                rect = label.get_rect(center=(x_start_zoomed - 50 * zoom_factor, y_zoomed))
                window.blit(label, rect)
        elif self.road_type == 'secondary':
            max_dist = self.config.ROAD_LENGTH_SECONDARY
            for pos in range(0, int(max_dist) + 1, 50):  # Změněno na interval 50 m pro lepší přehlednost
                x = pos * self.config.SCALE_SECONDARY
                x_zoomed = apply_zoom(x, self.config.SECONDARY_ROAD_Y, zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)[0]
                y_start = self.config.SECONDARY_ROAD_Y - 20
                y_end = self.config.SECONDARY_ROAD_Y + 20

                pygame.draw.line(
                    window,
                    self.config.COLORS['GRAY'],
                    (x_zoomed, y_start * zoom_factor + (self.config.SECONDARY_ROAD_Y * (1 - zoom_factor))),
                    (x_zoomed, y_end * zoom_factor + (self.config.SECONDARY_ROAD_Y * (1 - zoom_factor))),
                    max(1, int(2 * zoom_factor))
                )
                dist_from_int = int(max_dist - pos)
                label = font.render(f"{dist_from_int} m", True, self.config.COLORS['BLACK'])
                rect = label.get_rect(center=(x_zoomed, self.config.SECONDARY_ROAD_Y + 30 * zoom_factor))
                window.blit(label, rect)

        # Draw all cars on the road
        for car in self.cars:
            car.draw(window, font, zoom_factor)

# --- Simulation Class ---
class Simulation:
    def __init__(self):
        self.config = Config()
        if self.config.VISUALIZE:
            pygame.init()
            pygame.font.init()

            self.window = pygame.display.set_mode((self.config.WIDTH, self.config.HEIGHT))
            pygame.display.set_caption("T-Intersection Simulation with Zoom and Speed Control")
            self.font = pygame.font.SysFont(None, 20)
            self.clock = pygame.time.Clock()
        else:
            # Pokud nemáme vizualizaci, neinitializujeme Pygame
            self.window = None
            self.font = None
            self.clock = None

        # Create roads
        self.main_road = Road(self.config, 'main')
        self.secondary_road = Road(self.config, 'secondary')

        self.collision_occurred = False
        self.collision_x = 0
        self.collision_y = 0

        # --- Zoom Attributes ---
        self.zoom_factor = 1.0
        self.min_zoom = 1.0   # Nastaveno na 1
        self.max_zoom = 6.0   # Nastaveno na 6
        self.zoom_step = 0.1

        # --- Slider Attributes ---
        self.slider_rect = pygame.Rect(800, 100, 20, 400)  # (x, y, width, height)
        self.slider_handle_rect = pygame.Rect(790, 300, 40, 10)  # Initial handle position
        self.slider_dragging = False

        # Initialize handle position based on initial zoom
        self.update_slider_handle()

        # --- Speed Multiplier Attributes ---
        self.speed_multiplier = 1  # Starts at 1x
        self.max_multiplier = 32
        self.min_multiplier = 1
        self.button_width = 100
        self.button_height = 50
        self.button_rect = pygame.Rect(10, 10, self.button_width, self.button_height)  # Top-left corner

        # --- Simulation Time ---
        self.simulation_time = 0.0  # Celkový simulační čas

        # --- Data Tracking ---
        self.main_pass_times = []          # Časy průjezdů hlavní silnice
        self.secondary_merge_times = []    # Časy mergování z vedlejší silnice
        self.gap_data = []                 # Seznam pro ukládání (time_gap, merge_count) tupleů

        # --- Spawn Counter ---
        self.last_main_car_spawned = False  # Indikuje, zda bylo spawnováno poslední modré auto

    def update_slider_handle(self):
        """
        Updates the slider handle position based on the current zoom factor.
        """
        # Map zoom_factor to handle y position
        # y=100 corresponds to max_zoom
        # y=500 corresponds to min_zoom
        ratio = (self.zoom_factor - self.min_zoom) / (self.max_zoom - self.min_zoom)
        handle_y = 500 - (ratio * 400)  # 400 is the height of the slider
        self.slider_handle_rect.y = handle_y - self.slider_handle_rect.height // 2

    def run(self):
        running = True
        self.main_road.spawn_car()
        self.secondary_road.spawn_car()

        while running:
            if self.config.VISUALIZE:
                # Cap the frame rate and get delta time
                dt = self.clock.tick(60) / 1000.0  # Seconds per frame
                adjusted_dt = dt * self.speed_multiplier  # Adjusted delta time based on speed multiplier
                self.simulation_time += adjusted_dt  # Aktualizace simulačního času správně

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    # Handle mouse button down events for the slider handle and speed-up button
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left mouse button
                            mouse_pos = event.pos
                            if self.slider_handle_rect.collidepoint(mouse_pos):
                                self.slider_dragging = True
                            elif self.button_rect.collidepoint(mouse_pos):
                                self.update_speed_multiplier()

                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:  # Left mouse button
                            self.slider_dragging = False

                    # Handle mouse motion for dragging the slider handle
                    elif event.type == pygame.MOUSEMOTION:
                        if self.slider_dragging:
                            mouse_x, mouse_y = event.pos
                            # Clamp the handle within the slider_rect
                            new_y = max(self.slider_rect.y, min(mouse_y, self.slider_rect.y + self.slider_rect.height))
                            self.slider_handle_rect.y = new_y - self.slider_handle_rect.height // 2

                            # Map the handle position to zoom_factor
                            # Invert the y-axis: higher y -> lower zoom
                            relative_y = new_y - self.slider_rect.y
                            ratio = 1 - (relative_y / self.slider_rect.height)
                            self.zoom_factor = self.min_zoom + ratio * (self.max_zoom - self.min_zoom)
                            self.zoom_factor = round(self.zoom_factor, 2)
                            # Update other zoom-dependent elements if necessary

                    # Handle mouse wheel for zooming
                    elif event.type == pygame.MOUSEWHEEL:
                        if event.y > 0:
                            # Zoom in
                            self.zoom_factor = min(self.zoom_factor * 1.1, self.max_zoom)
                        elif event.y < 0:
                            # Zoom out
                            self.zoom_factor = max(self.zoom_factor / 1.1, self.min_zoom)
                        self.zoom_factor = round(self.zoom_factor, 2)
                        # Update slider handle position
                        self.update_slider_handle()

                if self.slider_dragging:
                    # Continuously update slider handle and zoom_factor
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    new_y = max(self.slider_rect.y, min(mouse_y, self.slider_rect.y + self.slider_rect.height))
                    self.slider_handle_rect.y = new_y - self.slider_handle_rect.height // 2

                    relative_y = new_y - self.slider_rect.y
                    ratio = 1 - (relative_y / self.slider_rect.height)
                    self.zoom_factor = self.min_zoom + ratio * (self.max_zoom - self.min_zoom)
                    self.zoom_factor = round(self.zoom_factor, 2)

                if not self.collision_occurred:
                    # Update logic with adjusted delta time
                    was_last_spawn = self.main_road.update_cars(adjusted_dt, self.simulation_time, self)
                    self.secondary_road.update_cars(adjusted_dt, self.simulation_time, self, main_road=self.main_road)

                    # Collision check
                    self.check_collisions()

                    # Ukončení simulace, pokud bylo spawnováno poslední modré auto
                    if not self.last_main_car_spawned and was_last_spawn:
                        self.last_main_car_spawned = True
                        print("Last blue car spawned. Ending simulation.")
                        running = False

                # Draw everything
                self.draw()

            else:
                # Režim bez vizualizace
                # Použijeme malou iteraci pro zpracování událostí, ale bez vykreslování
                # Aby simulace běžela co nejrychleji
                try:
                    while True:
                        event = pygame.event.poll()
                        if event.type == pygame.QUIT:
                            running = False
                            break
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            running = False
                            break
                        # Můžete přidat další zpracování událostí, pokud je potřeba
                        else:
                            break
                except:
                    pass  # Pokud není Pygame inicializován, ignorujte

                if not self.collision_occurred:
                    # Neomezený časový krok pro maximální rychlost
                    adjusted_dt = 0.01 * self.speed_multiplier  # Nastavte menší krok pro přesnost
                    self.simulation_time += adjusted_dt

                    # Aktualizujte auta
                    was_last_spawn = self.main_road.update_cars(adjusted_dt, self.simulation_time, self)
                    self.secondary_road.update_cars(adjusted_dt, self.simulation_time, self, main_road=self.main_road)

                    # Collision check
                    self.check_collisions()

                    # Ukončení simulace, pokud bylo spawnováno poslední modré auto
                    if not self.last_main_car_spawned and was_last_spawn:
                        self.last_main_car_spawned = True
                        print("Last blue car spawned. Ending simulation.")
                        running = False

        # Po ukončení simulace ukončíme Pygame a uložíme data do CSV
        if self.config.VISUALIZE:
            pygame.quit()
        self.save_data_to_csv()
        sys.exit()

    def record_main_pass(self, time):
        """
        Zaznamená čas průjezdu vozidla na hlavní silnici a uloží
        časový rozestup a počet mergovaných vozidel do `gap_data`.
        """
        self.main_pass_times.append(time)
        # Pokud existuje předchozí průjezd, spočítáme rozestup a počet mergovaných vozidel
        if len(self.main_pass_times) >= 2:
            previous_time = self.main_pass_times[-2]
            current_time = self.main_pass_times[-1]
            time_gap = current_time - previous_time
            # Spočítáme, kolik sekundních vozidel mergovalo mezi předchozím a aktuálním časem
            merge_count = sum(1 for merge_time in self.secondary_merge_times if previous_time < merge_time <= current_time)
            self.gap_data.append((time_gap, merge_count))
            # Výpis pouze aktuálního průchodu
            print(f"Time Gap: {time_gap:.2f}s, Secondary Merges: {merge_count}")

    def record_secondary_merge(self, time):
        """
        Zaznamená čas mergování vozidla z vedlejší silnice.
        """
        self.secondary_merge_times.append(time)

    def check_collisions(self):
        # 1) Check collision on the main road
        main_cars = sorted(self.main_road.cars, key=lambda c: c.position)
        for i in range(len(main_cars) - 1):
            car_a = main_cars[i]
            car_b = main_cars[i + 1]
            # If car_b is overlapping car_a
            if car_b.position - car_a.position - car_a.length < 0:
                self.collision_occurred = True
                print("Collision occurred on main road!")
                collision_pos = (car_a.position + car_b.position) / 2
                self.collision_x = self.config.MAIN_ROAD_X
                self.collision_y = self.config.HEIGHT - int(collision_pos * self.config.SCALE_MAIN * self.zoom_factor)
                break

        # 2) Check collision on the secondary road
        if not self.collision_occurred:
            sec_cars = sorted(self.secondary_road.cars, key=lambda c: c.position)
            for i in range(len(sec_cars) - 1):
                car_a = sec_cars[i]
                car_b = sec_cars[i + 1]
                if car_b.position - car_a.position - car_a.length < 0:
                    self.collision_occurred = True
                    print("Collision occurred on secondary road!")
                    collision_pos = (car_a.position + car_b.position) / 2
                    self.collision_x = int(collision_pos * self.config.SCALE_SECONDARY * self.zoom_factor)
                    self.collision_y = self.config.SECONDARY_ROAD_Y
                    break

        # 3) Check if any secondary car has overshot the intersection from left to right
        #    We'll consider it a crash if the car passes beyond some small margin (e.g., 1 meter).
        if not self.collision_occurred:
            for car in self.secondary_road.cars:
                if car.position > (self.secondary_road.stopping_point + 1.0):
                    # The car has crossed the intersection significantly
                    self.collision_occurred = True
                    print("Collision occurred: Secondary car overshot the intersection!")
                    collision_pos = car.position  # approximate crash position
                    self.collision_x = int(collision_pos * self.config.SCALE_SECONDARY * self.zoom_factor)
                    self.collision_y = self.config.SECONDARY_ROAD_Y
                    break

    def draw(self):
        if not self.config.VISUALIZE:
            return  # Pokud nemáme vizualizaci, neprovádíme kreslení

        self.window.fill(self.config.COLORS['WHITE'])
        # Draw roads
        self.main_road.draw(self.window, self.font, self.zoom_factor)
        self.secondary_road.draw(self.window, self.font, self.zoom_factor)

        # Draw the speed-up button
        self.draw_speed_button()

        # Draw the zoom slider
        self.draw_slider()

        if self.collision_occurred:
            # Draw collision indicator
            pygame.draw.circle(self.window, self.config.COLORS['RED'], (self.collision_x, self.collision_y), int(30 * self.zoom_factor), max(1, int(5 * self.zoom_factor)))
            crash_font = pygame.font.SysFont(None, 50)
            crash_text = crash_font.render("CRASH!", True, self.config.COLORS['RED'])
            crash_rect = crash_text.get_rect(center=(self.collision_x, self.collision_y - int(40 * self.zoom_factor)))
            self.window.blit(crash_text, crash_rect)

        pygame.display.flip()

    def draw_speed_button(self):
        # Draw button rectangle
        pygame.draw.rect(self.window, self.config.COLORS['GREEN'], self.button_rect)

        # Draw button text
        button_text = self.font.render(f"Speed: {self.speed_multiplier}x", True, self.config.COLORS['BLACK'])
        text_rect = button_text.get_rect(center=self.button_rect.center)
        self.window.blit(button_text, text_rect)

    def draw_slider(self):
        # Draw slider background
        pygame.draw.rect(self.window, self.config.COLORS['GRAY'], self.slider_rect)

        # Draw slider handle
        pygame.draw.rect(self.window, self.config.COLORS['GREEN'], self.slider_handle_rect)

        # Draw slider border
        pygame.draw.rect(self.window, self.config.COLORS['BLACK'], self.slider_rect, 2)
        pygame.draw.rect(self.window, self.config.COLORS['BLACK'], self.slider_handle_rect, 2)

        # Draw zoom level text
        zoom_text = self.font.render(f"Zoom: {self.zoom_factor:.2f}x", True, self.config.COLORS['BLACK'])
        zoom_rect = zoom_text.get_rect(center=(self.slider_rect.centerx, self.slider_rect.y - 30))
        self.window.blit(zoom_text, zoom_rect)

    def update_speed_multiplier(self):
        if self.speed_multiplier < self.max_multiplier:
            self.speed_multiplier *= 2
            if self.speed_multiplier > self.max_multiplier:
                self.speed_multiplier = self.max_multiplier
            print(f"Simulation speed increased to {self.speed_multiplier}x")
        else:
            self.speed_multiplier = self.min_multiplier
            print(f"Simulation speed reset to {self.speed_multiplier}x")

    def save_data_to_csv(self):
        """
        Saves all recorded time gaps and secondary merges to a CSV file.
        """
        filename = "data/simulation_data.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time Gap (s)", "Secondary Merges"])
            for gap, count in self.gap_data:
                writer.writerow([f"{gap:.2f}", count])
        print(f"Data saved to {filename}")

        # Výpis konečných statistik
        if self.gap_data:
            last_gap, last_merge = self.gap_data[-1]
            print(f"Final Time Gap: {last_gap:.2f}s, Total Secondary Merges: {last_merge}")

# --- Main Execution ---
if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()
