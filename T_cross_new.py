#%%
import pygame
import numpy as np
from scipy.stats import geninvgauss, lognorm
import csv
import sys

def apply_zoom(x, y, zoom_factor, center_x, center_y):
    """
    Applies zoom to a point (x, y) centered at (center_x, center_y).
    (Použito při vykreslování, když VISUALIZE=True.)
    """
    x -= center_x
    y -= center_y
    x *= zoom_factor
    y *= zoom_factor
    x += center_x
    y += center_y
    return int(x), int(y)

class Config:
    """
    Centrální konfigurace simulace.
    """
    # --- Vizualizace ---
    VISUALIZE = False  # Pokud nastavíte na True, otevře se okno a simulace poběží reálně; False - simulace jede rychleji, ale nezobrazuje se

    WIDTH, HEIGHT = 900, 600
    SCALE_MAIN = 1
    SCALE_SECONDARY = 1.33
    MAX_SPEED = 50.0          # m/s (~180 km/h)
    MAX_DECELERATION = 4.0    # m/s^2

    # --- Délky silnic (v metrech) ---
    ROAD_LENGTH_MAIN = 600
    ROAD_LENGTH_SECONDARY = 300

    # --- IDM parametry pro hlavní a vedlejší silnici ---
    MAIN_IDM_PARAMETERS = {
        'min_gap': 1.625,
        'acceleration': 2.0,
        'comfortable_deceleration': 4.0,  # pokud preferujete, můžete i 2.0 atd.
        'react_time': 0.3,
        'desired_speed': 19.44,  # ~70 km/h
        'delta': 2,
    }
    SECONDARY_IDM_PARAMETERS = {
        'min_gap': 1.625,
        'acceleration': 4.0,
        'comfortable_deceleration': 4.0,
        'react_time': 0.3,
        'desired_speed': 13.88,  # ~50 km/h
        'delta': 1,
    }

    # --- Pomalá zóna (při dojezdu ke křižovatce z vedlejší) ---
    SEC_SLOWDOWN_START_DISTANCE = 50.0
    SEC_SLOWDOWN_TARGET_SPEED = 10.0 / 3.6  # 10 km/h → 2.78 m/s

    # --- Rozdělení intervalu spawnu (GIG) ---
    MAIN_GIG_PARAMETERS = {
        'lambda_': 2.71,
        'beta': 1.03
    }
    SECONDARY_GIG_PARAMETERS = {
        'lambda_': 11.5,
        'beta': 4.7,
    }

    # --- Rozdělení počátečních rychlostí (lognorm) ---
    MAIN_LOGNORM_PARAMETERS = {
        'mu_log': np.log(13.89) - (0.107**2)/2,
        'sigma_log': 0.107,
    }
    SECONDARY_LOGNORM_PARAMETERS = {
        'mu_log': np.log(11.11) - (0.107**2)/2,
        'sigma_log': 0.107,
    }

    # --- Mergování ---
    LINEUP = True
    MERGE_CHECK_DISTANCE = 50.0
    MERGE_DISTANCE_THRESHOLD = 1.0
    MERGING_SPEED = 10 / 3.6  # 10 km/h

    # --- Pozice silnic v okně ---
    MAIN_ROAD_X = 400
    SECONDARY_ROAD_Y = 300
    INTERSECTION_POSITION_MAIN_ROAD = 300.0  # v metrech od začátku hlavní cesty

    # --- Barvy pro Pygame kreslení ---
    COLORS = {
        'WHITE': (255, 255, 255),
        'BLACK': (0, 0, 0),
        'BLUE': (0, 0, 255),
        'RED': (255, 0, 0),
        'GRAY': (200, 200, 200),
        'GREEN': (0, 255, 0),
    }

    # --- Omezení počtu aut na hlavní silnici ---
    MAX_TOTAL_SPAWNED_CARS_MAIN = 5000

class IDM:
    """
    Intelligent Driver Model
    """
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
        if self.v0 < 1e-3:
            return -self.b
        if s <= 0:
            s = 0.1  # zamezení dělení nulou
        s_star = self.s0 + max(0, v*self.t0 + (v*delta_v)/(2*np.sqrt(self.a*self.b)))
        accel = self.a * (1 - (v/self.v0)**self.delta - (s_star/s)**2)
        return max(accel, -self.b)

    def update_parameters(self, params):
        self.v0 = params.get('desired_speed', self.v0)
        self.a = params.get('acceleration', self.a)
        self.b = params.get('comfortable_deceleration', self.b)
        self.delta = params.get('delta', self.delta)
        self.s0 = params.get('min_gap', self.s0)
        self.t0 = params.get('react_time', self.t0)

class Car:
    """
    Třída pro auto na silnici.
    """
    WIDTH = 10
    HEIGHT = 5  # zmenšeno pro přehlednost vykreslování

    def __init__(self, position, road_type, config, initial_speed=0.0):
        self.position = position
        self.road_type = road_type
        self.length = 5.0
        self.v = initial_speed
        self.a = 0.0
        self.config = config
        self.idm = IDM(config, road_type)
        self.color = config.COLORS['BLUE'] if road_type == 'main' else config.COLORS['RED']
        self.has_passed_intersection = False  # sleduje, zda modré auto už projelo křižovatkou

    def update(self, dt, lead_car=None, simulation_time=None, simulation=None):
        """
        Aktualizace stavu vozidla (poloha, rychlost, zrychlení).
        """
        if lead_car:
            s = lead_car.position - self.position - lead_car.length
            delta_v = self.v - lead_car.v
            self.a = self.idm.compute_acceleration(self.v, delta_v, s)
        else:
            # free flow
            if self.idm.v0 < 1e-3:
                self.a = -self.config.MAX_DECELERATION
            else:
                self.a = self.idm.a * (1 - (self.v / self.idm.v0)**self.idm.delta)

        # Omezíme deceleraci
        self.a = max(self.a, -self.config.MAX_DECELERATION)

        # Aktualizace rychlosti
        self.v += self.a * dt
        self.v = max(min(self.v, self.config.MAX_SPEED), 0)

        # Aktualizace polohy
        self.position += self.v * dt + 0.5 * self.a * dt**2

        # Kontrola, zda modré auto projelo křižovatkou
        if self.road_type == 'main' and self.color == self.config.COLORS['BLUE'] and not self.has_passed_intersection:
            if self.position >= self.config.INTERSECTION_POSITION_MAIN_ROAD:
                self.has_passed_intersection = True
                if simulation:
                    simulation.record_main_pass(simulation_time)

    def update_idm_parameters(self, params):
        self.idm.update_parameters(params)

    # Následující dvě metody (`draw`, `get_position`) se využívají jen, pokud je VISUALIZE=True.
    def draw(self, window, font, zoom_factor):
        if self.road_type == 'main':
            x = self.config.MAIN_ROAD_X
            y = self.config.HEIGHT - self.position * self.config.SCALE_MAIN
        else:
            x = self.position * self.config.SCALE_SECONDARY
            y = self.config.SECONDARY_ROAD_Y

        x_zoomed, y_zoomed = apply_zoom(x, y, zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)
        scaled_width = max(1, int(self.WIDTH * zoom_factor))
        scaled_height = max(1, int(self.HEIGHT * zoom_factor))

        rect = pygame.Rect(x_zoomed - scaled_width // 2, y_zoomed - scaled_height // 2, scaled_width, scaled_height)
        pygame.draw.rect(window, self.color, rect)

        speed_kmh = self.v * 3.6
        text_surface = font.render(f"{speed_kmh:.1f} km/h", True, self.config.COLORS['BLACK'])
        text_rect = text_surface.get_rect(center=(rect.centerx, rect.centery - 15 * zoom_factor))
        window.blit(text_surface, text_rect)

    def get_position(self):
        if self.road_type == 'main':
            x = self.config.MAIN_ROAD_X
            y = self.config.HEIGHT - self.position * self.config.SCALE_MAIN
        else:
            x = self.position * self.config.SCALE_SECONDARY
            y = self.config.SECONDARY_ROAD_Y
        return x, y

class GIGDistribution:
    """
    Rozdělení generované Inverse Gaussian (pro spawnovací intervaly).
    """
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

class Road:
    """
    Třída reprezentující silnici (hlavní nebo vedlejší).
    Obsahuje seznam aut a logiku spawnu/aut.
    """
    def __init__(self, config, road_type):
        self.config = config
        self.road_type = road_type
        self.cars = []

        # Nastavení parametrů pro "main" vs "secondary"
        if road_type == 'main':
            gig_params = config.MAIN_GIG_PARAMETERS
            log_params = config.MAIN_LOGNORM_PARAMETERS
            self.scale = config.SCALE_MAIN
            self.road_length = config.ROAD_LENGTH_MAIN
            self.max_total_spawned_cars_main = config.MAX_TOTAL_SPAWNED_CARS_MAIN
        else:
            gig_params = config.SECONDARY_GIG_PARAMETERS
            log_params = config.SECONDARY_LOGNORM_PARAMETERS
            self.scale = config.SCALE_SECONDARY
            self.road_length = config.ROAD_LENGTH_SECONDARY
            self.max_total_spawned_cars_main = None

        self.gig_distribution = GIGDistribution(gig_params['lambda_'], gig_params['beta'])
        self.lognorm_mu_log = log_params['mu_log']
        self.lognorm_sigma_log = log_params['sigma_log']

        # Čas do dalšího spawnu
        self.next_spawn_time = self.gig_distribution.sample()[0]
        self.time_since_last_spawn = 0.0

        self.total_spawned_cars = 0

        if self.road_type == 'secondary':
            self.stopping_point = self.config.INTERSECTION_POSITION_MAIN_ROAD
        else:
            self.stopping_point = None

    def spawn_car(self):
        """
        Pokus o spawn nového auta. Pokud bylo dosaženo maxima (hlavní silnice),
        skript se ukončí (můžete změnit chování podle potřeby).
        """
        if self.road_type == 'main':
            if (self.max_total_spawned_cars_main is not None and
                self.total_spawned_cars >= self.max_total_spawned_cars_main):
                print("Reached max number of main road cars.")
                if self.config.VISUALIZE:
                    pygame.quit()
                sys.exit()

        # Náhodná počáteční rychlost z lognorm
        speed_sample = lognorm.rvs(s=self.lognorm_sigma_log, scale=np.exp(self.lognorm_mu_log))
        speed_sample = max(speed_sample, 0.0)

        new_car = Car(position=0.0, road_type=self.road_type, config=self.config, initial_speed=speed_sample)
        self.cars.append(new_car)
        self.time_since_last_spawn = 0.0

        # Počítadlo pro hlavní silnici
        if self.road_type == 'main' and new_car.color == self.config.COLORS['BLUE']:
            self.total_spawned_cars += 1
            print(f"Spawned cars: {self.total_spawned_cars}/{self.max_total_spawned_cars_main}")

        # Nastavíme další spawn čas
        self.next_spawn_time = self.gig_distribution.sample()[0]

        # Pokud jsme dosáhli maxima
        if self.road_type == 'main' and self.total_spawned_cars >= self.max_total_spawned_cars_main:
            return True
        return False

    def get_linear_desired_speed(self, distance):
        """
        Pro vedlejší cestu: lineární zpomalení z ~50 km/h na 10 km/h
        na úseku "SEC_SLOWDOWN_START_DISTANCE".
        """
        original_speed = self.config.SECONDARY_IDM_PARAMETERS['desired_speed']  # 13.88 m/s
        target_speed = self.config.SEC_SLOWDOWN_TARGET_SPEED
        start_dist = self.config.SEC_SLOWDOWN_START_DISTANCE

        if distance >= start_dist:
            return original_speed
        if distance <= 0:
            return target_speed
        ratio = distance / start_dist
        return target_speed + ratio * (original_speed - target_speed)

    def recheck_merge(self, car, main_road):
        """
        Zda je volno pro merge na hlavní silnici.
        """
        if main_road and len(main_road.cars) == 0:
            return True
        intersection = self.config.INTERSECTION_POSITION_MAIN_ROAD
        lead_car, lag_car = None, None

        for main_car in main_road.cars:
            if main_car.position > intersection:
                # hledáme lead
                if not lead_car or main_car.position < lead_car.position:
                    lead_car = main_car
            else:
                # hledáme lag
                if not lag_car or main_car.position > lag_car.position:
                    lag_car = main_car

        gap_ahead = float('inf') if not lead_car else (lead_car.position - intersection - lead_car.length)
        gap_behind = float('inf') if not lag_car else (intersection - lag_car.position - car.length - 3*(lag_car.v - car.v))
        min_gap = car.idm.s0

        if gap_ahead >= 0 and gap_behind >= min_gap:
            return True
        return False

    def update_cars(self, dt, simulation_time, simulation, main_road=None):
        """
        Aktualizace všech aut na silnici, detekce merge.
        """
        if not self.cars:
            return False

        self.cars.sort(key=lambda c: c.position, reverse=True)

        i = 0
        while i < len(self.cars):
            car = self.cars[i]

            if self.road_type == 'secondary' and i == 0:
                # Auto, které je nejblíže křižovatce
                distance_to_int = self.stopping_point - car.position
                new_speed = self.get_linear_desired_speed(distance_to_int)
                updated_params = self.config.SECONDARY_IDM_PARAMETERS.copy()
                updated_params['desired_speed'] = new_speed
                car.update_idm_parameters(updated_params)

                # Vytvoříme virtuální auto v křižovatce
                virtual_car = Car(position=self.stopping_point, road_type='secondary', config=self.config, initial_speed=0.0)
                virtual_car.v = 0.0
                virtual_car.length = 0.0

                if distance_to_int <= self.config.MERGE_CHECK_DISTANCE:
                    can_merge = False
                    if main_road:
                        can_merge = self.recheck_merge(car, main_road)
                    if can_merge:
                        # Nastavíme auto, aby dojelo do křižovatky nízkou rychlostí
                        merge_params = self.config.SECONDARY_IDM_PARAMETERS.copy()
                        merge_params['desired_speed'] = self.config.MERGING_SPEED
                        car.update_idm_parameters(merge_params)
                        lead_car = None
                    else:
                        lead_car = virtual_car
                else:
                    lead_car = virtual_car

                car.update(dt, lead_car=lead_car, simulation_time=simulation_time, simulation=simulation)

                # Finální merge přímo v křižovatce
                if (self.config.LINEUP
                    and main_road
                    and abs(car.position - self.stopping_point) <= self.config.MERGE_DISTANCE_THRESHOLD):
                    can_merge_final = False
                    if main_road:
                        can_merge_final = self.recheck_merge(car, main_road)
                    if can_merge_final:
                        self.cars.pop(i)
                        car.road_type = 'main'
                        car.position = self.config.INTERSECTION_POSITION_MAIN_ROAD
                        car.v = min(car.v, self.config.MERGING_SPEED)
                        car.idm = IDM(self.config, road_type='main')
                        main_road.cars.append(car)
                        print("Car from secondary road merged onto main road.")
                        # Zaznamenáme merge
                        simulation.record_secondary_merge(simulation_time)
                        continue
            else:
                # Normální update
                if i == 0:
                    car.update(dt, lead_car=None, simulation_time=simulation_time, simulation=simulation)
                else:
                    lead_car = self.cars[i - 1]
                    car.update(dt, lead_car=lead_car, simulation_time=simulation_time, simulation=simulation)

            i += 1

        self.time_since_last_spawn += dt

        # Zkusíme spawnout nové auto
        if self.time_since_last_spawn >= self.next_spawn_time:
            was_last_spawn = self.spawn_car()
            if was_last_spawn:
                # Vracejte True, pokud jsme zrovna spawnuli poslední auto
                return True
        return False

    # Kreslení silnice a aut (využito jen při VISUALIZE=True):
    def draw(self, window, font, zoom_factor):
        # Nakreslíme linku silnice
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

        # Nakreslíme značky (po 50 m)
        if self.road_type == 'main':
            max_dist = self.config.ROAD_LENGTH_MAIN
            for pos in range(0, int(max_dist) + 1, 50):
                y = self.config.HEIGHT - pos * self.config.SCALE_MAIN
                x_start = self.config.MAIN_ROAD_X - 20
                x_end = self.config.MAIN_ROAD_X + 20

                y_zoomed = apply_zoom(self.config.MAIN_ROAD_X, y, zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)[1]
                x_start_zoomed = apply_zoom(x_start, y, zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)[0]
                x_end_zoomed = apply_zoom(x_end, y, zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)[0]

                pygame.draw.line(window, self.config.COLORS['GRAY'], (x_start_zoomed, y_zoomed), (x_end_zoomed, y_zoomed), max(1, int(2 * zoom_factor)))
                label = font.render(f"{pos} m", True, self.config.COLORS['BLACK'])
                rect = label.get_rect(center=(x_start_zoomed - 50 * zoom_factor, y_zoomed))
                window.blit(label, rect)

        else:  # vedlejší
            max_dist = self.config.ROAD_LENGTH_SECONDARY
            for pos in range(0, int(max_dist) + 1, 50):
                x = pos * self.config.SCALE_SECONDARY
                x_zoomed = apply_zoom(x, self.config.SECONDARY_ROAD_Y, zoom_factor, self.config.MAIN_ROAD_X, self.config.SECONDARY_ROAD_Y)[0]
                y_start = self.config.SECONDARY_ROAD_Y - 20
                y_end = self.config.SECONDARY_ROAD_Y + 20

                pygame.draw.line(window, self.config.COLORS['GRAY'],
                                 (x_zoomed, y_start * zoom_factor + (self.config.SECONDARY_ROAD_Y * (1 - zoom_factor))),
                                 (x_zoomed, y_end * zoom_factor + (self.config.SECONDARY_ROAD_Y * (1 - zoom_factor))),
                                 max(1, int(2 * zoom_factor)))

                dist_from_int = int(max_dist - pos)
                label = font.render(f"{dist_from_int} m", True, self.config.COLORS['BLACK'])
                rect = label.get_rect(center=(x_zoomed, self.config.SECONDARY_ROAD_Y + 30 * zoom_factor))
                window.blit(label, rect)

        # Nakreslíme všechna auta
        for car in self.cars:
            car.draw(window, font, zoom_factor)

class Simulation:
    """
    Vlastní simulace: drží hlavní i vedlejší silnici, spouští update a vykreslování.
    """
    def __init__(self):
        self.config = Config()
        # Pokud chceme vizualizaci
        if self.config.VISUALIZE:
            pygame.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.config.WIDTH, self.config.HEIGHT))
            pygame.display.set_caption("T-Intersection Simulation Optimized")
            self.font = pygame.font.SysFont(None, 20)
            self.clock = pygame.time.Clock()
        else:
            self.window = None
            self.font = None
            self.clock = None

        self.main_road = Road(self.config, 'main')
        self.secondary_road = Road(self.config, 'secondary')

        self.collision_occurred = False
        self.collision_x = 0
        self.collision_y = 0

        # Parametry zoomu a slideru (využité jen při VISUALIZE=True)
        self.zoom_factor = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 6.0
        self.zoom_step = 0.1

        self.slider_rect = pygame.Rect(800, 100, 20, 400)
        self.slider_handle_rect = pygame.Rect(790, 300, 40, 10)
        self.slider_dragging = False

        self.speed_multiplier = 1
        self.max_multiplier = 32
        self.min_multiplier = 1
        self.button_width = 100
        self.button_height = 50
        self.button_rect = pygame.Rect(10, 10, self.button_width, self.button_height)

        self.simulation_time = 0.0

        self.last_main_car_spawned = False

        # CSV soubor pro průběžné ukládání [Time Gap, Secondary Merges]
        self.csv_file = open("simulation_data.csv", mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Time Gap (s)", "Secondary Merges"])

        # Držíme jen poslední čas průjezdu a čítač
        self.last_pass_time = None
        self.merge_count = 0

    def run(self):
        """
        Hlavní smyčka simulace.
        """
        running = True
        # Pro start si spawneme aspoň 1 auto na hlavní i vedlejší
        self.main_road.spawn_car()
        self.secondary_road.spawn_car()

        while running:
            if self.config.VISUALIZE:
                dt = self.clock.tick(60) / 1000.0
            else:
                # Bez vizualizace můžeme nastavit menší dt pro rychlou simulaci
                dt = 0.1

            adjusted_dt = dt * self.speed_multiplier
            self.simulation_time += adjusted_dt

            # Event loop - pokud je VISUALIZE=False, tak tu vlastně nic nedělá
            if self.config.VISUALIZE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            mouse_pos = event.pos
                            if self.slider_handle_rect.collidepoint(mouse_pos):
                                self.slider_dragging = True
                            elif self.button_rect.collidepoint(mouse_pos):
                                self.update_speed_multiplier()
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            self.slider_dragging = False
                    elif event.type == pygame.MOUSEMOTION:
                        if self.slider_dragging:
                            mouse_x, mouse_y = event.pos
                            new_y = max(self.slider_rect.y, min(mouse_y, self.slider_rect.y + self.slider_rect.height))
                            self.slider_handle_rect.y = new_y - self.slider_handle_rect.height // 2
                            relative_y = new_y - self.slider_rect.y
                            ratio = 1 - (relative_y / self.slider_rect.height)
                            self.zoom_factor = self.min_zoom + ratio * (self.max_zoom - self.min_zoom)
                            self.zoom_factor = round(self.zoom_factor, 2)
                    elif event.type == pygame.MOUSEWHEEL:
                        if event.y > 0:
                            self.zoom_factor = min(self.zoom_factor * 1.1, self.max_zoom)
                        elif event.y < 0:
                            self.zoom_factor = max(self.zoom_factor / 1.1, self.min_zoom)
                        self.zoom_factor = round(self.zoom_factor, 2)

            if not self.collision_occurred:
                # Update hlavní a vedlejší silnice
                was_last_spawn = self.main_road.update_cars(adjusted_dt, self.simulation_time, self)
                self.secondary_road.update_cars(adjusted_dt, self.simulation_time, self, main_road=self.main_road)
                self.check_collisions()

                # Pokud bylo spawnováno poslední modré auto (dle MAX_TOTAL_SPAWNED_CARS_MAIN)
                if not self.last_main_car_spawned and was_last_spawn:
                    self.last_main_car_spawned = True
                    print("Last blue car spawned. Ending simulation.")
                    running = False

            if self.config.VISUALIZE:
                self.draw()

        # Konec simulace
        if self.config.VISUALIZE:
            pygame.quit()

        # Zavření CSV souboru
        self.csv_file.close()
        sys.exit()

    # --- Záznamy o průjezdu a merge ---
    def record_main_pass(self, time):
        """
        Modré auto projelo křižovatkou - spočítáme time gap a uložíme
        + zapíšeme merge_count.
        """
        if self.last_pass_time is None:
            # První průjezd
            self.last_pass_time = time
        else:
            time_gap = time - self.last_pass_time
            self.last_pass_time = time
            # Zapsat do CSV
            self.csv_writer.writerow([f"{time_gap:.2f}", self.merge_count])
            # Vynulovat merge_count
            self.merge_count = 0

    def record_secondary_merge(self, time):
        """
        Zvýšíme jen čítač sloučení.
        """
        self.merge_count += 1

    # --- Kontrola kolizí ---
    def check_collisions(self):
        main_cars = sorted(self.main_road.cars, key=lambda c: c.position)
        for i in range(len(main_cars) - 1):
            car_a = main_cars[i]
            car_b = main_cars[i + 1]
            if car_b.position - car_a.position - car_a.length < 0:
                self.collision_occurred = True
                print("Collision on main road!")
                break

        if not self.collision_occurred:
            sec_cars = sorted(self.secondary_road.cars, key=lambda c: c.position)
            for i in range(len(sec_cars) - 1):
                car_a = sec_cars[i]
                car_b = sec_cars[i + 1]
                if car_b.position - car_a.position - car_a.length < 0:
                    self.collision_occurred = True
                    print("Collision on secondary road!")
                    break

        if not self.collision_occurred:
            for car in self.secondary_road.cars:
                if car.position > (self.secondary_road.stopping_point + 1.0):
                    self.collision_occurred = True
                    print("Collision: Secondary car overshot intersection!")
                    break

    # --- Kreslení (pokud VISUALIZE=True) ---
    def draw(self):
        if not self.config.VISUALIZE:
            return
        self.window.fill(self.config.COLORS['WHITE'])
        self.main_road.draw(self.window, self.font, self.zoom_factor)
        self.secondary_road.draw(self.window, self.font, self.zoom_factor)
        self.draw_speed_button()
        self.draw_slider()
        pygame.display.flip()

    def draw_speed_button(self):
        pygame.draw.rect(self.window, self.config.COLORS['GREEN'], self.button_rect)
        button_text = self.font.render(f"Speed: {self.speed_multiplier}x", True, self.config.COLORS['BLACK'])
        text_rect = button_text.get_rect(center=self.button_rect.center)
        self.window.blit(button_text, text_rect)

    def draw_slider(self):
        pygame.draw.rect(self.window, self.config.COLORS['GRAY'], self.slider_rect)
        pygame.draw.rect(self.window, self.config.COLORS['GREEN'], self.slider_handle_rect)
        pygame.draw.rect(self.window, self.config.COLORS['BLACK'], self.slider_rect, 2)
        pygame.draw.rect(self.window, self.config.COLORS['BLACK'], self.slider_handle_rect, 2)

    def update_speed_multiplier(self):
        if self.speed_multiplier < self.max_multiplier:
            self.speed_multiplier *= 2
            if self.speed_multiplier > self.max_multiplier:
                self.speed_multiplier = self.max_multiplier
            print(f"Simulation speed increased to {self.speed_multiplier}x")
        else:
            self.speed_multiplier = self.min_multiplier
            print(f"Simulation speed reset to {self.speed_multiplier}x")

# --- Hlavní spuštění skriptu ---
if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()
