import pygame
import numpy as np
from scipy.stats import lognorm, kstest, geninvgauss
import matplotlib.pyplot as plt
import pandas as pd  # For loading Excel files
import sys  # For exiting the program
from collections import deque

# --- PARAMETERS FOR SIMULATION ---
class Config:
    VIZUALIZE = True

    # Simulation Parameters
    MAX_CARS = 1000  # Nastaveno na 5000 pro plné testování

    # GIG Distribution Parameters for Spawn Time Intervals
    GIG_TIME_PARAMETERS = {
        'lambda_': 2.71,    # Shape parameter (lambda_) - Optimized
        'beta': 1.03       # Scale parameter (beta) - Optimized
    }

    # LOGNORM Distribution Parameters for spawn speeds - mean=50km/h, std=1,5m/s
    LOGNORM_PARAMETERS = {
        "mu_log": np.log(13.89) - (0.107**2) / 2,  # Approximately 2.62425
        "sigma_log": 0.107
    }

    # IDM parameters
    #test - 2,75	3	4	2	25	4
    MIN_GAP = 2.75  # Minimum gap between cars
    MAX_ACCELERATION = 3.0  # Maximum acceleration in m/s²
    MAX_DECELERATION = 4.0  # Maximum deceleration in m/s²
    REACT_TIME = 2.0  # Driver's react time (time headway)
    DESIRED_SPEED = 25.0
    DELTA = 4

    # Road Settings
    ROAD_LENGTH = 300  # Road length in meters
    SCALE = 5          # Conversion factor (5 pixels = 1 meter)

    # Roadcross Position
    ROADCROSS_X_PX = ROAD_LENGTH * SCALE  # Roadcross position in pixels
    ROADCROSS_POSITION = ROAD_LENGTH  # Position in meters

    # Colors
    COLORS = {
        'WHITE': (255, 255, 255),
        'BLACK': (0, 0, 0),
        'BLUE': (0, 0, 255),
        'RED': (255, 0, 0),
        'GRAY': (200, 200, 200)
    }

    # Screen Dimensions
    WIDTH = 800
    HEIGHT = 600

# --- Intelligent Driver Model (IDM) ---
class IDM:
    def __init__(self, config):
        self.v0 = config.DESIRED_SPEED  # Desired speed (m/s)
        self.a = config.MAX_ACCELERATION  # Maximum acceleration (m/s²)
        self.b = config.MAX_DECELERATION  # Comfortable deceleration (m/s²)
        self.delta = config.DELTA  # Exponent
        self.s0 = config.MIN_GAP  # Minimum gap (m)
        self.t0 = config.REACT_TIME

    def compute_acceleration(self, v, delta_v, s):
        if s <= 0:
            s = 0.1  # Prevent division by zero
        s_star = self.s0 + max(0, v * self.t0 + (v * delta_v) / (2 * np.sqrt(self.a * self.b)))
        acceleration = self.a * (1 - (v / self.v0) ** self.delta - (s_star / s) ** 2)
        # Limit deceleration
        acceleration = max(acceleration, -self.b)
        return acceleration

# --- Car Class ---
class Car:
    WIDTH = 20
    HEIGHT = 10

    def __init__(self, position, road, config, initial_speed):
        self.position = position  # Position on the road (m)
        self.road = road          # Reference to the Road object
        self.length = 5.0         # Car length in meters for spacing calculations
        self.v = initial_speed    # Speed (m/s)
        self.a = 0.0              # Acceleration (m/s²)
        self.idm = IDM(config)
        self.color = config.COLORS['BLUE']
        self.has_recorded_distance = False  # Flag to check if distance has been recorded

        # Additional attributes
        self.should_remove = False          # Indicates if the car should be removed
        self.has_left_screen = False        # Indicates if the car has left the screen
        self.roadcross_recorded = False     # Indicates if roadcross distance has been recorded

    def update(self, dt, lead_car=None):
        if lead_car:
            s = lead_car.position - self.position - lead_car.length
            delta_v = self.v - lead_car.v
            self.a = self.idm.compute_acceleration(self.v, delta_v, s)
        else:
            self.a = self.idm.a * (1 - (self.v / self.idm.v0) ** self.idm.delta)

        # Limit deceleration
        self.a = max(self.a, -self.idm.b)
        # Update speed and position
        self.v += self.a * dt
        self.v = max(self.v, 0)  # Apply speed limiter
        self.position += self.v * dt + 0.5 * self.a * dt ** 2

        # Check if the car has left the screen
        if self.position > self.road.config.ROADCROSS_POSITION + self.road.config.ROAD_LENGTH:
            self.has_left_screen = True

    def draw(self, window, font):
        x = int(self.position * self.road.config.SCALE)
        y = self.road.config.HEIGHT // 2
        rect = pygame.Rect(x - self.WIDTH // 2, y - self.HEIGHT // 2, self.WIDTH, self.HEIGHT)
        pygame.draw.rect(window, self.color, rect)

        # Display car speed
        speed_kmh = self.v * 3.6  # Convert to km/h
        speed_text = font.render(f"{speed_kmh:.1f} km/h", True, self.road.config.COLORS['BLACK'])
        text_rect = speed_text.get_rect(center=(rect.centerx, rect.centery - 15))
        window.blit(speed_text, text_rect)

    def get_position(self):
        x = int(self.position * self.road.config.SCALE)
        y = self.road.config.HEIGHT // 2
        return x, y

# --- GIG Distribution Class ---
class GIGDistribution:
    def __init__(self, lambda_, beta):
        """
        Initialize the GIGDistribution with parameters lambda_ and beta.

        Parameters:
        - lambda_ (float): Shape parameter (λ)
        - beta (float): Scale parameter (β)
        """
        self.lambda_ = lambda_
        self.beta = beta
        self.geninvgauss_dist = geninvgauss(self.lambda_, self.beta)
        self.fitted_params = None  # Placeholder for fitted parameters

    def pdf(self, x):
        """
        Probability Density Function for the GIG distribution using SciPy's geninvgauss.
        """
        return self.geninvgauss_dist.pdf(x)

    def sample(self, num_samples=1):
        """
        Sample values from the GIG(lambda_, beta) distribution using SciPy's geninvgauss.

        Parameters:
        - num_samples (int): Number of samples to generate

        Returns:
        - samples (list): List of sampled values
        """
        try:
            samples = self.geninvgauss_dist.rvs(size=num_samples)
            # Ensure sampled values are positive and respect the minimum interval
            samples = np.clip(samples, 0.5, None)  # Set minimum interval to 0.5 seconds
            # Optional: Remove extremely large intervals if necessary
            # samples = np.clip(samples, 0.5, 20.0)
            # print(samples)  # Remove or comment out to reduce console clutter
            if len(samples) < num_samples:
                # If some samples are invalid, resample the missing ones
                additional_samples = self.sample(num_samples - len(samples))
                samples = np.concatenate((samples, additional_samples))
            return samples.tolist()
        except ValueError as e:
            if hasattr(self, 'road') and not self.road.config.SILENT:
                print(f"Error sampling next spawn time interval: {e}. Setting default interval of 10.0 seconds.")
            return [10.0] * num_samples  # Default interval if sampling fails

    def fit(self, data):
        """
        Fit the GIG distribution to the data using MLE with fixed scale=1.

        Parameters:
        - data (array-like): Data points

        Returns:
        - fitted_params (tuple): Fitted parameters (lambda_, b, loc), or None if fitting failed
        """
        # Filter data to ensure all values are positive
        data = np.array(data)
        data = data[data > 0]
        if len(data) == 0:
            if hasattr(self, 'road') and not self.road.config.SILENT:
                print("No positive data available for fitting.")
            return None

        try:
            # Fit using SciPy's geninvgauss with fixed scale=1 and loc unfixed
            lambda_, b, loc, scale = geninvgauss.fit(data, floc=0, fscale=1)

            # Store fitted parameters and return them
            self.lambda_ = lambda_
            self.beta = 1  # Scale is fixed at 1
            self.geninvgauss_dist = geninvgauss(lambda_, b, loc=0, scale=1)
            self.fitted_params = (lambda_, b, loc)

            return (lambda_, b, loc)
        except Exception as e:
            if hasattr(self, 'road') and not self.road.config.SILENT:
                print(f"Error during fitting: {e}")
            return None  # Explicitly return None in case of failure

    def plot_fit(self, data, fitted_params, theoretical_params=None, title='', xlabel='', ylabel='', ax=None, xlim=(0, 35)):
        """
        Plot histogram of data and the fitted GIG PDF. Optionally, plot the theoretical GIG PDF.

        Parameters:
        - data (array-like): Data points
        - fitted_params (tuple): Fitted parameters (lambda_, b, loc)
        - theoretical_params (tuple): Optional tuple of theoretical parameters (lambda_, beta)
        - title (str): Plot title
        - xlabel (str): X-axis label
        - ylabel (str): Y-axis label
        - ax (matplotlib.axes.Axes): Matplotlib axis to plot on
        - xlim (tuple): Limits for the x-axis

        Returns:
        - None
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Histogram of data
        ax.hist(data, bins=50, density=True, alpha=0.6, color='g', edgecolor='black', label='Data')

        # Fitted GIG PDF
        x = np.linspace(0, max(xlim), 1000)  # Start x from 0 up to xlim[1]
        lambda_fit, b_fit, loc_fit = fitted_params
        pdf_fitted = geninvgauss.pdf(x, lambda_fit, b_fit, loc=loc_fit, scale=1)
        ax.plot(x, pdf_fitted, 'r-', lw=2, label=f'Fitted GIG\nλ={lambda_fit:.2f}, b={b_fit:.2f}, loc={loc_fit:.2f}')

        # Theoretical GIG PDF (if provided)
        if theoretical_params is not None:
            lambda_theo, beta_theo = theoretical_params
            # Note: The theoretical b can be either fixed or use b_fit, depending on interpretation
            # Here, we use b_fit with theoretical lambda_ and beta
            pdf_theoretical = geninvgauss.pdf(x, lambda_theo, beta_theo, loc=loc_fit, scale=1)
            ax.plot(x, pdf_theoretical, 'b--', lw=2, label=f'Theoretical GIG\nλ={lambda_theo:.2f}, β={beta_theo:.2f}')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)  # Set the x-axis to range from 0 to 35
        ax.legend()

# --- Data Manager Class ---
class DataManager:
    def __init__(self, config):
        self.config = config
        self.roadcross_time_intervals = []  # Renamed for clarity
        self.spawn_time_intervals = []
        self.real_data = []
        self.initial_speeds = []  # List for initial speeds

    def load_real_data(self, file_path):
        if self.config.SILENT:
            return
        try:
            # Load data from Excel file
            df = pd.read_excel(file_path, header=None)
            if df.shape[1] < 2:
                # If only one column, assume it's individual distances
                data = df.iloc[:, 0].values
            else:
                # Only load unique distances from the first column
                data = df.iloc[:, 0].values

            if not self.config.SILENT:
                print(f"\nLoaded {len(data)} unique distances from '{file_path}'.")
            self.real_data = data
        except FileNotFoundError:
            if not self.config.SILENT:
                print(f"\nFile '{file_path}' not found.")
        except Exception as e:
            if not self.config.SILENT:
                print(f"\nError loading file '{file_path}': {e}")

    def perform_ks_test(self, data, a, b, loc, title):
        """
        Perform Kolmogorov-Smirnov test for the GIG distribution.

        Parameters:
        - data (array-like): Data points
        - a (float): Fitted parameter 'a' of GIG
        - b (float): Fitted parameter 'b' of GIG
        - loc (float): Fitted location parameter of GIG
        - title (str): Description of the data

        Returns:
        - None (prints the results)
        """
        if self.config.SILENT:
            return

        if len(data) == 0:
            print(f"\nData for '{title}' is empty. Cannot perform KS test.")
            return

        try:
            # Perform KS test using the theoretical CDF with fitted parameters
            D, p_value = kstest(data, lambda x: geninvgauss.cdf(x, a, b, loc=loc, scale=1))

            print(f"\nKolmogorov-Smirnov test for '{title}':")
            print(f"D-statistic: {D:.4f}")
            print(f"P-value: {p_value:.4f}")
            if p_value > 0.05:
                print("Cannot reject the null hypothesis. The distribution fits the data well.\n")
            else:
                print("Reject the null hypothesis. The distribution may not fit the data well.\n")
        except Exception as e:
            print(f"Error during KS test for '{title}': {e}")

# --- Plotter Class ---
class Plotter:
    def __init__(self, config):
        self.config = config

    def plot_histograms(self, data_manager, gig_spawn, gig_roadcross, gig_real, config):
        if self.config.SILENT:
            return

        plt.figure(figsize=(18, 12))  # Increased size for more subplots

        # Subplot 2x2 grid
        # Subplot for Spawn Time Intervals (Simulation)
        plt.subplot(2, 2, 1)
        if len(data_manager.spawn_time_intervals) > 0:
            try:
                fitted_params = gig_spawn.fit(data_manager.spawn_time_intervals)
                if fitted_params:  # Check if fitting was successful
                    gig_spawn.plot_fit(
                        data=data_manager.spawn_time_intervals,
                        fitted_params=fitted_params,
                        theoretical_params=(config.GIG_TIME_PARAMETERS['lambda_'], config.GIG_TIME_PARAMETERS['beta']),
                        title='Spawn Time Intervals Histogram (Simulation)',
                        xlabel='Time Interval (s)',
                        ylabel='Density',
                        ax=plt.gca(),
                        xlim=(0, 35)
                    )
                    # Unpack fitted_params
                    a, b, loc = fitted_params
                    data_manager.perform_ks_test(data_manager.spawn_time_intervals, a, b, loc, 'Spawn Time Intervals (Simulation)')
                else:
                    print("Spawn Time Intervals fitting was not successful.")
            except RuntimeError as e:
                print(str(e))
        else:
            print("\nSimulation - Spawn Time Intervals are empty.")

        # Subplot for Roadcross Time Intervals (Simulation)
        plt.subplot(2, 2, 2)
        if len(data_manager.roadcross_time_intervals) > 0:
            try:
                fitted_params = gig_roadcross.fit(data_manager.roadcross_time_intervals)
                if fitted_params:
                    gig_roadcross.plot_fit(
                        data=data_manager.roadcross_time_intervals,
                        fitted_params=fitted_params,
                        theoretical_params=None,  # No theoretical distribution provided
                        title='Roadcross Time Intervals Histogram (Simulation)',
                        xlabel='Time Interval (s)',
                        ylabel='Density',
                        ax=plt.gca(),
                        xlim=(0, 35)
                    )
                    # Unpack fitted_params
                    a, b, loc = fitted_params
                    data_manager.perform_ks_test(data_manager.roadcross_time_intervals, a, b, loc, 'Roadcross Time Intervals (Simulation)')
                else:
                    print("Roadcross Time Intervals fitting was not successful.")
            except RuntimeError as e:
                print(str(e))
        else:
            print("\nSimulation - Roadcross Time Intervals are empty.")

        # Subplot for Real Data (Real Data)
        plt.subplot(2, 2, 3)
        if len(data_manager.real_data) > 0:
            try:
                fitted_params = gig_real.fit(data_manager.real_data)
                if fitted_params:
                    gig_real.plot_fit(
                        data=data_manager.real_data,
                        fitted_params=fitted_params,
                        theoretical_params=None,  # Assuming no theoretical distribution for real data
                        title='Distances Histogram (Real Data)',
                        xlabel='Distance (m)',
                        ylabel='Density',
                        ax=plt.gca(),
                        xlim=(0, 35)
                    )
                    # Unpack fitted_params
                    a, b, loc = fitted_params
                    data_manager.perform_ks_test(data_manager.real_data, a, b, loc, 'Distances (Real Data)')
                else:
                    print("Real Data fitting was not successful.")
            except RuntimeError as e:
                print(str(e))
        else:
            print("\nReal Data is not available for fitting.")

        # Subplot for Initial Speeds (Simulation)
        plt.subplot(2, 2, 4)
        if len(data_manager.initial_speeds) > 0:
            try:
                # Histogram of initial speeds
                ax = plt.gca()
                ax.hist(data_manager.initial_speeds, bins=30, density=True, alpha=0.6, color='c', edgecolor='black', label='Initial Speeds')

                # Fitted Log-Normal PDF
                mu_log = config.LOGNORM_PARAMETERS["mu_log"]
                sigma_log = config.LOGNORM_PARAMETERS["sigma_log"]
                x = np.linspace(0, max(data_manager.initial_speeds), 1000)
                pdf_fitted = lognorm.pdf(x, s=sigma_log, scale=np.exp(mu_log))
                ax.plot(x, pdf_fitted, 'm-', lw=2, label='Fitted Log-Normal PDF')

                ax.set_title('Initial Speeds Histogram (Simulation)')
                ax.set_xlabel('Speed (m/s)')
                ax.set_ylabel('Density')
                ax.set_xlim((0, 35))  # Nastaveno na 0-35
                ax.legend()

                # Perform KS test
                # In this case, we compare the data with the theoretical log-normal distribution
                D, p_value = kstest(data_manager.initial_speeds, lambda x: lognorm.cdf(x, s=sigma_log, scale=np.exp(mu_log)))
                print(f"\nKolmogorov-Smirnov test for 'Initial Speeds (Simulation)':")
                print(f"D-statistic: {D:.4f}")
                print(f"P-value: {p_value:.4f}")
                if p_value > 0.05:
                    print("Cannot reject the null hypothesis. The distribution fits the data well.\n")
                else:
                    print("Reject the null hypothesis. The distribution may not fit the data well.\n")

            except Exception as e:
                print(f"Error during plotting initial speeds: {e}")
        else:
            print("\nSimulation - Initial Speeds are empty.")

        plt.tight_layout()
        plt.show()

# --- Road Class ---
class Road:
    def __init__(self, road_type, config):
        self.road_type = road_type
        self.config = config
        self.cars = deque()  # Using deque for efficient removals
        self.gig_distribution = GIGDistribution(lambda_=config.GIG_TIME_PARAMETERS['lambda_'],
                                                beta=config.GIG_TIME_PARAMETERS['beta'])
        try:
            self.next_spawn_time_interval = self.gig_distribution.sample(num_samples=1)[0]
        except ValueError as e:
            if not self.config.SILENT:
                print(f"Error sampling initial spawn time interval: {e}")
            self.next_spawn_time_interval = 10.0  # Default time interval
        self.time_since_last_spawn = 0.0
        self.cars_spawned = 0
        self.cross_times = deque(maxlen=2)  # Stores the last two cross times

    def spawn_car(self, data_manager):
        if self.cars_spawned >= self.config.MAX_CARS:
            if not self.config.SILENT:
                print("Maximum number of cars spawned. No more cars will be spawned.")
            return

        # Sample initial speed from log-normal distribution
        initial_speed = lognorm.rvs(s=self.config.LOGNORM_PARAMETERS["sigma_log"], scale=np.exp(self.config.LOGNORM_PARAMETERS["mu_log"]))
        initial_speed = max(initial_speed, 0.0)

        spawn_time_interval = self.next_spawn_time_interval
        data_manager.spawn_time_intervals.append(spawn_time_interval)
        data_manager.initial_speeds.append(initial_speed)

        if not self.config.SILENT:
            print(f"Spawned car {self.cars_spawned + 1}/{self.config.MAX_CARS} with speed {initial_speed * 3.6:.2f} km/h and spawn interval {spawn_time_interval:.2f} seconds.")

        # Create and add the new car at the start of the road
        new_car = Car(position=0.0, road=self, config=self.config, initial_speed=initial_speed)
        self.cars.append(new_car)
        self.cars_spawned += 1

        self.time_since_last_spawn = 0.0
        try:
            self.next_spawn_time_interval = self.gig_distribution.sample(num_samples=1)[0]
        except ValueError as e:
            if not self.config.SILENT:
                print(f"Error sampling next spawn time interval: {e}. Setting default interval of 5.0 seconds.")
            self.next_spawn_time_interval = 5.0  # Nastaveno na 5 sekund pro rychlejší spawnování

    def update_cars(self, dt_sim, data_manager, current_time):
        if not self.cars:
            if not self.config.SILENT:
                print("No cars on the road.")
            # Even if no cars on the road, continue simulation to spawn more cars
            self.time_since_last_spawn += dt_sim
            if self.cars_spawned < self.config.MAX_CARS and self.time_since_last_spawn >= self.next_spawn_time_interval:
                self.spawn_car(data_manager)
            return False, None  # No collision or end condition

        # Sort cars from front to back
        self.cars = deque(sorted(self.cars, key=lambda car: car.position, reverse=True))

        # Update car positions
        for i, car in enumerate(self.cars):
            lead_car = self.cars[i - 1] if i > 0 else None
            car.update(dt_sim, lead_car)

            # Record crossing Roadcross
            if not car.roadcross_recorded and car.position >= self.config.ROADCROSS_POSITION:
                car.roadcross_recorded = True
                self.cross_times.append(current_time)

                if len(self.cross_times) == 2:
                    time_interval = self.cross_times[1] - self.cross_times[0]
                    data_manager.roadcross_time_intervals.append(time_interval)

        # Remove cars that have left the screen
        while self.cars and self.cars[0].has_left_screen:
            removed_car = self.cars.popleft()

        # Increment time since last spawn
        self.time_since_last_spawn += dt_sim
        # Check if it's time to spawn a new car
        if self.cars_spawned < self.config.MAX_CARS:
            if self.time_since_last_spawn >= self.next_spawn_time_interval:
                self.spawn_car(data_manager)

        # Check if all cars have passed Roadcross
        all_passed = all(car.position > self.config.ROADCROSS_POSITION for car in self.cars)
        if all_passed and self.cars_spawned >= self.config.MAX_CARS:
            if not self.config.SILENT:
                print("All cars have passed Roadcross.")
            return True, "natural_end"  # End of simulation without collision

        return False, None  # No collision and simulation continues

    def draw(self, window, font):
        # Draw the main road
        y = self.config.HEIGHT // 2
        pygame.draw.line(window, self.config.COLORS['BLACK'], (0, y), (self.config.WIDTH, y), 5)

        # Draw road markings every 50 meters
        for pos in range(0, int(self.config.ROAD_LENGTH) + 1, 50):
            x = int(pos * self.config.SCALE)
            pygame.draw.line(window, self.config.COLORS['GRAY'], (x, y - 20), (x, y + 20), 2)
            label = font.render(f"{pos} m", True, self.config.COLORS['BLACK'])
            window.blit(label, (x, y - 35))

        # Draw Roadcross
        roadcross_x = int(self.config.ROADCROSS_X_PX)
        roadcross_y = y
        pygame.draw.circle(window, self.config.COLORS['RED'], (roadcross_x, roadcross_y), 5)
        roadcross_font = pygame.font.SysFont(None, 24)
        roadcross_label = roadcross_font.render("Roadcross", True, self.config.COLORS['RED'])
        window.blit(roadcross_label, (roadcross_x - 60, roadcross_y - 20))

        # Draw all cars (only if visualization is enabled)
        if not self.config.VISUALIZE:
            for car in self.cars:
                car.draw(window, font)

# --- Simulation class ---
class Simulation:
    def __init__(self, silent=False, config=None):
        pygame.init()
        pygame.font.init()

        if config is None:
            self.config = Config()
        else:
            self.config = config
        self.config.SILENT = silent  # Nastavení silent režimu podle parametru
        self.config.VISUALIZE = silent

        self.data_manager = DataManager(config=self.config)
        self.plotter = Plotter(config=self.config)
        self.gig_spawn = GIGDistribution(**self.config.GIG_TIME_PARAMETERS)
        self.gig_roadcross = GIGDistribution(**self.config.GIG_TIME_PARAMETERS)
        self.gig_real = GIGDistribution(**self.config.GIG_TIME_PARAMETERS)
        self.road = Road(road_type='main', config=self.config)

        self.window = pygame.display.set_mode((self.config.WIDTH, self.config.HEIGHT))
        pygame.display.set_caption("Traffic Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)
        self.speed_multiplier = 1
        self.running = True
        self.collision_occurred = False
        self.collision_timer = 0
        self.collision_duration = 3  # seconds
        self.current_simulation_time = 0.0  # Initialize simulation time

        # Button settings (only relevant if VISUALIZE=True)
        self.button_rect = pygame.Rect(10, 10, 150, 50)
        self.button_color = self.config.COLORS['GRAY']
        self.button_text = "Speed: 1x"

        # Debug: Print initialized parameters
        print(f"Simulation initialized with parameters: max_acc={self.config.MAX_ACCELERATION}, "
              f"max_dec={self.config.MAX_DECELERATION}, react_time={self.config.REACT_TIME}, "
              f"min_gap={self.config.MIN_GAP}, delta={self.config.DELTA}, "
              f"desired_speed={self.config.DESIRED_SPEED}")

    def run(self):
        try:
            self.road.spawn_car(self.data_manager)

            # Fixed simulation step for consistency
            dt = 0.016  # Set to 16 ms (60 FPS) for stable realistic step

            while self.running:
                if not self.config.VISUALIZE:
                    # Set framerate to respect `speed_multiplier` in visualization
                    self.clock.tick(60 * self.speed_multiplier)
                else:
                    # If `VISUALIZE=False`, allow maximum framerate
                    self.clock.tick()

                # Process events (speed change in visualization)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if self.button_rect.collidepoint(event.pos):
                            self.change_speed()

                # Increment simulation time
                dt_sim = dt * self.speed_multiplier
                self.current_simulation_time += dt_sim

                # Update cars based on simulation delta time `dt_sim`
                if not self.collision_occurred:
                    ended, reason = self.road.update_cars(dt_sim, self.data_manager, self.current_simulation_time)
                    if ended:
                        if reason == "collision":
                            self.collision_occurred = True
                            if not self.config.SILENT:
                                print("Collision occurred! Simulation will end shortly.")
                        elif reason == "natural_end":
                            if not self.config.SILENT:
                                print("Simulation completed without any collisions.")
                            self.running = False
                else:
                    self.collision_timer += dt_sim
                    if self.collision_timer >= self.collision_duration:
                        self.running = False

                # Draw window if visualization is enabled
                if not self.config.VISUALIZE:
                    self.window.fill(self.config.COLORS['WHITE'])
                    self.road.draw(self.window, self.font)
                    self.draw_button()

                    if self.collision_occurred:
                        crash_font = pygame.font.SysFont(None, 50)
                        crash_text = crash_font.render("CRASH!", True, self.config.COLORS['RED'])
                        crash_rect = crash_text.get_rect(center=(self.config.WIDTH // 2, self.config.HEIGHT // 2 - 40))
                        self.window.blit(crash_text, crash_rect)

                    pygame.display.flip()
            pygame.quit()

            # After simulation ends, process and plot results if not in silent mode
            if not self.collision_occurred and not self.config.SILENT:
                self.data_manager.load_real_data("data/real_data.xlsx")
                self.plotter.plot_histograms(
                    data_manager=self.data_manager,
                    gig_spawn=self.gig_spawn,
                    gig_roadcross=self.gig_roadcross,
                    gig_real=self.gig_real,
                    config=self.config
                )
                self.display_parameters()

        except Exception as e:
            if not self.config.SILENT:
                print(f"An error occurred during the simulation: {e}")
            pygame.quit()
            sys.exit(1)

    def change_speed(self):
        if self.speed_multiplier < 64:
            self.speed_multiplier *= 2
        else:
            self.speed_multiplier = 1
        self.button_text = f"Speed: {self.speed_multiplier}x"
        if not self.config.SILENT:
            print(f"Speed multiplier changed to {self.speed_multiplier}x")

    def draw_button(self):
        pygame.draw.rect(self.window, self.button_color, self.button_rect)
        button_text_surface = self.font.render(self.button_text, True, self.config.COLORS['BLACK'])
        text_rect = button_text_surface.get_rect(center=self.button_rect.center)
        self.window.blit(button_text_surface, text_rect)

    def display_parameters(self):
        if self.config.SILENT:
            return

        print("\n--- GIG Distribution Fitting Results ---")
        print("\nTheoretical Parameters (GIG_TIME_PARAMETERS):")
        print(f"Lambda (λ): {self.config.GIG_TIME_PARAMETERS['lambda_']:.4f}")
        print(f"Beta (β): {self.config.GIG_TIME_PARAMETERS['beta']:.4f}")

        # Fitted Parameters for Spawn Time Intervals
        if hasattr(self.gig_spawn, 'fitted_params') and self.gig_spawn.fitted_params is not None:
            a, b, loc = self.gig_spawn.fitted_params
            print("\nSimulation - Spawn Time Intervals (Fitted):")
            print(f"Lambda (λ): {a:.4f}")
            print(f"b: {b:.4f}")
            print(f"Loc: {loc:.4f}")
        else:
            print("\nSimulation - Spawn Time Intervals fitting was not successful.")

        # Fitted Parameters for Roadcross Time Intervals
        if hasattr(self.gig_roadcross, 'fitted_params') and self.gig_roadcross.fitted_params is not None:
            a, b, loc = self.gig_roadcross.fitted_params
            print("\nSimulation - Roadcross Time Intervals (Fitted):")
            print(f"Lambda (λ): {a:.4f}")
            print(f"b: {b:.4f}")
            print(f"Loc: {loc:.4f}")
        else:
            print("\nSimulation - Roadcross Time Intervals fitting was not successful.")

        # Fitted Parameters for Real Data
        if hasattr(self.gig_real, 'fitted_params') and len(
                self.data_manager.real_data) > 0 and self.gig_real.fitted_params is not None:
            a, b, loc = self.gig_real.fitted_params
            print("\nReal Data - Distances (Fitted):")
            print(f"Lambda (λ): {a:.4f}")
            print(f"b: {b:.4f}")
            print(f"Loc: {loc:.4f}")
        else:
            print("\nReal Data fitting was not successful or data is unavailable.")

# --- Main Execution ---
if __name__ == "__main__":
    # Example usage:
    # To run simulation normally:
    # simulation = Simulation(silent=False)
    # simulation.run()

    # To run simulation silently:
    # simulation = Simulation(silent=True)
    # simulation.run()

    # For optimization purposes, you would run it silently to collect roadcross_time_intervals
    simulation = Simulation(silent=False)
    simulation.run()
