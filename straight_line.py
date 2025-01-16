import pygame
import numpy as np
from scipy.stats import lognorm, kstest, geninvgauss
import matplotlib.pyplot as plt
import pandas as pd  # Pro načítání Excel souborů
import sys  # Pro ukončení programu
from collections import deque

# --- PARAMETRY PRO SIMULACI ---
class Config:
    VIZUALIZE = True

    # Parametry simulace
    MAX_CARS = 1000  # Nastaveno na 5000 pro plné testování

    # Parametry GIG rozdělení pro intervaly spawn času
    GIG_TIME_PARAMETERS = {
        'lambda_': 2.71,    # Tvarový parametr (lambda_) - Optimalizováno
        'beta': 1.03       # Měřítkový parametr (beta) - Optimalizováno
    }

    # Parametry LOGNORM rozdělení pro rychlosti spawn - průměr=50km/h, std=1,5m/s
    LOGNORM_PARAMETERS = {
        "mu_log": np.log(13.89) - (0.107**2) / 2,  # Přibližně 2.62425
        "sigma_log": 0.107
    }

    # IDM parametry
    '''
    'min_gap': 1.625,
    'acceleration': 2.0,
    'comfortable_deceleration': 4.0,
    'react_time': 0.3,
    'desired_speed': 19.44,  # ~50 km/h
    'delta': 2,
    '''
    MIN_GAP = 1.625  # Minimální mezera mezi auty
    MAX_ACCELERATION = 2.0  # Maximální zrychlení v m/s²
    MAX_DECELERATION = 4.0  # Komfortní brzdění v m/s²
    REACT_TIME = 0.3  # Reakční doba řidiče (časový předstih)
    DESIRED_SPEED = 19.44
    DELTA = 2

    # Nastavení silnice
    ROAD_LENGTH = 300  # Délka silnice v metrech
    SCALE = 5          # Konverzní faktor (5 pixelů = 1 metr)

    # Pozice silniční křižovatky
    ROADCROSS_X_PX = ROAD_LENGTH * SCALE  # Pozice křižovatky v pixelech
    ROADCROSS_POSITION = ROAD_LENGTH  # Pozice křižovatky v metrech

    # Barvy
    COLORS = {
        'WHITE': (255, 255, 255),
        'BLACK': (0, 0, 0),
        'BLUE': (0, 0, 255),
        'RED': (255, 0, 0),
        'GRAY': (200, 200, 200)
    }

    # Rozměry obrazovky
    WIDTH = 800
    HEIGHT = 600

# --- Inteligentní model řidiče (IDM) ---
class IDM:
    def __init__(self, config):
        self.v0 = config.DESIRED_SPEED  # Žádoucí rychlost (m/s)
        self.a = config.MAX_ACCELERATION  # Maximální zrychlení (m/s²)
        self.b = config.MAX_DECELERATION  # Komfortní brzdění (m/s²)
        self.delta = config.DELTA  # Exponent
        self.s0 = config.MIN_GAP  # Minimální mezera (m)
        self.t0 = config.REACT_TIME

    def compute_acceleration(self, v, delta_v, s):
        if s <= 0:
            s = 0.1  # Zabraň dělení nulou
        s_star = self.s0 + max(0, v * self.t0 + (v * delta_v) / (2 * np.sqrt(self.a * self.b)))
        acceleration = self.a * (1 - (v / self.v0) ** self.delta - (s_star / s) ** 2)
        # Omez brzdění
        acceleration = max(acceleration, -self.b)
        return acceleration

# --- Třída Auto ---
class Car:
    WIDTH = 20
    HEIGHT = 10

    def __init__(self, position, road, config, initial_speed):
        self.position = position  # Pozice na silnici (m)
        self.road = road          # Reference na objekt Road
        self.length = 5.0         # Délka auta pro výpočty mezery v metrech
        self.v = initial_speed    # Rychlost (m/s)
        self.a = 0.0              # Zrychlení (m/s²)
        self.idm = IDM(config)
        self.color = config.COLORS['BLUE']
        self.has_recorded_interval = False  # Značka, zda byl interval zaznamenán

        # Další atributy
        self.should_remove = False          # Indikuje, zda by auto mělo být odstraněno
        self.has_left_screen = False        # Indikuje, zda auto opustilo obrazovku
        self.roadcross_recorded = False     # Indikuje, zda byl interval křižovatky zaznamenán

    def update(self, dt, lead_car=None):
        if lead_car:
            s = lead_car.position - self.position - lead_car.length
            delta_v = self.v - lead_car.v
            self.a = self.idm.compute_acceleration(self.v, delta_v, s)
        else:
            self.a = self.idm.a * (1 - (self.v / self.idm.v0) ** self.idm.delta)

        # Omez brzdění
        self.a = max(self.a, -self.idm.b)
        # Aktualizuj rychlost a pozici
        self.v += self.a * dt
        self.v = max(self.v, 0)  # Aplikuj limit rychlosti
        self.position += self.v * dt + 0.5 * self.a * dt ** 2

        # Zkontroluj, zda auto opustilo obrazovku
        if self.position > self.road.config.ROADCROSS_POSITION + self.road.config.ROAD_LENGTH:
            self.has_left_screen = True

    def draw(self, window, font):
        x = int(self.position * self.road.config.SCALE)
        y = self.road.config.HEIGHT // 2
        rect = pygame.Rect(x - self.WIDTH // 2, y - self.HEIGHT // 2, self.WIDTH, self.HEIGHT)
        pygame.draw.rect(window, self.color, rect)

        # Zobraz rychlost auta
        speed_kmh = self.v * 3.6  # Převod na km/h
        speed_text = font.render(f"{speed_kmh:.1f} km/h", True, self.road.config.COLORS['BLACK'])
        text_rect = speed_text.get_rect(center=(rect.centerx, rect.centery - 15))
        window.blit(speed_text, text_rect)

    def get_position(self):
        x = int(self.position * self.road.config.SCALE)
        y = self.road.config.HEIGHT // 2
        return x, y

# --- Třída GIG Rozdělení ---
class GIGDistribution:
    def __init__(self, lambda_, beta):
        """
        Inicializuje GIGDistribution s parametry lambda_ a beta.

        Parametry:
        - lambda_ (float): Tvarový parametr (λ)
        - beta (float): Měřítkový parametr (β)
        """
        self.lambda_ = lambda_
        self.beta = beta
        self.geninvgauss_dist = geninvgauss(self.lambda_, self.beta)
        self.fitted_params = None  # Místo pro nahrání odhadnutých parametrů

    def pdf(self, x):
        """
        Hustotní funkce pravděpodobnosti pro GIG rozdělení pomocí SciPy's geninvgauss.
        """
        return self.geninvgauss_dist.pdf(x)

    def sample(self, num_samples=1):
        """
        Generuje vzorky z GIG(lambda_, beta) rozdělení pomocí SciPy's geninvgauss.

        Parametry:
        - num_samples (int): Počet vzorků k vygenerování

        Návratové hodnoty:
        - samples (list): Seznam vygenerovaných vzorků
        """
        try:
            samples = self.geninvgauss_dist.rvs(size=num_samples)
            # Zajisti, aby vzorky byly kladné a respektovaly minimální interval
            samples = np.clip(samples, 0.5, None)  # Nastav minimální interval na 0.5 sekundy
            # Volitelné: Odstraň extrémně velké intervaly, pokud je to nutné
            # samples = np.clip(samples, 0.5, 20.0)
            # print(samples)  # Odstraň nebo zakomentuj pro snížení hluku v konzoli
            if len(samples) < num_samples:
                # Pokud některé vzorky jsou neplatné, generuj chybějící
                additional_samples = self.sample(num_samples - len(samples))
                samples = np.concatenate((samples, additional_samples))
            return samples.tolist()
        except ValueError as e:
            if hasattr(self, 'road') and not self.road.config.SILENT:
                print(f"Chyba při vzorkování dalšího intervalu spawn času: {e}. Nastavuji výchozí interval 10.0 sekund.")
            return [10.0] * num_samples  # Výchozí interval v případě selhání vzorkování

    def fit(self, data):
        """
        Přizpůsobí GIG rozdělení datům pomocí MLE s pevnou škálou=1.

        Parametry:
        - data (array-like): Data

        Návratové hodnoty:
        - fitted_params (tuple): Přizpůsobené parametry (lambda_, b, loc), nebo None pokud přizpůsobení selže
        """
        # Filtruj data, aby všechny hodnoty byly kladné
        data = np.array(data)
        data = data[data > 0]
        if len(data) == 0:
            if hasattr(self, 'road') and not self.road.config.SILENT:
                print("Žádná kladná data k dispozici pro přizpůsobení.")
            return None

        try:
            # Přizpůsobení pomocí SciPy's geninvgauss s pevnou škálou=1 a nefixovaným loc
            lambda_, b, loc, scale = geninvgauss.fit(data, floc=0, fscale=1)

            # Ulož přizpůsobené parametry a vrať je
            self.lambda_ = lambda_
            self.beta = 1  # Škála je pevně nastavena na 1
            self.geninvgauss_dist = geninvgauss(lambda_, b, loc=0, scale=1)
            self.fitted_params = (lambda_, b, loc)

            return (lambda_, b, loc)
        except Exception as e:
            if hasattr(self, 'road') and not self.road.config.SILENT:
                print(f"Chyba během přizpůsobení: {e}")
            return None  # Explicitně vrať None v případě selhání

    def plot_fit(self, data, fitted_params, theoretical_params=None, title='', xlabel='', ylabel='', ax=None, xlim=(0, 35)):
        """
        Vykreslí histogram dat a přizpůsobenou GIG PDF. Volitelně vykreslí teoretickou GIG PDF.

        Parametry:
        - data (array-like): Data
        - fitted_params (tuple): Přizpůsobené parametry (lambda_, b, loc)
        - theoretical_params (tuple): Volitelný n-tice teoretických parametrů (lambda_, beta)
        - title (str): Titulek grafu
        - xlabel (str): Popisek osy X
        - ylabel (str): Popisek osy Y
        - ax (matplotlib.axes.Axes): Matplotlib osa pro vykreslení
        - xlim (tuple): Limity pro osu X

        Návratové hodnoty:
        - None
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Histogram dat
        ax.hist(data, bins=50, density=True, alpha=0.6, color='g', edgecolor='black', label='Data')

        # Přizpůsobená GIG PDF
        x = np.linspace(0, max(xlim), 1000)  # Začni x od 0 až po xlim[1]
        lambda_fit, b_fit, loc_fit = fitted_params
        pdf_fitted = geninvgauss.pdf(x, lambda_fit, b_fit, loc=loc_fit, scale=1)
        ax.plot(x, pdf_fitted, 'r-', lw=2, label=f'Přizpůsobená GIG\nλ={lambda_fit:.2f}, b={b_fit:.2f}, loc={loc_fit:.2f}')

        # Teoretická GIG PDF (pokud je poskytnuta)
        if theoretical_params is not None:
            lambda_theo, beta_theo = theoretical_params
            # Poznámka: Teoretické b může být buď pevné nebo použít b_fit, v závislosti na interpretaci
            # Zde používáme b_fit s teoretickým lambda_ a beta
            pdf_theoretical = geninvgauss.pdf(x, lambda_theo, beta_theo, loc=loc_fit, scale=1)
            ax.plot(x, pdf_theoretical, 'b--', lw=2, label=f'Teoretická GIG\nλ={lambda_theo:.2f}, β={beta_theo:.2f}')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)  # Nastav osu X od 0 do 35
        ax.legend()

# --- Třída Data Manager ---
class DataManager:
    def __init__(self, config):
        self.config = config
        self.roadcross_time_intervals = []  # Přejmenováno pro jasnost
        self.spawn_time_intervals = []
        self.real_data = []
        self.initial_speeds = []  # Seznam pro počáteční rychlosti

    def load_real_data(self, file_path):
        if self.config.SILENT:
            return
        try:
            # Načti data z Excel souboru
            df = pd.read_excel(file_path, header=None)
            if df.shape[1] < 2:
                # Pokud je pouze jeden sloupec, předpokládej, že jsou to jednotlivé intervaly
                data = df.iloc[:, 0].values
            else:
                # Načti pouze unikátní intervaly z prvního sloupce
                data = df.iloc[:, 0].values

            if not self.config.SILENT:
                print(f"\nNačteno {len(data)} unikátních intervalů ze souboru '{file_path}'.")
            self.real_data = data
        except FileNotFoundError:
            if not self.config.SILENT:
                print(f"\nSoubor '{file_path}' nebyl nalezen.")
        except Exception as e:
            if not self.config.SILENT:
                print(f"\nChyba při načítání souboru '{file_path}': {e}")

    def perform_ks_test(self, data, a, b, loc, title):
        """
        Provede Kolmogorov-Smirnov test pro GIG rozdělení.

        Parametry:
        - data (array-like): Data
        - a (float): Přizpůsobený parametr 'a' GIG
        - b (float): Přizpůsobený parametr 'b' GIG
        - loc (float): Přizpůsobený parametr loc GIG
        - title (str): Popis dat

        Návratové hodnoty:
        - None (vypíše výsledky)
        """
        if self.config.SILENT:
            return

        if len(data) == 0:
            print(f"\nData pro '{title}' jsou prázdná. Nelze provést KS test.")
            return

        try:
            # Proveď KS test pomocí teoretické CDF s přizpůsobenými parametry
            D, p_value = kstest(data, lambda x: geninvgauss.cdf(x, a, b, loc=loc, scale=1))

            print(f"\nKolmogorov-Smirnov test pro '{title}':")
            print(f"D-statistika: {D:.4f}")
            print(f"P-hodnota: {p_value:.4f}")
            if p_value > 0.05:
                print("Nelze zamítnout nulovou hypotézu. Rozdělení dobře sedí na data.\n")
            else:
                print("Zamítnutí nulové hypotézy. Rozdělení nemusí dobře sedět na data.\n")
        except Exception as e:
            print(f"Chyba během KS testu pro '{title}': {e}")

# --- Třída Plotter ---
class Plotter:
    def __init__(self, config):
        self.config = config

    def plot_histograms(self, data_manager, gig_spawn, gig_roadcross, gig_real, config):
        if self.config.SILENT:
            return

        plt.figure(figsize=(18, 12))  # Zvýšená velikost pro více subplotů

        # Subplot 2x2 mřížka
        # Subplot pro Spawn Time Intervals (Simulace)
        plt.subplot(2, 2, 1)
        if len(data_manager.spawn_time_intervals) > 0:
            try:
                fitted_params = gig_spawn.fit(data_manager.spawn_time_intervals)
                if fitted_params:  # Zkontroluj, zda přizpůsobení bylo úspěšné
                    gig_spawn.plot_fit(
                        data=data_manager.spawn_time_intervals,
                        fitted_params=fitted_params,
                        theoretical_params=(config.GIG_TIME_PARAMETERS['lambda_'], config.GIG_TIME_PARAMETERS['beta']),
                        title='Histogram Intervalů Spawn Časů (Simulace)',
                        xlabel='Interval Času (s)',
                        ylabel='Hustota',
                        ax=plt.gca(),
                        xlim=(0, 35)
                    )
                    # Rozbal přizpůsobené parametry
                    a, b, loc = fitted_params
                    data_manager.perform_ks_test(data_manager.spawn_time_intervals, a, b, loc, 'Spawn Time Intervals (Simulace)')
                else:
                    print("Přizpůsobení Intervalů Spawn Časů nebylo úspěšné.")
            except RuntimeError as e:
                print(str(e))
        else:
            print("\nSimulace - Intervaly Spawn Časů jsou prázdné.")

        # Subplot pro Roadcross Time Intervals (Simulace)
        plt.subplot(2, 2, 2)
        if len(data_manager.roadcross_time_intervals) > 0:
            try:
                fitted_params = gig_roadcross.fit(data_manager.roadcross_time_intervals)
                if fitted_params:
                    gig_roadcross.plot_fit(
                        data=data_manager.roadcross_time_intervals,
                        fitted_params=fitted_params,
                        theoretical_params=None,  # Žádné teoretické rozdělení poskytnuto
                        title='Histogram Intervalů Roadcross Časů (Simulace)',
                        xlabel='Interval Času (s)',
                        ylabel='Hustota',
                        ax=plt.gca(),
                        xlim=(0, 35)
                    )
                    # Rozbal přizpůsobené parametry
                    a, b, loc = fitted_params
                    data_manager.perform_ks_test(data_manager.roadcross_time_intervals, a, b, loc, 'Roadcross Time Intervals (Simulace)')
                else:
                    print("Přizpůsobení Intervalů Roadcross Časů nebylo úspěšné.")
            except RuntimeError as e:
                print(str(e))
        else:
            print("\nSimulace - Intervaly Roadcross Časů jsou prázdné.")

        # Subplot pro Real Data (Reálná data)
        plt.subplot(2, 2, 3)
        if len(data_manager.real_data) > 0:
            try:
                fitted_params = gig_real.fit(data_manager.real_data)
                if fitted_params:
                    gig_real.plot_fit(
                        data=data_manager.real_data,
                        fitted_params=fitted_params,
                        theoretical_params=None,  # Předpokládá se žádné teoretické rozdělení pro reálná data
                        title='Histogram Intervalů (Reálná Data)',
                        xlabel='Interval (s)',
                        ylabel='Hustota',
                        ax=plt.gca(),
                        xlim=(0, 35)
                    )
                    # Rozbal přizpůsobené parametry
                    a, b, loc = fitted_params
                    data_manager.perform_ks_test(data_manager.real_data, a, b, loc, 'intervals (Reálná Data)')
                else:
                    print("Přizpůsobení Reálných Dat nebylo úspěšné.")
            except RuntimeError as e:
                print(str(e))
        else:
            print("\nReálná Data nejsou dostupná pro přizpůsobení.")

        # Subplot pro Počáteční Rychlosti (Simulace)
        plt.subplot(2, 2, 4)
        if len(data_manager.initial_speeds) > 0:
            try:
                # Histogram počátečních rychlostí
                ax = plt.gca()
                ax.hist(data_manager.initial_speeds, bins=30, density=True, alpha=0.6, color='c', edgecolor='black', label='Počáteční Rychlosti')

                # Přizpůsobená Log-Normal PDF
                mu_log = config.LOGNORM_PARAMETERS["mu_log"]
                sigma_log = config.LOGNORM_PARAMETERS["sigma_log"]
                x = np.linspace(0, max(data_manager.initial_speeds), 1000)
                pdf_fitted = lognorm.pdf(x, s=sigma_log, scale=np.exp(mu_log))
                ax.plot(x, pdf_fitted, 'm-', lw=2, label='Přizpůsobená Log-Normální PDF')

                ax.set_title('Histogram Počátečních Rychlostí (Simulace)')
                ax.set_xlabel('Rychlost (m/s)')
                ax.set_ylabel('Hustota')
                ax.set_xlim((0, 35))  # Nastaveno na 0-35
                ax.legend()

                # Proveď KS test
                # V tomto případě porovnáme data s teoretickým log-normálním rozdělením
                D, p_value = kstest(data_manager.initial_speeds, lambda x: lognorm.cdf(x, s=sigma_log, scale=np.exp(mu_log)))
                print(f"\nKolmogorov-Smirnov test pro 'Počáteční Rychlosti (Simulace)':")
                print(f"D-statistika: {D:.4f}")
                print(f"P-hodnota: {p_value:.4f}")
                if p_value > 0.05:
                    print("Nelze zamítnout nulovou hypotézu. Rozdělení dobře sedí na data.\n")
                else:
                    print("Zamítnutí nulové hypotézy. Rozdělení nemusí dobře sedět na data.\n")

            except Exception as e:
                print(f"Chyba při vykreslování počátečních rychlostí: {e}")
        else:
            print("\nSimulace - Počáteční Rychlosti jsou prázdné.")

        plt.tight_layout()
        plt.show()

# --- Třída Road ---
class Road:
    def __init__(self, road_type, config):
        self.road_type = road_type
        self.config = config
        self.cars = deque()  # Použití deque pro efektivní odstraňování
        self.gig_distribution = GIGDistribution(lambda_=config.GIG_TIME_PARAMETERS['lambda_'],
                                                beta=config.GIG_TIME_PARAMETERS['beta'])
        try:
            self.next_spawn_time_interval = self.gig_distribution.sample(num_samples=1)[0]
        except ValueError as e:
            if not self.config.SILENT:
                print(f"Chyba při vzorkování počátečního intervalu spawn času: {e}")
            self.next_spawn_time_interval = 10.0  # Výchozí časový interval
        self.time_since_last_spawn = 0.0
        self.cars_spawned = 0
        self.cross_times = deque(maxlen=2)  # Ukládá poslední dva časy průjezdu křižovatkou

    def spawn_car(self, data_manager):
        if self.cars_spawned >= self.config.MAX_CARS:
            if not self.config.SILENT:
                print("Maximální počet aut bylo spawnováno. Další auta nebudou spawnována.")
            return

        # Vzorkuj počáteční rychlost z log-normálního rozdělení
        initial_speed = lognorm.rvs(s=self.config.LOGNORM_PARAMETERS["sigma_log"], scale=np.exp(self.config.LOGNORM_PARAMETERS["mu_log"]))
        initial_speed = max(initial_speed, 0.0)

        spawn_time_interval = self.next_spawn_time_interval
        data_manager.spawn_time_intervals.append(spawn_time_interval)
        data_manager.initial_speeds.append(initial_speed)

        if not self.config.SILENT:
            print(f"Spawnováno auto {self.cars_spawned + 1}/{self.config.MAX_CARS} s rychlostí {initial_speed * 3.6:.2f} km/h a spawn intervalem {spawn_time_interval:.2f} sekund.")

        # Vytvoř a přidej nové auto na začátek silnice
        new_car = Car(position=0.0, road=self, config=self.config, initial_speed=initial_speed)
        self.cars.append(new_car)
        self.cars_spawned += 1

        self.time_since_last_spawn = 0.0
        try:
            self.next_spawn_time_interval = self.gig_distribution.sample(num_samples=1)[0]
        except ValueError as e:
            if not self.config.SILENT:
                print(f"Chyba při vzorkování dalšího intervalu spawn času: {e}. Nastavuji výchozí interval 5.0 sekund.")
            self.next_spawn_time_interval = 5.0  # Nastaveno na 5 sekund pro rychlejší spawnování

    def update_cars(self, dt_sim, data_manager, current_time):
        if not self.cars:
            if not self.config.SILENT:
                print("Na silnici nejsou žádná auta.")
            # I když na silnici nejsou žádná auta, pokračuj v simulaci a spawnuj další auta
            self.time_since_last_spawn += dt_sim
            if self.cars_spawned < self.config.MAX_CARS and self.time_since_last_spawn >= self.next_spawn_time_interval:
                self.spawn_car(data_manager)
            return False, None  # Žádná kolize nebo konec podmínky

        # Seřaď auta od čela k zadní části
        self.cars = deque(sorted(self.cars, key=lambda car: car.position, reverse=True))

        # Aktualizuj pozice aut
        for i, car in enumerate(self.cars):
            lead_car = self.cars[i - 1] if i > 0 else None
            car.update(dt_sim, lead_car)

            # Zaznamenej průjezd křižovatkou
            if not car.roadcross_recorded and car.position >= self.config.ROADCROSS_POSITION:
                car.roadcross_recorded = True
                self.cross_times.append(current_time)

                if len(self.cross_times) == 2:
                    time_interval = self.cross_times[1] - self.cross_times[0]
                    data_manager.roadcross_time_intervals.append(time_interval)

        # Odstraň auta, která opustila obrazovku
        while self.cars and self.cars[0].has_left_screen:
            removed_car = self.cars.popleft()

        # Inkrementuj čas od posledního spawnování
        self.time_since_last_spawn += dt_sim
        # Zkontroluj, zda je čas spawnovat nové auto
        if self.cars_spawned < self.config.MAX_CARS:
            if self.time_since_last_spawn >= self.next_spawn_time_interval:
                self.spawn_car(data_manager)

        # Zkontroluj, zda všechna auta prošla křižovatkou
        all_passed = all(car.position > self.config.ROADCROSS_POSITION for car in self.cars)
        if all_passed and self.cars_spawned >= self.config.MAX_CARS:
            if not self.config.SILENT:
                print("Všechna auta prošla křižovatkou.")
            return True, "natural_end"  # Konec simulace bez kolize

        return False, None  # Žádná kolize a simulace pokračuje

    def draw(self, window, font):
        # Nakresli hlavní silnici
        y = self.config.HEIGHT // 2
        pygame.draw.line(window, self.config.COLORS['BLACK'], (0, y), (self.config.WIDTH, y), 5)

        # Nakresli silniční značky každých 50 metrů
        for pos in range(0, int(self.config.ROAD_LENGTH) + 1, 50):
            x = int(pos * self.config.SCALE)
            pygame.draw.line(window, self.config.COLORS['GRAY'], (x, y - 20), (x, y + 20), 2)
            label = font.render(f"{pos} m", True, self.config.COLORS['BLACK'])
            window.blit(label, (x, y - 35))

        # Nakresli Roadcross
        roadcross_x = int(self.config.ROADCROSS_X_PX)
        roadcross_y = y
        pygame.draw.circle(window, self.config.COLORS['RED'], (roadcross_x, roadcross_y), 5)
        roadcross_font = pygame.font.SysFont(None, 24)
        roadcross_label = roadcross_font.render("Roadcross", True, self.config.COLORS['RED'])
        window.blit(roadcross_label, (roadcross_x - 60, roadcross_y - 20))

        # Nakresli všechna auta (pouze pokud je vizualizace povolena)
        if not self.config.VISUALIZE:
            for car in self.cars:
                car.draw(window, font)

# --- Třída Simulation ---
class Simulation:
    def __init__(self, silent=False, config=None):
        pygame.init()
        pygame.font.init()

        if config is None:
            self.config = Config()
        else:
            self.config = config
        self.config.SILENT = silent  # Nastavení režimu silent podle parametru
        self.config.VISUALIZE = silent

        self.data_manager = DataManager(config=self.config)
        self.plotter = Plotter(config=self.config)
        self.gig_spawn = GIGDistribution(**self.config.GIG_TIME_PARAMETERS)
        self.gig_roadcross = GIGDistribution(**self.config.GIG_TIME_PARAMETERS)
        self.gig_real = GIGDistribution(**self.config.GIG_TIME_PARAMETERS)
        self.road = Road(road_type='main', config=self.config)

        self.window = pygame.display.set_mode((self.config.WIDTH, self.config.HEIGHT))
        pygame.display.set_caption("Simulace Dopravy")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)
        self.speed_multiplier = 1
        self.running = True
        self.collision_occurred = False
        self.collision_timer = 0
        self.collision_duration = 3  # sekundy
        self.current_simulation_time = 0.0  # Inicializuj simulační čas

        # Nastavení tlačítka (relevantní pouze pokud VIZUALIZE=True)
        self.button_rect = pygame.Rect(10, 10, 150, 50)
        self.button_color = self.config.COLORS['GRAY']
        self.button_text = "Rychlost: 1x"

        # Debug: Vytiskni inicializované parametry
        print(f"Simulace inicializována s parametry: max_acc={self.config.MAX_ACCELERATION}, "
              f"max_dec={self.config.MAX_DECELERATION}, react_time={self.config.REACT_TIME}, "
              f"min_gap={self.config.MIN_GAP}, delta={self.config.DELTA}, "
              f"desired_speed={self.config.DESIRED_SPEED}")

    def run(self):
        try:
            self.road.spawn_car(self.data_manager)

            # Pevný simulační krok pro konzistenci
            dt = 0.016  # Nastaveno na 16 ms (60 FPS) pro stabilní realistický krok

            while self.running:
                if not self.config.VISUALIZE:
                    # Nastav snímkovou frekvenci podle `speed_multiplier` ve vizualizaci
                    self.clock.tick(60 * self.speed_multiplier)
                else:
                    # Pokud `VIZUALIZE=False`, povol maximální snímkovou frekvenci
                    self.clock.tick()

                # Zpracuj události (změna rychlosti ve vizualizaci)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if self.button_rect.collidepoint(event.pos):
                            self.change_speed()

                # Inkrementuj simulační čas
                dt_sim = dt * self.speed_multiplier
                self.current_simulation_time += dt_sim

                # Aktualizuj auta na základě simulačního delta času `dt_sim`
                if not self.collision_occurred:
                    ended, reason = self.road.update_cars(dt_sim, self.data_manager, self.current_simulation_time)
                    if ended:
                        if reason == "collision":
                            self.collision_occurred = True
                            if not self.config.SILENT:
                                print("Kolidování nastalo! Simulace se brzy ukončí.")
                        elif reason == "natural_end":
                            if not self.config.SILENT:
                                print("Simulace dokončena bez kolizí.")
                            self.running = False
                else:
                    self.collision_timer += dt_sim
                    if self.collision_timer >= self.collision_duration:
                        self.running = False

                # Nakresli okno, pokud je vizualizace povolena
                if not self.config.VISUALIZE:
                    self.window.fill(self.config.COLORS['WHITE'])
                    self.road.draw(self.window, self.font)
                    self.draw_button()

                    if self.collision_occurred:
                        crash_font = pygame.font.SysFont(None, 50)
                        crash_text = crash_font.render("NÁRAZ!", True, self.config.COLORS['RED'])
                        crash_rect = crash_text.get_rect(center=(self.config.WIDTH // 2, self.config.HEIGHT // 2 - 40))
                        self.window.blit(crash_text, crash_rect)

                    pygame.display.flip()
            pygame.quit()

            # Po ukončení simulace, zpracuj a vykresli výsledky, pokud není v silent režimu
            if not self.collision_occurred and not self.config.SILENT:
                self.data_manager.load_real_data("real_data.xlsx")
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
                print(f"Došlo k chybě během simulace: {e}")
            pygame.quit()
            sys.exit(1)

    def change_speed(self):
        if self.speed_multiplier < 64:
            self.speed_multiplier *= 2
        else:
            self.speed_multiplier = 1
        self.button_text = f"Rychlost: {self.speed_multiplier}x"
        if not self.config.SILENT:
            print(f"Rychlostní násobitel změněn na {self.speed_multiplier}x")

    def draw_button(self):
        pygame.draw.rect(self.window, self.button_color, self.button_rect)
        button_text_surface = self.font.render(self.button_text, True, self.config.COLORS['BLACK'])
        text_rect = button_text_surface.get_rect(center=self.button_rect.center)
        self.window.blit(button_text_surface, text_rect)

    def display_parameters(self):
        if self.config.SILENT:
            return

        print("\n--- Výsledky Přizpůsobení GIG Rozdělení ---")
        print("\nTeoretické Parametry (GIG_TIME_PARAMETERS):")
        print(f"Lambda (λ): {self.config.GIG_TIME_PARAMETERS['lambda_']:.4f}")
        print(f"Beta (β): {self.config.GIG_TIME_PARAMETERS['beta']:.4f}")

        # Přizpůsobené Parametry pro Intervaly Spawn Časů
        if hasattr(self.gig_spawn, 'fitted_params') and self.gig_spawn.fitted_params is not None:
            a, b, loc = self.gig_spawn.fitted_params
            print("\nSimulace - Intervaly Spawn Časů (Přizpůsobené):")
            print(f"Lambda (λ): {a:.4f}")
            print(f"b: {b:.4f}")
            print(f"Loc: {loc:.4f}")
        else:
            print("\nSimulace - Přizpůsobení Intervalů Spawn Časů nebylo úspěšné.")

        # Přizpůsobené Parametry pro Intervaly Roadcross Časů
        if hasattr(self.gig_roadcross, 'fitted_params') and self.gig_roadcross.fitted_params is not None:
            a, b, loc = self.gig_roadcross.fitted_params
            print("\nSimulace - Intervaly Roadcross Časů (Přizpůsobené):")
            print(f"Lambda (λ): {a:.4f}")
            print(f"b: {b:.4f}")
            print(f"Loc: {loc:.4f}")
        else:
            print("\nSimulace - Přizpůsobení Intervalů Roadcross Časů nebylo úspěšné.")

        # Přizpůsobené Parametry pro Reálná Data
        if hasattr(self.gig_real, 'fitted_params') and len(
                self.data_manager.real_data) > 0 and self.gig_real.fitted_params is not None:
            a, b, loc = self.gig_real.fitted_params
            print("\nReálná Data - Intervaly (Přizpůsobené):")
            print(f"Lambda (λ): {a:.4f}")
            print(f"b: {b:.4f}")
            print(f"Loc: {loc:.4f}")
        else:
            print("\nPřizpůsobení Reálných Dat nebylo úspěšné nebo data nejsou dostupná.")

# --- Hlavní Spuštění ---
if __name__ == "__main__":
    # Příklad použití:
    # Pro spuštění simulace normálně:
    # simulation = Simulation(silent=False)
    # simulation.run()

    # Pro spuštění simulace v tichém režimu:
    # simulation = Simulation(silent=True)
    # simulation.run()

    # Pro účely optimalizace byste ji spustili v tichém režimu pro sběr roadcross_time_intervals
    simulation = Simulation(silent=False)
    simulation.run()
