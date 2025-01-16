import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys


def load_data(real_file, simulated_file):
    """
    Načte reálná a simulovaná data bez záhlaví a přiřadí názvy sloupců.
    """
    try:
        df_real = pd.read_excel(real_file, header=None)
    except Exception as e:
        print(f"Chyba při načítání reálných dat z {real_file}: {e}")
        sys.exit(1)

    try:
        df_sim = pd.read_csv(simulated_file, header=None)
    except Exception as e:
        print(f"Chyba při načítání simulovaných dat z {simulated_file}: {e}")
        sys.exit(1)

    # Pojmenování sloupců
    df_real.columns = ['Interval', 'Count']
    df_sim.columns = ['Interval', 'Count']

    return df_real, df_sim


def group_by_count(df_real, df_sim):
    """
    Rozdělí oba DataFrame do skupin podle 'Count' a identifikuje společné kategorie.
    """
    real_groups = df_real.groupby('Count')
    sim_groups = df_sim.groupby('Count')

    real_counts = set(df_real['Count'].unique())
    sim_counts = set(df_sim['Count'].unique())
    common_counts = real_counts.intersection(sim_counts)

    print("Společné kategorie (Count) v obou souborech:", sorted(common_counts))
    print()

    return real_groups, sim_groups, sorted(common_counts)


def perform_ks_test(real_values, sim_values):
    """
    Provede Kolmogorov-Smirnovův test mezi dvěma soubory dat.
    """
    ks_stat, p_value = stats.ks_2samp(real_values, sim_values)
    return ks_stat, p_value


def compare_distributions(df_real, df_sim, common_counts):
    """
    Porovná rozdělení 'Interval' mezi reálnými a simulovanými daty pro každou kategorii 'Count'.
    """
    results = []

    for count in common_counts:
        print(f"=== POČET AUT = {count} ===")

        real_intervals = df_real[df_real['Count'] == count]['Interval'].values
        sim_intervals = df_sim[df_sim['Count'] == count]['Interval'].values

        n_real = len(real_intervals)
        n_sim = len(sim_intervals)

        print(f"  Reálné intervaly: {n_real}")
        print(f"  Simulované intervaly: {n_sim}")

        if n_real == 0 or n_sim == 0:
            print("  Nedostatek dat pro porovnání. Přeskakuji.\n")
            continue

        # Provedení KS testu
        ks_stat, p_value = perform_ks_test(real_intervals, sim_intervals)
        print(f"  KS Test: Statistic = {ks_stat:.4f}, p-value = {p_value:.4f}")

        # Uložení výsledků
        results.append({
            'Count': count,
            'Real_Count': n_real,
            'Simulated_Count': n_sim,
            'KS_Statistic': ks_stat,
            'p_value': p_value
        })

        # Volitelná vizualizace pro tuto kategorii
        plot_distributions(real_intervals, sim_intervals, count)

        print()

    return pd.DataFrame(results)


def plot_distributions(real, sim, count):
    """
    Vykreslí histogramy rozdělení 'Interval' pro reálná a simulovaná data.
    """
    plt.figure(figsize=(10, 6))
    bins = np.histogram_bin_edges(np.concatenate((real, sim)), bins='auto')

    plt.hist(real, bins=bins, alpha=0.5, label='Reálná data', color='blue', density=True)
    plt.hist(sim, bins=bins, alpha=0.5, label='Simulovaná data', color='orange', density=True)

    plt.title(f'Distribuce Intervalů pro Počet Aut = {count}')
    plt.xlabel('Interval')
    plt.ylabel('Relativní četnost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Cesty k souborům
    real_file = 'real_data.xlsx'
    simulated_file = 'simulation_data.csv'

    # Načtení dat
    df_real, df_sim = load_data(real_file, simulated_file)

    # Rozdělení do skupin a identifikace společných kategorií
    real_groups, sim_groups, common_counts = group_by_count(df_real, df_sim)

    if not common_counts:
        print("Žádné společné kategorie 'Count' mezi reálnými a simulovanými daty. Konec.")
        sys.exit(0)

    # Porovnání rozdělení pro každou kategorii
    comparison_results = compare_distributions(df_real, df_sim, common_counts)

    # Uložení výsledků do CSV (volitelné)
    comparison_results.to_csv('comparison_results.csv', index=False)
    print("Výsledky porovnání byly uloženy do 'comparison_results.csv'.")


if __name__ == "__main__":
    main()
