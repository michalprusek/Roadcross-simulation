import numpy as np
from scipy.stats import geninvgauss
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def objective(params, desired_stats):
    """
    Objektivní funkce pro optimalizaci parametrů GIG distribuce tak,
    aby byla minimalizována absolutní odchylka od požadovaných hodnot.

    Parameters:
    - params (list or array): Parametry [lambda_, beta] pro GIG distribuci.
    - desired_stats (list or array): Požadované [mean, variance].

    Returns:
    - float: Součet absolutních rozdílů mezi aktuálními a požadovanými statistiky.
    """
    lambda_, beta = params
    # Zajistit, aby parametry byly pozitivní
    if lambda_ <= 0 or beta <= 0:
        return np.inf

    try:
        # Inicializace GIG distribuce
        gig_dist = geninvgauss(lambda_, beta)
        # Výpočet aktuálních statistických hodnot (střední hodnota a rozptyl)
        mean, var = gig_dist.stats(moments='mv')
        # Vektor aktuálních statistických hodnot
        current_stats = np.array([mean, var])
        # Vektor požadovaných statistických hodnot
        target_stats = np.array(desired_stats)
        # Spočítat absolutní rozdíly mezi aktuálními a požadovanými hodnotami
        abs_diff = np.abs(current_stats - target_stats)
        # Součet absolutních rozdílů jako cílová hodnota pro minimalizaci
        return np.sum(abs_diff)
    except:
        return np.inf


def optimize_GIG(desired_mean, desired_var, initial_guess):
    """
    Optimalizuje parametry GIG distribuce pro dosažení požadovaných statistických hodnot.

    Parameters:
    - desired_mean (float): Požadovaná střední hodnota.
    - desired_var (float): Požadovaný rozptyl.
    - initial_guess (list): Počáteční odhad parametrů [lambda_, beta].

    Returns:
    - dict: Optimalizované parametry a dosažené statistiky.
    """
    desired_stats = [desired_mean, desired_var]

    # Proveďte optimalizaci pomocí metody Nelder-Mead
    result = minimize(
        objective,
        initial_guess,
        args=(desired_stats,),
        method='Nelder-Mead',
        bounds=((1e-3, None), (1e-3, None)),  # lambda_ a beta musí být kladné
        options={'disp': True, 'maxiter': 1000}
    )

    if result.success:
        optimized_lambda, optimized_beta = result.x
        # Inicializace optimalizované GIG distribuce pro ověření výsledků
        optimized_gig = geninvgauss(optimized_lambda, optimized_beta)
        # Výpočet střední hodnoty a rozptylu s optimalizovanými parametry
        computed_mean, computed_var = optimized_gig.stats(moments='mv')

        return {
            'lambda_': optimized_lambda,
            'beta': optimized_beta,
            'computed_mean': computed_mean,
            'computed_var': computed_var,
            'optimized_gig': optimized_gig
        }
    else:
        raise ValueError("Optimalizace selhala. Zkontrolujte počáteční odhad nebo omezení.")


def plot_distribution(gig_dist, desired_mean, desired_var):
    """
    Vizualizuje PDF optimalizované GIG distribuce.

    Parameters:
    - gig_dist (scipy.stats.rv_continuous_frozen): Optimalizovaná GIG distribuce.
    - desired_mean (float): Požadovaná střední hodnota.
    - desired_var (float): Požadovaný rozptyl.
    """
    x = np.linspace(0.1, desired_mean * 3, 1000)
    pdf = gig_dist.pdf(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, label='Optimized GIG PDF', color='blue')
    plt.title('Optimalizovaná GIG Distribuce')
    plt.xlabel('Spawn Distance (meters)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Požadované statistiky
    desired_mean = 5  # sec
    desired_var = 2  # sec^2

    # Počáteční odhad parametrů [lambda_, beta]
    initial_guess = [4.0, 300.0]

    # Optimalizujte parametry GIG distribuce
    optimal_params = optimize_GIG(desired_mean, desired_var, initial_guess)

    # Vytiskněte optimalizované parametry a dosažené statistiky
    print("\n--- Optimalizované Parametry GIG Distribuce ---")
    print(f"Lambda (λ): {optimal_params['lambda_']:.4f}")
    print(f"Beta (β): {optimal_params['beta']:.4f}")
    print(f"Computed Mean: {optimal_params['computed_mean']:.4f} meters")
    print(f"Computed Variance: {optimal_params['computed_var']:.4f} meters^2")

    # Vizualizujte optimalizovanou distribuci
    plot_distribution(optimal_params['optimized_gig'], desired_mean, desired_var)


if __name__ == "__main__":
    main()
