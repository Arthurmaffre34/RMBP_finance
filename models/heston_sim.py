import numpy as np

def heston_simulation_dict(params_dict, n_paths, T, dt):
    """
    Simule le modèle de Heston pour plusieurs actifs en utilisant un dictionnaire de paramètres.

    Args:
        params_dict: Dictionnaire contenant les paramètres pour chaque ticker.
            Format attendu:
            {
                "ticker1": {"S0": ..., "v0": ..., "mu": ..., "kappa": ..., "theta": ..., "sigma": ..., "rho": ...},
                "ticker2": {"S0": ..., "v0": ..., "mu": ..., "kappa": ..., "theta": ..., "sigma": ..., "rho": ...},
                ...
            }
        n_paths: Nombre de trajectoires simulées pour chaque actif.
        T: Horizon temporel (en années).
        dt: Pas de temps.

    Returns:
        simulations: Dictionnaire structuré contenant les trajectoires de prix et de variance pour chaque ticker.
    """
    simulations = {}
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps, dtype=np.float32)

    for ticker, params in params_dict.item():
        #Récupération des paramètres pour le ticker
        S0 = params["S0"]
        v0 = params["v0"]
        mu = params["mu"]
        kappa = params["kappa"]
        theta = params["theta"]
        sigma = params["sigma"]
        rho = params["rho"]

        