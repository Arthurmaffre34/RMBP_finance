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

    for ticker, params in params_dict.items():
        #Récupération des paramètres pour le ticker
        S0 = params["S0"]
        v0 = params["v0"]
        mu = params["mu"]
        kappa = params["kappa"]
        theta = params["theta"]
        sigma = params["sigma"]
        rho = params["rho"]

        #initialisation des matrices pour les prix et la variance
        S = np.zeros((n_paths, n_steps), dtype=np.float32)
        V = np.zeros((n_paths, n_steps), dtype=np.float32)

        #conditions initiales
        S[:, 0] = S0
        V[:, 0] = v0

        #simuler les mouvements browniens corrélés
        Z1 = np.random.normal(size=(n_paths, n_steps - 1)).astype(np.float32)
        Z2 = np.random.normal(size=(n_paths, n_steps - 1)).astype(np.float32)
        W1 = Z1 * np.sqrt(dt).astype(np.float32)
        W2 = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)

        #Pré-calcul de certaines constantes
        max_variance_epsilon = 1e-6 #évite les valeurs négatives

        #boucle temporelle pour simuler les trajectoires
        for i in range(1, n_steps):
            #simulation de la variance v_t
            V[:, i] = (
                V[:, i-1]
                + kappa * (theta - V[:, i-1]) * dt
                + sigma * np.sqrt(np.maximum(V[:, i-1], 0)) * W2[:, i-1] 
            )
            V[:, i] = np.maximum(V[:, i], max_variance_epsilon) #empèche les valeurs négatives

            #simulation du prix s_t
            S[:, i] = S[:, i-1] * np.exp(
                (mu - 0.5 * V[:, i-1]) * dt + np.sqrt(V[:, i-1]) * W1[:, i-1]
            )
        
        #construire la structure des trajectoires
        trajectories = []
        for path_id in range(n_paths):
            trajectories.append({
                "path_id": path_id,
                "S": S[path_id, :].tolist(),
                "V": V[path_id, :].tolist()
            })
        
        #Ajouter au dictionnaire principal
        simulations[ticker] = {"trajectories": trajectories, "t": t.tolist()}

    return simulations