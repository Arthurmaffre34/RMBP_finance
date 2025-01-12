import matplotlib.pyplot as plt
from heston_sim import heston_simulation_dict

#dictionnaire d'exemple
params_dict = {
    "AAPL": {"S0": 150, "v0": 0.04, "mu": 0.05, "kappa": 2, "theta": 0.04, "sigma": 0.2, "rho": -0.7},
    "MSFT": {"S0": 200, "v0": 0.03, "mu": 0.06, "kappa": 1.5, "theta": 0.03, "sigma": 0.15, "rho": -0.5},
}


#appel de la fonction
simulations = heston_simulation_dict(params_dict, n_paths=1000, T=1, dt=1/252)

#Accéder aux résultats pour un ticker
aapl_simulations = simulations["AAPL"]
print("Temps :")
print(aapl_simulations["t"])
print("Trajectoires (première simulation) :")
print(aapl_simulations["trajectories"][0])



#création des sous-graphiques
fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 2 lignes, 2 colonnes

#graphique des prix pour AAPL
for trajectory in simulations["AAPL"]["trajectories"]:
    axes[0, 0].plot(simulations["AAPL"]["t"], trajectory["S"], label=f"Path {trajectory['path_id']}")
axes[0, 0].set_title("Trajectoires de prix - AAPL")
axes[0, 0].set_xlabel("Temps (années)")
axes[0, 0].set_ylabel("Prix de l'actif")
axes[0, 0].legend()

#Graphique de la variance pour AAPL
for trajectory in simulations["AAPL"]["trajectories"]:
    axes[0, 1].plot(simulations["AAPL"]["t"], trajectory["V"], label=f"Path {trajectory['path_id']}")
axes[0, 1].set_title("Trajectoires de variance - AAPL")
axes[0, 1].set_xlabel("Temps (années)")
axes[0, 1].set_ylabel("Variance")
axes[0, 1].legend()

#Graphique des prix pour MSFT
for trajectory in simulations["MSFT"]["trajectories"]:
    axes[1, 0].plot(simulations["MSFT"]["t"], trajectory["S"], label=f"Path {trajectory['path_id']}")
axes[1, 0].set_title("Trajectoires de prix - MSFT")
axes[1, 0].set_xlabel("Temps (années)")
axes[1, 0].set_ylabel("Prix de l'actif")
axes[1, 0].legend()

#graphique de la variance pour MSFT
for trajectory in simulations["MSFT"]["trajectories"]:
    axes[1, 1].plot(simulations["MSFT"]["t"], trajectory["V"], label=f"Path {trajectory['path_id']}")
axes[1, 1].set_title("Trajectoires de variance - MSFT")
axes[1, 1].set_xlabel("Temps (années)")
axes[1, 1].set_ylabel("Variance")
axes[1, 1].legend()

#ajustement des espaces entre les sousgraphiques
plt.tight_layout()

#affichage
plt.show()