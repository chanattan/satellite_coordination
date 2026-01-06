import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple
import ESOPInstance

def plot_schedule(instance: ESOPInstance, user_plans, title="Satellite schedule", show_exclusives=True):
    """
    Affiche un planning avec les fenêtres exclusives en arrière-plan.
    
    user_plans : dict {user_id -> {sat_id -> [(obs, t_start), ...]}}
    show_exclusives : afficher/masquer les fenêtres exclusives
    """
    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    
    # Couleurs pour les utilisateurs
    user_ids = [u.uid for u in instance.users]
    color_map = {
        uid: base_colors[i % len(base_colors)]
        for i, uid in enumerate(sorted(user_ids))
    }
    
    # Couleurs pour les fenêtres exclusives : version claire de la couleur utilisateur
    def lighten_color(color_hex):
        """Convertit une couleur hex en version plus claire (alpha visuel)."""
        import matplotlib.colors as mcolors
        rgb = mcolors.hex2color(color_hex)
        # Mélange avec blanc (0.6 ratio) pour éclaircir
        light_rgb = tuple(c * 0.4 + 0.6 for c in rgb)
        return mcolors.to_hex(light_rgb)
    
    exclusive_colors = {
        uid: lighten_color(color_map[uid]) 
        for uid in user_ids if uid != "u0"
    }
    
    fig, ax = plt.subplots(figsize=(12, 4 + len(instance.satellites) * 0.5))
    
    # Axe Y : une ligne par satellite
    sat_ids = [s.sid for s in instance.satellites]
    sat_to_y = {sid: i for i, sid in enumerate(sat_ids)}
    
    horizon = instance.horizon
    ax.set_xlim(0, horizon)
    ax.set_ylim(-1, len(sat_ids))
    ax.set_yticks(range(len(sat_ids)))
    ax.set_yticklabels(sat_ids)
    ax.set_xlabel("Time")
    ax.set_title(title)
    
    # Lignes de base pour chaque satellite
    for sid, y in sat_to_y.items():
        ax.axhline(y, color="lightgray", linewidth=0.8)
    
    # 🚀 Affichage des fenêtres exclusives (arrière-plan)
    if show_exclusives:
        for user in instance.users:
            if user.uid != "u0":  # Seulement les utilisateurs exclusifs
                for window in user.exclusive_windows:
                    y = sat_to_y[window.satellite]
                    # Rectangle hachuré pour les exclusives
                    rect = Rectangle(
                        (window.t_start, y - 0.45),
                        window.t_end - window.t_start,
                        0.9,
                        facecolor=exclusive_colors.get(user.uid, "#ffff99"),
                        edgecolor="darkred",
                        alpha=0.3,
                        hatch="////",
                        linewidth=1.5
                    )
                    ax.add_patch(rect)
                    
                    # Label de l'utilisateur exclusif
                    ax.text(
                        window.t_start + (window.t_end - window.t_start)/2,
                        y - 0.75,
                        user.uid,
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="darkred",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
                    )
    
    # Observations planifiées (premier plan)
    for uid, plans in user_plans.items():
        for sid, plan in plans.items():
            if sid in sat_to_y:  # Sécurité
                y = sat_to_y[sid]
                for obs, t_start in plan:
                    rect = Rectangle(
                        (t_start, y - 0.3),
                        obs.duration,
                        0.6,
                        facecolor=color_map[uid],
                        edgecolor="black",
                        alpha=0.8,
                        linewidth=1
                    )
                    ax.add_patch(rect)
                    
                    # Label de tâche (raccourci)
                    ax.text(
                        t_start + obs.duration / 2,
                        y,
                        obs.task_id[-3:],
                        ha="center", va="center",
                        fontsize=7,
                        color="white",
                        fontweight="bold"
                    )
    
    # Légende corrigée
    legend_handles = []
    legend_labels = []
    
    # Utilisateurs
    for uid in sorted(user_ids):
        legend_handles.append(Rectangle((0, 0), 1, 1, color=color_map[uid]))
        legend_labels.append(f"User {uid}")
    
    # Fenêtres exclusives (seulement celles qui existent)
    if show_exclusives:
        for user in instance.users:
            if user.uid != "u0" and user.exclusive_windows:
                color = exclusive_colors.get(user.uid, "#ffff99")
                handle = Rectangle((0, 0), 1, 1, facecolor=color, 
                                 edgecolor="darkred", alpha=0.3, hatch="////")
                legend_handles.append(handle)
                legend_labels.append(f"Exclusive {user.uid}")
    
    ax.legend(legend_handles, legend_labels, loc="upper right", bbox_to_anchor=(1.15, 1))
    
    # Grille temporelle
    for t in range(0, horizon+1, 50):
        ax.axvline(t, color="gray", alpha=0.3, linestyle=":")
    
    plt.tight_layout()
    plt.show()
