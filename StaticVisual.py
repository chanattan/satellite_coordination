import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple
import ESOPInstance


def plot_schedule(instance: ESOPInstance, user_plans, title = "Satellite schedule"):
    """
    Affiche un planning (ou plusieurs) sur un graphique 2D.
    
    user_plans : dict {user_id -> {sat_id -> [(obs, t_start), ...]}}

    Chaque observation planifiée est un rectangle coloré selon l'utilisateur.
    """
    base_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    user_ids = [u.uid for u in instance.users]
    color_map = {
        uid: base_colors[i % len(base_colors)]
        for i, uid in enumerate(sorted(user_ids))
    }

    fig, ax = plt.subplots(figsize=(10, 4 + len(instance.satellites) * 0.4))

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

    # Oobservations planifiées
    for uid, plans in user_plans.items():
        for sid, plan in plans.items():
            y = sat_to_y[sid]
            for obs, t_start in plan:
                rect = Rectangle(
                    (t_start, y - 0.3),
                    obs.duration,
                    0.6,
                    facecolor=color_map[uid],
                    edgecolor="black",
                    alpha=0.8
                )
                ax.add_patch(rect)
                # label optionnel
                ax.text(
                    t_start + obs.duration / 2,
                    y,
                    obs.task_id,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white"
                )

    legend_handles = [
        Rectangle((0, 0), 1, 1, color=color_map[uid])
        for uid in sorted(user_ids)
    ]
    ax.legend(
        legend_handles,
        sorted(user_ids),
        title="Users",
        loc="upper right"
    )

    plt.tight_layout()
    plt.show()
