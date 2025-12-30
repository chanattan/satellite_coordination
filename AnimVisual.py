from matplotlib import animation
from ESOPInstance import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple

def animate_user_schedule_on_satellite(
    instance: ESOPInstance,
    user_id: str,
    satellite_id: str,
    interval_ms: int = 500,
    save_gif: str= None
):
    """
    Anime la construction du plan glouton d'un utilisateur sur un satellite.
    
    - interval_ms : durée entre frames (en ms).
    - save_gif : chemin pour sauvegarder en GIF (optionnel).
    """

    # On récupère toutes les observations candidates (non planifiées au départ)
    candidate_obs = [
        o for o in instance.observations
        if o.owner == user_id and o.satellite == satellite_id
    ]

    sat = next(s for s in instance.satellites if s.sid == satellite_id)
    tau = sat.transition_time
    sat_start = sat.t_start
    sat_end = sat.t_end
    kappa = sat.capacity

    # même tri que dans l'algorithme glouton
    candidate_obs.sort(key=lambda o: (-o.reward, o.t_start))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, instance.horizon)
    ax.set_ylim(-1, 1)
    ax.set_yticks([0])
    ax.set_yticklabels([satellite_id])
    ax.set_xlabel("Time")
    ax.set_title(f"Building schedule for {user_id} on {satellite_id}")

    ax.axhline(0, color="lightgray", linewidth=0.8)

    # plan courant (liste de (obs, t_start))
    plan: List[Tuple[Observation, int]] = []

    # utilitaire d'insertion (identique à la version statique)
    def try_insert(obs: Observation) -> int:
        if len(plan) >= kappa:
            return None

        if not plan:
            t0 = max(obs.t_start, sat_start)
            if t0 + obs.duration <= min(obs.t_end, sat_end):
                return t0
            return None

        sorted_plan = sorted(plan, key=lambda p: p[1])

        # avant la première
        first_obs, first_t = sorted_plan[0]
        earliest_start = max(obs.t_start, sat_start)
        latest_end = min(obs.t_end, first_t - tau)
        if earliest_start + obs.duration <= latest_end:
            return earliest_start

        # entre obs successives
        for (o_prev, t_prev), (o_next, t_next) in zip(sorted_plan, sorted_plan[1:]):
            window_start = max(
                obs.t_start,
                t_prev + o_prev.duration + tau
            )
            window_end = min(
                obs.t_end,
                t_next - tau
            )
            if window_start + obs.duration <= window_end:
                return window_start

        # après la dernière
        last_obs, last_t = sorted_plan[-1]
        window_start = max(
            obs.t_start,
            last_t + last_obs.duration + tau
        )
        window_end = min(obs.t_end, sat_end)
        if window_start + obs.duration <= window_end:
            return window_start

        return None

    rects: List[Rectangle] = []

    def init():
        # aucune observation au début
        return rects

    def update(frame_idx):
        # À chaque frame, on tente d'ajouter une observation de plus
        if frame_idx < len(candidate_obs):
            obs = candidate_obs[frame_idx]
            t_insert = try_insert(obs)
            if t_insert is not None:
                plan.append((obs, t_insert))
                rect = Rectangle(
                    (t_insert, -0.3),
                    obs.duration,
                    0.6,
                    facecolor="#1f77b4",
                    edgecolor="black",
                    alpha=0.8
                )
                ax.add_patch(rect)
                rects.append(rect)
                ax.text(
                    t_insert + obs.duration / 2,
                    0,
                    obs.task_id,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white"
                )
        return rects

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(candidate_obs) + 1,
        interval=interval_ms,
        blit=False,
        repeat=False
    )

    plt.close(fig)  # évite le double affichage en notebook

    if save_gif is not None:
        # nécessite imagemagick installé pour writer="imagemagick"
        anim.save(save_gif, writer="imagemagick")

    return anim
