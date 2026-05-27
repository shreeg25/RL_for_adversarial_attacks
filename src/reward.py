# src/reward.py
"""
R_t = w1 * IoU_survival  -  w2 * id_switches  -  w3 * action_cost
"""
from src.transformations import ACTION_COST


def compute_reward(
    prev_id_set: set,
    current_ids: list,
    action: int,
    w1: float = 1.0,
    w2: float = 5.0,
    w3: float = 0.5,
) -> tuple[float, int]:
    """
    Returns:
        reward (float), id_switch_count (int)
    """
    current_set = set(current_ids)

    # Survival: reward proportional to fraction of tracks retained
    if prev_id_set:
        retained = len(prev_id_set & current_set)
        iou_reward = retained / len(prev_id_set)
    else:
        iou_reward = 1.0 if current_set else 0.0

    # ID switches: tracks in current that were NOT in previous frame
    new_ids = current_set - prev_id_set
    id_switches = len(new_ids)

    cost = ACTION_COST[action]
    reward = w1 * iou_reward - w2 * id_switches - w3 * cost
    return float(reward), id_switches