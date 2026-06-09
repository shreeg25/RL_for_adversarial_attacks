# src/reward.py
"""
TRACE Reward Function — Complete Redesign

ROOT CAUSE of passive T0 agent:
  OLD: R_t = w1*survival - w2*id_switches - w3*action_cost
       w2=5.0 destroyed any positive signal. Every defense attempt costs ~5.0.
       T0 optimal because it incurs zero penalty.

  OLD BUG: id_switches = len(current_set - prev_id_set)
       This counts NEW pedestrians entering the frame as "switches".
       Agent punished for scene events it cannot control.

NEW STRUCTURE:
  R_t = w1*survival + w4*defense_bonus - w2*lost_tracks - w3*action_cost

  w1 = 2.0  survival reward        (up from 1.0)
  w2 = 1.5  lost-track penalty     (down from 5.0, correct definition)
  w3 = 0.01 action cost            (near-zero — agent must not fear its tools)
  w4 = 3.0  defense success bonus  (NEW — reward successful non-T0 defense)

  lost_tracks = prev_id_set - current_set   (tracks that DISAPPEARED)
  NOT:          current_set - prev_id_set   (new tracks = scene events)

  defense_bonus logic:
    - action != T0 AND survival >= 0.75 AND no lost tracks  →  +w4 (perfect)
    - action != T0 AND survival >= 0.50                     →  +w4*0.4 (partial)
    - action != T0 (exploration incentive, even failed)     →  +0.2 (tiny bonus)

  Initialization guard: first 5 frames return neutral reward.
  Tracker needs 3 frames (n_init=3) to confirm tracks; penalizing
  this phase trained the agent that "all states are bad from the start".
"""

from src.transformations import ACTION_COST


def compute_reward(
    prev_id_set:  set,
    current_ids:  list,
    action:       int,
    frame_idx:    int   = 0,
    w1:           float = 2.0,
    w2:           float = 1.5,
    w3:           float = 0.01,
    w4:           float = 3.0,
) -> tuple[float, int]:
    """
    Returns: (reward float, lost_track_count int)

    Args:
        prev_id_set  : confirmed track IDs from previous frame
        current_ids  : confirmed track IDs from current frame
        action       : integer in {0, 1, 2, 3}
        frame_idx    : current frame index (0-based) — used for init guard
        w1           : survival reward weight
        w2           : lost-track penalty weight
        w3           : action cost weight (should be near-zero)
        w4           : defense success bonus weight
    """
    current_set = set(current_ids)

    # ── Initialization guard ──────────────────────────────────────────
    # DeepSORT requires n_init=3 frames before confirming any track.
    # Penalizing this phase taught the old agent that all actions are bad.
    if frame_idx < 5:
        return 0.5, 0

    # ── Survival reward ───────────────────────────────────────────────
    if prev_id_set:
        retained  = len(prev_id_set & current_set)
        survival  = retained / len(prev_id_set)

        # FIXED DEFINITION: count tracks that DISAPPEARED (agent's fault)
        # NOT new tracks appearing (pedestrians entering frame = scene event)
        lost_tracks = len(prev_id_set - current_set)
    else:
        survival    = 1.0 if current_set else 0.0
        lost_tracks = 0

    # ── Defense bonus ─────────────────────────────────────────────────
    # The agent must learn that using T1/T2/T3 is WORTH IT.
    # Without this, T0 is always optimal in a purely-negative reward space.
    defense_bonus = 0.0
    if action != 0:
        if lost_tracks == 0 and survival >= 0.75:
            # Perfect defense: used a transformation AND maintained all tracks
            defense_bonus = w4
        elif survival >= 0.50:
            # Partial defense: used transformation, tracking mostly survived
            defense_bonus = w4 * 0.4
        else:
            # Exploration incentive: agent tried to defend even if it failed
            # Small positive to keep exploration alive and prevent T0 lock-in
            defense_bonus = 0.2

    # ── Action cost (near-zero) ───────────────────────────────────────
    # w3=0.01 makes this essentially negligible.
    # Old w3=0.5 penalized warping more than a small ID switch.
    action_cost = ACTION_COST[action] * w3

    reward = (w1 * survival
              + defense_bonus
              - w2 * lost_tracks
              - action_cost)

    return float(reward), int(lost_tracks)