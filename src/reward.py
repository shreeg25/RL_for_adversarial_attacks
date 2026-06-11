# src/reward.py
"""
ATTACK-AWARE COUNTERFACTUAL REWARD
Calculates the causal impact of the agent's defense.
"""

def _iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0

def _match(dets, confs, gt_boxes, iou_thresh=0.5):
    if not gt_boxes:
        return 0.0, len(dets)
    if not dets:
        return 0.0, 0

    pairs = []
    for gi, g in enumerate(gt_boxes):
        for di, d in enumerate(dets):
            iou = _iou_xywh(g, d)
            if iou >= iou_thresh:
                pairs.append((iou, gi, di))
    pairs.sort(reverse=True)

    matched_gt, matched_det = set(), set()
    covered_conf = 0.0
    for iou, gi, di in pairs:
        if gi in matched_gt or di in matched_det:
            continue
        matched_gt.add(gi)
        matched_det.add(di)
        covered_conf += float(confs[di])

    n_fp = len(dets) - len(matched_det)
    return covered_conf, n_fp

def compute_reward(
    det_action, conf_action,      
    det_t0,     conf_t0,          
    gt_boxes,                     
    action: int,
    prev_id_set: set,
    current_ids: list,
    frame_idx: int = 0,
    action_cost_table: dict | None = None,
    w_rec:  float = 5.0,
    w_fp:   float = 2.0,
    w_lost: float = 0.5,
    w_cost: float = 1.0,
):
    if action_cost_table is None:
        action_cost_table = {0: 0.0, 1: 0.05, 2: 0.03, 3: 0.04}

    if frame_idx < 5:
        return 0.0, {"recovery": 0.0, "fp_delta": 0.0, "lost": 0, "phase": "init"}

    cov_a, fp_a = _match(det_action, conf_action, gt_boxes)
    cov_0, fp_0 = _match(det_t0,     conf_t0,     gt_boxes)

    recovery = cov_a - cov_0                 
    fp_delta = max(0, fp_a - fp_0)           

    current_set = set(current_ids)
    lost = len(prev_id_set - current_set) if prev_id_set else 0

    cost = action_cost_table.get(action, 0.0)

    reward = (w_rec * recovery
              - w_fp  * fp_delta
              - w_lost * lost
              - w_cost * cost)

    info = {
        "recovery": round(recovery, 4),
        "fp_delta": int(fp_delta),
        "lost":     int(lost),
        "cov_a":    round(cov_a, 3),
        "cov_0":    round(cov_0, 3),
        "phase":    "active",
    }
    return float(reward), info