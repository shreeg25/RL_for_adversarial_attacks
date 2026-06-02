# Modified segment within scripts/evaluate_attack_success.py
def run_sequence(seq_path: str, model=None, label: str = "") -> dict:
    cfg = yaml.safe_load(open("config.yaml"))
    env = MOT17Env(
        seq_path,
        w1=cfg["reward"]["w1"],
        w2=cfg["reward"]["w2"],
        w3=cfg["reward"]["w3"],
    )
    obs, _ = env.reset()
    
    if hasattr(env, "_extractor") and env._extractor is not None:
        env._extractor.reset()

    try:
        target = find_optimal_target(seq_path, min_frames=50, min_visibility=0.5)
        tid    = target["target_id"]
        s_f    = target["start_frame"]
        e_f    = target["end_frame"]
    except Exception:
        tid, s_f, e_f = None, 1, env._n_frames

    total_target_frames = 0
    lost_frames         = 0
    total_id_sw         = 0
    
    # Fixed accumulators to prevent MOTA corruption
    total_gt, total_tp, total_fp, total_fn = 0, 0, 0, 0
    from scripts.evaluate_accuracy import load_ground_truth, match_detections
    gt = load_ground_truth(seq_path)

    frame_no = 1
    done     = False

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=False)
            action = int(action)
        else:
            action = 0

        obs, reward, done, _, info = env.step(action)
        total_id_sw += info["id_switches"]

        gt_boxes = gt.get(frame_no, [])
        confirmed = [t for t in env._extractor.tracker.tracker.tracks if t.is_confirmed()]
        pred_boxes = [t.to_tlwh().tolist() for t in confirmed]
        
        matched_ious, fp, fn = match_detections(gt_boxes, pred_boxes)
        
        total_gt += len(gt_boxes)
        total_fp += fp
        total_fn += fn
        total_tp += len(matched_ious)

        if tid is not None and s_f <= frame_no <= e_f:
            total_target_frames += 1
            confirmed_ids = [t.track_id for t in confirmed]
            if tid not in confirmed_ids:
                lost_frames += 1

        frame_no += 1

    asr  = lost_frames / max(total_target_frames, 1)
    mota = max(0.0, 1.0 - (total_fn + total_fp + total_id_sw) / max(total_gt, 1))

    result = {
        "label":          label,
        "asr":            round(asr * 100, 2),
        "id_switches":    total_id_sw,
        "mota":           round(mota * 100, 1),
        "target_frames":  total_target_frames,
        "lost_frames":    lost_frames,
    }
    print(f"  [{label:30s}]  ASR={asr*100:.1f}%  MOTA={mota*100:.1f}%  ID-sw={total_id_sw}")
    return result