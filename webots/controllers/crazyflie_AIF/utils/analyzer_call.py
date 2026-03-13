from snapshot_analyzer import analyze_snapshot_file

print(analyze_snapshot_file(
    "webots/controllers/crazyflie_AIF/logs/log_run_aif_saccade_20260313_151540/snapshots/snapshot_step004332_stuck40_20260313_151713.npz",
    save_json=True,
))