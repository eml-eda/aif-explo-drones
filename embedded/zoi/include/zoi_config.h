/* zoi_config.h: compile-time sizes of the ZOI C port.
 *
 * Everything is statically allocated from these sizes; there is no malloc.
 * The values match the Webots setup (webots/controllers/crazyflie_AIF/crazyflie_AIF.py):
 * a 5.5 x 6.5 m room on 0.5 m cells, 8 candidate yaws, a 32 x 32 (rho, theta) grid.
 */
#ifndef ZOI_CONFIG_H
#define ZOI_CONFIG_H

/* Map: rows along y, columns along x. Cell index = cy * ZOI_COLS + cx.
 * 13 x 11 is the Webots room; a larger room can be built with -DZOI_ROWS / -DZOI_COLS
 * (the grid stays centred on the origin, see zoi_config_default). */
#ifndef ZOI_ROWS
#define ZOI_ROWS 13
#endif
#ifndef ZOI_COLS
#define ZOI_COLS 11
#endif
#define ZOI_CELLS (ZOI_ROWS * ZOI_COLS)

/* Planner candidates: every cell centre x 8 yaws (k * pi / 4). Candidate index =
 * cell_index * ZOI_N_YAW + yaw_index, the order of the Python planner. */
#define ZOI_N_YAW 8
#define ZOI_N_CAND (ZOI_CELLS * ZOI_N_YAW)

/* (rho, theta) grid: ZOI_GRID_N midpoints per side over (0, 1). Only feasible
 * points are kept. Row theta_j has an open feasible rho-interval of length
 * 1 - theta_j, which holds at most N - j midpoints, so for any pi there are at
 * most N (N + 1) / 2 points (528 for N = 32; 512 at pi = 0.5, 513 at pi = 0.9). */
#define ZOI_GRID_N 32
#define ZOI_MAX_GRID_POINTS (ZOI_GRID_N * (ZOI_GRID_N + 1) / 2)

/* Per-frame limits. */
#define ZOI_MAX_BOXES 32      /* detector boxes per frame */
#define ZOI_MAX_FOOT 36       /* cells one camera footprint can touch (floor plane, any yaw) */
#define ZOI_MAX_PATTERN 24    /* cells seen from one candidate (floor or table plane) */
#define ZOI_MAX_PATTERN_ENTRIES 112 /* all yaws together (88 in the Webots setup) */
#define ZOI_MAX_PROBE_CELLS 32 /* cells hit by the probes of one desk box */

/* States of a cell: surface * 3 + content; surface 0 table, 1 floor;
 * content 0 empty, 1 bottle, 2 can. Observations: desk * 3 + target. */
#define ZOI_NS 6
#define ZOI_NO 6

/* Detector classes (crazyflie_AIF.py). */
#define ZOI_CLASS_BOTTLE 0
#define ZOI_CLASS_CAN 1
#define ZOI_CLASS_DESK 2

#endif /* ZOI_CONFIG_H */
