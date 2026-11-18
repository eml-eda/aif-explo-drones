/* zoi.h: ZOI on board, float32, no dynamic memory.
 *
 * C port of the Python policy (webots/controllers/crazyflie_AIF/policies/zoi*.py)
 * and of the per-frame geometry the controller runs before it
 * (env/env_manager.py, env/surface_projection.py, vision/perception_projection.py,
 * task/claim_manager.py, utils/navigator.py, policies/planner_time.py).
 *
 * Use:
 *     zoi_config_t cfg;  zoi_config_default(&cfg, ZOI_ENV_HARD, ZOI_DET_MOBILENET);
 *     static zoi_state_t st;  zoi_init(&st, &cfg);
 *     every detector frame:  zoi_perception_frame(&st, &pose, boxes, n_boxes);
 *     every control step:    zoi_control_step(&st, &pose, time_s, &cmd);
 *
 * Coordinates: world x, y in metres, yaw in radians (as the Webots GPS / IMU).
 * Cells: (cx, cy) = (column, row), index cy * ZOI_COLS + cx.
 */
#ifndef ZOI_H
#define ZOI_H

#include <stdint.h>

#include "zoi_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------ */
/* Configuration                                                            */
/* ------------------------------------------------------------------------ */

typedef enum { ZOI_ENV_EASY = 0, ZOI_ENV_MEDIUM = 1, ZOI_ENV_HARD = 2 } zoi_env_t;
typedef enum { ZOI_DET_MOBILENET = 0, ZOI_DET_YOLO_NANO = 1 } zoi_det_t;

/* Camera footprint on a plane, body frame (+x forward, +y left), metres. */
typedef struct {
    float half_left, half_right, forward, backward;
} zoi_fov_t;

/* Counts from one detector on the offline calibration set
 * (configuration/zoi_params.py, DETECTOR_CALIBRATIONS). Rates are
 * derived from them with the rule of succession (k + 1) / (n + 2). */
typedef struct {
    int targets, targets_detected, class_correct, false_boxes, empty_cell_frames;
    int table_cells, table_cells_covered, floor_cells, floor_cells_covered; /* 0 without desks */
} zoi_calibration_t;

typedef struct {
    /* Grid (world frame). */
    float min_x, min_y, cell_size;
    /* Camera. */
    int img_w, img_h;
    zoi_fov_t fov_floor, fov_table;
    /* Controller gates (crazyflie_AIF.py). */
    float coverage_threshold;       /* COVERAGE_UPDATE_THRESHOLD */
    float desk_conf_threshold;      /* SURFACE_DETECTION_CONF_THRESHOLD */
    float shadow_forward_tol_m;     /* TABLE_SHADOW_FORWARD_TOLERANCE_M */
    float claim_conf_threshold;     /* CLAIM_CONFIDENCE_THRESHOLD */
    int bbox_max_samples_per_axis;  /* desk-box probes per axis */
    float desk_overlap_min_frac;    /* bbox_overlaps_any_desk */
    /* Navigator (policies/zoi.py NAVIGATOR_KWARGS). */
    float max_forward, max_sideways, max_yaw_rate, k_pos, k_yaw;
    float dist_eps_m, yaw_eps_rad, min_cell_coverage;
    /* Planner. */
    float detector_period_s;        /* DETECTION_EVERY_S */
    float replan_cooldown_s;
    /* Model: calibrated detector rates and the scenario share of targets on desks. */
    float recall, class_accuracy, false_alarm, desk_recall, desk_false_alarm;
    float p_on_table;
    int table_context;              /* 0 in an environment without desks (easy) */
} zoi_config_t;

/* Defaults of the Webots setup with the calibration of (env, det). Returns 0 on success. */
int zoi_config_default(zoi_config_t *cfg, zoi_env_t env, zoi_det_t det);
float zoi_rule_of_succession(int successes, int trials);

/* ------------------------------------------------------------------------ */
/* Inputs and outputs                                                       */
/* ------------------------------------------------------------------------ */

typedef struct {
    float x, y, yaw;
} zoi_pose_t;

/* One detector box, pixel coordinates [x1, y1, x2, y2] in a img_w x img_h image. */
typedef struct {
    int class_id;   /* ZOI_CLASS_BOTTLE, ZOI_CLASS_CAN, ZOI_CLASS_DESK */
    float conf;
    float x1, y1, x2, y2;
} zoi_box_t;

typedef struct {
    float forward, sideways, yaw_rate;  /* body-frame velocity commands */
    int replanned;                      /* 1 if a new target was chosen this step */
    int target_cx, target_cy, target_yaw_index;
} zoi_command_t;

/* ------------------------------------------------------------------------ */
/* State                                                                    */
/* ------------------------------------------------------------------------ */

/* One cell seen from a candidate pose, relative to the candidate's cell. */
typedef struct {
    int8_t dr, dc;          /* row and column offsets */
    uint8_t lik;            /* index of its likelihood in zoi_state_t.pat_L */
    float w_floor, w_table; /* coverage on the floor and table planes */
} zoi_pattern_entry_t;

typedef struct {
    int n;
    zoi_pattern_entry_t e[ZOI_MAX_PATTERN];
} zoi_pattern_t;

/* Target event of the current frame, projected on a cell. */
typedef struct {
    int class_id;
    float conf;
    int cell;      /* cell index */
    int on_table;  /* surface assumption used for the projection */
} zoi_target_event_t;

typedef struct {
    zoi_config_t cfg;

    /* Prior terms: prior(rho, theta) = base + rho * d_rho + theta * d_theta. */
    float base[ZOI_NS], d_rho[ZOI_NS], d_theta[ZOI_NS];

    /* (rho, theta) grid, posterior. log_ev is the running sum over cells of
     * log Z_j, re-centred to max 0 after every frame (weights unchanged). */
    int n_grid;
    float g_rho[ZOI_MAX_GRID_POINTS], g_theta[ZOI_MAX_GRID_POINTS];
    float log_ev[ZOI_MAX_GRID_POINTS], weight[ZOI_MAX_GRID_POINTS];

    /* Per cell: log of the product of the likelihood rows received (max 0),
     * the same in linear scale (max 1), the evidence coefficients
     * Z_j = a + rho b + theta c, the belief over the 6 joint states. */
    float log_ell[ZOI_CELLS][ZOI_NS];
    float ell[ZOI_CELLS][ZOI_NS];
    float coef[ZOI_CELLS][3];
    float q[ZOI_CELLS][ZOI_NS];
    uint8_t seen[ZOI_CELLS];
    int beliefs_valid;     /* q matches the current posterior */
    int n_seen, n_active;  /* diagnostics of the last belief refresh: seen cells, active grid points */

    /* Controller map: persistent table cells, desk confidence; claims. */
    uint8_t table_map[ZOI_CELLS];
    float table_conf[ZOI_CELLS];
    uint8_t claimed[ZOI_CELLS];   /* bit 0 bottle, bit 1 can */

    /* Current frame (valid after zoi_perception_frame). */
    float cov_floor[ZOI_CELLS], cov_table[ZOI_CELLS], cov_merged[ZOI_CELLS];
    uint8_t in_view[ZOI_CELLS];   /* touched by the floor or table footprint */
    uint8_t visible[ZOI_CELLS];   /* merged coverage >= coverage_threshold */
    uint8_t shadow[ZOI_CELLS];    /* transient table-shadow cells */
    uint8_t desk_cov[ZOI_CELLS];  /* covered by a desk box of this frame */
    uint8_t frame_desk_cells[ZOI_CELLS]; /* desk covered + primary cells (planner table map) */
    int frame_desk_valid;         /* frame_desk_cells apply to the next plan only */
    int n_tevents;
    zoi_target_event_t tevents[ZOI_MAX_BOXES];
    int n_new_claims;
    int new_claim_cell[ZOI_MAX_BOXES], new_claim_class[ZOI_MAX_BOXES];

    /* Planner: view patterns per yaw, candidate scores of the last plan. */
    float yaw_options[ZOI_N_YAW];
    zoi_pattern_t pattern[ZOI_N_YAW];
    int n_pattern_entries;
    /* Likelihood of each pattern entry (it depends only on the entry and the
     * calibration) and its column entropies H(O | s). */
    float pat_L[ZOI_MAX_PATTERN_ENTRIES][ZOI_NO][ZOI_NS];
    float pat_H[ZOI_MAX_PATTERN_ENTRIES][ZOI_NS];
    float cand_ig[ZOI_N_CAND], cand_T[ZOI_N_CAND];
    int last_best;                /* candidate index of the last plan */
    int n_ig_evals, n_T_evals;    /* candidates whose exact IG / exact T the last plan computed */
    /* Per candidate: bit 0 exact IG known (in cand_ig), bit 1 exact T computed by the
     * last plan, bit 2 exact IG computed by the last plan. */
    uint8_t cand_eval[ZOI_N_CAND];
    /* IG reuse between plans: the cached IGs (bit 0 of cand_eval) and prune_H match
     * the current q; the table map those IGs were computed with; candidates whose
     * cached IG the last plan dropped because of a table-map change (diagnostics). */
    int ig_cache_valid;
    int n_ig_invalidated;
    uint8_t ig_cache_table[ZOI_CELLS];
    /* Pruned search: optimistic rate per candidate, candidates in decreasing
     * optimistic rate, per-cell lower bound on the move time, per-yaw turn time,
     * per-cell belief entropy. */
    float prune_key[ZOI_N_CAND];
    uint16_t prune_order[ZOI_N_CAND];
    float prune_move[ZOI_CELLS], prune_turn[ZOI_N_YAW], prune_H[ZOI_CELLS];

    /* Navigator. */
    int has_target, target_cell, target_yaw_index;
    float target_yaw;
    float last_replan_time_s;
    int n_replans;
} zoi_state_t;

/* ------------------------------------------------------------------------ */
/* API                                                                      */
/* ------------------------------------------------------------------------ */

/* Returns 0 on success, < 0 if a compile-time size is too small. */
int zoi_init(zoi_state_t *st, const zoi_config_t *cfg);

/* One processed detector frame: geometry, desk map, claims, belief update. */
void zoi_perception_frame(zoi_state_t *st, const zoi_pose_t *pose, const zoi_box_t *boxes, int n_boxes);

/* One control step: replan when needed (as compute_action), then the navigator command. */
void zoi_control_step(zoi_state_t *st, const zoi_pose_t *pose, float time_s, zoi_command_t *cmd);

/* Best candidate from a pose (side-effect free on the belief); returns its index
 * and fills st->cand_ig / st->cand_T. */
int zoi_plan(zoi_state_t *st, const zoi_pose_t *pose);

/* Navigator only (no replanning): arrival test and velocity command towards the
 * current target. Returns 1 if arrived. zoi_control_step uses the same code. */
int zoi_navigator_step(zoi_state_t *st, const zoi_pose_t *pose, zoi_command_t *cmd);

/* Posterior summary of rho and theta. */
void zoi_rho_theta(const zoi_state_t *st, float *rho_mean, float *rho_sd, float *theta_mean, float *theta_sd);

/* Make st->q current (computed lazily; cheap if already valid). */
void zoi_refresh_beliefs(zoi_state_t *st);

/* ------------------------------------------------------------------------ */
/* Internals shared between the source files (also used by the tests)        */
/* ------------------------------------------------------------------------ */

typedef struct {
    int cell;
    float frac;
} zoi_cov_entry_t;

/* Cells touched by the footprint at (x, y, yaw), in the Python order (rows, then columns). */
int zoi_cell_coverages(const zoi_config_t *cfg, const zoi_fov_t *fov, float x, float y, float yaw,
                       zoi_cov_entry_t *out, int max_out);
/* Coverage of one cell (0 if not touched). */
float zoi_one_cell_coverage(const zoi_config_t *cfg, const zoi_fov_t *fov, float x, float y, float yaw,
                            int row, int col);
/* Cell index of a world point, -1 outside the grid. */
int zoi_cell_from_xy(const zoi_config_t *cfg, float x, float y);
/* Cell of a bbox centre projected on the plane of fov, -1 outside the grid. */
int zoi_cell_from_bbox(const zoi_config_t *cfg, const zoi_fov_t *fov, float x1, float y1, float x2, float y2,
                       float drone_x, float drone_y, float drone_yaw);

int zoi_belief_init(zoi_state_t *st);
int zoi_build_patterns(zoi_state_t *st);
void zoi_project_frame(zoi_state_t *st, const zoi_pose_t *pose, const zoi_box_t *boxes, int n_boxes);
void zoi_belief_update(zoi_state_t *st);
void zoi_joint_likelihood(const zoi_config_t *cfg, float w_table, float w_floor, int desk_in_view,
                          float L[ZOI_NO][ZOI_NS]);
float zoi_time_to(const zoi_config_t *cfg, const zoi_pose_t *pose, float tx, float ty, float target_yaw);

/* Parallel kernels: each processes its share [core, ncores) of the work. On the
 * host they run with ncores = 1; on GAP8 the same functions run on the cluster
 * (see gap8/main.c). zoi_plan / zoi_belief_update call them through st_fork. */
typedef void (*zoi_kernel_fn)(void *arg, int core, int ncores);
typedef void (*zoi_fork_fn)(zoi_kernel_fn fn, void *arg);
extern zoi_fork_fn zoi_fork;   /* default: serial call fn(arg, 0, 1) */

/* Optional profiling hook, called at the start (start = 1) and end (start = 0)
 * of each stage; NULL by default. gap8/main.c uses it to read cycle counters. */
typedef enum {
    ZOI_STAGE_GEOMETRY = 0, /* coverage, desk map, target projection, claims */
    ZOI_STAGE_UPDATE,       /* likelihoods, log ell, (rho, theta) evidence, weights */
    ZOI_STAGE_BELIEFS,      /* per-cell beliefs averaged over (rho, theta) */
    ZOI_STAGE_IG,           /* planner: IG cache upkeep, entropy bounds, candidate order */
    ZOI_STAGE_TIME,         /* planner: search (exact IGs and flight times of the candidates that can win) */
    ZOI_STAGE_NAVIGATOR,    /* arrival test + velocity command */
    ZOI_N_STAGES
} zoi_stage_t;
typedef void (*zoi_profile_fn)(zoi_stage_t stage, int start);
extern zoi_profile_fn zoi_profile;
#define ZOI_PROF(stage, start) do { if (zoi_profile) zoi_profile((stage), (start)); } while (0)

#ifdef __cplusplus
}
#endif

#endif /* ZOI_H */
