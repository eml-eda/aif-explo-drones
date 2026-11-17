/* zoi_config.c: defaults of the Webots setup and the detector calibrations. */
#include "zoi.h"

/* (k + 1) / (n + 2): posterior mean of a rate under a uniform prior, also the
 * predictive probability of the next trial (configuration/zoi_params.py). */
float zoi_rule_of_succession(int successes, int trials)
{
    return ((float)successes + 1.0f) / ((float)trials + 2.0f);
}

/* Counts on the seed-101 calibration sets, 400 images each, confidence 0.45, NMS 0.40
 * (configuration/zoi_params.py, DETECTOR_CALIBRATIONS). Index [env][det]. */
static const zoi_calibration_t CALIBRATIONS[3][2] = {
    /* easy: no desks */
    {{333, 301, 301, 23, 3346, 0, 0, 0, 0},
     {333, 321, 321, 17, 3346, 0, 0, 0, 0}},
    /* medium */
    {{187, 182, 182, 0, 3415, 238, 227, 1414, 252},
     {187, 187, 187, 3, 3415, 238, 229, 1414, 260}},
    /* hard */
    {{167, 140, 140, 2, 3491, 189, 181, 1488, 188},
     {167, 167, 167, 6, 3491, 189, 182, 1488, 198}},
};

/* Share of targets on desks given to the policy (DEFAULT_P_ON_TABLE in Python): 0.5 tells it
 * nothing about where targets are more likely. */
#define ZOI_DEFAULT_P_ON_TABLE 0.5f

int zoi_config_default(zoi_config_t *cfg, zoi_env_t env, zoi_det_t det)
{
    if ((int)env < 0 || (int)env > 2 || (int)det < 0 || (int)det > 1)
        return -1;
    const zoi_calibration_t *c = &CALIBRATIONS[env][det];

    /* Grid: 5.5 x 6.5 m room centred on the origin, 0.5 m cells. */
    cfg->cell_size = 0.5f;
    cfg->min_x = -0.5f * (float)ZOI_COLS * cfg->cell_size;
    cfg->min_y = -0.5f * (float)ZOI_ROWS * cfg->cell_size;

    /* Camera 320 x 240; floor footprint measured at 1.5 m (FOV_WIDTH_M, FOV_FORWARD_M,
     * FOV_BACKWARD_M); table footprint scaled by the camera-to-surface distance,
     * (1.5 - 0.72) / 1.5 (env/surface_projection.py, scaled_fov_spec_for_surface). */
    cfg->img_w = 320;
    cfg->img_h = 240;
    cfg->fov_floor.half_left = 0.5f * 1.394f;
    cfg->fov_floor.half_right = 0.5f * 1.394f;
    cfg->fov_floor.forward = 0.553f;
    cfg->fov_floor.backward = 0.493f;
    const float scale = (1.5f - 0.72f) / 1.5f;
    cfg->fov_table.half_left = 0.5f * (1.394f * scale);
    cfg->fov_table.half_right = 0.5f * (1.394f * scale);
    cfg->fov_table.forward = 0.553f * scale;
    cfg->fov_table.backward = 0.493f * scale;

    cfg->coverage_threshold = 0.05f;
    cfg->desk_conf_threshold = 0.30f;
    cfg->shadow_forward_tol_m = 0.10f;
    cfg->claim_conf_threshold = 0.45f;
    cfg->bbox_max_samples_per_axis = 5;
    cfg->desk_overlap_min_frac = 0.05f;

    cfg->max_forward = 0.50f;
    cfg->max_sideways = 0.50f;
    cfg->max_yaw_rate = 0.60f;
    cfg->k_pos = 1.2f;
    cfg->k_yaw = 1.5f;
    cfg->dist_eps_m = 0.2f;
    cfg->yaw_eps_rad = 0.12f;
    cfg->min_cell_coverage = 0.60f;

    cfg->detector_period_s = 0.625f;
    cfg->replan_cooldown_s = 0.4f;

    cfg->recall = zoi_rule_of_succession(c->targets_detected, c->targets);
    cfg->class_accuracy = zoi_rule_of_succession(c->class_correct, c->targets_detected);
    cfg->false_alarm = zoi_rule_of_succession(c->false_boxes, c->empty_cell_frames);
    cfg->table_context = (c->table_cells > 0);
    if (cfg->table_context) {
        cfg->desk_recall = zoi_rule_of_succession(c->table_cells_covered, c->table_cells);
        cfg->desk_false_alarm = zoi_rule_of_succession(c->floor_cells_covered, c->floor_cells);
        cfg->p_on_table = ZOI_DEFAULT_P_ON_TABLE;
    } else {
        cfg->desk_recall = 0.0f;
        cfg->desk_false_alarm = 0.0f;
        cfg->p_on_table = 0.0f;
    }
    return 0;
}
