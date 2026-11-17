/* zoi_v2.c: top-level API and navigator.
 *
 * zoi_perception_frame = the controller's processed-frame geometry + the policy's
 * belief update; zoi_control_step = AIFZOIV2Policy.compute_action's replanning
 * rule + utils/navigator.py.
 */
#include <math.h>
#include <string.h>

#include "zoi_v2.h"

#define ZOI_PI_F 3.14159265358979f

static void serial_fork(zoi_kernel_fn fn, void *arg)
{
    fn(arg, 0, 1);
}

zoi_fork_fn zoi_fork = serial_fork;
zoi_profile_fn zoi_profile = 0;

int zoi_init(zoi_state_t *st, const zoi_config_t *cfg)
{
    memset(st, 0, sizeof(*st));
    st->cfg = *cfg;
    for (int k = 0; k < ZOI_N_YAW; k++)
        st->yaw_options[k] = (float)k * (ZOI_PI_F / 4.0f);
    if (zoi_belief_init(st) != 0)
        return -1;
    if (zoi_build_patterns(st) != 0)
        return -2;
    st->last_replan_time_s = -1e9f;
    st->last_best = -1;
    return 0;
}

void zoi_perception_frame(zoi_state_t *st, const zoi_pose_t *pose, const zoi_box_t *boxes, int n_boxes)
{
    ZOI_PROF(ZOI_STAGE_GEOMETRY, 1);
    zoi_project_frame(st, pose, boxes, n_boxes);
    ZOI_PROF(ZOI_STAGE_GEOMETRY, 0);
    ZOI_PROF(ZOI_STAGE_UPDATE, 1);
    zoi_belief_update(st);
    ZOI_PROF(ZOI_STAGE_UPDATE, 0);
}

/* ---- Navigator ---------------------------------------------------------- */

static float wrap_to_pi(float a)
{
    float r = fmodf(a + ZOI_PI_F, 2.0f * ZOI_PI_F);
    if (r < 0.0f)
        r += 2.0f * ZOI_PI_F;
    return r - ZOI_PI_F;
}

static float clampf(float x, float lo, float hi)
{
    return fmaxf(lo, fminf(hi, x));
}

static void cell_centre(const zoi_config_t *cfg, int cell, float *x, float *y)
{
    *x = cfg->min_x + ((float)(cell % ZOI_COLS) + 0.5f) * cfg->cell_size;
    *y = cfg->min_y + ((float)(cell / ZOI_COLS) + 0.5f) * cfg->cell_size;
}

/* Arrival: target cell coverage >= min, distance <= eps, yaw error <= eps. The
 * coverage is the controller's merged coverage of this step: table plane on
 * table-map cells (and on this frame's shadow cells right after a detector frame). */
static int navigator_arrived(const zoi_state_t *st, const zoi_pose_t *p)
{
    const zoi_config_t *cfg = &st->cfg;
    const int cell = st->target_cell;
    const int table = st->table_map[cell] || (st->frame_desk_valid && st->shadow[cell]);
    const float cov = zoi_one_cell_coverage(cfg, table ? &cfg->fov_table : &cfg->fov_floor, p->x, p->y, p->yaw,
                                            cell / ZOI_COLS, cell % ZOI_COLS);
    float tx, ty;
    cell_centre(cfg, cell, &tx, &ty);
    const float dist = hypotf(tx - p->x, ty - p->y);
    const float yaw_err = fabsf(wrap_to_pi(st->target_yaw - p->yaw));
    return cov >= cfg->min_cell_coverage && dist <= cfg->dist_eps_m && yaw_err <= cfg->yaw_eps_rad;
}

/* Yaw first, then translate towards the cell centre (body-frame P control). */
static void navigator_command(const zoi_state_t *st, const zoi_pose_t *p, zoi_command_t *cmd)
{
    const zoi_config_t *cfg = &st->cfg;
    cmd->forward = cmd->sideways = cmd->yaw_rate = 0.0f;
    if (!st->has_target)
        return;
    const float yaw_err = wrap_to_pi(st->target_yaw - p->yaw);
    if (fabsf(yaw_err) > cfg->yaw_eps_rad) {
        cmd->yaw_rate = clampf(cfg->k_yaw * yaw_err, -cfg->max_yaw_rate, cfg->max_yaw_rate);
        return;
    }
    float tx, ty;
    cell_centre(cfg, st->target_cell, &tx, &ty);
    const float dx = tx - p->x, dy = ty - p->y;
    const float c = cosf(p->yaw), s = sinf(p->yaw);
    const float dxb = c * dx + s * dy, dyb = -s * dx + c * dy;
    cmd->forward = clampf(cfg->k_pos * dxb, -cfg->max_forward, cfg->max_forward);
    cmd->sideways = clampf(cfg->k_pos * dyb, -cfg->max_sideways, cfg->max_sideways);
}

int zoi_navigator_step(zoi_state_t *st, const zoi_pose_t *pose, zoi_command_t *cmd)
{
    ZOI_PROF(ZOI_STAGE_NAVIGATOR, 1);
    const int arrived = st->has_target ? navigator_arrived(st, pose) : 0;
    navigator_command(st, pose, cmd);
    ZOI_PROF(ZOI_STAGE_NAVIGATOR, 0);
    return arrived;
}

void zoi_control_step(zoi_state_t *st, const zoi_pose_t *pose, float time_s, zoi_command_t *cmd)
{
    cmd->replanned = 0;
    const int current = zoi_cell_from_xy(&st->cfg, pose->x, pose->y);
    ZOI_PROF(ZOI_STAGE_NAVIGATOR, 1);
    const int arrived = st->has_target ? navigator_arrived(st, pose) : 0;
    ZOI_PROF(ZOI_STAGE_NAVIGATOR, 0);
    const int cooldown_elapsed = (time_s - st->last_replan_time_s) >= st->cfg.replan_cooldown_s;
    if ((!st->has_target || (arrived && cooldown_elapsed)) && current >= 0) {
        const int best = zoi_plan(st, pose);
        st->has_target = 1;
        st->target_cell = best / ZOI_N_YAW;
        st->target_yaw_index = best % ZOI_N_YAW;
        st->target_yaw = st->yaw_options[st->target_yaw_index];
        st->last_replan_time_s = time_s;
        st->n_replans++;
        cmd->replanned = 1;
    }
    cmd->target_cx = st->has_target ? st->target_cell % ZOI_COLS : -1;
    cmd->target_cy = st->has_target ? st->target_cell / ZOI_COLS : -1;
    cmd->target_yaw_index = st->has_target ? st->target_yaw_index : -1;
    ZOI_PROF(ZOI_STAGE_NAVIGATOR, 1);
    navigator_command(st, pose, cmd);
    ZOI_PROF(ZOI_STAGE_NAVIGATOR, 0);
    /* This frame's desk cells and shadow cells apply only to the step that processed it. */
    st->frame_desk_valid = 0;
}
