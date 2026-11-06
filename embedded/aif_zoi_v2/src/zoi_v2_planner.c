/* zoi_v2_planner.c: expected information gain per second of flight.
 *
 * Port of AIFZOIV2Policy._pick_next_action / _candidate_information_gain
 * (policies/aif_zoi_v2.py, aif_zoi_v2_vectorized.py), expected_information_gain
 * (policies/zoi_v2_model.py) and NavigatorTimeModel (policies/planner_time.py).
 *
 * Every (cell centre, yaw) candidate is scored by IG / T: IG summed over the
 * cells the controller would update from there (merged coverage >= gate, table
 * plane on table-map cells), T the navigator's flight time plus one detector
 * period. Best rate wins; ties go to the shorter T, then to the first candidate.
 *
 * View patterns: the coverage of the cells seen from a candidate, relative to the
 * candidate's cell, is the same for every cell (checked in Python to 1e-12 for
 * all 1144 candidates: 88 (yaw, offset) entries), so it is computed once per yaw
 * at a reference cell, instead of a 11,080-entry per-candidate table. Near the
 * walls the entries outside the grid are skipped, as in the per-candidate table.
 */
#include <math.h>
#include <string.h>

#include "zoi_v2.h"

#define ZOI_PI_F 3.14159265358979f

/* ---- Navigator time model ------------------------------------------------ */

static float wrap_to_pi(float a)
{
    float r = fmodf(a + ZOI_PI_F, 2.0f * ZOI_PI_F);
    if (r < 0.0f)
        r += 2.0f * ZOI_PI_F;
    return r - ZOI_PI_F;
}

/* |e(t)| for e' = -clamp(gain e, +-limit) from |e(0)| = |e0|. */
static float saturated_p_error(float e0, float gain, float limit, float t)
{
    const float a = fabsf(e0);
    if (t <= 0.0f || a == 0.0f)
        return a;
    const float knee = limit / gain;
    if (a > knee) {
        const float t_linear = (a - knee) / limit;
        if (t <= t_linear)
            return a - limit * t;
        return knee * expf(-gain * (t - t_linear));
    }
    return a * expf(-gain * t);
}

/* First time with |e| <= tol for e' = -clamp(gain e, +-limit). */
static float saturated_p_time_to(float e0, float gain, float limit, float tol)
{
    const float a = fabsf(e0);
    if (a <= tol)
        return 0.0f;
    const float knee = limit / gain;
    if (a > knee) {
        if (tol >= knee)
            return (a - tol) / limit;
        return (a - knee) / limit + logf(knee / tol) / gain;
    }
    return logf(a / tol) / gain;
}

#if ZOI_V2_REFERENCE_KERNELS
#define ZOI_BISECTION_STEPS 50
#else
/* The bracket starts below ~20 s; 24 halvings reach the float32 resolution. */
#define ZOI_BISECTION_STEPS 24
#endif

static float move_time(const zoi_config_t *cfg, float dx, float dy)
{
    const float eps = cfg->dist_eps_m;
#if ZOI_V2_REFERENCE_KERNELS
    if (hypotf(dx, dy) <= eps)
        return 0.0f;
#else
    const float eps2 = eps * eps;
    if (dx * dx + dy * dy <= eps2)
        return 0.0f;
#endif
    const float tol = eps / sqrtf(2.0f);
    float lo = 0.0f;
    float hi = fmaxf(saturated_p_time_to(dx, cfg->k_pos, cfg->max_forward, tol),
                     saturated_p_time_to(dy, cfg->k_pos, cfg->max_sideways, tol));
    for (int it = 0; it < ZOI_BISECTION_STEPS; it++) {
        const float mid = 0.5f * (lo + hi);
        const float ex = saturated_p_error(dx, cfg->k_pos, cfg->max_forward, mid);
        const float ey = saturated_p_error(dy, cfg->k_pos, cfg->max_sideways, mid);
#if ZOI_V2_REFERENCE_KERNELS
        const int arrived = hypotf(ex, ey) <= eps;
#else
        const int arrived = ex * ex + ey * ey <= eps2;
#endif
        if (arrived)
            hi = mid;
        else
            lo = mid;
    }
    return hi;
}

/* Navigator arrival time (turn first, then translate in the body frame) plus one detector period. */
float zoi_time_to(const zoi_config_t *cfg, const zoi_pose_t *pose, float tx, float ty, float target_yaw)
{
    const float err = wrap_to_pi(target_yaw - pose->yaw);
    float t_turn = 0.0f, heading = pose->yaw;
    if (fabsf(err) > cfg->yaw_eps_rad) {
        t_turn = saturated_p_time_to(err, cfg->k_yaw, cfg->max_yaw_rate, cfg->yaw_eps_rad);
        heading = target_yaw - copysignf(cfg->yaw_eps_rad, err); /* the turn stops at the tolerance */
    }
    const float dx = tx - pose->x, dy = ty - pose->y;
    const float c = cosf(heading), s = sinf(heading);
    const float t_move = move_time(cfg, c * dx + s * dy, -s * dx + c * dy);
    return t_turn + t_move + cfg->detector_period_s;
}

/* ---- Expected information gain ----------------------------------------- */

/* Expected KL(posterior || prior) over the predicted observation. */
float zoi_expected_information_gain(const float q_in[ZOI_NS], const float L[ZOI_NO][ZOI_NS])
{
    float z = 0.0f;
    for (int s = 0; s < ZOI_NS; s++)
        z += q_in[s];
    if (z <= 0.0f)
        return 0.0f;
    float q[ZOI_NS], log_q[ZOI_NS];
    for (int s = 0; s < ZOI_NS; s++) {
        q[s] = q_in[s] / z;
        log_q[s] = logf(q[s] + 1e-12f);
    }
    float ig = 0.0f;
    for (int o = 0; o < ZOI_NO; o++) {
        float po = 0.0f;
        for (int s = 0; s < ZOI_NS; s++)
            po += L[o][s] * q[s];
        if (!(po > 1e-12f))
            continue;
        float kl = 0.0f;
        for (int s = 0; s < ZOI_NS; s++) {
            const float post = L[o][s] * q[s] / po;
            kl += post * (logf(post + 1e-12f) - log_q[s]);
        }
        ig += po * kl;
    }
    return ig;
}

#if !ZOI_V2_REFERENCE_KERNELS
/* The same quantity as a mutual information, I(S; O) = H(O) - sum_s q_s H(O | s),
 * with H(O | s) (the column entropies of L) precomputed per pattern entry: 6 logs
 * instead of 42. */
static float information_gain_mi(const float q_in[ZOI_NS], const float L[ZOI_NO][ZOI_NS], const float H_col[ZOI_NS])
{
    float z = 0.0f;
    for (int s = 0; s < ZOI_NS; s++)
        z += q_in[s];
    if (z <= 0.0f)
        return 0.0f;
    const float inv_z = 1.0f / z;
    float h_cond = 0.0f, h_obs = 0.0f;
    for (int s = 0; s < ZOI_NS; s++)
        h_cond += q_in[s] * H_col[s];
    for (int o = 0; o < ZOI_NO; o++) {
        float po = 0.0f;
        for (int s = 0; s < ZOI_NS; s++)
            po += L[o][s] * q_in[s];
        po *= inv_z;
        if (po > 1e-12f)
            h_obs -= po * logf(po);
    }
    const float ig = h_obs - h_cond * inv_z;
    return ig > 0.0f ? ig : 0.0f;
}
#endif

/* ---- View patterns -------------------------------------------------------- */

int zoi_build_patterns(zoi_state_t *st)
{
    const zoi_config_t *cfg = &st->cfg;
    const int r0 = ZOI_ROWS / 2, c0 = ZOI_COLS / 2;
    const float x = cfg->min_x + ((float)c0 + 0.5f) * cfg->cell_size;
    const float y = cfg->min_y + ((float)r0 + 0.5f) * cfg->cell_size;
    zoi_cov_entry_t fc[ZOI_MAX_FOOT], tc[ZOI_MAX_FOOT];
    st->n_pattern_entries = 0;
    for (int k = 0; k < ZOI_N_YAW; k++) {
        const float yaw = st->yaw_options[k];
        const int nf = zoi_cell_coverages(cfg, &cfg->fov_floor, x, y, yaw, fc, ZOI_MAX_FOOT);
        const int nt = zoi_cell_coverages(cfg, &cfg->fov_table, x, y, yaw, tc, ZOI_MAX_FOOT);
        zoi_pattern_t *p = &st->pattern[k];
        p->n = 0;
        /* Union of the cells touched on either plane, in cell order. */
        for (int cell = 0; cell < ZOI_CELLS; cell++) {
            float wf = 0.0f, wt = 0.0f;
            int hit = 0;
            for (int i = 0; i < nf; i++)
                if (fc[i].cell == cell) {
                    wf = fc[i].frac;
                    hit = 1;
                }
            for (int i = 0; i < nt; i++)
                if (tc[i].cell == cell) {
                    wt = tc[i].frac;
                    hit = 1;
                }
            if (!hit)
                continue;
            if (p->n >= ZOI_MAX_PATTERN || st->n_pattern_entries >= ZOI_MAX_PATTERN_ENTRIES)
                return -1;
            /* The reference cell must keep the whole footprint inside the grid. */
            const int row = cell / ZOI_COLS, col = cell % ZOI_COLS;
            if (row == 0 || row == ZOI_ROWS - 1 || col == 0 || col == ZOI_COLS - 1)
                return -2;
            const int li = st->n_pattern_entries++;
            zoi_pattern_entry_t *e = &p->e[p->n];
            e->lik = (uint8_t)li;
            e->dr = (int8_t)(cell / ZOI_COLS - r0);
            e->dc = (int8_t)(cell % ZOI_COLS - c0);
            e->w_floor = wf;
            e->w_table = wt;
            const int desk_in_view = cfg->table_context && wt > 0.0f && wt >= cfg->coverage_threshold;
            zoi_joint_likelihood(cfg, wt, wf, desk_in_view, st->pat_L[li]);
            for (int s = 0; s < ZOI_NS; s++) {
                float h = 0.0f;
                for (int o = 0; o < ZOI_NO; o++) {
                    const float l = st->pat_L[li][o][s];
                    if (l > 0.0f)
                        h -= l * logf(l);
                }
                st->pat_H[li][s] = h;
            }
            p->n++;
        }
    }
    return 0;
}

/* ---- Planner ---------------------------------------------------------- */

typedef struct {
    zoi_state_t *st;
    const zoi_pose_t *pose;
    const uint8_t *is_table;
} plan_arg_t;

static void ig_kernel(void *p, int core, int ncores)
{
    plan_arg_t *arg = (plan_arg_t *)p;
    zoi_state_t *st = arg->st;
    const float thr = st->cfg.coverage_threshold;
    const int chunk = (ZOI_N_CAND + ncores - 1) / ncores;
    const int k0 = core * chunk, k1 = k0 + chunk < ZOI_N_CAND ? k0 + chunk : ZOI_N_CAND;
    for (int k = k0; k < k1; k++) {
        const int cell = k / ZOI_N_YAW, yaw = k % ZOI_N_YAW;
        const int r = cell / ZOI_COLS, c = cell % ZOI_COLS;
        const zoi_pattern_t *pat = &st->pattern[yaw];
        float ig = 0.0f;
        for (int i = 0; i < pat->n; i++) {
            const zoi_pattern_entry_t *e = &pat->e[i];
            const int rr = r + e->dr, cc = c + e->dc;
            if (rr < 0 || rr >= ZOI_ROWS || cc < 0 || cc >= ZOI_COLS)
                continue;
            const int j = rr * ZOI_COLS + cc;
            const float merged = arg->is_table[j] ? e->w_table : e->w_floor;
            if (!(merged > 0.0f && merged >= thr))
                continue;
#if ZOI_V2_REFERENCE_KERNELS
            ig += zoi_expected_information_gain(st->q[j], (const float (*)[ZOI_NS])st->pat_L[e->lik]);
#else
            ig += information_gain_mi(st->q[j], (const float (*)[ZOI_NS])st->pat_L[e->lik], st->pat_H[e->lik]);
#endif
        }
        st->cand_ig[k] = ig;
    }
}

static void time_kernel(void *p, int core, int ncores)
{
    plan_arg_t *arg = (plan_arg_t *)p;
    zoi_state_t *st = arg->st;
    const zoi_config_t *cfg = &st->cfg;
    const int chunk = (ZOI_N_CAND + ncores - 1) / ncores;
    const int k0 = core * chunk, k1 = k0 + chunk < ZOI_N_CAND ? k0 + chunk : ZOI_N_CAND;
    for (int k = k0; k < k1; k++) {
        const int cell = k / ZOI_N_YAW, yaw = k % ZOI_N_YAW;
        const float tx = cfg->min_x + ((float)(cell % ZOI_COLS) + 0.5f) * cfg->cell_size;
        const float ty = cfg->min_y + ((float)(cell / ZOI_COLS) + 0.5f) * cfg->cell_size;
        st->cand_T[k] = zoi_time_to(cfg, arg->pose, tx, ty, st->yaw_options[yaw]);
    }
}

int zoi_plan(zoi_state_t *st, const zoi_pose_t *pose)
{
    /* The controller's table map, as the planner sees it: persistent cells plus,
     * right after a detector frame, that frame's desk cells (_table_cells_from_grid). */
    uint8_t is_table[ZOI_CELLS];
    for (int j = 0; j < ZOI_CELLS; j++)
        is_table[j] = st->table_map[j] || (st->frame_desk_valid && st->frame_desk_cells[j]);

    zoi_refresh_beliefs(st);
    plan_arg_t arg = {st, pose, is_table};
    ZOI_PROF(ZOI_STAGE_IG, 1);
    zoi_fork(ig_kernel, &arg);
    ZOI_PROF(ZOI_STAGE_IG, 0);
    ZOI_PROF(ZOI_STAGE_TIME, 1);
    zoi_fork(time_kernel, &arg);
    ZOI_PROF(ZOI_STAGE_TIME, 0);

    int best = 0;
    float best_rate = -INFINITY, best_T = INFINITY;
    for (int k = 0; k < ZOI_N_CAND; k++) {
        const float rate = st->cand_ig[k] / st->cand_T[k];
        if (rate > best_rate || (rate == best_rate && st->cand_T[k] < best_T)) {
            best = k;
            best_rate = rate;
            best_T = st->cand_T[k];
        }
    }
    st->last_best = best;
    return best;
}
