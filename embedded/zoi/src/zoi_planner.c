/* zoi_planner.c: expected information gain per second of flight.
 *
 * Port of ZOIPolicy._pick_next_action / _candidate_information_gain
 * (policies/zoi.py, zoi_vectorized.py), expected_information_gain
 * (policies/zoi_model.py) and NavigatorTimeModel (policies/planner_time.py).
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

#include "zoi.h"

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

#if ZOI_REFERENCE_KERNELS
#define ZOI_BISECTION_STEPS 50
#else
/* The bracket starts below ~20 s; 24 halvings reach the float32 resolution. */
#define ZOI_BISECTION_STEPS 24
#endif

static float move_time(const zoi_config_t *cfg, float dx, float dy)
{
    const float eps = cfg->dist_eps_m;
#if ZOI_REFERENCE_KERNELS
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
#if ZOI_REFERENCE_KERNELS
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

#if !ZOI_REFERENCE_KERNELS
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
    int reuse; /* ZOI_IG_CACHE: q unchanged since the last plan (cached IGs, prune_H) */
} plan_arg_t;

/* The controller's update gate on a cell's merged coverage. */
static inline int coverage_gate(float merged, float thr)
{
    return merged > 0.0f && merged >= thr;
}

/* Exact information gain of candidate k: the sum over the cells the controller
 * would update from there (merged coverage >= gate, table plane on table-map cells). */
static float candidate_ig(const zoi_state_t *st, const uint8_t *is_table, int k)
{
    const float thr = st->cfg.coverage_threshold;
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
        const float merged = is_table[j] ? e->w_table : e->w_floor;
        if (!coverage_gate(merged, thr))
            continue;
#if ZOI_REFERENCE_KERNELS
        ig += zoi_expected_information_gain(st->q[j], (const float (*)[ZOI_NS])st->pat_L[e->lik]);
#else
        ig += information_gain_mi(st->q[j], (const float (*)[ZOI_NS])st->pat_L[e->lik], st->pat_H[e->lik]);
#endif
    }
    return ig;
}

static float candidate_T(const zoi_state_t *st, const zoi_pose_t *pose, int k)
{
    const zoi_config_t *cfg = &st->cfg;
    const int cell = k / ZOI_N_YAW, yaw = k % ZOI_N_YAW;
    const float tx = cfg->min_x + ((float)(cell % ZOI_COLS) + 0.5f) * cfg->cell_size;
    const float ty = cfg->min_y + ((float)(cell / ZOI_COLS) + 0.5f) * cfg->cell_size;
    return zoi_time_to(cfg, pose, tx, ty, st->yaw_options[yaw]);
}

#if ZOI_PLANNER_PRUNING < 2
static void ig_kernel(void *p, int core, int ncores)
{
    plan_arg_t *arg = (plan_arg_t *)p;
    zoi_state_t *st = arg->st;
    const int chunk = (ZOI_N_CAND + ncores - 1) / ncores;
    const int k0 = core * chunk, k1 = k0 + chunk < ZOI_N_CAND ? k0 + chunk : ZOI_N_CAND;
    for (int k = k0; k < k1; k++) {
#if ZOI_IG_CACHE
        if (st->cand_eval[k] & 1)
            continue; /* reused from the previous plan */
#endif
        st->cand_ig[k] = candidate_ig(st, arg->is_table, k);
        st->cand_eval[k] |= 5;
    }
}
#endif

/* Better candidate under the exhaustive planner's rule, independent of the order
 * in which candidates are examined: higher rate, then shorter T, then lower index. */
static int better(float rate, float T, int k, float best_rate, float best_T, int best)
{
    if (rate != best_rate)
        return rate > best_rate;
    if (T != best_T)
        return T < best_T;
    return best < 0 || k < best;
}

#if !ZOI_PLANNER_PRUNING

static void time_kernel(void *p, int core, int ncores)
{
    plan_arg_t *arg = (plan_arg_t *)p;
    zoi_state_t *st = arg->st;
    const int chunk = (ZOI_N_CAND + ncores - 1) / ncores;
    const int k0 = core * chunk, k1 = k0 + chunk < ZOI_N_CAND ? k0 + chunk : ZOI_N_CAND;
    for (int k = k0; k < k1; k++) {
        st->cand_T[k] = candidate_T(st, arg->pose, k);
        st->cand_eval[k] |= 2;
    }
}

static int search(zoi_state_t *st, plan_arg_t *arg)
{
    ZOI_PROF(ZOI_STAGE_IG, 1);
    zoi_fork(ig_kernel, arg);
    ZOI_PROF(ZOI_STAGE_IG, 0);
    ZOI_PROF(ZOI_STAGE_TIME, 1);
    zoi_fork(time_kernel, arg);
    ZOI_PROF(ZOI_STAGE_TIME, 0);
    int best = -1;
    float best_rate = -INFINITY, best_T = INFINITY;
    for (int k = 0; k < ZOI_N_CAND; k++) {
        const float rate = st->cand_ig[k] / st->cand_T[k];
        if (better(rate, st->cand_T[k], k, best_rate, best_T, best)) {
            best = k;
            best_rate = rate;
            best_T = st->cand_T[k];
        }
    }
    return best;
}

#else /* ZOI_PLANNER_PRUNING */

/* Safety margins against float32 rounding: a bound may only err on the optimistic
 * side. Relative on every bound, plus an absolute slack per cell on the entropy
 * bound (the computed IG of a nearly certain cell can exceed its entropy by
 * rounding, ~1e-6). They cost a few extra evaluations, never a different decision. */
#define PRUNE_REL 1.0001f
#define PRUNE_ABS_PER_CELL 1e-5f
/* Candidates examined per parallel step (cores take them in turn). The best rate
 * is updated between steps; inside a step the best at its start is used, which
 * only errs on the side of computing more. */
#define PRUNE_BATCH 8

/* Lower bound on the flight time of candidate k: the exact turn time, plus the
 * distance to cover (arrival within dist_eps) at the navigator's top speed
 * sqrt(max_forward^2 + max_sideways^2), plus one detector period. */
static float t_min(const zoi_state_t *st, int k)
{
    return st->prune_turn[k % ZOI_N_YAW] + st->prune_move[k / ZOI_N_YAW] + st->cfg.detector_period_s;
}

static void prepare_t_min(zoi_state_t *st, const zoi_pose_t *pose)
{
    const zoi_config_t *cfg = &st->cfg;
    const float vmax = sqrtf(cfg->max_forward * cfg->max_forward + cfg->max_sideways * cfg->max_sideways);
    for (int j = 0; j < ZOI_CELLS; j++) {
        const float tx = cfg->min_x + ((float)(j % ZOI_COLS) + 0.5f) * cfg->cell_size;
        const float ty = cfg->min_y + ((float)(j / ZOI_COLS) + 0.5f) * cfg->cell_size;
        const float dx = tx - pose->x, dy = ty - pose->y;
        const float d = sqrtf(dx * dx + dy * dy);
        st->prune_move[j] = d > cfg->dist_eps_m ? (d - cfg->dist_eps_m) / vmax : 0.0f;
    }
    for (int y = 0; y < ZOI_N_YAW; y++) {
        const float err = wrap_to_pi(st->yaw_options[y] - pose->yaw);
        st->prune_turn[y] =
            fabsf(err) > cfg->yaw_eps_rad ? saturated_p_time_to(err, cfg->k_yaw, cfg->max_yaw_rate, cfg->yaw_eps_rad) : 0.0f;
    }
}

/* Heapsort of the candidates by decreasing key (ties: lower index first). */
static int goes_after(const float *key, uint16_t a, uint16_t b)
{
    return key[a] < key[b] || (key[a] == key[b] && a > b);
}

static void sift_down(const float *key, uint16_t *h, int i, int n)
{
    for (;;) {
        int m = i;
        const int l = 2 * i + 1, r = l + 1;
        /* min-heap under goes_after: the root is the candidate that goes last */
        if (l < n && goes_after(key, h[l], h[m]))
            m = l;
        if (r < n && goes_after(key, h[r], h[m]))
            m = r;
        if (m == i)
            return;
        const uint16_t t = h[i];
        h[i] = h[m];
        h[m] = t;
        i = m;
    }
}

static void sort_by_key(const float *key, uint16_t *order, int n)
{
    for (int k = 0; k < n; k++)
        order[k] = (uint16_t)k;
    for (int i = n / 2 - 1; i >= 0; i--)
        sift_down(key, order, i, n);
    for (int end = n - 1; end > 0; end--) {
        const uint16_t t = order[0];
        order[0] = order[end];
        order[end] = t;
        sift_down(key, order, 0, end);
    }
}

#if ZOI_PLANNER_PRUNING >= 2
/* Entropy of each cell's belief: no single view can give more information about the cell. */
static void entropy_kernel(void *p, int core, int ncores)
{
    zoi_state_t *st = ((plan_arg_t *)p)->st;
    const int chunk = (ZOI_CELLS + ncores - 1) / ncores;
    const int j0 = core * chunk, j1 = j0 + chunk < ZOI_CELLS ? j0 + chunk : ZOI_CELLS;
    for (int j = j0; j < j1; j++) {
        float z = 0.0f;
        for (int s = 0; s < ZOI_NS; s++)
            z += st->q[j][s];
        float h = 0.0f;
        if (z > 0.0f)
            for (int s = 0; s < ZOI_NS; s++) {
                const float qs = st->q[j][s] / z;
                if (qs > 0.0f)
                    h -= qs * logf(qs);
            }
        st->prune_H[j] = h;
    }
}

/* Optimistic rate: (sum of the entropies of the cells the candidate would update) / T_min. */
static void bound_kernel(void *p, int core, int ncores)
{
    plan_arg_t *arg = (plan_arg_t *)p;
    zoi_state_t *st = arg->st;
    const float thr = st->cfg.coverage_threshold;
    const int chunk = (ZOI_N_CAND + ncores - 1) / ncores;
    const int k0 = core * chunk, k1 = k0 + chunk < ZOI_N_CAND ? k0 + chunk : ZOI_N_CAND;
    for (int k = k0; k < k1; k++) {
#if ZOI_IG_CACHE
        if (st->cand_eval[k] & 1) {
            /* Exact IG reused from the previous plan: the Level 1 key, IG / T_min,
             * the same bound the scan applies before computing T. */
            st->prune_key[k] = st->cand_ig[k] / t_min(st, k) * PRUNE_REL;
            continue;
        }
#endif
        const int cell = k / ZOI_N_YAW, yaw = k % ZOI_N_YAW;
        const int r = cell / ZOI_COLS, c = cell % ZOI_COLS;
        const zoi_pattern_t *pat = &st->pattern[yaw];
        float ub = 0.0f;
        for (int i = 0; i < pat->n; i++) {
            const zoi_pattern_entry_t *e = &pat->e[i];
            const int rr = r + e->dr, cc = c + e->dc;
            if (rr < 0 || rr >= ZOI_ROWS || cc < 0 || cc >= ZOI_COLS)
                continue;
            const int j = rr * ZOI_COLS + cc;
            const float merged = arg->is_table[j] ? e->w_table : e->w_floor;
            if (coverage_gate(merged, thr))
                ub += st->prune_H[j] * PRUNE_REL + PRUNE_ABS_PER_CELL;
        }
        st->prune_key[k] = ub / t_min(st, k) * PRUNE_REL;
    }
}
#endif

/* One step of the search: candidates order[start .. start + n). Each gets its
 * exact IG if not known yet; its exact T only if IG / T_min can still reach the
 * best rate known at the start of the step. */
typedef struct {
    plan_arg_t *plan;
    int start, n;
    float best_rate;
} scan_arg_t;

static void scan_kernel(void *p, int core, int ncores)
{
    scan_arg_t *a = (scan_arg_t *)p;
    zoi_state_t *st = a->plan->st;
    for (int i = core; i < a->n; i += ncores) {
        const int k = st->prune_order[a->start + i];
        if (!(st->cand_eval[k] & 1)) {
            st->cand_ig[k] = candidate_ig(st, a->plan->is_table, k);
            st->cand_eval[k] |= 5;
        }
        if (st->cand_ig[k] / t_min(st, k) * PRUNE_REL < a->best_rate)
            continue; /* cannot win: its exact T is not needed */
        st->cand_T[k] = candidate_T(st, a->plan->pose, k);
        st->cand_eval[k] |= 2;
    }
}

static int search(zoi_state_t *st, plan_arg_t *arg)
{
    ZOI_PROF(ZOI_STAGE_IG, 1);
    prepare_t_min(st, arg->pose);
#if ZOI_PLANNER_PRUNING == 1
    /* Level 1: exact IG of every candidate; the key is the best rate it could reach. */
    zoi_fork(ig_kernel, arg);
    for (int k = 0; k < ZOI_N_CAND; k++)
        st->prune_key[k] = st->cand_ig[k] / t_min(st, k) * PRUNE_REL;
#else
    /* Level 2: the key bounds the IG too (sum of entropies). The entropies
     * depend only on q: with the IG cache they are kept while q is. */
#if ZOI_IG_CACHE
    if (!arg->reuse)
#endif
        zoi_fork(entropy_kernel, arg);
    zoi_fork(bound_kernel, arg);
#endif
    sort_by_key(st->prune_key, st->prune_order, ZOI_N_CAND);
    ZOI_PROF(ZOI_STAGE_IG, 0);

    ZOI_PROF(ZOI_STAGE_TIME, 1);
    int best = -1;
    float best_rate = -INFINITY, best_T = INFINITY;
    for (int start = 0; start < ZOI_N_CAND; start += PRUNE_BATCH) {
        /* Sorted keys: once a key is below the best exact rate, no candidate from
         * there on can reach it (strictly below: possible ties are still examined). */
        if (st->prune_key[st->prune_order[start]] < best_rate)
            break;
        int n = ZOI_N_CAND - start < PRUNE_BATCH ? ZOI_N_CAND - start : PRUNE_BATCH;
        while (n > 0 && st->prune_key[st->prune_order[start + n - 1]] < best_rate)
            n--;
        scan_arg_t sa = {arg, start, n, best_rate};
        zoi_fork(scan_kernel, &sa);
        for (int i = 0; i < n; i++) {
            const int k = st->prune_order[start + i];
            if (!(st->cand_eval[k] & 2))
                continue;
            const float rate = st->cand_ig[k] / st->cand_T[k];
            if (better(rate, st->cand_T[k], k, best_rate, best_T, best)) {
                best = k;
                best_rate = rate;
                best_T = st->cand_T[k];
            }
        }
    }
    ZOI_PROF(ZOI_STAGE_TIME, 0);
    return best;
}

#endif /* ZOI_PLANNER_PRUNING */

#if ZOI_IG_CACHE
/* Drop the cached IG of every candidate that reads cell j through a view whose
 * coverage gate differs between the floor and table planes: that gate is the
 * only use of is_table[j] in candidate_ig (the likelihood is the entry's own).
 * Returns the number of cached IGs dropped. */
static int invalidate_cell(zoi_state_t *st, int j)
{
    const float thr = st->cfg.coverage_threshold;
    const int r = j / ZOI_COLS, c = j % ZOI_COLS;
    int n = 0;
    for (int y = 0; y < ZOI_N_YAW; y++) {
        const zoi_pattern_t *pat = &st->pattern[y];
        for (int i = 0; i < pat->n; i++) {
            const zoi_pattern_entry_t *e = &pat->e[i];
            if (coverage_gate(e->w_table, thr) == coverage_gate(e->w_floor, thr))
                continue;
            const int rc = r - e->dr, cc = c - e->dc; /* the candidate cell that sees j through e */
            if (rc < 0 || rc >= ZOI_ROWS || cc < 0 || cc >= ZOI_COLS)
                continue;
            const int k = (rc * ZOI_COLS + cc) * ZOI_N_YAW + y;
            if (st->cand_eval[k] & 1) {
                st->cand_eval[k] = (uint8_t)(st->cand_eval[k] & ~1u);
                n++;
            }
        }
    }
    return n;
}
#endif

int zoi_plan(zoi_state_t *st, const zoi_pose_t *pose)
{
    /* The controller's table map, as the planner sees it: persistent cells plus,
     * right after a detector frame, that frame's desk cells (_table_cells_from_grid). */
    uint8_t is_table[ZOI_CELLS];
    for (int j = 0; j < ZOI_CELLS; j++)
        is_table[j] = st->table_map[j] || (st->frame_desk_valid && st->frame_desk_cells[j]);

    zoi_refresh_beliefs(st); /* clears ig_cache_valid if it recomputes q */
#if ZOI_IG_CACHE
    /* Keep the exact IGs of the previous plan while q is unchanged, except those
     * a table-map change affects. Counted in the IG stage. */
    ZOI_PROF(ZOI_STAGE_IG, 1);
    plan_arg_t arg = {st, pose, is_table, st->ig_cache_valid};
    st->n_ig_invalidated = 0;
    if (arg.reuse) {
        for (int k = 0; k < ZOI_N_CAND; k++)
            st->cand_eval[k] &= 1;
        for (int j = 0; j < ZOI_CELLS; j++)
            if (is_table[j] != st->ig_cache_table[j])
                st->n_ig_invalidated += invalidate_cell(st, j);
    } else {
        memset(st->cand_eval, 0, sizeof(st->cand_eval));
    }
    memcpy(st->ig_cache_table, is_table, sizeof(st->ig_cache_table));
    ZOI_PROF(ZOI_STAGE_IG, 0);
#else
    memset(st->cand_eval, 0, sizeof(st->cand_eval));
    plan_arg_t arg = {st, pose, is_table, 0};
#endif
    const int best = search(st, &arg);
#if ZOI_IG_CACHE
    st->ig_cache_valid = 1;
#endif

    st->n_ig_evals = st->n_T_evals = 0;
    for (int k = 0; k < ZOI_N_CAND; k++) {
        st->n_ig_evals += (st->cand_eval[k] >> 2) & 1;
        st->n_T_evals += (st->cand_eval[k] >> 1) & 1;
    }
    st->last_best = best;
    return best;
}
