/* zoi_belief.c: observation model, belief update, learned (rho, theta).
 *
 * Port of policies/zoi_model.py and ZOIPolicy._update_beliefs_from_observation /
 * _update_evidence / _refresh_beliefs (policies/zoi.py).
 *
 * States s = surface * 3 + content (surface 0 table, 1 floor; content 0 empty,
 * 1 bottle, 2 can). Observations o = desk * 3 + target (desk 0 none, 1 covered;
 * target 0 nothing, 1 "bottle" box, 2 "can" box).
 *
 * Differences from the Python code, all for float32 and none in the model:
 * - each cell keeps log(ell) instead of ell, normalised to max 0; ell = exp(log ell)
 *   is derived from it (an entry that underflows to 0 in ell keeps its value in
 *   log ell, so it can recover: in float32 ell itself would reach 0 after ~30
 *   frames on a floor cell next to a desk);
 * - log_ev is re-centred to max 0 after every frame (the weights do not change);
 * - beliefs are computed only when they are read (zoi_refresh_beliefs), which is
 *   exact: in Python they are recomputed every frame but only the planner reads them.
 */
#include <math.h>
#include <string.h>

#include "zoi.h"

/* ---- Observation model ------------------------------------------------ */

/* Probability that a present target in a cell with visibility v gives a box. */
static float detect_prob(const zoi_config_t *cfg, float v)
{
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    return fmaxf(cfg->false_alarm, v * cfg->recall);
}

/* L[o][s] = P(desk obs | s) * P(target obs | s); table-plane visibility for the
 * table states, floor-plane visibility for the floor states (joint_likelihood). */
void zoi_joint_likelihood(const zoi_config_t *cfg, float w_table, float w_floor, int desk_in_view,
                          float L[ZOI_NO][ZOI_NS])
{
    const float f = cfg->false_alarm, a = cfg->class_accuracy;
    for (int surf = 0; surf < 2; surf++) {
        const float p = detect_prob(cfg, surf == 0 ? w_table : w_floor);
        const float T[3][3] = {
            {1.0f - f, 1.0f - p, 1.0f - p},
            {0.5f * f, p * a, p * (1.0f - a)},
            {0.5f * f, p * (1.0f - a), p * a},
        };
        const float covered = desk_in_view ? (surf == 0 ? cfg->desk_recall : cfg->desk_false_alarm) : 0.0f;
        const float D[2] = {1.0f - covered, covered};
        for (int d = 0; d < 2; d++)
            for (int t = 0; t < 3; t++)
                for (int c = 0; c < 3; c++)
                    L[d * 3 + t][surf * 3 + c] = D[d] * T[t][c];
    }
}

/* ---- Prior and (rho, theta) grid -------------------------------------- */

/* prior(rho, theta) = base + rho * d_rho + theta * d_theta (joint_prior_terms). */
static void prior_terms(zoi_state_t *st, float pi)
{
    const float b[ZOI_NS] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    const float r[ZOI_NS] = {1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f};
    const float t[ZOI_NS] = {-pi, 0.5f * pi, 0.5f * pi, -(1.0f - pi), 0.5f * (1.0f - pi), 0.5f * (1.0f - pi)};
    memcpy(st->base, b, sizeof(b));
    memcpy(st->d_rho, r, sizeof(r));
    memcpy(st->d_theta, t, sizeof(t));
}

/* Feasible grid points (feasible_grid / floor_only_grid), rho outer, theta inner. */
static int build_grid(zoi_state_t *st)
{
    const int n = ZOI_GRID_N;
    st->n_grid = 0;
    if (!st->cfg.table_context) {
        for (int j = 0; j < n; j++) {
            st->g_rho[j] = 0.0f;
            st->g_theta[j] = ((float)j + 0.5f) / (float)n;
        }
        st->n_grid = n;
        return 0;
    }
    const float pi = st->cfg.p_on_table;
    for (int i = 0; i < n; i++) {
        const float rho = ((float)i + 0.5f) / (float)n;
        for (int j = 0; j < n; j++) {
            const float theta = ((float)j + 0.5f) / (float)n;
            if (rho > theta * pi && rho < 1.0f - theta * (1.0f - pi)) {
                if (st->n_grid >= ZOI_MAX_GRID_POINTS)
                    return -1;
                st->g_rho[st->n_grid] = rho;
                st->g_theta[st->n_grid] = theta;
                st->n_grid++;
            }
        }
    }
    return 0;
}

static void evidence_coefficients(const zoi_state_t *st, const float ell[ZOI_NS], float coef[3])
{
    float a = 0.0f, b = 0.0f, c = 0.0f;
    for (int s = 0; s < ZOI_NS; s++) {
        a += ell[s] * st->base[s];
        b += ell[s] * st->d_rho[s];
        c += ell[s] * st->d_theta[s];
    }
    coef[0] = a;
    coef[1] = b;
    coef[2] = c;
}

static void normalise_weights(zoi_state_t *st)
{
    float m = st->log_ev[0];
    for (int g = 1; g < st->n_grid; g++)
        m = fmaxf(m, st->log_ev[g]);
    float sum = 0.0f;
    for (int g = 0; g < st->n_grid; g++) {
        st->log_ev[g] -= m;
        st->weight[g] = expf(st->log_ev[g]);
        sum += st->weight[g];
    }
    for (int g = 0; g < st->n_grid; g++)
        st->weight[g] /= sum;
}

int zoi_belief_init(zoi_state_t *st)
{
    prior_terms(st, st->cfg.table_context ? st->cfg.p_on_table : 0.0f);
    if (build_grid(st) != 0)
        return -1;
    const float ones[ZOI_NS] = {1, 1, 1, 1, 1, 1};
    for (int j = 0; j < ZOI_CELLS; j++) {
        for (int s = 0; s < ZOI_NS; s++) {
            st->log_ell[j][s] = 0.0f;
            st->ell[j][s] = 1.0f;
        }
        evidence_coefficients(st, ones, st->coef[j]);
        st->seen[j] = 0;
    }
    for (int g = 0; g < st->n_grid; g++)
        st->log_ev[g] = 0.0f; /* unseen cells have Z = 1 */
    normalise_weights(st);
    st->beliefs_valid = 0;
    return 0;
}

/* ---- Belief update ---------------------------------------------------- */

typedef struct {
    zoi_state_t *st;
    int n;
    const int *cells;
    const float (*old_coef)[3];
} evidence_arg_t;

/* log_ev[g] += sum_j log Z_j,new(g) - sum_j log Z_j,old(g) over the updated cells. */
static void evidence_kernel(void *p, int core, int ncores)
{
    evidence_arg_t *arg = (evidence_arg_t *)p;
    zoi_state_t *st = arg->st;
    const int chunk = (st->n_grid + ncores - 1) / ncores;
    const int g0 = core * chunk, g1 = g0 + chunk < st->n_grid ? g0 + chunk : st->n_grid;
    for (int g = g0; g < g1; g++) {
        const float rho = st->g_rho[g], theta = st->g_theta[g];
#if ZOI_REFERENCE_KERNELS
        float s_new = 0.0f, s_old = 0.0f;
        for (int k = 0; k < arg->n; k++) {
            const float *cn = st->coef[arg->cells[k]];
            const float *co = arg->old_coef[k];
            s_new += logf(cn[0] + cn[1] * rho + cn[2] * theta);
            s_old += logf(co[0] + co[1] * rho + co[2] * theta);
        }
        st->log_ev[g] += s_new - s_old;
#else
        /* One log per cell: log(Z_new / Z_old). */
        float ratio_log = 0.0f;
        for (int k = 0; k < arg->n; k++) {
            const float *cn = st->coef[arg->cells[k]];
            const float *co = arg->old_coef[k];
            ratio_log += logf((cn[0] + cn[1] * rho + cn[2] * theta) / (co[0] + co[1] * rho + co[2] * theta));
        }
        st->log_ev[g] += ratio_log;
#endif
    }
}

/* Target observation of a cell: the class of its highest-confidence box (first on ties). */
static int cell_target_obs(const zoi_state_t *st, int cell)
{
    int obs = 0;
    float best = -INFINITY;
    for (int i = 0; i < st->n_tevents; i++) {
        const zoi_target_event_t *e = &st->tevents[i];
        if (e->cell != cell)
            continue;
        if (e->conf > best) {
            best = e->conf;
            obs = e->class_id == ZOI_CLASS_BOTTLE ? 1 : 2;
        }
    }
    return obs;
}

void zoi_belief_update(zoi_state_t *st)
{
    const zoi_config_t *cfg = &st->cfg;
    int cells[ZOI_CELLS];
    float old_coef[ZOI_CELLS][3];
    int n = 0;
    float L[ZOI_NO][ZOI_NS];

    for (int j = 0; j < ZOI_CELLS; j++) {
        if (!st->visible[j])
            continue;
        const float w_table = st->cov_table[j], w_floor = st->cov_floor[j];
        const int in_view = cfg->table_context && w_table > 0.0f && w_table >= cfg->coverage_threshold;
        const int covered = in_view && st->desk_cov[j];
        const int o = (covered ? 3 : 0) + cell_target_obs(st, j);
        zoi_joint_likelihood(cfg, w_table, w_floor, in_view, L);
        float m = -INFINITY;
        for (int s = 0; s < ZOI_NS; s++) {
            st->log_ell[j][s] += logf(L[o][s]);
            m = fmaxf(m, st->log_ell[j][s]);
        }
        for (int s = 0; s < ZOI_NS; s++) {
            st->log_ell[j][s] -= m;
            st->ell[j][s] = expf(st->log_ell[j][s]);
        }
        cells[n] = j;
        memcpy(old_coef[n], st->coef[j], sizeof(old_coef[n]));
        evidence_coefficients(st, st->ell[j], st->coef[j]);
        st->seen[j] = 1;
        n++;
    }
    if (n == 0)
        return;
    evidence_arg_t arg = {st, n, cells, (const float (*)[3])old_coef};
    zoi_fork(evidence_kernel, &arg);
    normalise_weights(st);
    st->beliefs_valid = 0;
}

/* ---- Beliefs ---------------------------------------------------------- */

typedef struct {
    zoi_state_t *st;
    int n_active;
    const int *active;
    const float *w_act;
    float q_unseen[ZOI_NS];
} belief_arg_t;

/* q_j(s) = ell_j(s) * (base_s S0 + d_rho_s S1 + d_theta_s S2),
 * S0 = sum_g w_g / Z_j(g), S1 = sum_g w_g rho_g / Z_j(g), S2 = sum_g w_g theta_g / Z_j(g). */
static void belief_kernel(void *p, int core, int ncores)
{
    belief_arg_t *arg = (belief_arg_t *)p;
    zoi_state_t *st = arg->st;
    const int chunk = (ZOI_CELLS + ncores - 1) / ncores;
    const int j0 = core * chunk, j1 = j0 + chunk < ZOI_CELLS ? j0 + chunk : ZOI_CELLS;
    for (int j = j0; j < j1; j++) {
        if (!st->seen[j]) {
            memcpy(st->q[j], arg->q_unseen, sizeof(arg->q_unseen));
            continue;
        }
        const float *c = st->coef[j];
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
        for (int k = 0; k < arg->n_active; k++) {
            const int g = arg->active[k];
            const float wz = arg->w_act[k] / (c[0] + c[1] * st->g_rho[g] + c[2] * st->g_theta[g]);
            s0 += wz;
            s1 += wz * st->g_rho[g];
            s2 += wz * st->g_theta[g];
        }
        for (int s = 0; s < ZOI_NS; s++)
            st->q[j][s] = st->ell[j][s] * (st->base[s] * s0 + st->d_rho[s] * s1 + st->d_theta[s] * s2);
    }
}

void zoi_refresh_beliefs(zoi_state_t *st)
{
    if (st->beliefs_valid)
        return;
    ZOI_PROF(ZOI_STAGE_BELIEFS, 1);
    static int active[ZOI_MAX_GRID_POINTS];
    static float w_act[ZOI_MAX_GRID_POINTS];
    belief_arg_t arg;
    arg.st = st;

    /* Unseen cells (Z = 1) share one belief, over all grid points. */
    float e_rho = 0.0f, e_theta = 0.0f, w_sum = 0.0f, w_max = 0.0f;
    for (int g = 0; g < st->n_grid; g++) {
        w_sum += st->weight[g];
        e_rho += st->weight[g] * st->g_rho[g];
        e_theta += st->weight[g] * st->g_theta[g];
        w_max = fmaxf(w_max, st->weight[g]);
    }
    for (int s = 0; s < ZOI_NS; s++)
        arg.q_unseen[s] = st->base[s] * w_sum + st->d_rho[s] * e_rho + st->d_theta[s] * e_theta;

    /* Seen cells: grid points with weight above 1e-12 of the largest (the
     * dropped mass changes beliefs by < 1e-8), renormalised. */
    int n_active = 0;
    float act_sum = 0.0f;
    for (int g = 0; g < st->n_grid; g++) {
        if (st->weight[g] > 1e-12f * w_max) {
            active[n_active] = g;
            w_act[n_active] = st->weight[g];
            act_sum += st->weight[g];
            n_active++;
        }
    }
    for (int k = 0; k < n_active; k++)
        w_act[k] /= act_sum;
    arg.n_active = n_active;
    st->n_active = n_active;
    st->n_seen = 0;
    for (int j = 0; j < ZOI_CELLS; j++)
        st->n_seen += st->seen[j];
    arg.active = active;
    arg.w_act = w_act;
    zoi_fork(belief_kernel, &arg);
    st->beliefs_valid = 1;
#if ZOI_IG_CACHE
    st->ig_cache_valid = 0; /* q changed: the planner's cached IGs are stale */
#endif
    ZOI_PROF(ZOI_STAGE_BELIEFS, 0);
}

void zoi_rho_theta(const zoi_state_t *st, float *rho_mean, float *rho_sd, float *theta_mean, float *theta_sd)
{
    float mr = 0.0f, mt = 0.0f;
    for (int g = 0; g < st->n_grid; g++) {
        mr += st->weight[g] * st->g_rho[g];
        mt += st->weight[g] * st->g_theta[g];
    }
    float vr = 0.0f, vt = 0.0f;
    for (int g = 0; g < st->n_grid; g++) {
        const float dr = st->g_rho[g] - mr, dt = st->g_theta[g] - mt;
        vr += st->weight[g] * dr * dr;
        vt += st->weight[g] * dt * dt;
    }
    *rho_mean = mr;
    *rho_sd = sqrtf(fmaxf(0.0f, vr));
    *theta_mean = mt;
    *theta_sd = sqrtf(fmaxf(0.0f, vt));
}
