/* replay_test.c: run the C port on a Python reference trace and compare.
 *
 * The trace comes from tools/v2_c/export_trace.py: logged poses and boxes of one
 * flight, replayed through the Python code, with its geometry, beliefs, (rho, theta)
 * posterior and planner decision at every frame. The C state evolves on its own
 * from the same inputs (only the poses and boxes are fed in).
 *
 *   ./replay_test trace.txt [-v]
 *
 * Exit code 0 when every check passes its tolerance.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zoi_v2.h"

#define TOL_COV 1e-5f      /* coverage */
#define TOL_EDGE 1e-5f     /* how close to a threshold a flip is excused */
#define TOL_Q 1e-4f        /* belief entries */
#define TOL_RT 1e-4f       /* rho / theta mean and sd */
#define TOL_TIE 1e-4f      /* relative rate gap of an excused plan mismatch */

static int verbose = 0;

typedef struct {
    int frames, cov_n, cov_extra, vis_flip, vis_flip_edge;
    float cov_err;
    int tevt_mismatch, set_mismatch[3], claim_mismatch;
    float rt_err, q_err, q_err_frame_max_first_half;
    int plans, plan_same, plan_tie, plan_diff;
    float rate_rel_err, cand_ig_abs, cand_ig_rel, cand_T_err;
    int cand_n;
    float min_log_ell;
} stats_t;

static stats_t S;

static float hexf(const char *tok)
{
    return strtof(tok, NULL);
}

/* Expected per-frame values collected until the frame is closed. */
static uint8_t exp_in_view[ZOI_CELLS];
static int exp_tevt_n;
static zoi_target_event_t exp_tevt[ZOI_MAX_BOXES];
static int exp_claim_n, exp_claim_cell[ZOI_MAX_BOXES], exp_claim_class[ZOI_MAX_BOXES];
static int frame_open = 0, frame_idx = -1, q_refreshed = 0;
static zoi_pose_t frame_pose;

static void close_frame(zoi_state_t *st)
{
    if (!frame_open)
        return;
    /* Cells C sees that Python does not (only tiny coverage is excused). */
    for (int j = 0; j < ZOI_CELLS; j++) {
        if (!exp_in_view[j] && st->in_view[j]) {
            const float m = fmaxf(st->cov_floor[j], st->cov_table[j]);
            if (m > TOL_COV) {
                S.cov_extra++;
                if (verbose)
                    printf("frame %d: C sees cell %d (cov %.3g), Python does not\n", frame_idx, j, (double)m);
            }
        }
    }
    /* Target events, in order. */
    int bad = exp_tevt_n != st->n_tevents;
    for (int i = 0; !bad && i < exp_tevt_n; i++)
        bad = exp_tevt[i].cell != st->tevents[i].cell || exp_tevt[i].class_id != st->tevents[i].class_id ||
              exp_tevt[i].on_table != st->tevents[i].on_table;
    if (bad) {
        S.tevt_mismatch++;
        if (verbose)
            printf("frame %d: target events differ (%d vs %d)\n", frame_idx, exp_tevt_n, st->n_tevents);
    }
    bad = exp_claim_n != st->n_new_claims;
    for (int i = 0; !bad && i < exp_claim_n; i++)
        bad = exp_claim_cell[i] != st->new_claim_cell[i] || exp_claim_class[i] != st->new_claim_class[i];
    if (bad) {
        S.claim_mismatch++;
        if (verbose)
            printf("frame %d: claims differ\n", frame_idx);
    }
    frame_open = 0;
}

/* "TAG n cx cy ..." against a C cell mask. */
static void compare_set(const char *line, const uint8_t *mask, int which)
{
    uint8_t exp[ZOI_CELLS];
    memset(exp, 0, sizeof(exp));
    char *p = strchr(line, ' ');
    int n = (int)strtol(p, &p, 10);
    for (int i = 0; i < n; i++) {
        const int cx = (int)strtol(p, &p, 10), cy = (int)strtol(p, &p, 10);
        exp[cy * ZOI_COLS + cx] = 1;
    }
    for (int j = 0; j < ZOI_CELLS; j++) {
        if ((exp[j] != 0) != (mask[j] != 0)) {
            S.set_mismatch[which]++;
            if (verbose)
                printf("frame %d: %.6s cell (%d,%d) Python %d C %d\n", frame_idx, line, j % ZOI_COLS, j / ZOI_COLS,
                       exp[j], mask[j]);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s trace.txt [-v]\n", argv[0]);
        return 2;
    }
    verbose = argc > 2 && strcmp(argv[2], "-v") == 0;
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror(argv[1]);
        return 2;
    }
    static char line[16384];
    static zoi_state_t st;
    zoi_config_t cfg;
    char env[32] = "", det[32] = "";
    int initialised = 0;
    zoi_box_t boxes[ZOI_MAX_BOXES];

    while (fgets(line, sizeof(line), f)) {
        char *tok[16];
        int nt = 0;
        char copy[sizeof(line)];
        strcpy(copy, line);
        for (char *t = strtok(copy, " \n"); t && nt < 16; t = strtok(NULL, " \n"))
            tok[nt++] = t;
        if (nt == 0)
            continue;
        const char *tag = tok[0];

        if (!strcmp(tag, "ENV")) {
            strncpy(env, tok[1], sizeof(env) - 1);
            strncpy(det, tok[2], sizeof(det) - 1);
        } else if (!strcmp(tag, "PARAMS")) {
            const zoi_env_t e = !strcmp(env, "easy") ? ZOI_ENV_EASY : !strcmp(env, "medium") ? ZOI_ENV_MEDIUM : ZOI_ENV_HARD;
            const zoi_det_t d = !strcmp(det, "yolo_nano") ? ZOI_DET_YOLO_NANO : ZOI_DET_MOBILENET;
            zoi_config_default(&cfg, e, d);
            /* The trace's parameters win (a run may override pi, e.g. the pi sweep). */
            const float r = hexf(tok[1]), a = hexf(tok[2]), fa = hexf(tok[3]), dr = hexf(tok[4]), dg = hexf(tok[5]);
            const float pi = hexf(tok[6]);
            const int grid = atoi(tok[7]), ctx = atoi(tok[8]);
            if (fabsf(r - cfg.recall) > 1e-6f || fabsf(a - cfg.class_accuracy) > 1e-6f || fabsf(fa - cfg.false_alarm) > 1e-9f ||
                (ctx && (fabsf(dr - cfg.desk_recall) > 1e-6f || fabsf(dg - cfg.desk_false_alarm) > 1e-6f)))
                printf("note: trace calibration differs from the C defaults; using the trace values\n");
            cfg.recall = r;
            cfg.class_accuracy = a;
            cfg.false_alarm = fa;
            cfg.table_context = ctx;
            if (ctx) {
                cfg.desk_recall = dr;
                cfg.desk_false_alarm = dg;
                cfg.p_on_table = pi;
            }
            if (grid != ZOI_GRID_N) {
                fprintf(stderr, "trace grid %d, C compiled with ZOI_GRID_N %d\n", grid, ZOI_GRID_N);
                return 2;
            }
            const int rc = zoi_init(&st, &cfg);
            if (rc != 0) {
                fprintf(stderr, "zoi_init failed (%d)\n", rc);
                return 2;
            }
            initialised = 1;
            printf("%s %s/%s: pi %.2f, grid %d (%d points), %d pattern entries, sizeof(zoi_state_t) %zu B\n",
                   argv[1], env, det, (double)cfg.p_on_table, ZOI_GRID_N, st.n_grid, st.n_pattern_entries,
                   sizeof(zoi_state_t));
        } else if (!strcmp(tag, "FRAME")) {
            close_frame(&st);
            if (!initialised)
                return 2;
            frame_idx = atoi(tok[1]);
            frame_pose.x = hexf(tok[3]);
            frame_pose.y = hexf(tok[4]);
            frame_pose.yaw = hexf(tok[5]);
            const int nb = atoi(tok[6]);
            for (int b = 0; b < nb; b++) {
                if (!fgets(line, sizeof(line), f))
                    return 2;
                char *p = line + 4;
                zoi_box_t *bx = &boxes[b < ZOI_MAX_BOXES ? b : ZOI_MAX_BOXES - 1];
                bx->class_id = (int)strtol(p, &p, 10);
                bx->conf = strtof(p, &p);
                bx->x1 = strtof(p, &p);
                bx->y1 = strtof(p, &p);
                bx->x2 = strtof(p, &p);
                bx->y2 = strtof(p, &p);
            }
            zoi_perception_frame(&st, &frame_pose, boxes, nb < ZOI_MAX_BOXES ? nb : ZOI_MAX_BOXES);
            for (int j = 0; j < ZOI_CELLS; j++)
                for (int s = 0; s < ZOI_NS; s++)
                    S.min_log_ell = fminf(S.min_log_ell, st.log_ell[j][s]);
            memset(exp_in_view, 0, sizeof(exp_in_view));
            exp_tevt_n = exp_claim_n = 0;
            frame_open = 1;
            q_refreshed = 0;
            S.frames++;
        } else if (!strcmp(tag, "COV")) {
            const int j = atoi(tok[2]) * ZOI_COLS + atoi(tok[1]);
            const float fl = hexf(tok[3]), tb = hexf(tok[4]), mg = hexf(tok[5]);
            const int vis = atoi(tok[6]);
            exp_in_view[j] = 1;
            S.cov_n++;
            S.cov_err = fmaxf(S.cov_err, fabsf(fl - st.cov_floor[j]));
            S.cov_err = fmaxf(S.cov_err, fabsf(tb - st.cov_table[j]));
            S.cov_err = fmaxf(S.cov_err, fabsf(mg - st.cov_merged[j]));
            if (vis != st.visible[j]) {
                if (fabsf(mg - cfg.coverage_threshold) < TOL_EDGE)
                    S.vis_flip_edge++;
                else
                    S.vis_flip++;
                if (verbose)
                    printf("frame %d: visible cell %d Python %d C %d (merged %.7f)\n", frame_idx, j, vis, st.visible[j],
                           (double)mg);
            }
        } else if (!strcmp(tag, "TEVT")) {
            if (exp_tevt_n < ZOI_MAX_BOXES) {
                zoi_target_event_t *e = &exp_tevt[exp_tevt_n++];
                e->class_id = atoi(tok[1]);
                e->conf = hexf(tok[2]);
                e->cell = atoi(tok[4]) * ZOI_COLS + atoi(tok[3]);
                e->on_table = atoi(tok[5]);
            }
        } else if (!strcmp(tag, "DESKCOV")) {
            compare_set(line, st.desk_cov, 0);
        } else if (!strcmp(tag, "SHADOW")) {
            compare_set(line, st.shadow, 1);
        } else if (!strcmp(tag, "TMAP")) {
            compare_set(line, st.table_map, 2);
        } else if (!strcmp(tag, "CLAIM")) {
            if (exp_claim_n < ZOI_MAX_BOXES) {
                exp_claim_cell[exp_claim_n] = atoi(tok[2]) * ZOI_COLS + atoi(tok[1]);
                exp_claim_class[exp_claim_n] = atoi(tok[3]);
                exp_claim_n++;
            }
        } else if (!strcmp(tag, "RT")) {
            float rm, rs, tm, ts;
            zoi_rho_theta(&st, &rm, &rs, &tm, &ts);
            const float e = fmaxf(fmaxf(fabsf(rm - hexf(tok[1])), fabsf(rs - hexf(tok[2]))),
                                  fmaxf(fabsf(tm - hexf(tok[3])), fabsf(ts - hexf(tok[4]))));
            S.rt_err = fmaxf(S.rt_err, e);
            if (verbose && e > TOL_RT)
                printf("frame %d: rho/theta differ by %.3g\n", frame_idx, (double)e);
        } else if (!strcmp(tag, "Q")) {
            if (!q_refreshed) {
                zoi_refresh_beliefs(&st);
                q_refreshed = 1;
            }
            const int j = atoi(tok[2]) * ZOI_COLS + atoi(tok[1]);
            for (int s = 0; s < ZOI_NS; s++) {
                const float e = fabsf(hexf(tok[3 + s]) - st.q[j][s]);
                if (e > S.q_err && verbose && e > TOL_Q)
                    printf("frame %d: q cell %d state %d: Python %.6g C %.6g\n", frame_idx, j, s,
                           (double)hexf(tok[3 + s]), (double)st.q[j][s]);
                S.q_err = fmaxf(S.q_err, e);
            }
        } else if (!strcmp(tag, "PLAN")) {
            const int k_py = ((atoi(tok[4]) * ZOI_COLS + atoi(tok[3])) * ZOI_N_YAW) + atoi(tok[5]);
            const float rate_py = hexf(tok[6]);
            const int k_c = zoi_plan(&st, &frame_pose);
            const float rate_c = st.cand_ig[k_c] / st.cand_T[k_c];
            S.plans++;
            if (rate_py > 0.0f)
                S.rate_rel_err = fmaxf(S.rate_rel_err, fabsf(rate_c - rate_py) / rate_py);
            if (k_c == k_py) {
                S.plan_same++;
            } else {
                const float rate_c_at_py = st.cand_ig[k_py] / st.cand_T[k_py];
                const float gap = (rate_c - rate_c_at_py) / fmaxf(rate_c, 1e-30f);
                if (gap < TOL_TIE)
                    S.plan_tie++;
                else
                    S.plan_diff++;
                if (verbose || gap >= TOL_TIE)
                    printf("frame %d: plan Python k=%d C k=%d, relative rate gap %.3g%s\n", frame_idx, k_py, k_c,
                           (double)gap, gap < TOL_TIE ? " (tie)" : "");
            }
        } else if (!strcmp(tag, "CAND")) {
            const int k = atoi(tok[1]);
            const float ig = hexf(tok[2]), T = hexf(tok[3]);
            S.cand_n++;
            /* Relative error where IG is not tiny; absolute error everywhere
             * (float32 cancellation dominates for IG near 0). */
            const float ig_abs = fabsf(ig - st.cand_ig[k]);
            S.cand_ig_abs = fmaxf(S.cand_ig_abs, ig_abs);
            if (ig >= 1e-3f)
                S.cand_ig_rel = fmaxf(S.cand_ig_rel, ig_abs / ig);
            S.cand_T_err = fmaxf(S.cand_T_err, fabsf(T - st.cand_T[k]));
        } else if (!strcmp(tag, "END")) {
            close_frame(&st);
        }
    }
    fclose(f);

    const int set_bad = S.set_mismatch[0] + S.set_mismatch[1] + S.set_mismatch[2];
    printf("frames %d\n", S.frames);
    printf("  coverage: %d cells, max |err| %.2e; cells only C sees %d; visible flips %d (+%d within %.0e of the gate)\n",
           S.cov_n, (double)S.cov_err, S.cov_extra, S.vis_flip, S.vis_flip_edge, (double)TOL_EDGE);
    printf("  target-event frames differing %d; desk-covered / shadow / table-map cell mismatches %d / %d / %d; claim frames differing %d\n",
           S.tevt_mismatch, S.set_mismatch[0], S.set_mismatch[1], S.set_mismatch[2], S.claim_mismatch);
    printf("  belief: max |q err| %.2e; rho/theta max |err| %.2e; min log ell %.1f (a linear float32 ell underflows below -87.3)\n",
           (double)S.q_err, (double)S.rt_err, (double)S.min_log_ell);
    printf("  planner: %d plans, same %d, ties %d, different %d; best-rate rel err %.2e; candidates %d: IG abs err %.2e (rel %.2e where IG >= 1e-3), T err %.2e s\n",
           S.plans, S.plan_same, S.plan_tie, S.plan_diff, (double)S.rate_rel_err, S.cand_n, (double)S.cand_ig_abs, (double)S.cand_ig_rel,
           (double)S.cand_T_err);
    const int ok = S.cov_err <= TOL_COV && S.cov_extra == 0 && S.vis_flip == 0 && S.tevt_mismatch == 0 && set_bad == 0 &&
                   S.claim_mismatch == 0 && S.q_err <= TOL_Q && S.rt_err <= TOL_RT && S.plan_diff == 0;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
