/* zoi_perception.c: detector boxes -> cells, desk map, shadow cells, visible mask, claims.
 *
 * Port of the controller's processed-frame logic (crazyflie_AIF.py main loop,
 * vision/perception_projection.py project_perception_frame for a non-ground-truth
 * detector, env/surface_projection.py, task/claim_manager.py), in the same order:
 *
 *   1. floor- and table-plane coverage of every cell in view;
 *   2. desk boxes: cells hit by the box probes (table plane) join the persistent
 *      table map; the desk event keeps its covered cells;
 *   3. target boxes: projected on the table plane if the box overlaps a desk box
 *      or its table-plane cell is already a table cell, else on the floor plane;
 *   4. desk events: floor-plane counterparts of their probes in front of the
 *      drone become this frame's shadow cells;
 *   5. merged coverage (table plane on table-map and shadow cells), visible mask;
 *   6. claims: a target event with confidence >= claim threshold claims (cell, class).
 */
#include <math.h>
#include <string.h>

#include "zoi.h"

static int is_target_class(int class_id)
{
    return class_id == ZOI_CLASS_BOTTLE || class_id == ZOI_CLASS_CAN;
}

/* Bbox clamped to the image; returns 0 if empty. */
static int clamp_bbox(const zoi_config_t *cfg, const zoi_box_t *b, float out[4])
{
    const float w = (float)cfg->img_w, h = (float)cfg->img_h;
    out[0] = fmaxf(0.0f, fminf(w, b->x1));
    out[2] = fmaxf(0.0f, fminf(w, b->x2));
    out[1] = fmaxf(0.0f, fminf(h, b->y1));
    out[3] = fmaxf(0.0f, fminf(h, b->y2));
    return !(out[2] <= out[0] || out[3] <= out[1]);
}

/* Probe centres over the clamped bbox (env/surface_projection.py _bbox_probe_grid):
 * n per axis = clamp(round(size / 32) + 1, 2, max), round half to even like Python. */
static int probe_centres(const zoi_config_t *cfg, const float bb[4], float px[], float py[], int max_probes)
{
    const float w = fmaxf(1.0f, bb[2] - bb[0]);
    const float h = fmaxf(1.0f, bb[3] - bb[1]);
    const int ms = cfg->bbox_max_samples_per_axis;
    int nx = (int)rintf(w / 32.0f) + 1, ny = (int)rintf(h / 32.0f) + 1;
    nx = nx < ms ? nx : ms;
    ny = ny < ms ? ny : ms;
    nx = nx > 2 ? nx : 2;
    ny = ny > 2 ? ny : 2;
    int n = 0;
    for (int ix = 0; ix < nx; ix++) {
        const float x = bb[0] + (((float)ix + 0.5f) / (float)nx) * w;
        for (int iy = 0; iy < ny; iy++) {
            if (n < max_probes) {
                px[n] = x;
                py[n] = bb[1] + (((float)iy + 0.5f) / (float)ny) * h;
                n++;
            }
        }
    }
    return n;
}

static int cell_of_point(const zoi_state_t *st, const zoi_fov_t *fov, float u, float v, const zoi_pose_t *p)
{
    return zoi_cell_from_bbox(&st->cfg, fov, u, v, u, v, p->x, p->y, p->yaw);
}

/* Forward distance of a cell centre from the drone, along its heading. */
static float forward_distance(const zoi_config_t *cfg, int cell, const zoi_pose_t *p)
{
    const int row = cell / ZOI_COLS, col = cell % ZOI_COLS;
    const float cx = cfg->min_x + ((float)col + 0.5f) * cfg->cell_size;
    const float cy = cfg->min_y + ((float)row + 0.5f) * cfg->cell_size;
    return (cx - p->x) * cosf(p->yaw) + (cy - p->y) * sinf(p->yaw);
}

static int bbox_overlaps_any_desk(const zoi_config_t *cfg, const zoi_box_t *t, const zoi_box_t *boxes, int n)
{
    const float area = fmaxf(1e-6f, fmaxf(0.0f, t->x2 - t->x1) * fmaxf(0.0f, t->y2 - t->y1));
    const float cx = 0.5f * (t->x1 + t->x2), cy = 0.5f * (t->y1 + t->y2);
    for (int i = 0; i < n; i++) {
        const zoi_box_t *d = &boxes[i];
        if (d->class_id != ZOI_CLASS_DESK)
            continue;
        if (d->x1 <= cx && cx <= d->x2 && d->y1 <= cy && cy <= d->y2)
            return 1;
        const float iw = fmaxf(0.0f, fminf(t->x2, d->x2) - fmaxf(t->x1, d->x1));
        const float ih = fmaxf(0.0f, fminf(t->y2, d->y2) - fmaxf(t->y1, d->y1));
        if ((iw * ih) / area >= cfg->desk_overlap_min_frac)
            return 1;
    }
    return 0;
}

/* Front-only floor counterparts of a desk box's probes (sample_front_only_bbox_floor_counterparts_rc). */
static void add_shadow_pair(zoi_state_t *st, const zoi_pose_t *p, float u, float v)
{
    const zoi_config_t *cfg = &st->cfg;
    const int tc = cell_of_point(st, &cfg->fov_table, u, v, p);
    const int fc = cell_of_point(st, &cfg->fov_floor, u, v, p);
    if (tc < 0 || fc < 0)
        return;
    const float tol = cfg->shadow_forward_tol_m;
    const float t_fwd = forward_distance(cfg, tc, p), f_fwd = forward_distance(cfg, fc, p);
    if (t_fwd < -tol || f_fwd < -tol || f_fwd < t_fwd - tol)
        return;
    if (st->cov_floor[fc] > 0.0f) /* only cells in the floor footprint */
        st->shadow[fc] = 1;
}

void zoi_project_frame(zoi_state_t *st, const zoi_pose_t *pose, const zoi_box_t *boxes, int n_boxes)
{
    const zoi_config_t *cfg = &st->cfg;
    zoi_cov_entry_t cov[ZOI_MAX_FOOT];
    float px[ZOI_MAX_PROBE_CELLS], py[ZOI_MAX_PROBE_CELLS];

    if (n_boxes > ZOI_MAX_BOXES)
        n_boxes = ZOI_MAX_BOXES;

    /* 1. Coverage on both planes. */
    memset(st->cov_floor, 0, sizeof(st->cov_floor));
    memset(st->cov_table, 0, sizeof(st->cov_table));
    memset(st->in_view, 0, sizeof(st->in_view));
    int n = zoi_cell_coverages(cfg, &cfg->fov_floor, pose->x, pose->y, pose->yaw, cov, ZOI_MAX_FOOT);
    for (int i = 0; i < n; i++) {
        st->cov_floor[cov[i].cell] = cov[i].frac;
        st->in_view[cov[i].cell] = 1;
    }
    n = zoi_cell_coverages(cfg, &cfg->fov_table, pose->x, pose->y, pose->yaw, cov, ZOI_MAX_FOOT);
    for (int i = 0; i < n; i++) {
        st->cov_table[cov[i].cell] = cov[i].frac;
        st->in_view[cov[i].cell] = 1;
    }

    memset(st->shadow, 0, sizeof(st->shadow));
    memset(st->desk_cov, 0, sizeof(st->desk_cov));
    memset(st->frame_desk_cells, 0, sizeof(st->frame_desk_cells));
    uint8_t desk_event[ZOI_MAX_BOXES];
    memset(desk_event, 0, sizeof(desk_event));

    /* 2. Desk boxes, before the targets of the same frame. */
    for (int b = 0; b < n_boxes; b++) {
        const zoi_box_t *d = &boxes[b];
        if (d->class_id != ZOI_CLASS_DESK)
            continue;
        uint8_t covered[ZOI_CELLS];
        memset(covered, 0, sizeof(covered));
        int any = 0, first = -1;
        float bb[4];
        if (clamp_bbox(cfg, d, bb)) {
            const int np = probe_centres(cfg, bb, px, py, ZOI_MAX_PROBE_CELLS);
            for (int i = 0; i <= np; i++) {
                /* probes, then the clamped bbox centre */
                const float u = i < np ? px[i] : 0.5f * (bb[0] + bb[2]);
                const float v = i < np ? py[i] : 0.5f * (bb[1] + bb[3]);
                const int c = cell_of_point(st, &cfg->fov_table, u, v, pose);
                if (c >= 0) {
                    covered[c] = 1;
                    any = 1;
                }
            }
        }
        /* Primary cell: the raw bbox centre on the table plane, else the first
         * covered cell in (row, col) order, which is the cell index order. */
        int primary = zoi_cell_from_bbox(cfg, &cfg->fov_table, d->x1, d->y1, d->x2, d->y2,
                                         pose->x, pose->y, pose->yaw);
        if (primary < 0 && any) {
            for (int c = 0; c < ZOI_CELLS && first < 0; c++)
                if (covered[c])
                    first = c;
            primary = first;
        }
        if (d->conf >= cfg->desk_conf_threshold) {
            for (int c = 0; c < ZOI_CELLS; c++) {
                if (!covered[c])
                    continue;
                st->table_map[c] = 1;
                if (d->conf > st->table_conf[c])
                    st->table_conf[c] = d->conf;
            }
        }
        if (primary >= 0) {
            /* A desk event: its covered cells are this frame's desk observation,
             * and with its primary cell they join the planner's table map. */
            desk_event[b] = 1;
            for (int c = 0; c < ZOI_CELLS; c++)
                if (covered[c]) {
                    st->desk_cov[c] = 1;
                    st->frame_desk_cells[c] = 1;
                }
            st->frame_desk_cells[primary] = 1;
        }
    }

    /* 3. Target boxes. */
    st->n_tevents = 0;
    for (int b = 0; b < n_boxes; b++) {
        const zoi_box_t *t = &boxes[b];
        if (!is_target_class(t->class_id))
            continue;
        const int fc = zoi_cell_from_bbox(cfg, &cfg->fov_floor, t->x1, t->y1, t->x2, t->y2, pose->x, pose->y, pose->yaw);
        const int tc = zoi_cell_from_bbox(cfg, &cfg->fov_table, t->x1, t->y1, t->x2, t->y2, pose->x, pose->y, pose->yaw);
        const int overlaps = bbox_overlaps_any_desk(cfg, t, boxes, n_boxes);
        const int known = tc >= 0 && st->table_map[tc];
        const int use_table = overlaps || known;
        int cell = use_table ? tc : fc;
        if (cell < 0)
            cell = tc < 0 ? fc : tc;
        if (cell < 0)
            continue;
        zoi_target_event_t *e = &st->tevents[st->n_tevents++];
        e->class_id = t->class_id;
        e->conf = t->conf;
        e->cell = cell;
        e->on_table = use_table;
    }

    /* 4. Shadow cells of this frame's desk events. */
    for (int b = 0; b < n_boxes; b++) {
        const zoi_box_t *d = &boxes[b];
        if (!desk_event[b] || d->conf < cfg->desk_conf_threshold)
            continue;
        float bb[4];
        if (!clamp_bbox(cfg, d, bb))
            continue;
        const int np = probe_centres(cfg, bb, px, py, ZOI_MAX_PROBE_CELLS);
        for (int i = 0; i < np; i++)
            add_shadow_pair(st, pose, px[i], py[i]);
        add_shadow_pair(st, pose, 0.5f * (bb[0] + bb[2]), 0.5f * (bb[1] + bb[3]));
    }

    /* 5. Merged coverage and visible mask. */
    for (int c = 0; c < ZOI_CELLS; c++) {
        const float m = (st->table_map[c] || st->shadow[c]) ? st->cov_table[c] : st->cov_floor[c];
        st->cov_merged[c] = m;
        st->visible[c] = (m > 0.0f && m >= cfg->coverage_threshold);
    }

    /* 6. Claims. */
    st->n_new_claims = 0;
    for (int i = 0; i < st->n_tevents; i++) {
        const zoi_target_event_t *e = &st->tevents[i];
        if (e->conf < cfg->claim_conf_threshold)
            continue;
        const uint8_t bit = (uint8_t)(1u << e->class_id);
        if (st->claimed[e->cell] & bit)
            continue;
        st->claimed[e->cell] |= bit;
        st->new_claim_cell[st->n_new_claims] = e->cell;
        st->new_claim_class[st->n_new_claims] = e->class_id;
        st->n_new_claims++;
    }
    st->frame_desk_valid = 1;
}
