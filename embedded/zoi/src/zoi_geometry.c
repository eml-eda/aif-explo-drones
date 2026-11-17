/* zoi_geometry.c: camera footprint, cell coverage, bbox -> cell.
 *
 * Port of env/env_manager.py (GridEnvManager). The footprint is a rectangle in
 * the body frame, rotated by yaw; a cell's coverage is the area of its
 * intersection with the footprint (Sutherland-Hodgman clipping) over the cell
 * area. Polygons are expressed relative to the cell's lower corner, which keeps
 * float32 products small (the Python code works in world coordinates in float64).
 */
#include <math.h>

#include "zoi.h"

#define MAX_POLY 12

typedef struct {
    float x, y;
} pt_t;

int zoi_cell_from_xy(const zoi_config_t *cfg, float x, float y)
{
    const float max_x = cfg->min_x + (float)ZOI_COLS * cfg->cell_size;
    const float max_y = cfg->min_y + (float)ZOI_ROWS * cfg->cell_size;
    if (x < cfg->min_x || x >= max_x || y < cfg->min_y || y >= max_y)
        return -1;
    const int col = (int)((x - cfg->min_x) / cfg->cell_size);
    const int row = (int)((y - cfg->min_y) / cfg->cell_size);
    if (row < 0 || row >= ZOI_ROWS || col < 0 || col >= ZOI_COLS)
        return -1;
    return row * ZOI_COLS + col;
}

/* Footprint corners (front-left, front-right, back-right, back-left) relative to (ox, oy). */
static void fov_polygon(const zoi_fov_t *fov, float x, float y, float c, float s, float ox, float oy, pt_t out[4])
{
    const float bx[4] = {fov->forward, fov->forward, -fov->backward, -fov->backward};
    const float by[4] = {fov->half_left, -fov->half_right, -fov->half_right, fov->half_left};
    const float tx = x - ox, ty = y - oy;
    for (int i = 0; i < 4; i++) {
        out[i].x = tx + c * bx[i] - s * by[i];
        out[i].y = ty + s * bx[i] + c * by[i];
    }
}

static int inside(pt_t p, pt_t a, pt_t b)
{
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) >= 0.0f;
}

static pt_t intersection(pt_t p1, pt_t p2, pt_t a, pt_t b)
{
    const float den = (p1.x - p2.x) * (a.y - b.y) - (p1.y - p2.y) * (a.x - b.x);
    if (fabsf(den) < 1e-12f)
        return p2; /* nearly parallel; same fallback as the Python code */
    const float c12 = p1.x * p2.y - p1.y * p2.x;
    const float cab = a.x * b.y - a.y * b.x;
    pt_t r;
    r.x = (c12 * (a.x - b.x) - (p1.x - p2.x) * cab) / den;
    r.y = (c12 * (a.y - b.y) - (p1.y - p2.y) * cab) / den;
    return r;
}

/* Clip the convex subject polygon by the axis-aligned square [0, cs]^2 (CCW). */
static int clip_to_cell(const pt_t *subject, int n, float cs, pt_t *out)
{
    const pt_t clip[4] = {{0.0f, 0.0f}, {cs, 0.0f}, {cs, cs}, {0.0f, cs}};
    pt_t buf_a[MAX_POLY], buf_b[MAX_POLY];
    pt_t *input = buf_a, *output = buf_b;
    int n_out = n;
    for (int i = 0; i < n; i++)
        output[i] = subject[i];
    for (int e = 0; e < 4; e++) {
        const pt_t a = clip[e], b = clip[(e + 1) % 4];
        pt_t *tmp = input;
        input = output;
        output = tmp;
        const int n_in = n_out;
        n_out = 0;
        if (n_in == 0)
            break;
        pt_t s = input[n_in - 1];
        for (int i = 0; i < n_in; i++) {
            const pt_t p = input[i];
            if (inside(p, a, b)) {
                if (!inside(s, a, b))
                    output[n_out++] = intersection(s, p, a, b);
                output[n_out++] = p;
            } else if (inside(s, a, b)) {
                output[n_out++] = intersection(s, p, a, b);
            }
            s = p;
        }
    }
    for (int i = 0; i < n_out; i++)
        out[i] = output[i];
    return n_out;
}

static float polygon_area(const pt_t *p, int n)
{
    if (n < 3)
        return 0.0f;
    float area = 0.0f;
    for (int i = 0; i < n; i++) {
        const pt_t a = p[i], b = p[(i + 1) % n];
        area += a.x * b.y - b.x * a.y;
    }
    return 0.5f * area;
}

static float cell_coverage(const zoi_config_t *cfg, const zoi_fov_t *fov, float x, float y, float c, float s,
                           int row, int col)
{
    const float cs = cfg->cell_size;
    const float ox = cfg->min_x + (float)col * cs, oy = cfg->min_y + (float)row * cs;
    pt_t poly[4], inter[MAX_POLY];
    fov_polygon(fov, x, y, c, s, ox, oy, poly);
    const int n = clip_to_cell(poly, 4, cs, inter);
    if (n == 0)
        return 0.0f;
    const float area = fabsf(polygon_area(inter, n));
    if (area <= 0.0f)
        return 0.0f;
    float frac = area / (cs * cs);
    if (frac > 1.0f)
        frac = 1.0f;
    return frac;
}

int zoi_cell_coverages(const zoi_config_t *cfg, const zoi_fov_t *fov, float x, float y, float yaw,
                       zoi_cov_entry_t *out, int max_out)
{
    const float c = cosf(yaw), s = sinf(yaw);
    const float cs = cfg->cell_size;
    /* Bounding box of the footprint in world coordinates. */
    pt_t poly[4];
    fov_polygon(fov, x, y, c, s, 0.0f, 0.0f, poly);
    float minx = poly[0].x, maxx = poly[0].x, miny = poly[0].y, maxy = poly[0].y;
    for (int i = 1; i < 4; i++) {
        minx = fminf(minx, poly[i].x);
        maxx = fmaxf(maxx, poly[i].x);
        miny = fminf(miny, poly[i].y);
        maxy = fmaxf(maxy, poly[i].y);
    }
    int c0 = (int)floorf((minx - cfg->min_x) / cs), c1 = (int)floorf((maxx - cfg->min_x) / cs);
    int r0 = (int)floorf((miny - cfg->min_y) / cs), r1 = (int)floorf((maxy - cfg->min_y) / cs);
    if (c0 < 0) c0 = 0;
    if (r0 < 0) r0 = 0;
    if (c1 > ZOI_COLS - 1) c1 = ZOI_COLS - 1;
    if (r1 > ZOI_ROWS - 1) r1 = ZOI_ROWS - 1;

    int n = 0;
    for (int row = r0; row <= r1; row++) {
        for (int col = c0; col <= c1; col++) {
            const float frac = cell_coverage(cfg, fov, x, y, c, s, row, col);
            if (frac > 0.0f && n < max_out) {
                out[n].cell = row * ZOI_COLS + col;
                out[n].frac = frac;
                n++;
            }
        }
    }
    return n;
}

float zoi_one_cell_coverage(const zoi_config_t *cfg, const zoi_fov_t *fov, float x, float y, float yaw,
                            int row, int col)
{
    return cell_coverage(cfg, fov, x, y, cosf(yaw), sinf(yaw), row, col);
}

/* Bbox centre pixel -> body-frame point on the footprint (linear across it) -> world -> cell. */
int zoi_cell_from_bbox(const zoi_config_t *cfg, const zoi_fov_t *fov, float x1, float y1, float x2, float y2,
                       float drone_x, float drone_y, float drone_yaw)
{
    const float un = (0.5f * (x1 + x2)) / (float)cfg->img_w;
    const float vn = (0.5f * (y1 + y2)) / (float)cfg->img_h;
    const float y_body = (1.0f - un) * fov->half_left + un * (-fov->half_right);
    const float x_body = (1.0f - vn) * fov->forward + vn * (-fov->backward);
    const float c = cosf(drone_yaw), s = sinf(drone_yaw);
    const float wx = drone_x + c * x_body - s * y_body;
    const float wy = drone_y + s * x_body + c * y_body;
    return zoi_cell_from_xy(cfg, wx, wy);
}
