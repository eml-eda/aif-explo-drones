/* host/pmsis.h: a host stand-in for the few PMSIS calls gap8/main.c uses.
 *
 * Lets gap8/main.c build and run on a PC (make gap8-host), to check it before
 * going to gvsoc: the cluster task runs as a plain call, pi_cl_team_fork runs the
 * cores one after the other (so the work split over cores is exercised), and
 * "cycles" are nanoseconds of the host clock. Not the GAP SDK: on the real target
 * the SDK's pmsis.h is used instead.
 */
#ifndef ZOI_HOST_PMSIS_H
#define ZOI_HOST_PMSIS_H

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

struct pi_device {
    int unused;
};
struct pi_cluster_conf {
    int id;
};
struct pi_cluster_task {
    void (*entry)(void *);
    void *arg;
    int stack_size, slave_stack_size;
};

static int host_core_id = 0;

static inline void pi_cluster_conf_init(struct pi_cluster_conf *c) { c->id = 0; }
static inline void pi_open_from_conf(struct pi_device *d, void *c) { (void)d; (void)c; }
static inline int pi_cluster_open(struct pi_device *d) { (void)d; return 0; }
static inline void pi_cluster_close(struct pi_device *d) { (void)d; }
static inline struct pi_cluster_task *pi_cluster_task(struct pi_cluster_task *t, void (*entry)(void *), void *arg)
{
    t->entry = entry;
    t->arg = arg;
    return t;
}
static inline int pi_cluster_send_task_to_cl(struct pi_device *d, struct pi_cluster_task *t)
{
    (void)d;
    t->entry(t->arg);
    return 0;
}
static inline uint32_t pi_core_id(void) { return (uint32_t)host_core_id; }
static inline void pi_cl_team_fork(int n, void (*entry)(void *), void *arg)
{
    for (int c = 0; c < n; c++) {
        host_core_id = c;
        entry(arg);
    }
    host_core_id = 0;
}

#define PI_PERF_CYCLES 0
static inline void pi_perf_conf(unsigned events) { (void)events; }
static inline void pi_perf_reset(void) {}
static inline void pi_perf_start(void) {}
static inline uint32_t pi_perf_read(int event)
{
    (void)event;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec);
}

static inline void pmsis_exit(int code) { exit(code); }
static inline int pmsis_kickoff(void *fn)
{
    ((void (*)(void))fn)();
    return 0;
}

#endif
