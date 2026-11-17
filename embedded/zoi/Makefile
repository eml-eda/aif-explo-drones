# Host build of the ZOI C port (float32) and its replay test.
#
#   make            build the replay tests, build/replay_test_<variant>[_igc]:
#                    ref, refprune1, refprune2 (faithful kernels, planner pruning 0 / 1 / 2),
#                    opt, prune1, prune2 (optimised kernels, pruning 0 / 1 / 2);
#                    _igc: the same with ZOI_IG_CACHE=1
#   make test TRACES="a.txt b.txt" [MIN_PLANS=N]   run all of them on reference traces from
#                    tools/c_port/export_trace.py (MIN_PLANS: at least N plans per frame, stress test)
#   make gap8-host [PRUNING=0|1|2] [FLIGHT=0|1] [IG_CACHE=0|1]   build and run gap8/main.c on the PC with
#                    host/pmsis.h (a stand-in for the SDK: cores run one after the other, "cycles" are nanoseconds)
#   make gap8-check [PRUNING=0|1|2] [IG_CACHE=0|1]   compile the library for GAP8 with the PULP
#                    toolchain (no run) and list any double-precision soft-float call
#
# The GAP8 build and gvsoc run are in gap8/ (see README.md).

CC ?= gcc
CFLAGS_LIB = -std=c99 -O2 -Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Werror -ffp-contract=off -Iinclude
SRC = src/zoi_config.c src/zoi_geometry.c src/zoi_perception.c src/zoi_belief.c src/zoi_planner.c src/zoi.c

PULP_CC ?= /opt/pulp-toolchain/bin/riscv32-unknown-elf-gcc
PULP_OBJDUMP ?= /opt/pulp-toolchain/bin/riscv32-unknown-elf-objdump
PULP_SIZE ?= /opt/pulp-toolchain/bin/riscv32-unknown-elf-size
PULP_FLAGS = -march=rv32imcxgap8 -mabi=ilp32 -O3 -std=gnu99 -Wall -Wextra -Wdouble-promotion -Werror -Iinclude -fdata-sections -ffunction-sections

# Kernels and planner pruning of each variant; the _igc twin adds ZOI_IG_CACHE=1.
FLAGS_ref = -DZOI_REFERENCE_KERNELS=1 -DZOI_PLANNER_PRUNING=0
FLAGS_refprune1 = -DZOI_REFERENCE_KERNELS=1 -DZOI_PLANNER_PRUNING=1
FLAGS_refprune2 = -DZOI_REFERENCE_KERNELS=1 -DZOI_PLANNER_PRUNING=2
FLAGS_opt = -DZOI_REFERENCE_KERNELS=0 -DZOI_PLANNER_PRUNING=0
FLAGS_prune1 = -DZOI_REFERENCE_KERNELS=0 -DZOI_PLANNER_PRUNING=1
FLAGS_prune2 = -DZOI_REFERENCE_KERNELS=0 -DZOI_PLANNER_PRUNING=2
REF_VARIANTS = ref refprune1 refprune2 ref_igc refprune1_igc refprune2_igc
OPT_VARIANTS = opt prune1 prune2 opt_igc prune1_igc prune2_igc
VARIANTS = $(REF_VARIANTS) $(OPT_VARIANTS)
all: $(VARIANTS:%=build/replay_test_%)

build:
	mkdir -p build

build/replay_test_%: $(SRC) host/replay_test.c include/*.h | build
	$(CC) $(CFLAGS_LIB) $(FLAGS_$(patsubst %_igc,%,$*)) -DZOI_IG_CACHE=$(if $(filter %_igc,$*),1,0) \
	  -o $@ $(SRC) host/replay_test.c -lm

# Every variant must pass against Python. Per kernel variant, every pruning level,
# with and without the IG cache, must take exactly the decisions of the exhaustive
# planner without cache (ref, opt) on every plan, later plans of a frame included,
# with the same rate and T bit for bit.
MIN_PLANS ?= 0
test: all
	@for t in $(TRACES); do \
	  for v in $(VARIANTS); do ./build/replay_test_$$v $$t -m $(MIN_PLANS) -d build/decisions_$$v.txt > build/out_$$v.txt || { cat build/out_$$v.txt; exit 1; }; done; \
	  for v in $(filter-out ref,$(REF_VARIANTS)); do cmp -s build/decisions_ref.txt build/decisions_$$v.txt || { echo "$$t: $$v decisions differ from ref"; exit 1; }; done; \
	  for v in $(filter-out opt,$(OPT_VARIANTS)); do cmp -s build/decisions_opt.txt build/decisions_$$v.txt || { echo "$$t: $$v decisions differ from opt"; exit 1; }; done; \
	  echo "$$t: PASS, $$(wc -l < build/decisions_opt.txt) plans, decisions identical bit for bit across pruning levels and IG cache"; \
	  echo "    prune2:     $$(grep -h 'later plans' build/out_prune2.txt | sed 's/^ *//')"; \
	  echo "    prune2_igc: $$(grep -h 'later plans' build/out_prune2_igc.txt | sed 's/^ *//')"; \
	done

PRUNING ?= 0
FLIGHT ?= 0
IG_CACHE ?= 0
gap8-host: | build
	$(CC) -std=gnu99 -O2 -Wall -Wextra -Wdouble-promotion -Werror -ffp-contract=off -Ihost -Iinclude -Igap8 \
	  -DNB_CORES=8 -DZOI_REFERENCE_KERNELS=0 -DZOI_PLANNER_PRUNING=$(PRUNING) -DFLIGHT=$(FLIGHT) -DZOI_IG_CACHE=$(IG_CACHE) \
	  -o build/gap8_host gap8/main.c $(SRC) -lm
	./build/gap8_host | grep -v "^[0-9]"

gap8-check: | build
	@for k in 1 0; do \
	  rm -f build/gap8_*.o; \
	  for s in $(SRC); do $(PULP_CC) $(PULP_FLAGS) -DZOI_REFERENCE_KERNELS=$$k -DZOI_PLANNER_PRUNING=$(PRUNING) -DZOI_IG_CACHE=$(IG_CACHE) -c $$s -o build/gap8_$$(basename $$s .c).o || exit 1; done; \
	  echo "== ZOI_REFERENCE_KERNELS=$$k, ZOI_PLANNER_PRUNING=$(PRUNING), ZOI_IG_CACHE=$(IG_CACHE) (text = code, data + bss = static data; zoi_state_t is allocated by the caller)"; \
	  $(PULP_SIZE) -t build/gap8_*.o; \
	  echo "components: belief update = geometry + perception + belief, planner = planner, navigator + API = zoi, shared = config"; \
	  echo "double-precision calls:"; $(PULP_OBJDUMP) -d build/gap8_*.o | grep -oE "<__[a-z]+df[0-9]>" | sort | uniq -c || true; \
	  echo "single-precision soft-float routines used:"; $(PULP_OBJDUMP) -dr build/gap8_*.o | grep -oE "__[a-z]+sf[0-9]|__[a-z]+sf[a-z]+[0-9]?" | sort | uniq -c | tr '\n' ' '; echo; \
	done

clean:
	rm -rf build

.PHONY: all test gap8-host gap8-check clean
