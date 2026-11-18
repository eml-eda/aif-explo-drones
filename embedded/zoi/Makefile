# Host build of the ZOI C port (float32) and its replay test.
#
#   make            build the replay test, build/replay_test
#   make test TRACES="a.txt b.txt" [MIN_PLANS=N]   run it on reference traces from
#                    tools/c_port/export_trace.py (MIN_PLANS: at least N plans per frame, stress test)
#   make gap8-host [FLIGHT=0|1] [ROWS=13 COLS=11] [TRACE=gap8/trace_data.h]   build and run gap8/main.c on the PC with
#                    host/pmsis.h (a stand-in for the SDK: cores run one after the other, "cycles" are nanoseconds)
#   make gap8-check [ROWS=13 COLS=11]   compile the library for GAP8 with the PULP
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

all: build/replay_test

build:
	mkdir -p build

build/replay_test: $(SRC) host/replay_test.c include/*.h | build
	$(CC) $(CFLAGS_LIB) -o $@ $(SRC) host/replay_test.c -lm

# Every trace must pass against Python (beliefs, claims, every first plan of a frame).
MIN_PLANS ?= 0
test: all
	@for t in $(TRACES); do \
	  ./build/replay_test $$t -m $(MIN_PLANS) > build/out.txt || { cat build/out.txt; exit 1; }; \
	  echo "$$t: PASS, $$(grep -h 'planner:' build/out.txt | sed 's/^ *planner: //; s/;.*//')"; \
	  echo "    $$(grep -h 'later plans' build/out.txt | sed 's/^ *//')"; \
	done

FLIGHT ?= 0
ROWS ?= 13
COLS ?= 11
TRACE ?= gap8/trace_data.h
gap8-host: | build
	mkdir -p build/gap8_trace && cp gap8/main.c build/gap8_trace/main.c && cp $(TRACE) build/gap8_trace/trace_data.h
	$(CC) -std=gnu99 -O2 -Wall -Wextra -Wdouble-promotion -Werror -ffp-contract=off -Ihost -Iinclude \
	  -DNB_CORES=8 -DFLIGHT=$(FLIGHT) -DZOI_ROWS=$(ROWS) -DZOI_COLS=$(COLS) \
	  -o build/gap8_host build/gap8_trace/main.c $(SRC) -lm
	./build/gap8_host | grep -v "^[0-9]"

gap8-check: | build
	@rm -f build/gap8_*.o; \
	for s in $(SRC); do $(PULP_CC) $(PULP_FLAGS) -DZOI_ROWS=$(ROWS) -DZOI_COLS=$(COLS) -c $$s -o build/gap8_$$(basename $$s .c).o || exit 1; done; \
	echo "== text = code, data + bss = static data; zoi_state_t is allocated by the caller"; \
	$(PULP_SIZE) -t build/gap8_*.o; \
	echo "components: belief update = geometry + perception + belief, planner = planner, navigator + API = zoi, shared = config"; \
	echo "double-precision calls:"; $(PULP_OBJDUMP) -d build/gap8_*.o | grep -oE "<__[a-z]+df[0-9]>" | sort | uniq -c || true; \
	echo "single-precision soft-float routines used:"; $(PULP_OBJDUMP) -dr build/gap8_*.o | grep -oE "__[a-z]+sf[0-9]|__[a-z]+sf[a-z]+[0-9]?" | sort | uniq -c | tr '\n' ' '; echo

clean:
	rm -rf build

.PHONY: all test gap8-host gap8-check clean
