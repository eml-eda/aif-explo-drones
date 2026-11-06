# Host build of the aif_zoi_v2 C port (float32) and its replay test.
#
#   make            build build/replay_test_ref (faithful kernels) and build/replay_test_opt (optimised)
#   make test TRACES="a.txt b.txt"   run both on reference traces from tools/v2_c/export_trace.py
#   make gap8-host   build and run gap8/main.c on the PC with host/pmsis.h (a stand-in for
#                    the SDK: cores run one after the other, "cycles" are nanoseconds)
#   make gap8-check  compile the library for GAP8 with the PULP toolchain (no run) and list
#                    any double-precision soft-float call
#
# The GAP8 build and gvsoc run are in gap8/ (see GVSOC_GUIDE.md).

CC ?= gcc
CFLAGS_LIB = -std=c99 -O2 -Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Werror -ffp-contract=off -Iinclude
SRC = src/zoi_v2_config.c src/zoi_v2_geometry.c src/zoi_v2_perception.c src/zoi_v2_belief.c src/zoi_v2_planner.c src/zoi_v2.c

PULP_CC ?= /opt/pulp-toolchain/bin/riscv32-unknown-elf-gcc
PULP_OBJDUMP ?= /opt/pulp-toolchain/bin/riscv32-unknown-elf-objdump
PULP_SIZE ?= /opt/pulp-toolchain/bin/riscv32-unknown-elf-size
PULP_FLAGS = -march=rv32imcxgap8 -mabi=ilp32 -O3 -std=gnu99 -Wall -Wextra -Wdouble-promotion -Werror -Iinclude -fdata-sections -ffunction-sections

all: build/replay_test_ref build/replay_test_opt

build:
	mkdir -p build

build/replay_test_ref: $(SRC) host/replay_test.c include/*.h | build
	$(CC) $(CFLAGS_LIB) -DZOI_V2_REFERENCE_KERNELS=1 -o $@ $(SRC) host/replay_test.c -lm

build/replay_test_opt: $(SRC) host/replay_test.c include/*.h | build
	$(CC) $(CFLAGS_LIB) -DZOI_V2_REFERENCE_KERNELS=0 -o $@ $(SRC) host/replay_test.c -lm

test: all
	@for t in $(TRACES); do ./build/replay_test_ref $$t || exit 1; ./build/replay_test_opt $$t || exit 1; done

gap8-host: | build
	$(CC) -std=gnu99 -O2 -Wall -Wextra -Wdouble-promotion -Werror -ffp-contract=off -Ihost -Iinclude -Igap8 \
	  -DNB_CORES=8 -DZOI_V2_REFERENCE_KERNELS=0 -o build/gap8_host gap8/main.c $(SRC) -lm
	./build/gap8_host | tail -9

gap8-check: | build
	@for k in 1 0; do \
	  rm -f build/gap8_*.o; \
	  for s in $(SRC); do $(PULP_CC) $(PULP_FLAGS) -DZOI_V2_REFERENCE_KERNELS=$$k -c $$s -o build/gap8_$$(basename $$s .c).o || exit 1; done; \
	  echo "== ZOI_V2_REFERENCE_KERNELS=$$k"; $(PULP_SIZE) -t build/gap8_*.o | tail -1; \
	  echo "double-precision calls:"; $(PULP_OBJDUMP) -d build/gap8_*.o | grep -oE "<__[a-z]+df[0-9]>" | sort | uniq -c || true; \
	  echo "single-precision soft-float routines used:"; $(PULP_OBJDUMP) -dr build/gap8_*.o | grep -oE "__[a-z]+sf[0-9]|__[a-z]+sf[a-z]+[0-9]?" | sort | uniq -c | tr '\n' ' '; echo; \
	done

clean:
	rm -rf build

.PHONY: all test gap8-host gap8-check clean
