# GAP8 / gvsoc build of ZOI (GAP SDK, PMSIS). See ../README.md.
#
#   source <gap_sdk>/configs/ai_deck.sh        # once per shell
#   make clean all run platform=gvsoc                 # 8 cores, a plan every frame
#   make clean all run platform=gvsoc FLIGHT=1        # plans as in the recorded flight
#   make clean all run platform=gvsoc NB_CORES=1      # 1 core
#
# Options:
#   NB_CORES     cluster cores used by the parallel kernels (1..8), default 8
#   CL_FREQ_MHZ  cluster frequency used only to print ms, default 260
#   FLIGHT       0: a plan on every frame (worst case, default); 1: plans as in the recorded flight
#   MAX_FRAMES   run only the first N frames of the trace (0: all), for shorter gvsoc runs
#   ROWS, COLS   map size in 0.5 m cells, default 13 x 11 (the Webots room); must match trace_data.h

NB_CORES ?= 8
CL_FREQ_MHZ ?= 260
MAX_FRAMES ?= 0
FLIGHT ?= 0
ROWS ?= 13
COLS ?= 11

APP = zoi
APP_SRCS = main.c \
  ../src/zoi_config.c ../src/zoi_geometry.c ../src/zoi_perception.c \
  ../src/zoi_belief.c ../src/zoi_planner.c ../src/zoi.c
APP_INC = ../include .
APP_CFLAGS = -O3 -DNB_CORES=$(NB_CORES) -DCL_FREQ_MHZ=$(CL_FREQ_MHZ) -DMAX_FRAMES=$(MAX_FRAMES) -DFLIGHT=$(FLIGHT) \
  -DZOI_ROWS=$(ROWS) -DZOI_COLS=$(COLS) -Wall -Wextra -Wdouble-promotion
APP_LDFLAGS = -lm

include $(RULES_DIR)/pmsis_rules.mk
