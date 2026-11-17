# GAP8 / gvsoc build of aif_zoi_v2 (GAP SDK, PMSIS). See ../GVSOC_GUIDE.md.
#
#   source <gap_sdk>/configs/ai_deck.sh        # once per shell
#   make clean all run platform=gvsoc         # 8 cores, optimised kernels
#   make clean all run platform=gvsoc NB_CORES=1 REFERENCE=1
#
# Options:
#   NB_CORES     cluster cores used by the parallel kernels (1..8), default 8
#   REFERENCE    1: faithful kernels (same formulas as Python), 0: optimised (default)
#   CL_FREQ_MHZ  cluster frequency used only to print ms, default 260
#   PRUNING      planner search: 0 exhaustive (default), 1 skip hopeless flight times, 2 also skip hopeless IGs
#   FLIGHT       0: a plan on every frame (worst case, default); 1: plans as in the recorded flight
#   IG_CACHE     1: reuse the exact IGs between plans while the beliefs are unchanged (same decisions), 0: off (default)
#   MAX_FRAMES   run only the first N frames of the trace (0: all), for shorter gvsoc runs

NB_CORES ?= 8
REFERENCE ?= 0
CL_FREQ_MHZ ?= 260
MAX_FRAMES ?= 0
PRUNING ?= 0
FLIGHT ?= 0
IG_CACHE ?= 0

APP = zoi_v2
APP_SRCS = main.c \
  ../src/zoi_v2_config.c ../src/zoi_v2_geometry.c ../src/zoi_v2_perception.c \
  ../src/zoi_v2_belief.c ../src/zoi_v2_planner.c ../src/zoi_v2.c
APP_INC = ../include .
APP_CFLAGS = -O3 -DNB_CORES=$(NB_CORES) -DZOI_V2_REFERENCE_KERNELS=$(REFERENCE) -DCL_FREQ_MHZ=$(CL_FREQ_MHZ) -DMAX_FRAMES=$(MAX_FRAMES) -DZOI_V2_PLANNER_PRUNING=$(PRUNING) -DFLIGHT=$(FLIGHT) \
  -DZOI_V2_IG_CACHE=$(IG_CACHE) -Wall -Wextra -Wdouble-promotion
APP_LDFLAGS = -lm

include $(RULES_DIR)/pmsis_rules.mk
