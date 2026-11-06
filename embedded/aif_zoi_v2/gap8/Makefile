# GAP8 / gvsoc build of aif_zoi_v2 (GAP SDK, PMSIS). See ../GVSOC_GUIDE.md.
#
#   source <gap_sdk>/configs/ai_deck.sh        # once per shell
#   make clean all run platform=gvsoc         # 8 cores, optimised kernels
#   make clean all run platform=gvsoc NB_CORES=1 REFERENCE=1
#
# Options:
#   NB_CORES     cluster cores used by the parallel kernels (1..8), default 8
#   REFERENCE    1: faithful kernels (same formulas as Python), 0: optimised (default)
#   CL_FREQ_MHZ  cluster frequency used only to print ms, default 175

NB_CORES ?= 8
REFERENCE ?= 0
CL_FREQ_MHZ ?= 175

APP = zoi_v2
APP_SRCS = main.c \
  ../src/zoi_v2_config.c ../src/zoi_v2_geometry.c ../src/zoi_v2_perception.c \
  ../src/zoi_v2_belief.c ../src/zoi_v2_planner.c ../src/zoi_v2.c
APP_INC = ../include .
APP_CFLAGS = -O3 -DNB_CORES=$(NB_CORES) -DZOI_V2_REFERENCE_KERNELS=$(REFERENCE) -DCL_FREQ_MHZ=$(CL_FREQ_MHZ) \
  -Wall -Wextra -Wdouble-promotion
APP_LDFLAGS = -lm

include $(RULES_DIR)/pmsis_rules.mk
