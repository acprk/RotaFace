#!/bin/bash
# Re-run of multi-core scaling on a quiet host (load ~3), with CPU pinning, NUMA first-touch and permuted output.
set -e
B=code/target/release; O=results/p128/rev2/scale.csv
export RF_PARAMS=p128 RF_COMPACT=1
RF_TAG=socket0        taskset -c 0-25 $B/scale B 16 2 4 9 250000 1,2,4,8,16,26 3 $O
RF_TAG=both_default   taskset -c 0-51 $B/scale B 16 2 4 9 250000 32,48,52 3 $O
RF_TAG=both_firsttouch RF_FIRST_TOUCH=1 taskset -c 0-51 $B/scale B 16 2 4 9 250000 1,26,32,48,52 3 $O
RF_TAG=smt_firsttouch  RF_FIRST_TOUCH=1 $B/scale B 16 2 4 9 250000 104 3 $O
RF_TAG=perm_firsttouch RF_FIRST_TOUCH=1 RF_PERM=1 taskset -c 0-51 $B/scale B 16 2 4 9 250000 1,52 3 $O
echo DONE
