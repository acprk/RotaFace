#!/bin/bash
export RF_PARAMS=p128
B=../../../code/target/release
uptime > load.txt
$B/final_bench 9 100 final_bench.csv > final_bench.log 2>&1
RF_COMPACT=1 $B/scale B 16 2 4 9 250000 1,2,4,8,16,24,32,48 3 scale_final.csv > scale_final.log 2>&1
uptime >> load.txt
echo done > DONE
