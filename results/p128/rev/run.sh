#!/bin/bash
export RF_PARAMS=p128
B=../../../code/target/release
uptime > load.txt
for g in "2 18 1024" "3 12 4096" "4 9 4096" "6 6 4096" "9 4 1024"; do set -- $g
  $B/chain_compact $1 $2 $3 16 16 chain_compact.csv > chain_$1x$2.log 2>&1 &
done
wait
uptime >> load.txt
$B/final_bench 9 100 final_bench_rev.csv > final_bench_rev.log 2>&1
cd ../e2_ident && RF_RERAND=1 RF_COMPACT=1 RF_GADGET=6,6 $B/ident B 16 2 6 6 0,1024 24 COMPACT66_rr > ../rev/ident66.log 2>&1
cd ../rev; uptime >> load.txt; echo done > DONE
