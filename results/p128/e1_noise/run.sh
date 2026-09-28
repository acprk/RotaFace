#!/bin/bash
export RF_PARAMS=p128
BIN=../../../code/target/release/noise
mkdir -p parts
for r in "4 15" "5 12" "6 10" "8 7" "10 6" "12 5"; do set -- $r
  $BIN B 16 2 $1 $2 1024 32 32 parts/B_r$1x$2.csv raw_B_r$1x$2 > parts/B_r$1x$2.log 2>&1 &
done
for c in "8 4 4 15" "4 8 4 15" "4 8 8 7" "2 16 4 15"; do set -- $c
  $BIN A $1 $2 $3 $4 256 8 8 parts/A_q$1x$2_r$3x$4.csv > parts/A_q$1x$2_r$3x$4.log 2>&1 &
done
wait
head -1 parts/B_r4x15.csv > noise.csv; for f in parts/*.csv; do tail -n +2 $f >> noise.csv; done
echo done > DONE
