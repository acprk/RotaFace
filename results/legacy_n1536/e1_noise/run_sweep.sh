#!/bin/bash
# E1 sweep: each config is an independent single-threaded process writing its own CSV.
BIN=../../code/target/release/noise
mkdir -p parts
for r in "4 15" "5 12" "6 10" "8 7" "10 6" "12 5" "15 4" "20 3"; do
  set -- $r; $BIN B 16 2 $1 $2 1024 32 32 parts/B_r$1x$2.csv raw_B_r$1x$2 > parts/B_r$1x$2.log 2>&1 &
done
for q in "16 2" "11 3" "8 4" "4 8"; do for r in "4 15" "8 7"; do
  set -- $q $r; $BIN A $1 $2 $3 $4 256 16 16 parts/A_q$1x$2_r$3x$4.csv > parts/A_q$1x$2_r$3x$4.log 2>&1 &
done; done
wait
head -1 parts/B_r4x15.csv > noise.csv; for f in parts/*.csv; do tail -n +2 $f >> noise.csv; done
echo done
