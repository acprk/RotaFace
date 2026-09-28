#!/bin/bash
BIN=../../code/target/release/ident
for r in "4 15" "8 7" "10 6" "12 5"; do set -- $r
  $BIN B 16 2 $1 $2 0,16,64,256,1024 24 B_r$1x$2 2>&1 | tee -a e2.log
done
for c in "4 8 4 15" "8 4 4 15"; do set -- $c
  $BIN A $1 $2 $3 $4 0,16,64,256 24 A_q$1x$2_r$3x$4 2>&1 | tee -a e2.log
done
echo ALLDONE >> e2.log
