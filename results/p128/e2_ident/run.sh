#!/bin/bash
export RF_PARAMS=p128
BIN=../../../code/target/release/ident
for r in "4 15" "8 7" "10 6"; do set -- $r
  $BIN B 16 2 $1 $2 0,16,64,256,1024 24 B_r$1x$2 2>&1 | tee -a e2.log
done
$BIN A 4 8 4 15 0,16,64,256 24 A_q4x8_r4x15 2>&1 | tee -a e2.log
echo ALLDONE >> e2.log
