#!/bin/bash
# waits for E2 to finish (avoid contention), then measures scaling
while ! grep -q ALLDONE ../e2_ident/e2.log 2>/dev/null; do sleep 30; done
export RF_PARAMS=p128
BIN=../../../code/target/release/scale
uptime > load_before.txt
$BIN B 16 2 4 15 250000 1,2,4,8,16,24,32,48 3 scale.csv > scale_B.log 2>&1
$BIN A 4 8 4 15 20000 1,2,4,8,16,24,32,48 3 scale.csv > scale_A.log 2>&1
uptime > load_after.txt
echo done > DONE
