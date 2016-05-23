#!/bin/bash

for c in 0,1,2,3,4
do 
    echo "$c"
    python SiCNMF_start_folds.py -c $c -f SiCNMF_folds.config -p 5
done
