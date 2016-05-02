#!/bin/bash

for i in 0 1 2 3 4
do 
    echo "$s"
    python SiCNMF_start.py -i $i -f SiCNMF.config -p 1
done
