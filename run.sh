#!/bin/bash

for i in 0,1,2,3,4
do 
    echo "$i"
    python SiCNMF_start.py -i $i -f SiCNMF.config -p 5
done
