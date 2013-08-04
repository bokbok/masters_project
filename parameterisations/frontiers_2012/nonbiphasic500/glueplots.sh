#!/bin/bash

for filename in *plot*.dat

do

paste -d ' ' $filename total.dat > total.1.dat
mv total.1.dat total.dat

done

