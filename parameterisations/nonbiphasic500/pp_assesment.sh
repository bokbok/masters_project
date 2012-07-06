#!/bin/bash

#Calculate the percentage of LP and HB
#This is a very rough measure of complexity for the batch

echo ........................
awk '{if ($3=="HB") n++;if ($2=="Processing") s++;if ($3=="LP") l++;}
END{printf("HB: (%d / %d) %.2lf % \nLP: (%d / %d) %.2lf %\n", n, s, n/s*100, l, s, l/s*100)}' test.new.txt
echo ........................

echo

for filename in *.ode; do

echo ...................
echo ... Plotting ${filename:0:5} ...
echo ...................


awk '{printf("\n Value of p_{ee} = %.4lf\n", $2)}' ${filename:0:5}.pee

cd ${filename:0:5}/UZ-1/EPS-1/

printf "\n  BR   PT   TY   LAB    PAR(3)        L2-NORM         U(1)          U(2)          U(3)          U(4)          U(5)          U(6)\n"
grep HB d.par1
grep LP d.par1

auto ../../../plots.auto

#@pp par1

#@pp par2
#sleep 1 #wait 1 sec

#read -p "Next plot"

cd ../../../

done

