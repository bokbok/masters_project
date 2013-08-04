#!/bin/bash

#create a common directory
#!!! Don't forget to change
#the directory below !!!
#mkdir ORBITS-RvsFREQ
mkdir ORBITS-RvsHE

for filename in *.ode; do

#inform the user
echo ..... Processing ${filename:2:2} .....

#copy gnu script and 
#enter the main directory
cd ${filename:2:2}/

#extra dir to avoid issues
mkdir UZ-0/

#for each user point
for filename2 in UZ*; do

#enter the user-point folder
cd $filename2/

pwd

#copy the gnu script
cp ../../orbits.gnu .

#parse diagram file
#Only for no. of points vs freq
#awk '{if($12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)")print 1000/$12}' b.par1 > orbits.dat
#For R vs Freq
#awk '{if($12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && $2>0) print $5, 1000/$12}' b.par1 > unstable.dat
#awk '{if($12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && $2<0) print $5, 1000/$12}' b.par1 > stable.dat

#For R vs HE
awk '{if($1==1) print $5, $7}' b.par1 > steady.dat
awk '{if($12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && $2>0) print $5, $7}' b.par1 > unstable.dat
awk '{if($12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && $2<0) print $5, $7}' b.par1 > stable.dat

gnuplot orbits.gnu
ps2pdf orbits.ps
#rm orbits.dat
#rm stable.dat
#rm unstable.dat
rm *.par1
rm *.par2
rm orbits.gnu
rm orbits.ps
mv orbits.pdf ../../ORBITS-RvsHE/orb-${filename:2:2}-$filename2.pdf
echo $filename2 inside ${filename:2:2} completed!

#move back one directory
cd ../

done

#move back to the main directory
cd ../

echo .....  ${filename:2:2} completed! .....

done
