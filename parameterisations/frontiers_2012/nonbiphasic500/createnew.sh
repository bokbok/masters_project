#!/bin/bash

#self-contained scipt to create
#ode files from the table

# Extract table values (up to 656)
#split -1 parameters.txt

#Initial time of simulation
start_time0=$(date +%s)

#Initialise counter
let count=0

#Loop through files
for filename in *.pp
do
    (( count=$count+1 ))

# Enhancing equilibrium
awk '{print "param her="$3; print "param hir="$4; print "param heqee="$5; print "param heqei="$6; print "param heqie="$7; print "param heqii="$8; print "param naee="$9; print "param naei="$10; print "param nbee="$11; print "param nbei="$12; print "param nbie="$13; print "param nbii="$14; print "param lambdaee="$15; print "param lambdaei="$16; print "param vbar="$17/1000; print "param aee="$18; print "param aei="$19; print"param aie="$20; print"param aii="$21; print"param gee="$22/1000; print"param gei="$23/1000; print"param gie="$24/1000; print"param gii="$25/1000; print "param taue="$26*1000; print"param taui="$27*1000; print"param te="$28; print"param ti="$29; print"param rabs=0.0000"; print"param emax="$31/1000;print"param imax="$32/1000;print"param pee=-200.0"; print"param pei=200.0";print"param pie="$35/1000;print"param pii="$36/1000;print"param se="$37;print"param si="$38}' $filename > $count.par

# Extracting pee, pei, gee relevant to the par set
awk '{print "-3", $33/1000}' $filename > $count.pee
awk '{print "-4", $34/1000}' $filename > $count.pei
awk '{printf("       PAR(5)= %.8lfD0\n",$18)}' $filename > $count.aee 
awk '{printf("       PAR(6)= %.8lfD0\n",$19)}' $filename > $count.aei 
awk '{printf("       PAR(7)= %.8lfD0\n",$20)}' $filename > $count.aie 
awk '{printf("       PAR(8)= %.8lfD0\n",$21)}' $filename > $count.aii 


#creating the ode and removing useless files
cat pre.txt $count.par post.txt > $count.ode
rm $count.par
echo $filename processed
#rm $filename

#equilibrating via xppaut

echo equilibrating $count
./xppaut $count.ode -silent > log.txt

#Extract:
awk '/10000 /{printf("\tU(1)=%.24lfD0\n\tU(2)=%.24lfD0\n\tU(3)=%.24lfD0\n\tU(4)=%.24lfD0\n\tU(5)=%.24lfD0\n\tU(6)=%.24lfD0\n\tU(7)=%.24lfD0\n\tU(8)=%.24lfD0\n\tU(9)=%.24lfD0\n\tU(10)=%.24lfD0\n\tU(11)=%.24lfD0\n\tU(12)=%.24lfD0\n\tU(13)=%.24lfD0\n\tU(14)=%.24lfD0\n"),$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15}' output.dat > $count.init

#pause for debugging purposes
#read -p "key"

#removing useless files
mv output.dat $count.output
#rm output.dat
rm log.txt
#rm $filename

echo $count finished!
done

#rename files properly -1
for filename in ?.pee
do
mv $filename 0000$filename
done

for filename in ?.pei
do
mv $filename 0000$filename
done

for filename in ?.ode
do
mv $filename 0000$filename
done

for filename in ?.init
do
mv $filename 0000$filename
done

for filename in ?.aee
do
mv $filename 0000$filename
done

for filename in ?.aei
do
mv $filename 0000$filename
done

for filename in ?.aie
do
mv $filename 0000$filename
done

for filename in ?.aii
do
mv $filename 0000$filename
done

for filename in ?.output
do
mv $filename 0000$filename
done

#rename files properly -2
for filename in ??.pee
do
mv $filename 000$filename
done

for filename in ??.pei
do
mv $filename 000$filename
done

for filename in ??.ode
do
mv $filename 000$filename
done

for filename in ??.init
do
mv $filename 000$filename
done

for filename in ??.aee
do
mv $filename 000$filename
done

for filename in ??.aei
do
mv $filename 000$filename
done

for filename in ??.aie
do
mv $filename 000$filename
done

for filename in ??.aii
do
mv $filename 000$filename
done

for filename in ??.output
do
mv $filename 000$filename
done

#rename files properly -3
for filename in ???.pee
do
mv $filename 00$filename
done

for filename in ???.pei
do
mv $filename 00$filename
done

for filename in ???.ode
do
mv $filename 00$filename
done

for filename in ???.init
do
mv $filename 00$filename
done

for filename in ???.aee
do
mv $filename 00$filename
done

for filename in ???.aei
do
mv $filename 00$filename
done

for filename in ???.aie
do
mv $filename 00$filename
done

for filename in ???.aii
do
mv $filename 00$filename
done

for filename in ???.output
do
mv $filename 00$filename
done

#rename files properly -4
for filename in ????.pee
do
mv $filename 0$filename
done

for filename in ????.pei
do
mv $filename 0$filename
done

for filename in ????.ode
do
mv $filename 0$filename
done

for filename in ????.init
do
mv $filename 0$filename
done

for filename in ????.aee
do
mv $filename 0$filename
done

for filename in ????.aei
do
mv $filename 0$filename
done

for filename in ????.aie
do
mv $filename 0$filename
done

for filename in ????.aii
do
mv $filename 0$filename
done

for filename in ????.output
do
mv $filename 0$filename
done

final_time2=$(date +%s)

echo
echo .... TOTAL Execution Time in the process is $((final_time2-start_time0)) secs ...
echo

