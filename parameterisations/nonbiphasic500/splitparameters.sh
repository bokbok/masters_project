#!/bin/bash

#self-contained scipt to split
#each line of parameters.txt file into single 
#parameters files of type 00001.param.dat

# initialise count
let count=0
let num=0

#Loop through files
#and rename them as intermediate parameters files
for filename in parameters.txt
do 
split -l 500 parameters.txt
done

for filename2 in x??
do
   (( count=$count+1 ))
mv $filename2 $count.int
done

#@@@@@@@@@@@@@@@@@@
# Relabel int files to respect the right order
for filename in ?.int
do
mv $filename 00$filename
done

for filename in ??.int
do
mv $filename 0$filename
done
#@@@@@@@@@@@@@

#read -p "key"


let count=0
for filename4 in *.int
do
split -1 $filename4
(( count=$count+1 ))
for filename3 in x??
do
    (( num=$num+1 ))
mv $filename3 $num.pp
done
done

#Relabel accordingly
for filename in ?.pp
do
mv $filename 0000$filename
done

for filename in ??.pp
do
mv $filename 000$filename
done

for filename in ???.pp
do
mv $filename 00$filename
done

for filename in ????.pp
do
mv $filename 0$filename
done

#Remove intermediate files
rm *.int
