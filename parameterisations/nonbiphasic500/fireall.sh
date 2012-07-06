#!/bin/bash

./splitparameters.sh ; 
./createnew.sh ; 
./auto-start.sh > logbif.dat ; 
cat */red* > reduxall.dat ; 
#less reduxall.orbits.dat
