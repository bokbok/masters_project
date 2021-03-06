#!/bin/bash

#awk '{if($2>0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)") printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 > plot-test-un.dat
#awk '{if($2<0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)") printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 > plot-test-st.dat
#awk '{if($12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)") printf("%.18lf %.18lf\n",$5, 1000/$12)}' b.par1 > plot-test-all.dat

#cat plot-test-un.dat plot-test-st.dat > plot-test-all.dat

#awk '{gamma[k]=$1; f[k]=$2; k++}END{for(i=1; i<=NR-1; i++) {if((f[i+1]-f[i])>1e-10)printf("%.6lf %.6lf", gamma[i],(gamma[i+1]-gamma[i])/(f[i+1]-f[i]))}}' plot-test-all.dat > derfreq.dat  

#derivative within sensible limits
#awk '{gamma[k]=$1; f[k]=$2; k++}END{for(i=1; i<=NR-1; i++) {if((gamma[i+1]-gamma[i])>.00001 || (-gamma[i+1]+gamma[i])>.00001) printf("%.6lf %.6lf\n", gamma[i],(f[i+1]-f[i])/(gamma[i+1]-gamma[i]))}}' plot-test-all.dat > derfreq.dat  

#********************
#vector fields
#awk '{gamma[k]=$1; f[k]=$2; k++}END{for(i=1; i<=NR-1; i++) {if((gamma[i+1]-gamma[i])>.00001 || (-gamma[i+1]+gamma[i])>.00001) printf("%.6lf %.6lf %.6lf %.6lf\n", gamma[i],f[i],gamma[i+1],f[i+1])}}' plot-test-all.dat > vf-freq.dat  
#gnuplot command for vector field
#p './vf-freq.dat' u 1:2:($3*factor):($4*factor) w vector
#**********************

#vector derivative within sensible limits
#awk '{gamma[k]=$1; f[k]=$2; k++}END{for(i=1; i<=NR-1; i++) {if((gamma[i+1]-gamma[i])>.00001 || (-gamma[i+1]+gamma[i])>.00001) printf("%.6lf %.6lf %.6lf\n", gamma[i],(gamma[i+1]-gamma[i])/sqrt((gamma[i+1]-gamma[i])^2+(f[i+1]-f[i])^2),(f[i+1]-f[i])/sqrt((gamma[i+1]-gamma[i])^2+(f[i+1]-f[i])^2))}}' plot-test-all.dat > vecder.dat  

for filename in *.ode
do

#check if folder exists

if [ -s ${filename:0:5}/UZ-1 ]; then

#enter the right folders
cd ${filename:0:5}/UZ-1

#**********************************
# Divide according to bands
#*********************************
#unstable
#delta
awk '{if($2>0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && (1000/$12) >= 0 && (1000/$12) <= 4) printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 >> ../../un.delta.dat
#theta
awk '{if($2>0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && (1000/$12) >= 4 && (1000/$12) <= 8) printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 >> ../../un.theta.dat
#alpha
awk '{if($2>0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && (1000/$12) >= 8 && (1000/$12) <= 13) printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 >> ../../un.alpha.dat
#beta
awk '{if($2>0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && (1000/$12) >= 13 && (1000/$12) <= 30) printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 >> ../../un.beta.dat
#gamma
awk '{if($2>0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && (1000/$12) >= 30) printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 >> ../../un.gamma.dat

#stable
#delta
awk '{if($2<0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && (1000/$12) >= 0 && (1000/$12) <= 4) printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 >> ../../st.delta.dat
#theta
awk '{if($2<0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && (1000/$12) >= 4 && (1000/$12) <= 8) printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 >> ../../st.theta.dat
#alpha
awk '{if($2<0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && (1000/$12) >= 8 && (1000/$12) <= 13) printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 >> ../../st.alpha.dat
#beta
awk '{if($2<0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && (1000/$12) >= 13 && (1000/$12) <= 30) printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 >> ../../st.beta.dat
#gamma
awk '{if($2<0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)" && (1000/$12) >= 30) printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 >> ../../st.gamma.dat

#waterfall
#awk '{if($2>0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)") printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 > ../../un.${filename:0:5}.dat
#awk '{if($2<0 && $12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)") printf("%.6lf %.3lf\n",$5, 1000/$12)}' b.par1 > ../../st.${filename:0:5}.dat
#awk '{if($12>0 && $12!=1 && $12!="U(6)" && $12!="U(3)") printf("%.18lf %.18lf\n",$5, 1000/$12)}' b.par1 > ../../all.${filename:0:5}.dat


cd ../../

fi

done
