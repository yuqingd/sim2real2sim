#!/bin/sh
for i in 1 2 3 4 5 
do
if [ -r "/dev/nvidia$i" ] 
then
 export CUDA_VISIBLE_DEVICES=$i
 echo $CUDA_VISIBLE_DEVICES
fi
done