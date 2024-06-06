#!/bin/bash

model=$1
output_dir=$2

./pred.sh ${model} 00 ${output_dir} 0 &
./pred.sh ${model} 01 ${output_dir} 0 &
wait
./pred.sh ${model} 02 ${output_dir} 0 &
./pred.sh ${model} 03 ${output_dir} 0 &
wait
./pred.sh ${model} 04 ${output_dir} 0 &
./pred.sh ${model} 05 ${output_dir} 0 &
wait
./pred.sh ${model} 06 ${output_dir} 0 &
./pred.sh ${model} 07 ${output_dir} 0 &
wait
./pred.sh ${model} 08 ${output_dir} 0 &
./pred.sh ${model} 09 ${output_dir} 0 &
wait
