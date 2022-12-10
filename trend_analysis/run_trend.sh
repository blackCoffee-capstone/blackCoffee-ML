#!/bin/bash

source /home/iknow/anaconda3/etc/profile.d/conda.sh
conda activate test0

cd /home/iknow/Desktop/blackcoffee/trend_analysis/
python trend.py $1 $2
