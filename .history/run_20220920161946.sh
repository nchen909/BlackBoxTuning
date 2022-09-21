#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

#seed_lst=(8 13 42 50 60)
seed_lst=(60 50 42 13 8)
#seed_lst=(8)
#task_name_lst=(SST-2 Yelp AGNews TREC MRPC SNLI)  # SST-2 Yelp AGNews TREC MRPC SNLI
task_name_lst=(SNLI)
alg=openes
cuda=0


for task_name in "${task_name_lst[@]}"; do
    for seed in "${seed_lst[@]}"; do
        python -u bbt.py --seed $seed --task_name $task_name --alg $alg
    done
    python -u test.py --task_name $task_name --cuda $cuda
done


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 
