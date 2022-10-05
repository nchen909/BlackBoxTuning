#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

seed_lst=(8) # 8 13 42 50 60
task_name_lst=(SNLI)  # SST-2 Yelp AGNews TREC MRPC SNLI
device=cuda:0
cuda=0
model_name=roberta-large
model_path=roberta-large
n_prompt_tokens=50
intrinsic_dim=500
k_shot=16
loss_type=ce
random_proj=normal
sigma1=1
sigma2=0.2
popsize=20
bound=0
budget=8000
budget2=6000
print_every_lst=(24)
eval_every_lst=(100)

pop_mean=0

for task_name in "${task_name_lst[@]}"; do
    for seed in "${seed_lst[@]}"; do
        for print_every in "${print_every_lst[@]}"; do
	    for eval_every in "${eval_every_lst[@]}"; do
                #python -u deepbbt.py --seed $seed --task_name $task_name --device $device --budget $budget --model_name $model_name --model_path $model_path --budget2 $budget2 --pop_mean $pop_mean --print_every $print_every --eval_every $eval_every
            	python -u test_deepbbt.py --task_name $task_name --seed $seed --cuda $cuda --print_every $print_every --eval_every $eval_every
            done
	done
    done
done


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 
