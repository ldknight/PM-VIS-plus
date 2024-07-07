#!/bin/bash  

model_name_craete(){
# model_name_craete
model_name=()

# base_iter=1000 #多少次保存一个模型
# max_iter=15050

base_iter=$1
max_iter=$2
# echo $(expr $max_iter / $base_iter - 1) #向下取整
index_=$(expr $max_iter / $base_iter - 1)
# for num in {0..$index_}  
for num in `eval echo {0..$index_}`  
do  
# echo $num
res_num=$(expr $(expr $base_iter - 1) + $base_iter \* $num)
# echo $res_num
if [ $res_num -lt 1000 ];  
then  
model_name=("${model_name[@]}" model_0000$(expr $(expr $base_iter - 1) + $base_iter \* $num).pth)
elif [ $res_num -ge 1000 ]  && [ $res_num -lt 10000 ];  
then  
model_name=("${model_name[@]}" model_000$(expr $(expr $base_iter - 1) + $base_iter \* $num).pth)
elif [ $res_num -ge 10000 ]  && [ $res_num -lt 100000 ];  
then  
model_name=("${model_name[@]}" model_00$(expr $(expr $base_iter - 1) + $base_iter \* $num).pth)
elif [ $res_num -ge 100000 ]  && [ $res_num -lt 1000000 ];  
then  
model_name=("${model_name[@]}" model_0$(expr $(expr $base_iter - 1) + $base_iter \* $num).pth)
fi
# # echo $num  
# # echo $(expr 12 \* 5)
# # echo model_000$(expr $(expr $base_iter - 1) + $base_iter \* $num).pth
# model_name=("${model_name[@]}" model_000$(expr $(expr $base_iter - 1) + $base_iter \* $num).pth)
done  
echo ${model_name[@]} 
# return ${model_name[@]}
}