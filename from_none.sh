shell_epoch_nums=(1 1 1 1)
shell_save_paths=("resnet34" "vgg13" "mobilenet" "mobilenetv2")
shell_ban_nums=(3 3 3 3)

for(( i=0;i<${#shell_epoch_nums[@]};i++)) 
do
    shell_epoch_num=${shell_epoch_nums[i]}
    shell_save_path=${shell_save_paths[i]}
    shell_ban_num=${shell_ban_nums[i]}

    python ban.py --save_path ${shell_save_path} --epoch_num ${shell_epoch_num} --save_interval ${shell_epoch_num};
    python ban.py \
        --teacher results/${shell_save_path}/model_ep${shell_epoch_num}.pth \
        --save_path ${shell_save_path} \
        --ban_num ${shell_ban_num} \
        --epoch_num ${shell_epoch_num} \
        --save_interval ${shell_epoch_num};
    python evaluate.py --path ${shell_save_path};
    # scp -P 20022 -r results/${shell_save_path} yzq@10.177.21.233:~/Documents/results/determin/${shell_save_path}_${shell_epoch_num}ep/
    # rm -rf results/${shell_save_path}
done
