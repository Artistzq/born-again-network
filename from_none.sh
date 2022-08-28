shell_epoch_num=15
shell_save_path="vgg19"
sehll_ban_num=10

python ban.py --save_path ${shell_save_path} --epoch_num ${shell_epoch_num} --save_interval ${shell_epoch_num};
python ban.py \
    --teacher results/${shell_save_path}/model_ep${shell_epoch_num}.pth \
    --save_path ${shell_save_path} \
    --ban_num ${sehll_ban_num} \
    --epoch_num ${shell_epoch_num} \
    --save_interval ${shell_epoch_num};
python evaluate.py --path ${shell_save_path};
scp -P 20022 -r results/${shell_save_path} yzq@10.177.21.233:~/Documents/results/determin/${shell_save_path}_${shell_epoch_num}ep/
rm -rf results/${shell_save_path}

# shell_epoch_num=15
# shell_save_path="resnet18"
# sehll_ban_num=10

# python ban.py --save_path ${shell_save_path} --epoch_num ${shell_epoch_num} --save_interval ${shell_epoch_num};
# python ban.py \
#     --teacher results/${shell_save_path}/model_ep${shell_epoch_num}.pth \
#     --save_path ${shell_save_path} \
#     --ban_num ${sehll_ban_num} \
#     --epoch_num ${shell_epoch_num} \
#     --save_interval ${shell_epoch_num};
# python evaluate.py --path ${shell_save_path};
# scp -P 20022 -r results/${shell_save_path} yzq@10.177.21.233:~/Documents/results/determin/${shell_save_path}_${shell_epoch_num}ep/
# rm -rf results/${shell_save_path}

# shell_epoch_num=50
# shell_save_path="vgg11"
# sehll_ban_num=10

# python ban.py --save_path ${shell_save_path} --epoch_num ${shell_epoch_num} --save_interval ${shell_epoch_num};
# python ban.py \
#     --teacher results/${shell_save_path}/model_ep${shell_epoch_num}.pth \
#     --save_path ${shell_save_path} \
#     --ban_num ${sehll_ban_num} \
#     --epoch_num ${shell_epoch_num} \
#     --save_interval ${shell_epoch_num};
# python evaluate.py --path ${shell_save_path};
# scp -P 20022 -r results/${shell_save_path} yzq@10.177.21.233:~/Documents/results/determin/${shell_save_path}_${shell_epoch_num}ep/
# rm -rf results/${shell_save_path}

# shell_epoch_num=100
# shell_save_path="lenet"
# sehll_ban_num=5

# python ban.py --save_path ${shell_save_path} --epoch_num ${shell_epoch_num} --save_interval ${shell_epoch_num};
# python ban.py \
#     --teacher results/${shell_save_path}/model_ep${shell_epoch_num}.pth \
#     --save_path ${shell_save_path} \
#     --ban_num ${sehll_ban_num} \
#     --epoch_num ${shell_epoch_num} \
#     --save_interval ${shell_epoch_num};
# python evaluate.py --path ${shell_save_path};
# scp -P 20022 -r results/${shell_save_path} yzq@10.177.21.233:~/Documents/results/determin/${shell_save_path}_${shell_epoch_num}ep/
# rm -rf results/${shell_save_path}
