python ban.py --save_path vgg19 --epoch_num 200 --save_interval 200;
python ban.py --teacher results/vgg19/model_ep200.pth --save_path vgg19 --ban_num 5 --epoch_num 200 --save_interval 200

# python ban.py --save_path resnet18 --epoch_num 200 --save_interval 200;
# python ban.py --teacher results/resnet18/model_ep200.pth --save_path resnet18 --ban_num 5 --epoch_num 200 --save_interval 200