python main.py --roi_x=128 --roi_y=128 --roi_z=160 --in_channels=1 --out_channels=25 --model_name=Swin-X2S-Base --batch_size=1 --logdir="totalrib_swinx2sbase_view1/" --optim_lr=3e-5 --reg_weight=5e-1 --save_checkpoint --lrschedule="warmup_cosine" --data_dir="./dataset/Totalsegmentator_rib/data_list_.json" --use_ssl_pretrained --pretrained_dir="./pretrained_models/" --pretrained_model_name="swin_small_patch4_window7_224_22k.pth" --max_epochs=250 --val_start=0 --val_every=5 --warmup_epochs=20