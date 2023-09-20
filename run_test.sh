#inference Avenue
#python run_test.py --dataset_name Avenue --ckpt_path best_ckpt/avenue_91.23.pkl --feature_pretrain_model i3d --feature_modal combine --Vitblock_num 2

#inference shanghaitech
#python run_test.py --dataset_name shanghaitech --ckpt_path best_ckpt/ST.pth --feature_pretrain_model i3d --feature_modal combine --Vitblock_num 2 --cross_clip 4

#inference UCF-Crime
python run_test.py --dataset_name ucfcrime  --ckpt_path best_ckpt/UCF.pth --feature_pretrain_model i3d --feature_modal rgb --Vitblock_num 8 --cross_clip 4