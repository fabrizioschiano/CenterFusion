export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd src
# train
python main.py \
    ddd \
    --exp_id centerfusion \
    --shuffle_train \
    --train_split train \
    --val_split mini_val \
    --val_intervals 1 \
    --run_dataset_eval \
    --nuscenes_att \
    --velocity \
    --batch_size 4 \
    --lr 2.5e-16 \
    --num_epochs 10 \
    --lr_step 50 \
    --save_point 20,40,50 \
    --gpus 0 \
    --not_rand_crop \
    --flip 0.5 \
    --shift 0.1 \
    --pointcloud \
    --radar_sweeps 6 \
    --pc_z_offset 0.0 \
    --pillar_dims 1.0,0.2,0.2 \
    --max_pc_dist 60.0 \
    --load_model ../models/centerfusion_e60.pth \
    # --debug 4 \
    # --freeze_backbone \
    # --resume \

cd ..
