echo "LSQ Quan_Weight Quan_first_last  Training"
python /data/lsq_quantizer/lsq_main.py --dataset imagenet --network resnet50 --weight_bits 2 --prefix 50_w_wfl_ --quan_first_last 