SIZE=$1
IMG_DIR=$2

python evaluate_contrast_all_tasks.py ${SIZE} ./checkpoints/${SIZE}.bin --input_file ../data/cococon.json --out_file ./results/cococon_output_${SIZE}.json --img_dir ${IMG_DIR}

