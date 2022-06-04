python3 scripts/caption.py \
    --model data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar \
    --word_map data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json \
    --beam_size 1 \
    --img $1
