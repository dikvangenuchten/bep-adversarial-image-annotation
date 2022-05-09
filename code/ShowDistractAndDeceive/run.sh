docker build . -t show-distract-deceive
docker run \
    -it \
    --rm \
    --gpus all \
    --env-file .env \
    -v /home/dik/projects/university/bep-adversarial-image-annotation/coco_dataset/:/data/ \
    show-distract-deceive \
        python src/main.py