datasets=(coco) ## or any other dataset
types=(random popular adversarial)

## baseline
for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python  \  ## path to the provided inference script
            --model-path  \  ## path to model weights
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder  \  ## path to dataset images
            --answers-file  \  ## path to desired output file
            --temperature 0 \
            --conv-mode qwen_chat      ## vicuna_v1 for llava
    done
done