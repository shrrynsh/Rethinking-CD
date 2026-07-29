datasets=(coco) ## or any dataset
types=(random popular adversarial)

## vcd
for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python  \  ## path to the provided cd inference script
            --model-path  \  ## path to model weights
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder  \  ## path to dataset images
            --answers-file  \  ## path to desired output file
            --temperature 0 \
            --conv-mode qwen_chat \     ## vicuna_v1 for llava
            --use-vcd
    done
done

## icd

for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python  \  ## path to the provided cd inference script
            --model-path  \  ## path to model weights
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder  \  ## path to dataset images
            --answers-file  \  ## path to desired output file
            --temperature 0 \
            --conv-mode qwen_chat \    ## vicuna_v1 for llava
            --use-icd
    done
done

## sid
for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python  \  ## path to the provided cd inference script
            --model-path  \  ## path to model weights
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder  \  ## path to dataset images
            --answers-file  \  ## path to desired output file
            --temperature 0 \
            --conv-mode qwen_chat \   ## vicuna_v1 for llava
            --use-sid
    done
done