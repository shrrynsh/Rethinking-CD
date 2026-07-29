# Jaccard experiment

Code to run the Jaccard experiment for two vision-language models:

- `jaccard_on_llava/`: LLaVA-1.5-7B
- `jaccard_on_qwen/`: Qwen2.5-VL-7B

Each folder has the same layout:

```
inference/   code that runs the model and writes its answers
scripts/     shell scripts that run the full decoding sweep
analysis/    code that reads the answers and computes the Jaccard scores
data/        the LLaVA-Bench questions and images
```

Both models are run on the 60 questions of LLaVA-Bench (In-the-Wild) under
greedy search, direct sampling, the Adaptive Plausibility Constraint swept from
0.0 to 1.0, and VCD.

## How to run

You need the model weights and a GPU (an NVIDIA L4 with 24 GB works).

1. Set the path to the model weights:

   ```
   export MODEL_PATH=/path/to/llava-v1.5-7b          # for jaccard_on_llava
   export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct  # for jaccard_on_qwen
   ```

2. Install the model code.

   LLaVA:

   ```
   git clone https://github.com/haotian-liu/LLaVA
   pip install -e LLaVA
   ```

   Qwen:

   ```
   pip install transformers accelerate qwen-vl-utils torch
   ```

3. Run the decoding sweep from inside the model folder. This writes the answers
   into a new `outputs/` folder:

   ```
   cd jaccard_on_llava      # or jaccard_on_qwen
   bash scripts/run_all.sh
   ```

4. Compute the Jaccard scores from those answers:

   ```
   cd analysis
   python3 jaccard_apc_vs_greedy.py
   python3 jaccard_apc_vs_sample.py
   python3 jaccard_vcd_comparisons.py
   python3 plot_jaccard_apc_vs_greedy.py
   ```

   The analysis needs `transformers`, `numpy`, `pandas`, `matplotlib`, and
   `sentencepiece`. It loads the matching tokenizer, which downloads once from
   Hugging Face. To use a local tokenizer, set `MODEL_PATH` before running.

## Note

The images and questions are in `data/`. For the LLaVA folder you can also
re-download them with `scripts/download_llava_bench.py`.
