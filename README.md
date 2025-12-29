# llm_fromScratch

> Studying the original [MiniMind project](https://github.com/jingyaogong/minimind) by re-implementing the minimum needed for the pretraining (Pretrain) pipeline so far.

## Overview

This repo keeps the core [MiniMind](https://github.com/jingyaogong/minimind) training loop, modeling code, and tokenizer assets so you can study or extend a compact LLM stack. It currently tracks PyTorch 2.9, Transformers 4.57, and Python 3.12, with training/eval flows for the `MindConfig` causal language model plus the tokenizer exported under `model/`.

![MiniMind structure](img/LLM-structure.png)

## Project Layout

- `model/` – minimal MindConfig model and tokenizer assets.
- `dataset/` – pretraining dataset loader utilities.
- `trainer/` – training loop plus helper utilities.
- `eval.py` – inference/chat entry point.
- `main.py` – Torch environment smoke test.
- `out/` – checkpoints and experiment outputs.

## Setup

```bash
uv sync        # recommended
# or
pip install -e .
```

## Training

```bash
python trainer/train_pretrain.py \
  --data_path dataset/pretrain_hq.jsonl \
  --save_dir out \
  --epochs 1 \
  --batch_size 32 \
  --hidden_size 512 \
  --use_moe 0
```

Key flags to adjust: `--from_weight` for warm starts, `--accumulation_steps` for low-memory GPUs, and `--max_seq_len` to match your tokenizer context. The example `pretrain_hq.jsonl` comes from the [MiniMind dataset on ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) and should be downloaded before launching training. Checkpoints land in `out/pretrain_512.pth` (fp16) plus a mirrored entry under `checkpoints/`.

## Evaluation / Chat Demo

```bash
python eval.py --load_from model --weight pretrain --max_new_tokens 128
```

Set `--load_from path/to/hf-repo` to test community weights or toggle `--use_moe 1` to activate MoE checkpoints. The script prints prompts in auto mode or lets you enter chat turns when `--historys` > 0. Fine-tuned weights saved by the trainer land under `llm_fromScratch/out/` (e.g., `llm_fromScratch/out/pretrain_512.pth`) and can be referenced directly via the `--save_dir`/`--weight` flags.

## TODO

- Keep surfacing more [MiniMind](https://github.com/jingyaogong/minimind) internals (attention variants, optimizer knobs, tokenizer quirks) so every layer is understood through hands-on experiments.
- Implement fine-tuning recipes, starting with lightweight LoRA adapters before graduating to full-parameter updates.

Thanks again to the original [MiniMind](https://github.com/jingyaogong/minimind) project for the reference implementation and training insights.
