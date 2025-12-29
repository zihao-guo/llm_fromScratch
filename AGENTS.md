# Repository Guidelines

## Project Structure & Module Organization
Source lives under `model/` (MindConfig + tokenizer assets), `method/` (RoPE, RMSNorm, GQA layers), `dataset/` (pretraining dataset wrappers), and `trainer/` (training loop + utils). Entry points sit at `trainer/train_pretrain.py` for optimization, `eval.py` for chat/demo runs, and `main.py` for quick Torch sanity checks. Place any new tests under `tests/` mirroring this tree (`tests/method/test_rope.py`, `tests/dataset/test_lm_dataset.py`). Keep large checkpoints in `out/` and ignore raw corpora.

## Build, Test, and Development Commands
Install dependencies with `uv sync` (fallback `pip install -e .`). Typical workflows:
```bash
python trainer/train_pretrain.py --data_path dataset/pretrain_hq.jsonl --save_dir out --epochs 1
python eval.py --load_from model --weight pretrain --max_new_tokens 128
python -m pytest tests -q
```
The training script handles DDP, gradient accumulation, and checkpointing; `eval.py` loads local weights or Hugging Face repos for sampling; `pytest` should pass before sending a PR.

## Coding Style & Naming Conventions
Target Python 3.12 with 4-space indents and PEP 8 spacing. Favor snake_case for files/functions, PascalCase for configs/models (`MindForCausalLM`, `PretrainDataset`). Keep parser flags synchronized with the CLI argument names already defined in `train_pretrain.py`. Group imports stdlib/third-party/local, annotate tricky tensors, and run `black . --line-length 100` plus `ruff check .` before pushing.

## Testing Guidelines
Use `pytest` with deterministic seeds (see `setup_seed`). Cover tensor shapes, masking math, and distributed helpers; when adding new CLI flags or data pipelines, add at least a smoke test under `tests/<module>/test_<feature>.py`. After training, validate checkpoints via `python eval.py --load_from model --historys 0` to ensure serialization still works.

## Commit & Pull Request Guidelines
Follow the existing emoji-prefixed format (`:tada:readyTotrain`, `:fire:debug`): choose a fitting emoji + short verb phrase, no trailing period. PRs should link the motivating issue or experiment doc, describe behavior changes, list test evidence (loss curves, sample generations), and call out any new dependencies or flags.

## Security & Configuration Tips
Do not commit `pretrain_hq.jsonl` or other datasets—reference the ModelScope download (https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) and keep raw data outside version control. Store Hugging Face keys or custom configs in your shell profile or an untracked `.env`. Record CUDA/cuDNN versions and GPU types when sharing performance numbers to aid reproducibility.
