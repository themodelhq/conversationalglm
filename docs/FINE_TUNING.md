# Fine-Tuning Guide

## Supervised fine-tuning

Set `model.init_from` in `config/train_sft.yaml` to a prior `model.safetensors` file or a model directory, keep the matching tokenizer, and lower the learning rate to 1e-5 through 5e-5 for narrow-domain adaptation.

## Preference optimization

Preference records contain `prompt`, `chosen`, and `rejected` fields.

```bash
python -m training.validate examples/preferences.jsonl --task dpo
python -m training.train --config config/train_dpo.yaml
python -m training.reward --model checkpoints/glm-sft --train examples/preferences.jsonl --output checkpoints/reward
```

DPO uses the initialized base model as a frozen reference model and optimizes the policy log-probability margin. Tune `training.beta` between 0.05 and 0.2. Keep prompts, chosen answers, and rejected answers from the same policy and ensure the preference label reflects a documented evaluation rubric.

## Quality gates

Run held-out perplexity, task-specific evaluation, safety cases, tool-call tests, and multilingual review before promotion:

```bash
python -m evaluation.language --model checkpoints/glm-dpo --data data/processed/validation.jsonl
python -m evaluation.safety --model checkpoints/glm-dpo --cases examples/safety_cases.jsonl
pytest -q
```
