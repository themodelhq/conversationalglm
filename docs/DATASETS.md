# Dataset Guide

Only train on data whose license, consent, provenance, retention rules, and intended use permit model training and redistribution. Do not ingest private conversations, biometric recordings, copyrighted media without rights, secrets, credentials, or data collected in violation of law or platform policy.

## Manifest-based downloads

Record dataset identity, license, split, field mapping, and approval in a YAML manifest. `config/datasets.example.yaml` is the accepted format.

```bash
python scripts/download_datasets.py --manifest datasets.yaml --output data/raw
python scripts/clean_dataset.py --input data/raw/source.jsonl --output data/clean/source.jsonl
python scripts/shard_dataset.py --input data/clean/source.jsonl --output-dir data/shards
```

The download script only retrieves entries with `allowed_for_training: true` and an explicit `hf_id`. Retain source hashes, license copies, consent records, filtering policy, and split definitions outside training artifacts.

## Data quality

Deduplicate near-identical records, remove personal data, normalize Unicode, validate media decoding, balance languages and domains, isolate evaluation data before training, and prevent train/evaluation contamination. Store audio paths relative to controlled dataset roots and do not expose untrusted paths to public API callers.
