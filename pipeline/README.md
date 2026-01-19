# Pipeline

## Setup

```bash
pip install -r pipeline/requirements.txt
pip install -e ./kraken
```

```bash
export OPENAI_API_KEY="..."
export HANDWRITING_API_URL="http://localhost:8000"
```

## Handwriting Service

```bash
docker compose up --build handwriting-service
```

## Usage

```bash
python -m pipeline.openai_phrases --out data/phrases.jsonl --n 1000
python -m pipeline.render_dataset --phrases data/phrases.jsonl --out data/kraken_training
python -m pipeline.kraken_train --data data/kraken_training --out models/model.mlmodel
```
