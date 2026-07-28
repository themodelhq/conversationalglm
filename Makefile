PYTHON ?= python
install:
	$(PYTHON) -m pip install --upgrade pip && $(PYTHON) -m pip install -e '.[train,video,export,dev]'
api:
	$(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
train:
	$(PYTHON) -m training.train --config config/train_sft.yaml
prepare:
	$(PYTHON) -m training.prepare --config config/data.yaml
check:
	ruff check . && pytest -q
compose:
	docker compose up --build
