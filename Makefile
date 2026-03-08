setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	pytest

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
