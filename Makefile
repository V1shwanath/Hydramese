PIP = .\.venv\Scripts\pip
PYTHON = ./.venv/Scripts/python



install:
	python -m venv .venv && \
		.\.venv\Scripts\activate && \
		$(PIP) install -r requirements\requirements-gpu.txt

#format all python documents in the project
format:
	$(PYTHON) -m black tests/ *.py

test:
	$(PYTHON) -m pytest tests/ 

lint:
	$(PYTHON) -m pylint tests/ *.py

clean:
	rm -rf .venv