PIP = ./.venv/Scripts/pip
PYTHON = ./.venv/Scripts/python



install:
	python -m venv .venv && \
		. ./.venv/Scripts/activate && \
		$(PIP) install -r requirements/requirements-gpu.txt

#format all python documents in the project
format:
	$(PYTHON) -m black *.py

clean:
	rm -rf .venv