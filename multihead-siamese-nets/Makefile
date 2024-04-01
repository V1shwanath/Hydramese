PIP = ./.venv/Scripts/pip
PYTHON = ./.venv/Scripts/python

install:
	python -m venv .venv && \
		. ./.venv/Scripts/activate && \
		$(PIP) install -r requirements/requirements-gpu.txt


lint:
	
#format all python documents in the project
format:
	black *.py
	black layers/*.py
	$(PYTHON) -m pytest



clean:
	@echo "Cleaning up..."
	rm -rf .venv 