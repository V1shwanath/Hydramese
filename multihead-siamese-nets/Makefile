PIP = ./.venv/Scripts/pip
PYTHON = ./.venv/Scripts/python

hello:
	echo "Hello, world!"

venv:
	python -m venv .venv
	.\.venv\Scripts\activate

install: venv
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