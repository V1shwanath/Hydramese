# Variables
PYTHON_VERSION=3.6
PREPARE_DATA_SCRIPT=./bin/prepare_data.sh
REQUIREMENTS=requirements/requirements-cpu.txt

# Targets
environment:
    conda create -n myenv python=$(PYTHON_VERSION) -y

prepare_data:
    cd bin && chmod a+x prepare_data.sh && ./prepare_data.sh

install_requirements:
    pip install -r $(REQUIREMENTS)

format:
    black *.py

train_cnn_qqp:
    python run.py train cnn qqp

train_cnn_anli:
    python run.py train cnn anli

train_rnn_qqp:
    python run.py train rnn qqp

train_rnn_anli:
    python run.py train rnn anli

train_multihead_qqp:
    python run.py train multihead qqp

train_multihead_anli:
    python run.py train multihead anli

predict_cnn_qqp:
    python run.py predict cnn qqp

predict_cnn_anli:
    python run.py predict cnn anli

predict_rnn_qqp:
    python run.py predict rnn qqp

predict_rnn_anli:
    python run.py predict rnn anli

predict_multihead_qqp:
    python run.py predict multihead qqp

predict_multihead_anli:
    python run.py predict multihead anli

lint:
    pylint run.py

# Phony Targets
.PHONY: environment prepare_data install_requirements format lint \
        train_cnn_qqp train_cnn_anli train_rnn_qqp train_rnn_anli \
        train_multihead_qqp train_multihead_anli \
        predict_cnn_qqp predict_cnn_anli predict_rnn_qqp predict_rnn_anli \
        predict_multihead_qqp predict_multihead_anli


