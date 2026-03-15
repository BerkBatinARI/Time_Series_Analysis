VENV=.venv
PY=$(VENV)/bin/python

.PHONY: setup data features models tests plots all

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.download_data

features:
	$(PY) -m src.make_features

models:
	$(PY) -m src.risk_ewma
	$(PY) -m src.risk_t_var_es
	$(PY) -m src.risk_garch

tests:
	$(PY) -m src.backtest_var
	$(PY) -m src.backtest_var_t
	$(PY) -m src.backtest_var_garch
	$(PY) -m src.kupiec_test
	$(PY) -m src.christoffersen_test
	$(PY) -m src.add_pvalues

plots:
	$(PY) -m src.plot_breaches

all: data features models tests plots