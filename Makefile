.PHONY: doctor ingest features train sim report

PYTHON ?= python

doctor:
	$(PYTHON) -m a22a.tools.doctor

ingest:
	$(PYTHON) -m a22a.data.ingest

features:
	$(PYTHON) -m a22a.features.build

train:
	$(PYTHON) -m a22a.models.train_baseline

sim:
	$(PYTHON) -m a22a.sim.run

report:
	$(PYTHON) -m a22a.reports.weekly
