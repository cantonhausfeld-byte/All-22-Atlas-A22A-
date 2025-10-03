.PHONY: doctor ingest features train sim report

doctor:
	python -m a22a.tools.doctor

ingest:
	python -m a22a.data.ingest

features:
	python -m a22a.features.build

train:
	python -m a22a.models.train_baseline

sim:
	python -m a22a.sim.run

report:
	python -m a22a.reports.weekly

