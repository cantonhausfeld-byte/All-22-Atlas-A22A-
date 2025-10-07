.PHONY: doctor ingest features train sim report decision uer strategy context

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

decision:
	python -m a22a.decision.portfolio

uer:
	python -m a22a.units.uer

strategy:
	python -m a22a.strategy.coach_adapt

context:
	python -m a22a.context.game_state
