.PHONY: doctor ingest features train sim report decision portfolio uer strategy context injuries depth impact meta market clv \
        report_batch dashboard dashboard_check

doctor:
	python -m a22a.tools.doctor

meta:
	python -m a22a.meta.run

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

report_batch:
	python -m a22a.reports.compile

dashboard:
	streamlit run a22a/reports/app.py --server.headless true --server.port 8501

dashboard_check:
	python -m a22a.reports.smoke

decision:
	python -m a22a.decision.portfolio

portfolio:
	python -m a22a.portfolio.optimize

uer:
	python -m a22a.units.uer

strategy:
	python -m a22a.strategy.coach_adapt

context:
	python -m a22a.context.game_state

injuries:
	python -m a22a.health.injury_model

depth:
	python -m a22a.roster.depth_logic

impact:
	python -m a22a.impact.player_value

market:
	python -m a22a.market.ingest

clv:
	python -m a22a.market.clv
