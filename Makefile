viz:
	@firefox data/04_outputs/risk_return.html

viz_hull:
	@firefox data/04_outputs/convex_hull.html

run:
	@python -m investments.main 2>data/06_logs/log.info