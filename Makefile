viz:
	@firefox data/04_visualization/risk_return.html

viz_hull:
	@firefox data/04_visualization/convex_hull.html

run:
	@python -m investments.main

run_store_logs:
	@python -m investments.main 2>data/06_logs/log.info
