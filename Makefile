# Entry points for the project. src/ holds the importable library, these targets
# run the scripts/ that orchestrate it. Run `make help` to list targets.

.PHONY: help install check results evaluate significance operating_points figures model sensitivity_grid verify_vol_threshold test clean

help:
	@echo "make install           install dependencies"
	@echo "make check             lint, format, type and test gate (same as CI)"
	@echo "make test              run the test suite"
	@echo "make results           script pipeline: evaluation, significance, operating points, metrics figures"
	@echo "make evaluate          walk-forward evaluation -> reports/metrics/"
	@echo "make significance      dependence-aware significance testing -> reports/metrics/"
	@echo "make operating_points  alert-budget operating points -> reports/metrics/"
	@echo "make figures           render metrics figures -> reports/figures/ (SHAP and calibration figures come from notebook 03)"
	@echo "make model             train and save the final model -> models/artifacts/"
	@echo "make sensitivity_grid  threshold and horizon robustness grid -> reports/metrics/, reports/figures/"
	@echo "make verify_vol_threshold  derive, verify, and persist the label threshold into the dataset CSVs"
	@echo "make clean             remove regenerable artifacts (oof predictions, saved model)"

install:
	uv sync --locked --group notebooks

# Sub-makes keep the stages strictly sequential even under `make -j`, since
# significance, operating_points, and figures all read the metrics that evaluate writes.
results:
	$(MAKE) evaluate
	$(MAKE) significance
	$(MAKE) operating_points
	$(MAKE) figures

evaluate:
	uv run python scripts/run_evaluation.py

significance:
	uv run python scripts/significance.py

operating_points:
	uv run python scripts/operating_points.py

figures:
	uv run python scripts/make_figures.py

model:
	uv run python scripts/save_model.py

sensitivity_grid:
	uv run python scripts/sensitivity_grid.py

# Rewrites both dataset CSVs in place after verifying the threshold reproduces the labels.
verify_vol_threshold:
	uv run python scripts/verify_vol_threshold.py

check:
	scripts/check.sh

test:
	uv run pytest

clean:
	rm -f reports/metrics/oof_predictions.csv
	rm -f models/artifacts/final_model_*.joblib
