# La Liga football-analytics engine -- reproducible pipeline.
# Run `make help` for targets. Override the interpreter with `make PYTHON=python` inside an activated venv.
PYTHON ?= .venv/bin/python

.PHONY: help test features xg explain-data explain-rank predict-data predict-eval player-data \
        availability-eval absence-delta-data absence-delta-eval team-profile team-style chance-creation \
        exhibits team-card explain-match

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-20s %s\n", $$1, $$2}'

test:  ## Run all analytics selfchecks (leak-safety + data integrity = the test suite). Fast checks first.
	@set -e; for m in splits explanatory_dataset xg_features predictive_dataset player_features absence_delta_features \
	                  explanatory_rank predictive_rank squad_availability_eval absence_delta_eval team_profile team_style \
	                  chance_creation xpts team_card; do \
		printf "== %s ==\n" $$m; $(PYTHON) -m src.data_tool.$$m --selfcheck >/dev/null 2>&1 \
		&& echo "  PASS" || { echo "  FAIL ($$m)"; exit 1; }; done; echo "ALL SELFCHECKS PASS"

# --- Feature spine (leak-free Elo/DWA + rolling xG) ---
features:  ## Build the leak-free Elo/DWA rating spine -> data/feature_store/final_features.csv
	$(PYTHON) -m src.data_tool.features

xg:  ## Attach per-side rolling xG (xG_Diff / xGA_Diff / netxG_Diff) to the feature store
	$(PYTHON) -m src.data_tool.xg_features

# --- Explanatory layer: which in-match stats decide the 1X2 outcome ---
explain-data:  ## Build the explanatory match-stats dataset (home-minus-away diffs + 1X2)
	$(PYTHON) -m src.data_tool.explanatory_dataset

explain-rank:  ## Rank which stats decide a match (tree ensemble + permutation + TreeSHAP, chrono holdout)
	$(PYTHON) -m src.data_tool.explanatory_rank

# --- Predictive layer: T-1 1X2 forecast (single-source API) ---
predict-data:  ## Build the T-1 predictive spine from the API (Elo + DWA + rolling xG + candidates)
	$(PYTHON) -m src.data_tool.predictive_dataset

predict-eval:  ## Does rolling xG add predictive lift over Elo? (nested ladder, RPS/log-loss, bootstrap)
	$(PYTHON) -m src.data_tool.predictive_rank

# --- Player-level features (squad availability; pre-registered) ---
player-data:  ## Build the T-1 available-squad rating (announced XI x prior player ratings)
	$(PYTHON) -m src.data_tool.player_features

availability-eval:  ## Pre-registered: does the squad rating add predictive lift over full-history Elo?
	$(PYTHON) -m src.data_tool.squad_availability_eval

absence-delta-data:  ## Build the absence-delta feature (today's XI vs the team's own recent norm)
	$(PYTHON) -m src.data_tool.absence_delta_features

absence-delta-eval:  ## Pre-registered: does the absence delta add lift over Elo? (the final feature test)
	$(PYTHON) -m src.data_tool.absence_delta_eval

# --- Descriptive: what characterizes a great team (Elo as the target) ---
team-profile:  ## Profile what characterizes a great La Liga team (style signature vs quality core)
	$(PYTHON) -m src.data_tool.team_profile

team-style:  ## Map team-season style at fixed quality (continuum, not archetypes; the style map)
	$(PYTHON) -m src.data_tool.team_style

chance-creation:  ## Anatomy of chance creation: what great xG is made of (two levers; box penetration)
	$(PYTHON) -m src.data_tool.chance_creation

# --- Packaging: the demo CLI + the committed figure exhibits ---
exhibits:  ## Render the portfolio figures (style map, what-decides, xG table) -> docs/figures/
	$(PYTHON) -m src.data_tool.exhibits

team-card:  ## A team's quality+style fingerprint. Usage: make team-card TEAM="Atletico Madrid" [SEASON=24/25]
	$(PYTHON) -m src.data_tool.team_card card "$(TEAM)" $(if $(SEASON),--season $(SEASON),)

explain-match:  ## Explain a completed match. Usage: make explain-match HOME="Real Madrid" AWAY=Barcelona [SEASON=24/25]
	$(PYTHON) -m src.data_tool.team_card match "$(HOME)" "$(AWAY)" $(if $(SEASON),--season $(SEASON),)
