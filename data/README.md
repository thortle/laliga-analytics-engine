Licensed match data is not included in this public repo (TheStatsAPI, La Liga).
The code, the committed figures, and the executed notebook stand alone without it.
The data is a licensed subscription API (TheStatsAPI, JSON). The pipeline ingests it to
raw JSONL and flattens it to per-match CSVs under data/raw/thestatsapi/; the build steps
(make features, make explain-data, ...) read those CSVs. Reproducing them requires your own
TheStatsAPI subscription -- the public repo cannot ship the licensed data.
