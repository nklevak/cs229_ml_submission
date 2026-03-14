This is my cleaned up final project code for CS229, with only the filesn relevant to the CS229 assignment. It is run with uv.

This directory includes:
- writeups/
    - this includes the pdfs of the proposal and the milestone
- scripts/
    - preprocess.py (preprocesses the data to fit the ridge, decision tree, and lstm methods)
    - simulate.py (simulates all of the additional subjects we will try using to train)
    - lstm.py (trains the lstm model)
    - run_all_models.py (runs the ridge baseline + extended, gradient boosted decision tree + extended, and lstm)
    - make_plots.py (plots the MAE and r squared figures, as well as the feature importance plots)
- cleaned_exp_data/
    - has the original datasets we are training on
- splits/
    - has the established split IDs we are using for our pooled analysis (80% training, 20% testing)
- models/
    - has the saved models from different runs within different sub-folders


STEPS IN MY PIPELINE:
1) uv sync
2) run uv run python scripts/make_pooled_split.py to generate splits/pooled_80_20.json
3) run uv run python scripts/run_all_models.py --models-dir models/mar13_nosim
4) uv run python scripts/run_all_models.py --sim-multiplier 1 --models-dir models/mar13_1xsim
5) make more plots????

