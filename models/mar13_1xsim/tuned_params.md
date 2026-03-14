# Tuned values and params

## Tuning search
- Ridge alphas: 0.01,0.05,0.1,0.5,1,10,20,100
- GBM n_estimators: 100,200,300,400,500
- GBM max_depth: 1,2,3,4
- n_folds: 15

## Run
- load_split: splits/pooled_80_20
- simulated: 151 simulated subjects
- seed: 42

## pooled split (chosen values)
- Ridge alpha baseline: 0.01, extended: 20.0
- GBM n/d baseline: 100/4, extended: 200/3

## LSTM
- epochs: 100
- hidden: 64
- layers: 2
- dropout: 0.3
- lr: 0.002
- weight_decay: 0.001
- batch_size: 8
- val_frac: 0.2