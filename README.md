# Test Different GNN Kernels
This is the final project of MATH8013.

1. How does #conv-layers affect performance?
2. How dows Batch Normalization affect performance?
3. How does the aggregation method affect performance?
4. How does the normalized A affect performance?
5. How does the order of Linear and A affect performance?

## Environment
- pytorch 1.9.0
- pyg 2.0.3 (pytorch geometric)
- torchmetrics 0.7.3
- seaborn 0.11.2
- matplotlib 3.5.1

## Dataset
We focus on three semi-supervised node classification tasks.
1. CiteSeer
2. Cora
3. PubMed

## Run
Simply use ```bash run.sh``` to produce 54 experiments.

## Result
Check `core/visual.ipynb`

