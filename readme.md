# Edge-Centric Reweighting with Global Semantics for Graph Contrastive Learning

Implementation of ECRGS

## Requirements

This repository has been tested with the following packages:
- Python == 3.8
- PyTorch == 1.9.1
- DGL
- faiss-gpu or faiss-cpu
- cuda == 11.1

## Important Hyperparameters

- `dataname`: Name of the dataset. Could be one of `[citeseer, cs, photo, pubmed, computer]`. 
- `nclusters`:Number of clusters (prototypes) in k-means. 
- `alpha`:  coefficient between contrastive loss and edge-centric loss. 
- `lr1`: Learning rate for ECRGS.
- `lr2`: Learning rate for linear evaluator.
- `epoch1`: Training epochs for ECRGS.
- `epoch2`: Training epochs for linear evaluator.
- `der`: Edge drop rate of graph augmentation.
- `dfr`: Feature drop rate of graph augmentation.
- `clustering`: Whether to do the downstream node clustering task.
- `hc`: high-confidence clustering nodes selection portion.

Please refer to [args.py](args.py) for the full hyper-parameters.

## How to Run

Pass the above parameters to `main.py`. For example:

```python
# cora
python main.py --dataname cora --nclusters 14 --alpha 1 --epoch1 50 --epoch2 1000 
# citeseer
python main.py --dataname citeseer --nclusters 14 --alpha 1 --epoch1 50 --epoch2 1000 
# pubmed
python main.py --dataname pubmed --nclusters 90 --alpha 1 --epoch1 45 --epoch2 2000
# photo
python main.py --dataname photo --nclusters 70 --alpha 3 --epoch1 60 --lr1 1e-4 --epoch2 5000 --lr2 1e-3 --proj_dim 64 
# computer
python main.py --dataname comp --nclusters 40 --alpha 1 --epoch1 120 --lr1 1e-4 --epoch2 5000 --lr2 1e-3 
# WikiCS
python main.py --dataname wikics --nclusters 40 --alpha 1 --epoch1 30 --epoch2 1200 
```
