import numpy as np
from typing import Union, List

import torch
import torchmetrics
import pandas as pd

from torchmetrics import Metric, MeanSquaredError
from torch import Tensor
from icecream import ic

def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenation along the zero dimension."""
    x = x if isinstance(x, (list, tuple)) else [x]
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)


class CustomAUROC(Metric):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        for i in range(num_labels):
            self.add_state(f"preds_{i}", default=[], dist_reduce_fx='cat')
            self.add_state(f"targets_{i}", default=[], dist_reduce_fx='cat')
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        preds = preds.T
        targets = targets.T.long()
        mask = mask.T
        
        for i in range(self.num_labels):
            current_preds = getattr(self, f"preds_{i}")
            current_targets = getattr(self, f"targets_{i}")
            
            current_preds.append(preds[i][mask[i]])
            current_targets.append(targets[i][mask[i]])
            
            setattr(self, f"preds_{i}", current_preds)
            setattr(self, f"targets_{i}", current_targets)
    
    def compute(self):
        if not getattr(self, f"preds_0"):
            print("The ``compute`` method of metric CustomAUROC was called before the ``update`` method \
                  which may lead to errors, as metric states have not yet been updated")
            return torch.tensor(0)
        results = []
        for i in range(self.num_labels):
            preds = dim_zero_cat(getattr(self, f"preds_{i}"))
            targets = dim_zero_cat(getattr(self, f"targets_{i}"))
            
            if len(np.unique(targets.cpu())) == 1:
                print(f'Warning: Found a task with targets all 0s or all 1s')
                results.append(float('nan'))
                continue
                
            results.append(torchmetrics.functional.auroc(preds, targets, 'binary').cpu())
           
        return torch.tensor(np.nanmean(results)) 


class DeltaAveragePrecision(Metric):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        for i in range(num_labels):
            self.add_state(f"preds_{i}", default=[], dist_reduce_fx='cat')
            self.add_state(f"targets_{i}", default=[], dist_reduce_fx='cat')
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        preds = preds.T
        targets = targets.T.long()
        mask = mask.T
        
        for i in range(self.num_labels):
            current_preds = getattr(self, f"preds_{i}")
            current_targets = getattr(self, f"targets_{i}")
            
            current_preds.append(preds[i][mask[i]])
            current_targets.append(targets[i][mask[i]])
            
            setattr(self, f"preds_{i}", current_preds)
            setattr(self, f"targets_{i}", current_targets)
    
    def compute(self):
        if not getattr(self, f"preds_0"):
            print("The ``compute`` method of metric DeltaAveragePrecision was called before the ``update`` method \
                  which may lead to errors, as metric states have not yet been updated")
            return torch.tensor(0)
        results = []
        for i in range(self.num_labels):
            preds = dim_zero_cat(getattr(self, f"preds_{i}"))
            targets = dim_zero_cat(getattr(self, f"targets_{i}"))
            ap = torchmetrics.functional.average_precision(preds, targets, 'binary').cpu()
            base_rate = targets.cpu().numpy().mean()
            delta_ap = ap - base_rate
            results.append(delta_ap)
           
        return torch.tensor(np.nanmean(results))


