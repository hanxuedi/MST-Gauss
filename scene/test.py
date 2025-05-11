import torch
import itertools
coo_combs = list(itertools.combinations(
        range(4), 2)
    )
print(coo_combs)
