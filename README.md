## Diffusion-based Graph-agnostic Clustering

### Test environment


### Datasets 
Cora, Citeseer, Cornell and Texas datasets are included in dataset/ path. 
For other 8 datasets included in the paper, we load from torch_geometric.


To reproduce the results in the paper, please run the my_train7.py file, The hyperparameters required for each datasets are listed in Table 7. An example script on Texas graph are given. 

Run the following command to cluster Texas graph: 

```
bash texas_script.sh
```
