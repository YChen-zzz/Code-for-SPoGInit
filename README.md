# SPoGInit
Code for Exploring and Improving Initialization for Deep Graph Neural Networks: A Signal Propagation Perspective

# What is SPoGInit?
It is a Graph Neural Network (GNN) initialization searching method, called Signal Propagation on Graph-guided Initialization. Given a GNN architecture, it optimizes the initialization to stabilize **forward signal propagation (FSP)**, **backward signal propagation (BSP)**, **graph embedding variation(GEV)** as the model depth increases.

For more details, please read the paper.

This is an early-stage exploration where we employ a simple approach (black-box optimization) to improve GNN initialization. We hope this work inspires further research and discussion on initialization strategies for deep Graph Neural Networks.



# Examples to use the SPoGInit


## Setup
```
conda env create -n spog python==3.9
conda activate spog

pip install -r requirements.txt
```

We provide two examples to use SPoGInit

## examples for FSP, BSP, GEV in the initializations
We provide the code for comparing the FSP, BSP, GEV of GCN and ResGCN of different initializations on Cora dataset as the model depth increases.

For GCN, you could try:
```
SP_GCN_github.ipynb
```
and for ResGCN, you can try

```
SP_ResGCN_github.ipynb
```

## Examples for training deep GCN
We provide the codes for training ``GCN``, ``ResGCN``, ``gatResGCN`` on the OGBN-Arxiv and Arxiv-year datasets with SPoGIint and other initializations across different depths.

For the experiments on OGBN-Arxiv, you can try (after specifiying the ``data_path`` and ``save_path`` in the sh files)
```
./script/arxiv_GCN.sh

./script/arxiv_ResGCN_gatResGCN.sh 
```

And for the experients on Arxiv-year, you need to first download the data from https://drive.google.com/file/d/1sBxZMbd0m2tdrLFSHMKumKskl0ej3uT1/view?usp=sharing

Then, you can try

```
./script/arxiv_year_GCN.sh

./script/arxiv_year_ResGCN_gatResGCN.sh 
```

If you want to try experiment of different layers, you could try modifying ``num_layers``, 


# More on SPoGInit

There are many hyper-parameters in SPoGInit。
```python
Spog = SpogInit(model,data.edge_index,data.train_mask,data.x.shape[0],data.x.shape[1],dataset.num_classes, device, metric_way="divide_old")
```
When initialization the SPoGInit class,  you should specify the ``metric_way``, you can choose 

  ``divide_old`` by computing raitos of second to second-to-last layer 
  
  ``divide_stable`` by computing the max/min gradient/output norm

Then in the optimization line: 
```python
Spog.zeroincrease_initialization(data.x, data.y,w2=10,max_pati=20,steps=40, lr = 0.05,generate_data = True)
#lr: Learning rate
#steps: Maximum optimization steps
#max_pati: Patience for early stopping
#w1, w2, w3: Metric component weights
#generate_data: Whether to generate random data (x,y) each iteration
```
``SpogInit`` has different optimization functions: 

``zeroincrease_initialization`` Zero-order optimization with search directions across all layers.

``zerosingle_initialization`` Similar to zeroincrease_initialization but uses same direction for all parameters.

``secondorder_initialization`` Second-order initialization searching method. (It will cost much more GPU momery)


Since the GPU momery is not a very big problem in GNN,  we did not optimize GPU momery in this code. If you get CUDA OOM when using SPoGInit, you could try adding ``with torch.no_grad()`` on the SPoGInit's:
```python
  forward_norm = self.model.print_all_x(x, self.edge)
```
which is in the generate_metrics function. Or you could just let the print_all_x directly return the output norm rather than the whole output matrix, then it will have the same GPU memory with normal initializations.
