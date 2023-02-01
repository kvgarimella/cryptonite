## Artifact Evaluation

This directory contains scripts for generate key figures for our paper.

### Figure 3
Amount of pre-processing data that must be stored on the client device *per inference* for the Server-Garbler protocol. A majority of the cost comes from storing garbled circuits client-side. The cost analysis of GC storage can be found in `non_linear_storagecost_analysis.ipynb`. The number of ReLUs per network and dataset can be calculated using a script within the `pytorch_models` directory. 
Run the following to reproduce the figure:
```bash
python storage_per_inf_server_garbler.py
```
### Figure 4
Latency of homomorphic evaluations, GC garbling, and GC evaluation *per inference* for the Server-Garbler protocol. 
Run the following to reproduce the figure:
```bash
python latency_per_inf_server_garbler.py
```

### Figure 7
Runs a simulation of the Server-Garbler protocol for varying workloads of inference requests and plots the breakdown of end-to-end latency spent online, pre-processing, and waiting in the queue. 
Run:
```bash
sh generate_fig7.sh 
```
This will first run a simulation of the Server-Garbler protocol and average results over 30 independent runs (for our paper, we used 50 runs). The results of the simulation will be stored in `fig7_data` within this directory. Then a python script will launch to parse and plot the results. Expect this script to take roughly 5 to 10 minutes to run.


### Figure 8
Our proposed Client-Garbler protocol. 
Run the following:
```bash
python storage_optimization.py
```

### Figure 9
Our proposed Layer-Parallel Homomorphic Evaluation optimization.
Run the following:
```bash
python compute_optimization.py
```

### Figure 12a
This will compare the baseline Server-Garbler protocol versus our optimized protocol (Client-Garbler, Layer Parallel Homomorphic Evalution, and Wireless Slot Allocation). 
Run the following script:
```bash
sh generate_fig12.sh
```
This will run the Server-Garbler protocol with different client side storage capacities (16, 32, and 64GB) and the optimized protocol at 16GB. Each simulation will average results over 30 runs (we used 50 in the paper). The results will be stored in four new directories within the current one. Then a python script will parse and plot the results. Expect this script to take roughly 20 minutes to run.


### Figure 14
This plot estimates the benefits of future optimizations on private inference. Follow the `generate_14.ipynb` notebook to see the plot at the end. Each bar in Figure 14 is derived from the output of running `e2e_latency.py` within the `experiments/simulator` subdirectory.
