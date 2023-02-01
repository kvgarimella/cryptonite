# Characterizing and Optimizing End-to-End Systems for Private Inference

This anonymous repo contains the codes for our submission. The high-level directories are:

1. `garbled_circuits`: contains the raw data for benchmarking ReLU Garbling and Evaluation on the embedded device and the server
2. `layer_parallel_HE`: contains our code and the raw data for enable layer-parallel homomorophic evaluation of linear layers
3. `simulator`: our PI simulator used to generate all plots and results from the paper
4. `artifact` : contains scripts to generate key figures in the paper

Please refer to the READMEs within each subdirectory to learn how we collected data and how to run the simulator.

## Installation
First, clone this repo:
```bash
git clone https://github.com/asplos-anonymous-22/asplos-2022-sub.git
```
Then, change directories and install the Python dependencies:
```bash
cd asplos-2022-sub
pip install -r requirements.txt
```
As a simple test, navigate to the `simulator/experiments` sub-directory and run:
```bash
cd simulator/experiments
python simulate_server_garbler.py
```
This will run a single simulation of the Server-Garbler protocol and store the results in a directory called `tmp`.

## Artifact Evaluation
Please refer to the README within the `artifact` directory. 
