## Simulator

This directory contains the Private Inference simulator. The directories:
1. `data`: contains measurements from running the [Delphi codebase](https://github.com/mc2-project/delphi) on our specified networks. We ran the client and server on the same machine (separate processes) on our AMD EPYC 7502 32-Core server.
2. `experiments`: contains our simulators for the Server- and Client-Garbler protocols. Also contains a `utils` subdirectory which incorporates our measurements. 
3. `rust`: Rust files to add VGG-16, ResNet-18, and a shallow network (for understanding Client-Garbler) to Delphi.

### Parameters
Within the `experiments` directory, run the following to see the parameters for both the Server- and Client-Garbler:
```bash
python simulate_client_garbler.py -h
```
| Parameter                 | Description                                       | Units               |
|---------------------------|---------------------------------------------------|---------------------|
| `--sim-time`              | Total Simulated Time                              | Minutes             |
| `--client-storage`        | Client-Side Storage                               | GB                  |
| `--server-storage`        | Server-Side Storage                               | GB                  |
| `--client-nonlinear`      | Client-Side storage required per ReLU             | KB/ReLU             |
| `--server-nonlinear`      | Server-Side storage required per ReLU             | KB/ReLU             |
| `--upload-bandwidth`      | Bandwidth for Client to Server                    | Mbps                |
| `--download-bandwidth`    | Bandwidth for Server to Client                    | Mbps                |
| `--network`               | Neural Network (resnet32, vgg16, resnet18)                                |                 |
| `--dataset`               | Dataset (cifar10, tinyimagenet)                                    |              |
| `--num-threads-client`    | Number of threads for fancy-garbling client-side  |                |
| `--num-threads-server-he` | Number of threads for Layer-Parallel HE           |                |
| `--num-threads-server-gc` | Number of threads for fancy-garbling server-side  |                 |
| `--start`                 | Starting Arrival Rate                             | requests per second |
| `--end`                   | Ending Arrival Rate                               | requests per second |
| `--step`                  | Number of evenly spaced arrival rates             |                |
| `--number-of-runs`        | Number of independent experiments to average over |                |
| `--name`                  | Name of output directory                          |                |
### Example
To generate data for the Client-Garbler protocol using a ResNet32 network on CIFAR inputs, you could run:
```bash
python simulate_client_garbler.py --sim-time=1440 --client-storage=16 --server-storage=1000 --upload-bandwidth=865 --download-bandwidth=135 --network=resnet32 --dataset=cifar10 --num-threads-client=4 --num-threads-server-he=31 --num-threads-server-gc=32 --start=0.001 --end=0.01 --step=45 --number-of-runs=50 --name=examples
```
This will create an `examples` directory containing the results of the specified simulation. For each simulation run, the following information is collected:
1. average inference times (including time spent pre-processing)
2. arrival rates (as a sanity check)
3. average number of client inferences serviced
4. average time spent waiting in the queue
5. average number of outstanding requests
6. average time spent pre-processing during the online phase
