## Simulator

This directory contains the Private Inference simulator. Install the required packages by running:
```bash
pip install -r requirements.txt
``` 

### Parameters
Run the following to see the parameters for both the Server- and Client-Garbler:
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
python simulate_client_garbler.py --sim-time=1440 --client-storage=16 --server-storage=1000 --upload-bandwidth=865 --download-bandwidth=135 --network=resnet32 --dataset=cifar10 --num-threads-client=4 --num-threads-server-he=31 --num-threads-server-gc=32 --start=0.001 --end=0.01 --step=45 --number-of-runs=50 --name=example
```
This will create an `examples` directory containing the results of the specified simulation. 
