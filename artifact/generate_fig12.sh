python ../simulator/experiments/simulate_server_garbler.py --client-nonlinear=18.428 --server-nonlinear=3.59 --client-storage=16 --server-storage=10000 --dataset=cifar10 --network=resnet32 --download-bandwidth=500 --upload-bandwidth=500 --num-threads-client=4 --num-threads-server-gc=32 --num-threads-server-he=1 --number-of-runs=30  --start=0.001 --end=0.01 --step=45 --name=server_garbler_16gb

python ../simulator/experiments/simulate_server_garbler.py --client-nonlinear=18.428 --server-nonlinear=3.59 --client-storage=32 --server-storage=10000 --dataset=cifar10 --network=resnet32 --download-bandwidth=500 --upload-bandwidth=500 --num-threads-client=4 --num-threads-server-gc=32 --num-threads-server-he=1 --number-of-runs=30  --start=0.001 --end=0.01 --step=45 --name=server_garbler_32gb

python ../simulator/experiments/simulate_server_garbler.py --client-nonlinear=18.428 --server-nonlinear=3.59 --client-storage=64 --server-storage=10000 --dataset=cifar10 --network=resnet32 --download-bandwidth=500 --upload-bandwidth=500 --num-threads-client=4 --num-threads-server-gc=32 --num-threads-server-he=1 --number-of-runs=30  --start=0.001 --end=0.01 --step=45 --name=server_garbler_64gb

python ../simulator/experiments/simulate_client_garbler.py --client-nonlinear=3.59 --server-nonlinear=18.428 --client-storage=16 --server-storage=10000 --dataset=cifar10 --network=resnet32 --download-bandwidth=165 --upload-bandwidth=835 --num-threads-client=4 --num-threads-server-gc=32 --num-threads-server-he=31 --number-of-runs=30  --start=0.001 --end=0.01 --step=45 --name=client_garbler_16gb

python client_vs_server.py
