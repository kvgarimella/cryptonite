python ../simulator/experiments/simulate_server_garbler.py --client-nonlinear=18.428 --server-nonlinear=3.59 --client-storage=128 --server-storage=10000 --dataset=tinyimagenet --network=resnet18 --download-bandwidth=500 --upload-bandwidth=500 --num-threads-client=4 --num-threads-server-gc=4 --num-threads-server-he=1 --number-of-runs=30  --start=0.0001 --end=0.001 --step=45 --name=fig7_data

python latency_breakdown_server_garbler.py
