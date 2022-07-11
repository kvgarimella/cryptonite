## Layer Parallel Homomorphic Evaluation

This directory is based off of [this repo](https://github.com/mc2-project/delphi/tree/master/rust/protocols-sys/c%2B%2B).

Changes to `run_conv.cpp` and `benchmark.cpp` enabled layer-parallelization of homomorphic evaluation for the linear layers. Raw data is found in the `data` subdirectory. 

You will need to install both [`eigen`](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) and build the `c/c++` components in this repo. You can install both components using:
```bash
cmake -S . -B build
cmake --build build
sudo cmake --install build
```
at the root level and `eigen` directory.  To benchmark layer-parallel homomorphic evaluation, simply run:
``` bash
 ./bin/benchmark [0 or 1] [number_of_threads] 
 ```
where `0` and `1` represent CIFAR and TinyImageNet inputs, respectively. Enable the desired network in the `main`
function within `src/bin/benchmark.cpp`. Benchmarks are incorporated into our simulator at [`../simulator/experiments/utils/he_models.py`](https://github.com/asplos-anonymous-22/asplos-2022-sub/blob/main/simulator/experiments/utils/he_models.py)
