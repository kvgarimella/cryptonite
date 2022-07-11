## Garbled Circuit Benchmarks
This directory contains the raw data from benchmarking ReLU Garbling and Evaluation using the [`fancy-garbling`](https://github.com/GaloisInc/swanky/tree/master/fancy-garbling) rust library on two machines:

1) Intel Atom Z8350 2GB DDR3 32GB  (an embedded, client-like device)
2) AMD EPYC 7502 32-Core Processor (a server)

You will find the exact benchmarking file here:
https://github.com/mc2-project/delphi/blob/master/rust/crypto-primitives/benches/garbling.rs.

In order to run the benchmark, you will need to install rust. Simply run the following to benchmark the garble and eval functions:
``` bash
cargo bench --bench garbling
```

Reports are generated if gnuplot is installed. Otherwise, refer to the appropriate estimates.json file. Benchmarks are incorporated into our simulator at [`../simulator/experiments/utils/gc_models.py`](https://github.com/asplos-anonymous-22/asplos-2022-sub/blob/main/simulator/experiments/utils/gc_models.py)

