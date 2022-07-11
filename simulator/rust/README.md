## Rust Networks

The Rust files in this subdirectory are configured to work with the [Delphi codebase](https://github.com/mc2-project/delphi). Follow the installation instructions from their README.

## Adding these networks to their codebase
1. Navigate to `delphi/rust/experiments/src`.
2. Copy the rust files above containing the neural networks.
3. Add `pub mod {network_name};` to the `lib.rs` file.
4. Navigate to the `latency` directory.
5. Add a network directory and the appropriate `client.rs` and `server.rs` files (refer to the existing `resnet32` directory as an example).
6. Add the `client.rs` and `server.rs` files as executables in the `Cargo.toml` file found in the `delphi/rust/experiments` directory.

## Extracting Layer-Level Costs
1. Delphi already prints out the layer-level latency costs.
2. You can capture communication and storage between the server and client from [this file](https://github.com/mc2-project/delphi/blob/master/rust/protocols/src/bytes.rs).
1. In order to capture communication from Oblivious Transfer, you will have to add and modify the [ocelot](https://github.com/GaloisInc/ocelot/) library to the delphi codebase. In particular, you can capture any communication between the server and client using [this file](https://github.com/GaloisInc/ocelot/blob/master/src/ot/alsz.rs).
