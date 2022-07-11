// Used to extract layer-level communication / storage for the Client-Garbler protocol

use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_single_layer_conv<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
    kernel_size: usize,
    image_size: usize,
    in_channels: usize, 
    out_channels: usize
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let relu_layers = match num_poly {
        0 => vec![1],
        1 => vec![],
        _ => unreachable!(),
    };


    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    // Dimensions of input image.
    let input_dims = (batch_size, in_channels, image_size, image_size);

    let kernel_dims = (out_channels, in_channels, kernel_size, kernel_size);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (out_channels, out_channels, 1,1);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));

    for layer in &network.layers {
        println!("Layer dim: {:?}", layer.input_dimensions());
    }

    assert!(network.validate());

    network
}
