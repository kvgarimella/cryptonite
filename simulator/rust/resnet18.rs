use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

// It may be the case that down-sampling happens here.
fn conv_block<R: RngCore + CryptoRng>(
    nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    (k_h, k_w): (usize, usize),
    num_output_channels: usize,
    stride: usize,
    relu_layers: &[usize],
    rng: &mut R,
) {
    let cur_input_dims = nn.layers.last().as_ref().unwrap().output_dimensions();
    let c_in = cur_input_dims.1;

    let (conv_1, _) = sample_conv_layer(
        vs,
        cur_input_dims,
        (num_output_channels, c_in, k_h, k_w),
        stride,
        Padding::Same,
        rng,
    );
    nn.layers.push(Layer::LL(conv_1));
    add_activation_layer(nn, relu_layers);
    let cur_input_dims = nn.layers.last().as_ref().unwrap().output_dimensions();
    let c_in = cur_input_dims.1;

    let (conv_2, _) = sample_conv_layer(
        vs,
        cur_input_dims,
        (c_in, c_in, k_h, k_w), // Kernel dims
        1,                      // Stride = 1
        Padding::Same,
        rng,
    );
    nn.layers.push(Layer::LL(conv_2));
    add_activation_layer(nn, relu_layers);
}

// There's no down-sampling happening here, strides are always (1, 1).
fn iden_block<R: RngCore + CryptoRng>(
    nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    (k_h, k_w): (usize, usize),
    relu_layers: &[usize],
    rng: &mut R,
) {
    let cur_input_dims = nn.layers.last().as_ref().unwrap().output_dimensions();
    let c_in = cur_input_dims.1;

    let (conv_1, _) = sample_conv_layer(
        vs,
        cur_input_dims,
        (c_in, c_in, k_h, k_w), // Kernel dims
        1,                      // stride
        Padding::Same,
        rng,
    );
    nn.layers.push(Layer::LL(conv_1));
    add_activation_layer(nn, relu_layers);

    let (conv_2, _) = sample_conv_layer(
        vs,
        cur_input_dims,
        (c_in, c_in, k_h, k_w), // Kernel dims
        1,                      // stride
        Padding::Same,
        rng,
    );
    nn.layers.push(Layer::LL(conv_2));
    add_activation_layer(nn, relu_layers);
}

fn resnet_block<R: RngCore + CryptoRng>(
    nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    layer_size: usize,
    c_out: usize,
    kernel_size: (usize, usize),
    stride: usize,
    relu_layers: &[usize],
    rng: &mut R,
) {
    conv_block(nn, vs, kernel_size, c_out, stride, relu_layers, rng);
    for _ in 0..(layer_size - 1) {
        iden_block(nn, vs, kernel_size, relu_layers, rng)
    }
}

pub fn construct_new_resnet_18<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    use std::collections::HashSet;
    let mut relu_layers = Vec::new();
    //num_poly = 0 for our experiments
    let poly_layers = ((32 - num_poly)..32).collect::<Vec<_>>();
    
    let poly_layers: HashSet<_> = poly_layers.into_iter().collect();
    for l in 0..32 {
        if !poly_layers.contains(&l) {
            relu_layers.push(2 * l + 1);
        }
    }

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
    // Dimensions of input image. 32x32 for cifar10, 64x64 for tinyimagenet
    let input_dims = (batch_size, 3, 64,64);
    // Dimensions of first kernel
    let kernel_dims = (64, 3, 3, 3);

    // Sample a random kernel.
    let (conv_1, _) = sample_conv_layer(
        vs,
        input_dims,
        kernel_dims,
        1, // Stride
        Padding::Same,
        rng,
    );
    network.layers.push(Layer::LL(conv_1));
    add_activation_layer(&mut network, &relu_layers);
    resnet_block(
        &mut network,
        vs,
        2,      // layer_size,
        64,     // c_out
        (3, 3), // kernel_size
        1,      // stride
        &relu_layers,
        rng,
    );

    resnet_block(
        &mut network,
        vs,
        2,      // layer_size,
        128,     // c_out
        (3, 3), // kernel_size
        2,      // stride
        &relu_layers,
        rng,
    );

    resnet_block(
        &mut network,
        vs,
        2,      // layer_size,
        256,     // c_out
        (3, 3), // kernel_size
        2,      // stride
        &relu_layers,
        rng,
    );

    resnet_block(
        &mut network,
        vs,
        2,      // layer_size,
        512,     // c_out
        (3, 3), // kernel_size
        2,      // stride
        &relu_layers,
        rng,
    );

 
    let avg_pool_input_dims = network.layers.last().unwrap().output_dimensions();
    network.layers.push(Layer::LL(sample_avg_pool_layer(
        avg_pool_input_dims,
        (2, 2),
        2,
    )));

    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    // output_dims = 10 for cifar10, 200 for tinyimagenet
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 200, rng);
    network.layers.push(Layer::LL(fc));
    assert!(network.validate());
    for layer in &network.layers {
        println!("Layer dim: {:?} {:?}", layer.input_dimensions(), layer.is_linear());
    }

    network
}
