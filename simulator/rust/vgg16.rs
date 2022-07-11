use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_vgg16<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let relu_layers = match num_poly {
        0 => vec![1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29, 32, 34],
        16 => vec![],
        _ => unreachable!(),
    };

    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers:      vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    // Dimensions of input image.
    let resolution=64;
    let input_dims = (batch_size, 3, resolution, resolution);

    let kernel_dims = (64, 3, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (64, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (128, 64, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (128, 128, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (256, 128, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (256, 256, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (256, 256, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 256, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));


    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);
    
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (512, 512, 3, 3);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
    network.layers.push(Layer::LL(conv));
    add_activation_layer(&mut network, &relu_layers);

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));


    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    println!("fc input dims = {:?}", fc_input_dims);
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 4096, rng);
    network.layers.push(Layer::LL(fc));
    add_activation_layer(&mut network, &relu_layers);
    
    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 4096, rng);
    network.layers.push(Layer::LL(fc));
    add_activation_layer(&mut network, &relu_layers);
    
    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 200, rng);
    network.layers.push(Layer::LL(fc));
    assert!(network.validate());
    for layer in &network.layers {
        println!("Layer dim: {:?} {:?}", layer.input_dimensions(), layer.is_linear());
    }

    network


}
