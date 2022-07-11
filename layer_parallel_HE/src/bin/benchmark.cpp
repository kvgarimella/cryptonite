/*
 *  Benchmark homomorphic convolution
 *
 *  Created on: August 10, 2019
 *      Author: ryanleh */
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <limits>
#include <omp.h>


#include <math.h>
#include "conv2d.h"
#include "run_conv.cpp"

using namespace std;

/* Generates a random image and filters with the given dimensions and times
 * convolution operation */
void benchmark(int image_h, int image_w, int filter_h, int filter_w, 
        int inp_chans, int out_chans, int stride, bool padding_valid) {
    // Create uniform distribution
    // We only sample up to 20 bits because the plaintext evaluation
    // doesn't support 128 bit numbers so we need to make sure 
    // multiplication doesn't overfloow 64 bits
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<u64> dis(0, 1<<20);

    // Create Eigen inputs for the plaintext and raw arrays for HE
    EImage eimage(inp_chans);
    Image image = new Channel[inp_chans];
    for (int chan = 0; chan < inp_chans; chan++) {
        EChannel tmp_chan(image_h, image_w);
        image[chan] = new u64[image_h*image_w];
        for (int idx = 0; idx < image_h*image_w; idx++) {
            u64 val = dis(gen);
            tmp_chan(idx/image_w, idx%image_w) = val;
            image[chan][idx] = val;
        }
        eimage[chan] = tmp_chan;
    }

    EFilters efilters(out_chans);
    Filters filters = new Image[out_chans];
    for (int out_c = 0; out_c < out_chans; out_c++) {
        EImage tmp_img(inp_chans);
        filters[out_c] = new Channel[inp_chans];
        for (int inp_c = 0; inp_c < inp_chans; inp_c++) {
            EChannel tmp_chan(filter_h, filter_w);
            filters[out_c][inp_c] = new u64[filter_h*filter_w];
            for (int idx = 0; idx < filter_h*filter_w; idx++) {
                u64 val = dis(gen);
                tmp_chan(idx/filter_w, idx%filter_w) = val;
                filters[out_c][inp_c][idx] = val;
            }
            tmp_img[inp_c] = tmp_chan;
        }
        efilters[out_c] = tmp_img;
    }
    
    cout << "\n\n--------------------------------------------\n";
    cout << "Image shape: (" << image_h << "x" << image_w << ", " << inp_chans
        << ") - Filters shape: (" << filter_h << "x" << filter_w << ", " << out_chans
        << ") - Padding = " << (padding_valid ? "VALID" : "SAME") << ", Stride = (" <<
        stride << "x" << stride << ")\n";
    cout << "--------------------------------------------\n\n";

    bool pass = run_conv(image, filters, image_h, image_w, filter_h, filter_w, inp_chans, out_chans, padding_valid, stride, stride, 0);

    if (pass)
        cout << "PASS" << endl;
    else
        cout << "FAIL" << endl;
}




int main(int argc, char** argv)
{
    SerialCT key_share;
    ClientFHE cfhe = client_keygen(&key_share);
    ServerFHE sfhe = server_keygen(key_share); 

    /*  //VGG-16 Benchmarks
      int img_size[13]    = {64, 64, 32,   32,  16,  16,  16,   8,   8,   8,   4,   4,   4};
      int out_c[13]       = {64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512};
      int in_c[13]        = {3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512};
      int kernel_size[13] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
      int stride[13]      = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      int num_layers      = 13;
      */
     

    /*   //ResNet-32 Benchmarks
      int img_size[31]    = {64, 64,64,64,64,64,64,64,64,64,64, 64,32,32,32,32,32,32,32,32,32, 32,16,16,16,16,16,16,16,16,16};
      int in_c[31]        = {3, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64};
      int out_c[31]       = {16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};
      int kernel_size[31] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
      int stride[31]      = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
      int num_layers      = 31;
      */
      



     // ResNet-18 Benchmarks

      int img_size[17]    = { 64, 64,64,64,64, 64,32,32,32,     32,16,16,16,     16,8,8,8};
      int kernel_size[17] = { 3,  3,3,3,3,     3,3,3,3,         3,3,3,3,         3,3,3,3};
      int out_c[17]       = { 64, 64,64,64,64, 128,128,128,128, 256,256,256,256, 512,512,512,512};
      int in_c[17]        = { 3,  64,64,64,64, 64,128,128,128,  128,256,256,256, 256,512,512,512};
      int stride[17]      = { 1,  1,1,1,1,     2,2,2,2,         2,2,2,2,         2,2,2,2};
      int num_layers      = 17;
     



    printf("Number of Layers : %d\n", num_layers);

    printf("Number of args: %d\n", argc);



    int number_of_threads = atoi(argv[1]);
    int is_tiny = atoi(argv[2]);

    if (is_tiny == 0) {
     for (int i = 0; i < num_layers; i++) {
        img_size[i] = img_size[i] / 2;
    }
    }
     

    printf("CIFAR/Tiny (0/1): %d\n", is_tiny);
    printf("Number of Threads: %d\n", number_of_threads);


    ClientShares* ptr_client_shares;
    ptr_client_shares = (ClientShares*)malloc(num_layers * sizeof(ClientShares));

    ServerShares* ptr_server_shares;
    ptr_server_shares = (ServerShares*)malloc(num_layers * sizeof(ServerShares));

    printf("Encrypting...\n");
    double start_time_enc = omp_get_wtime();
    for (int i = 0; i < num_layers; i++) {
        ptr_client_shares[i] = client_encrypt(cfhe,  img_size[i], img_size[i], kernel_size[i],
        kernel_size[i], in_c[i], out_c[i], 0, stride[i], stride[i]);
    }
    double end_time_enc = omp_get_wtime();
    printf("All Encryptions: %f\n", end_time_enc - start_time_enc);

    for (int i = 0; i < num_layers; i++) {
        ptr_server_shares[i] = server_preprocess(cfhe, sfhe , img_size[i], img_size[i], kernel_size[i],
        kernel_size[i], in_c[i], out_c[i], 0, stride[i], stride[i]);
    }




    printf("Performing convolutions\n");
    double start_time_conv = omp_get_wtime();
    #pragma omp parallel for num_threads(number_of_threads) schedule(static)
    for (int i = 0; i < num_layers; i++)
    {

        he_conv(cfhe, sfhe, ptr_client_shares[i], ptr_server_shares[i],
                img_size[i], img_size[i], kernel_size[i], 
                kernel_size[i], in_c[i], out_c[i], 0, stride[i],stride[i]);
  
    }

    double end_time_conv = omp_get_wtime();
    printf("All Convs: %f\n", end_time_conv - start_time_conv);


    printf("Decrypting...\n");
    double start_time_dec = omp_get_wtime();
    for (int i = 0; i < num_layers; i++) {
        client_decrypt(cfhe, sfhe,  ptr_client_shares[i], ptr_server_shares[i],
                img_size[i], img_size[i], kernel_size[i], 
                kernel_size[i], in_c[i], out_c[i], 0, stride[i],stride[i]);
    }
    double end_time_dec = omp_get_wtime();
    printf("All Decryptions: %f\n", end_time_dec - start_time_dec);

    free(ptr_server_shares);
    free(ptr_client_shares);
    return 0;
}
