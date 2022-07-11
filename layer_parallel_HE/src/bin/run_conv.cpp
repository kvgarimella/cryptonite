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

#include "math.h"
#include <time.h>
#include "conv2d.h"
#include "fc_layer.h"
#include "im2col.h"
#include "interface.h"
#include <omp.h>

using namespace std;

bool pass = true;

/* Uses the C interface to perform convolution */
ClientShares interface_conv(ServerFHE &sfhe, ClientFHE &cfhe, Metadata data, Image image, Filters filters,
        Image linear_share) {
    chrono::high_resolution_clock::time_point time_start, time_end;

    double time_start_encrypt = omp_get_wtime();

    ClientShares client_shares = client_conv_preprocess(&cfhe, &data, image);
    
    double time_end_encrypt = omp_get_wtime();
    double time_diff_encrypt = time_end_encrypt - time_start_encrypt; 
    printf("Client Encrypt [ %f ]\n", time_diff_encrypt);
    // ------------------------------------------------ 
    double time_start_server_preprocess = omp_get_wtime();

    char**** masks = server_conv_preprocess(&sfhe, &data, filters);
    ServerShares server_shares = server_conv_preprocess_shares(&sfhe, &data, linear_share);
    
    double time_end_server_preprocess = omp_get_wtime();
    double time_diff_server_preprocess = time_end_server_preprocess - time_start_server_preprocess; 
    printf("Server Preprocessing[ %f ]\n", time_diff_server_preprocess);

    // ------------------------------------------------ 
    double time_start_eval_conv = omp_get_wtime();
    
    server_conv_online(&sfhe, &data, client_shares.input_ct, masks, &server_shares);
    
    double time_end_eval_conv = omp_get_wtime();
    double time_diff_eval_conv =  time_end_eval_conv - time_start_eval_conv;
    printf("Convolution [ %f ]\n", time_diff_eval_conv);
    // ------------------------------------------------ 
    client_shares.linear_ct.inner = (char*) malloc(sizeof(char)*server_shares.linear_ct.size);
    client_shares.linear_ct.size = server_shares.linear_ct.size;
    memcpy(client_shares.linear_ct.inner, server_shares.linear_ct.inner, server_shares.linear_ct.size);

    // ------------------------------------------------ 
    double time_start_decrypt = omp_get_wtime();

    client_conv_decrypt(&cfhe, &data, &client_shares);
    
    double time_end_decrypt = omp_get_wtime();
    double time_diff_decrypt = time_end_decrypt - time_end_encrypt; 
    printf("Client Decrypt [ %f ]\n", time_diff_decrypt);

    // ------------------------------------------------ 
  
    // Free C++ allocations
    server_conv_free(&data, masks, &server_shares);
    return client_shares;
}

void he_conv(ClientFHE &cfhe, ServerFHE &sfhe, ClientShares &client_shares,
    ServerShares &shares, int image_h, int image_w, int filter_h,
        int filter_w, int inp_chans, int out_chans, bool pad_valid, int stride_h,
        int stride_w) {

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<u64> dis(0, 1<<20);



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

    Metadata data = conv_metadata(cfhe.encoder, image_h, image_w, filter_h, filter_w, inp_chans, 
        out_chans, stride_h, stride_w, pad_valid);

    char**** masks = server_conv_preprocess(&sfhe, &data, filters);

    server_conv_online(&sfhe, &data, client_shares.input_ct, masks, &shares);

}

void free_share(Image share, int chans) {
    for (int chan = 0; chan < chans; chan++) {
        delete[] share[chan];
    }
    delete[] share;
}

ClientShares client_encrypt(ClientFHE &cfhe, int image_h, int image_w, int filter_h,
        int filter_w, int inp_chans, int out_chans, bool pad_valid, int stride_h,
        int stride_w) {

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

    Metadata data = conv_metadata(cfhe.encoder, image_h, image_w, filter_h, filter_w, inp_chans, 
        out_chans, stride_h, stride_w, pad_valid);

    ClientShares client_shares = client_conv_preprocess(&cfhe, &data, image);

    return client_shares;

}

void client_decrypt(ClientFHE &cfhe, ServerFHE &sfhe, ClientShares &client_shares,
    ServerShares &server_shares, int image_h, int image_w, int filter_h,
        int filter_w, int inp_chans, int out_chans, bool pad_valid, int stride_h,
        int stride_w) {

   Metadata data = conv_metadata(cfhe.encoder, image_h, image_w, filter_h, filter_w, inp_chans, 
        out_chans, stride_h, stride_w, pad_valid);

    client_shares.linear_ct.inner = (char*) malloc(sizeof(char)*server_shares.linear_ct.size);
    client_shares.linear_ct.size = server_shares.linear_ct.size;
    memcpy(client_shares.linear_ct.inner, server_shares.linear_ct.inner, server_shares.linear_ct.size);


   client_conv_decrypt(&cfhe, &data, &client_shares);
}



ServerShares server_preprocess(ClientFHE &cfhe, ServerFHE &sfhe , int image_h, int image_w, int filter_h,
        int filter_w, int inp_chans, int out_chans, bool pad_valid, int stride_h,
        int stride_w) {

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<u64> dis(0, 1<<20);

    Metadata data = conv_metadata(cfhe.encoder, image_h, image_w, filter_h, filter_w, inp_chans, 
        out_chans, stride_h, stride_w, pad_valid);

    Image linear_share = new u64*[out_chans];
    for (int chan = 0; chan < out_chans; chan++) {
        Channel channel = new u64[data.output_h * data.output_w];
        for (int idx = 0; idx < data.output_h*data.output_w; idx++) {
            channel[idx] = gen();
        }
        linear_share[chan] = channel;
    };
    ServerShares server_shares = server_conv_preprocess_shares(&sfhe, &data, linear_share);  
    return server_shares;
}


/* Runs plaintext and homomorphic convolution and compares result */
bool run_conv(Image image, Filters filters, int image_h, int image_w, int filter_h,
        int filter_w, int inp_chans, int out_chans, bool pad_valid, int stride_h,
        int stride_w, bool verbose) {
    
    // PRG for generating shares
    random_device rd;
    mt19937 engine(rd());
    uniform_int_distribution<u64> dist(0, PLAINTEXT_MODULUS);
    auto gen = [&dist, &engine](){
        return dist(engine);
    };

    /* --------------- KeyGen/Preprocessing -------------------- */
    SerialCT key_share;
    ClientFHE cfhe = client_keygen(&key_share);
    ServerFHE sfhe = server_keygen(key_share); 
    
    Metadata data = conv_metadata(cfhe.encoder, image_h, image_w, filter_h, filter_w, inp_chans, 
        out_chans, stride_h, stride_w, pad_valid);
    /* ---------------------------------------------------------- */

    // Generate server's linear secret shares
    Image linear_share = new u64*[out_chans];
    for (int chan = 0; chan < out_chans; chan++) {
        Channel channel = new u64[data.output_h * data.output_w];
        for (int idx = 0; idx < data.output_h*data.output_w; idx++) {
            channel[idx] = gen();
        }
        linear_share[chan] = channel;
    };

    // Convert the raw pointer to an Eigen matrix for plaintext evaluation
    EImage eimage(inp_chans);
    for (int chan = 0; chan < inp_chans; chan++) {
        EChannel echannel(image_h, image_w);
        for (int idx = 0; idx < image_h*image_w; idx++) {
            echannel(idx/image_w, idx%image_h) = image[chan][idx];
        }
        eimage[chan] = echannel;
    }

    EFilters efilters(out_chans);
    for (int o_chan = 0; o_chan < out_chans; o_chan++) {
        EImage eimage(inp_chans);
        for (int chan = 0; chan < inp_chans; chan++) {
            EChannel echannel(filter_h, filter_w);
            for (int idx = 0; idx < filter_h*filter_w; idx++) {
                echannel(idx/filter_h, idx%filter_w) = filters[o_chan][chan][idx];
            }
            eimage[chan] = echannel;
        }
        efilters[o_chan] = eimage;
    }

    cout << "Plaintext:\n";
    double time_start1 = omp_get_wtime();

    EImage pt_result = im2col_conv2D(&eimage, &efilters, pad_valid, stride_h, stride_w);
    // Compute linear shares 
    Image pt_linear_share = new u64*[data.out_chans];
    for (int chan = 0; chan < data.out_chans; chan++) {
        Channel channel = new u64[data.output_h * data.output_w];
        for (int idx = 0; idx < data.output_h*data.output_w; idx++) {
            // We first add the modulus so that we don't underflow the u64
            channel[idx] = (PLAINTEXT_MODULUS + pt_result[chan](idx/data.output_w,idx%data.output_w) - linear_share[chan][idx]) % PLAINTEXT_MODULUS;
        };
        pt_linear_share[chan] = channel;
    };
    
    double time_end1 = omp_get_wtime();
    double time_diff1 = time_end1 - time_start1;
    printf("Plaintext [ %f ]\n\n", time_diff1);

    double time_start_he = omp_get_wtime();

    auto shares = interface_conv(sfhe, cfhe, data, image, filters, linear_share);

    double time_end_he = omp_get_wtime();
    double time_diff_he = time_end_he - time_start_he; 
    printf("Homomorphic [ %f ]\n\n", time_diff_he);
    

    if (verbose) {
        print(pt_linear_share, data.out_chans, data.output_h, data.output_w);
    }

    if (verbose) {
        print(shares.linear, data.out_chans, data.output_h, data.output_w);
    }
    
    // Compare linear results
    for (int i = 0; i < data.out_chans; i++) {
        for (int j = 0; j < data.output_w*data.output_h; j++) {
            if (pt_linear_share[i][j] != shares.linear[i][j]) {
                pass = false;
            }
        }
    }


    // Free stuff
    free_ct(&key_share);
    client_free_keys(&cfhe);
    server_free_keys(&sfhe);
    free(shares.linear_ct.inner);
    client_conv_free(&data, &shares);
    free_share(linear_share, data.out_chans);
    free_share(pt_linear_share, data.out_chans);
    return pass;
}
