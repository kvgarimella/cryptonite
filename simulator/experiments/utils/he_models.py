import numpy as np

"""
Homomorphic Evaluation of Linear Layers 
as a function of the number of threads

example:
r18_he_latencies_cifar[10] = 
latency of performing the HE evaluations
of ResNet-18 on Cifar inputs given 11 thread 
"""


def he_eval_latency(dataset, network, num_threads):
    if dataset == "cifar10":
        if network == "resnet18":
            return r18_he_latencies_cifar[num_threads - 1]
        elif network == "resnet32":
            return r32_he_latencies_cifar[num_threads - 1]
        elif network == "vgg16":
            return vgg16_he_latencies_cifar[num_threads - 1]

    if dataset == "tinyimagenet":
        if network == "resnet18":
            return r18_he_latencies_tiny[num_threads - 1]
        elif network == "resnet32":
            return r32_he_latencies_tiny[num_threads - 1]
        elif network == "vgg16":
            return vgg16_he_latencies_tiny[num_threads - 1]




# ResNet-18 
r18_he_latencies_cifar = np.array([292.4065098, 151.7543008, 107.727452 ,  79.81203  ,  64.4129272,
                                  62.5728036,  50.7191106,  49.0400534,  48.728874 ,  48.339353 ,
                                  48.32074  ,  48.5412158,  36.1272766,  35.905063 ,  35.8559186,
                                  35.8915864,  36.511213 ])

r18_he_latencies_tiny = np.array([1066.19601533,  564.89279367,  404.44879833,  302.45587133,
                                  247.76990067,  239.505704  ,  193.583658  ,  191.649502  ,
                                  190.17166967,  188.967239  ,  187.22179767,  190.06091367,
                                  141.638602  ,  141.67529567,  140.42299067,  142.98676033,
                                  140.956612  ])

# ResNet-32
r32_he_latencies_cifar= np.array([39.054868 , 19.7755064, 14.1291778, 10.6794862,  8.4588826,
                                  7.6183646,  6.8421992,  5.9741398,  5.8623208,  4.738788 ,
                                  4.1570506,  4.040384 ,  4.0307288,  4.0289608,  3.9242828,
                                  3.2754502,  3.253016 ,  3.2304436,  3.2861864,  3.2429626,
                                  3.2394664,  3.2335546,  3.3094532,  3.260057 ,  3.2682844,
                                  3.2992954,  3.2893686,  3.3238594,  3.3338406,  2.8581502,
                                  2.6295826])

r32_he_latencies_tiny = np.array([111.50917267,  56.77609933,  38.15678233,  29.025188  ,
                                  24.35627233,  21.153193  ,  17.89531633,  15.17107033,
                                  14.74858133,  14.16876567,  11.44133267,  11.40533467,
                                  11.32923733,  11.27366833,  11.04904767,  11.13263667,
                                  11.204641  ,  11.18145633,  11.08171267,  10.971946  ,
                                  8.24483367,   8.438429  ,   8.185388  ,   8.36549633,
                                  8.27917167,   8.532651  ,   8.604815  ,   8.54038633,
                                  8.53414767,   8.72805033,   8.69798867])

# VGG-16
vgg16_he_latencies_cifar = np.array([187.1568062,  96.9525012,  69.7335044,  53.4916522,  46.7898048,
                                     38.9435804,  37.9284006,  36.1969372,  35.0284868,  29.0049442,
                                     28.5699802,  22.4760384,  22.534695 ])

vgg16_he_latencies_tiny = np.array([498.814645  , 259.96950267, 181.95388933, 154.15451033,
                                    126.231961  , 122.47389833,  99.78516667,  97.643725  ,
                                    72.73945633,  72.92380933,  73.67507133,  73.22918067,
                                    73.44313167])


