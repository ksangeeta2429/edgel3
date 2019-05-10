# edgel3


[![PyPI](https://img.shields.io/badge/python-2.7%2C%203.5%2C%203.6-blue.svg)](https://pypi.python.org/pypi/edgel3)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)
[![Build Status](https://travis-ci.com/ksangeeta2429/edgel3.svg?branch=master)](https://travis-ci.com/ksangeeta2429/edgel3)
[![Coverage Status](https://coveralls.io/repos/github/ksangeeta2429/edgel3/badge.svg)](https://coveralls.io/github/ksangeeta2429/edgel3)
[![Documentation Status](https://readthedocs.org/projects/edgel3/badge/?version=latest)](https://edgel3.readthedocs.io/en/latest/?badge=latest)

Look, Listen, and Learn (L3) [3],  a  recently  proposed  state-of-the-art  transfer learning technique, mitigates the first challenge by training self-supervised deep audio embedding through binary Audio-Visual Correspondence,  and  the  resulting  embedding  can  beused to train a variety of downstream audio classification tasks. However, with close to 4.7 million parameters, the multi-layer L3-Net  CNN is still prohibitively expensive to be run on small edge devices, such as 'motes' that use a single microcontroller and limited memory to achieve long-lived self-powered operation. 

In [EdgeL3](https://github.com/ksangeeta2429/Publications/raw/master/EdgeL3_Compressing_L3_Net_for_Mote_Scale.pdf) [1], we comprehensively explored the feasibility of compressing the L3-Net for mote-scale inference. We used pruning, ablation, and knowledge distillation techniques to show that the originally proposed L3-Net architecture is substantially overparameterized, not  only for AVC but for the target task of sound classification as evaluated on two popular downstream datasets, US8K and ESC50. EdgeL3, a 95% sparsified version of L3-Net, provides a useful reference model for approximating L3 audio embedding for transfer learning.

EdgeL3 is an open-source Python library for downloading the sparsified L3 models and computing deep audio embeddings from such models. The audio embedding models provided here are after sparsification and fine-tuning of L3 audio network. For additional implementation details, please refer to EdgeL3 [1]. The code for the model and training implementation can be found [here](https://github.com/ksangeeta2429/l3embedding/tree/dcompression)

Download the original L3 model used by EdgeL3 as baseline [here](https://github.com/ksangeeta2429/l3embedding/raw/dcompression/models/cnn_l3_melspec2_recent/model_best_valid_accuracy.h5)
. For non-sparse models and embedding, please refer to [OpenL3](https://github.com/marl/openl3) [2]

# Installing EdgeL3

Dependencies
------------
#### Tensorflow
Install Tensorflow (CPU-only/GPU) variant that best fits your usecase.

On most platforms, either of the following commands should properly install Tensorflow:

    pip install tensorflow # CPU-only version
    pip install tensorflow-gpu # GPU version

For more detailed information, please consult the
[Tensorflow installation documentation](https://www.tensorflow.org/install/).

#### libsndfile
EdgeL3 depends on the `pysoundfile` module to load audio files, which depends on the non-Python library ``libsndfile``. On Windows and macOS, these will be installed via ``pip`` and you can therefore skip this step.
However, on Linux this must be installed manually via your platform's package manager.
For Debian-based distributions (such as Ubuntu), this can be done by simply running

    apt-get install libsndfile1

For more detailed information, please consult the
[`pysoundfile` installation documentation](https://pysoundfile.readthedocs.io/en/0.9.0/#installation>).


Installing EdgeL3
-----------------
The simplest way to install EdgeL3 is by using ``pip``, which will also install the additional required dependencies
if needed. To install EdgeL3 using ``pip``, simply run

    pip install edgel3

To install the latest version of EdgeL3 from source:

1. Clone or pull the lastest version:

        git clone https://github.com/ksangeeta2429/edgel3.git

2. Install using pip to handle python dependencies:
        cd edgel3
        pip install -e .

# Using EdgeL3

To help you get started with EdgeL3 please see the [tutorial](https://edgel3.readthedocs.io/en/latest/tutorial.html) and [module usage](https://edgel3.readthedocs.io/en/latest/edgel3.html).


# References

Please cite the following papers when using EdgeL3 in your work:

[1] **[EdgeL3: Compressing L3-Net for Mote-Scale Urban Noise Monitoring](https://github.com/ksangeeta2429/Publications/raw/master/EdgeL3_Compressing_L3_Net_for_Mote_Scale.pdf)** <br/>
Sangeeta Kumari, Dhrubojyoti Roy, Mark Cartwright, Juan Pablo Bello, and Anish Arora. </br>
Parallel AI and Systems for the Edge (PAISE), Rio de Janeiro, Brazil, May 2019.

[2] **Look, Listen and Learn More: Design Choices for Deep Audio Embeddings** <br/>
Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.<br/>
IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.

[3] **Look, Listen and Learn**<br/>
Relja Arandjelović and Andrew Zisserman<br/>
IEEE International Conference on Computer Vision (ICCV), Venice, Italy, Oct. 2017.
