# edgel3


[![PyPI](https://img.shields.io/badge/python-3.6%2C%203.7%2C%203.8-blue.svg)](https://pypi.python.org/pypi/edgel3)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)
[![Build Status](https://travis-ci.com/ksangeeta2429/edgel3.svg?branch=master)](https://travis-ci.com/ksangeeta2429/edgel3)
[![Coverage Status](https://coveralls.io/repos/github/ksangeeta2429/edgel3/badge.svg?branch=master)](https://coveralls.io/github/ksangeeta2429/edgel3?branch=master)
[![Documentation Status](https://readthedocs.org/projects/edgel3/badge/?version=latest)](https://edgel3.readthedocs.io/en/latest/?badge=latest)

Look, Listen, and Learn (L3) [4] Audio subnetwork produces generic audio representations that can be used for myriad downstream tasks. However, L3-Net Audio requires 18 MB and 12 MB of static and dynamic memory respectively, making it infeasible for small edge devices with a single microcontroller. [EdgeL3](https://github.com/ksangeeta2429/Publications/raw/master/EdgeL3_Compressing_L3_Net_for_Mote_Scale.pdf) [2] is competetive with L3 Audio while being 95.45% sparse. However, it still has a high activation memory requirement.

To jointly handle both static and dynamic memory, we introduce [Specialized Embedding Approximation](https://github.com/ksangeeta2429/Publications/raw/master/SEA.pdf)[1], a teacher-student learning paradigm where the student audio embedding model is trained to approximate only the part of the teacher's embedding manifold which is relevant to the target data-domain. Notice the difference between data-domain and dataset. Restricting the specialization on a particular downstream dataset would compromise intra-domain generalizability.

``edgel3`` is an open-source Python library for downloading the smaller versions of L3 models and computing deep audio embeddings from such models. 
- The ``sea`` models are specialized for [SONYC-UST](https://zenodo.org/record/2590742#.YGlc1i1h2Tc) [5] data domain. Training pipelines can be found [[here](https://github.com/ksangeeta2429/embedding-approx)]. 
- The ``sparse`` models provided have been re-trained using two different mechanisms: fine-tuning ``ft`` and knowledge distillation ``kd``. Training pipelines can be found [[here](https://github.com/ksangeeta2429/l3embedding/tree/dcompression)].

For non-compressed L3-Net, please refer to [OpenL3](https://github.com/marl/openl3) [3]

# Installing edgel3

Dependencies
------------
#### Tensorflow
``edgel3`` has been tested with Tensorflow 2.0 and Keras 2.3.1. 

    pip install tensorflow==2.0.0

#### libsndfile
**edgel3** depends on the `pysoundfile` module to load audio files, which depends on the non-Python library ``libsndfile``. On Windows and macOS, these will be installed via ``pip`` and you can therefore skip this step.
However, on Linux this must be installed manually via your platform's package manager.
For Debian-based distributions (such as Ubuntu), this can be done by simply running

    apt-get install libsndfile1

For more detailed information, please consult the ``pysoundfile`` [installation documentation](https://pysoundfile.readthedocs.io/en/0.9.0/#installation>).


Installing edgel3
-----------------
The simplest way to install edgel3 is by using ``pip``, which will also install the additional required dependencies
if needed. To install edgel3 using ``pip``, simply run

    pip install edgel3

To install the latest version of edgel3 from source:

1. Clone or pull the lastest version:

        git clone https://github.com/ksangeeta2429/edgel3.git

2. Install using pip to handle python dependencies:
        cd edgel3
        pip install -e .

# Getting started with edgel3

Load a SONYC-UST specialized L3 audio (reduced input represenation and reduced architecture) that outputs an embedding of length 128
```python
model = edgel3.models.load_embedding_model(model_type='sea', emb_dim=128)
```

Load a 95.45% sparse L3 audio re-trained with fine-tuning
```python
model = edgel3.models.load_embedding_model(model_type='sparse', retrain_type='ft', sparsity=95.45)
```

Load a 87.0% sparse L3 audio re-trained with knowledge distillation
```python
model = edgel3.models.load_embedding_model(model_type='sparse', retrain_type='kd', sparsity=87.0)
```

For more examples, please see the [tutorial](https://edgel3.readthedocs.io/en/latest/tutorial.html) and [module usage](https://edgel3.readthedocs.io/en/latest/edgel3.html).

# References

If you use the SEA/EdgeL3 Github repos or the pre-trained models, please cite the relevant work:

[1] **[Specialized Embedding Approximation for Edge Intelligence: A case study in Urban Sound Classification](https://github.com/ksangeeta2429/Publications/raw/master/SEA.pdf)** <br/>
Sangeeta Srivastava, Dhrubojyoti Roy, Mark Cartwright, Juan Pablo Bello, and Anish Arora. </br>
To be published in IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), Toronto, Canada, June 2021.

[2] **[EdgeL3: Compressing L3-Net for Mote-Scale Urban Noise Monitoring](https://github.com/ksangeeta2429/Publications/raw/master/EdgeL3_Compressing_L3_Net_for_Mote_Scale.pdf)** <br/>
Sangeeta Kumari, Dhrubojyoti Roy, Mark Cartwright, Juan Pablo Bello, and Anish Arora. </br>
Parallel AI and Systems for the Edge (PAISE), Rio de Janeiro, Brazil, May 2019.

[3] **Look, Listen and Learn More: Design Choices for Deep Audio Embeddings** <br/>
Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.<br/>
IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.

[4] **Look, Listen and Learn**<br/>
Relja Arandjelović and Andrew Zisserman<br/>
IEEE International Conference on Computer Vision (ICCV), Venice, Italy, Oct. 2017.

[5] **SONYC Urban Sound Tagging (SONYC-UST): a multilabel dataset from an urban acoustic sensor network**</br>
Mark Cartwright, Ana Elisa Mendez Mendez, Graham Dove, Jason Cramer et al. 2019.

