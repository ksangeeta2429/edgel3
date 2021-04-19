import os
import sys
import gzip
from itertools import product
from setuptools import setup, find_packages
from urllib.request import urlretrieve

if sys.version_info[:2] >= (3, 3):
    import platform
    from importlib.machinery import SourceFileLoader
    def load_source(name, path):
        if not os.path.exists(path):
            return {}
        return vars(SourceFileLoader(name, path).load_module())
else:
    import imp
    def load_source(name, path):
        if not os.path.exists(path):
            return {}
        return vars(imp.load_source(name, path))

module_dir = 'edgel3'
retrain_type = ['ft', 'kd']
sparsity = ['53.5', '63.5', '72.3', '87.0', '95.45']
emb_dim_for_SEA = [512, 256, 128, 64]

# Add SEA student models
sparse_weight_files = ['edgel3_{}_audio_sparsity_{}.h5'.format(*tup) for tup in product(retrain_type, sparsity)]
sonyc_weight_files = ['edgel3_sea_ust_audio_emb_{}.h5'.format(emb_dim) for emb_dim in emb_dim_for_SEA]
weight_files = sparse_weight_files + sonyc_weight_files

base_url = 'https://github.com/ksangeeta2429/edgel3/raw/models/'

if len(sys.argv) > 1 and sys.argv[1] == 'sdist':
    # exclude the weight files in sdist
    weight_files = []
else:
    # in all other cases, decompress the weights file if necessary
    for weight_file in weight_files:
        weight_path = os.path.join(module_dir, weight_file)
        if not os.path.isfile(weight_path):
            weight_fname = os.path.splitext(weight_file)[0]
            compressed_file = '{}.h5.gz'.format(weight_fname)
            compressed_path = os.path.join(module_dir, compressed_file)
            print(base_url + compressed_file)
            if not os.path.isfile(compressed_file):
                print('Download path {} : '.format(compressed_path))
                print('Downloading weight file {} ...'.format(compressed_file))
                urlretrieve(base_url + compressed_file, compressed_path)
            print('Decompressing ...')
            with gzip.open(compressed_path, 'rb') as source:
                with open(weight_path, 'wb') as target:
                    target.write(source.read())
            print('Decompression complete')
            os.remove(compressed_path)
            print('Removing compressed file')

version = load_source('edgel3.version', os.path.join('edgel3', 'version.py'))

with open('README.md') as file:
    long_description = file.read()

setup(
    name='edgel3',
    version=version['version'],
    description='Audio embeddings based on sparse or UST specialized (SEA) Look, Listen, and Learn (L3) models for the Edge',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ksangeeta2429/edgel3',
    author='Sangeeta Srivastava',
    author_email='sangeeta.osu@gmail.com',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['edgel3=edgel3.cli:main'],
    },
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 3', #Removing support for python 2.x
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='audio embeddings machine learning tensorflow keras pruning compression embedding approximation knowledge distillation',
    project_urls={
        'Source': 'https://github.com/ksangeeta2429/edgel3',
        'Tracker': 'https://github.com/ksangeeta2429/edgel3/issues',
        'Documentation': 'https://readthedocs.org/projects/edgel3/'
    },
    install_requires=[
        'numpy>=1.13.0',
        'scipy>=0.19.1',
        'kapre==0.1.4', #pin kapre to 0.1.4
        'keras==2.3.1',
        'PySoundFile>=0.9.0.post1',
        'resampy>=0.2.1,<0.3.0',
        'h5py>=2.7.0,<3.0.0',
    ],
    extras_require={
        'docs': [
            'sphinx==1.2.3',  # autodoc was broken in 1.3.1
            'scikit-learn==0.19.0',
            'sphinxcontrib-napoleon',
            'sphinx_rtd_theme',
            'numpydoc',
            ],
        'tests': []
    },
    package_data={
        'edgel3': weight_files
    },
)
