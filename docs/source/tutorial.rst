.. _tutorial:

EdgeL3 tutorial
===============

Introduction
------------
Welcome to the EdgeL3 tutorial! We will show you how you can compute audio embeddings for resource constrained devices.
Note that only audio formats supported by `pysoundfile` are supported (e.g. WAV, OGG, FLAC).

.. _using_library:

Using the Library
-----------------

You can compute audio embeddings out of the EdgeL3 model, 95.45% pruned and fine-tuned L3 model by:

.. code-block:: python
    
    import edgel3
    import soundfile as sf
    
    audio, sr = sf.read('/path/to/file.wav')
    emb, ts = edgel3.get_embedding(audio, sr)

``get_embedding`` returns two objects. The first object ``emb`` is a T-by-D numpy array, where T is the number of analysis frames used to compute embeddings, and D is the dimensionality of the embedding.
The second object ``ts`` is a length-T numpy array containing timestamps corresponding to each embedding (to the center of the analysis window, by default).
To get embedding out of different sparse models, you can specify ``sparsity`` such as:

.. code-block:: python

    import edgel3
    import soundfile as sf

    audio, sr = sf.read('/path/to/file.wav')
    emb, ts = edgel3.get_embedding(audio, sr, sparsity=53.5)

Valid sparsity values are: 53.5, 63.5, 72.3, 81.0, 87.0, 90.5, or 95.45 (EdgeL3 model).
We had used two training schemes to train the pruned audio models of L3. The above is example of fine-tuning.
If you want to use the pruned models re-trained by knowledge distillation method, you can specify 
``retrain_type`` as ``kd`` which stands for knowledge distillation.

.. code-block:: python
    
    import edgel3
    import soundfile as sf
    
    audio, sr = sf.read('/path/to/file.wav')
    emb, ts = edgel3.get_embedding(audio, sr, retrain_type='kd', sparsity=53.5)

By default, EdgeL3 will pad the signal by half of the window size (one second) so that the 
center of the first window corresponds to the beginning of the signal, and the corresponding 
timestamps correspond to the center of the window. If you wish to disable this centering, you can 
use code like the following:

.. code-block:: python
    
    import edgel3
    import soundfile as sf
    emb, ts = edgel3.get_embedding(audio, sr, center=True)

To change the hop size use to compute embeddings (which is 0.1s by default), you can run:

.. code-block:: python
    
    import edgel3
    import soundfile as sf
    emb, ts = edgel3.get_embedding(audio, sr, hop_size=0.5)

where we changed the hop size to 0.5 seconds in this example. Finally, you can set the Keras model verbosity to either 0 or 1:

.. code-block:: python
    
    import edgel3
    import soundfile as sf
    emb, ts = edgel3.get_embedding(audio, sr, verbose=0)

By default, the corresponding model file is loaded every time this function is called. If you want to load the model only once when computing multiple embeddings, you can run:

.. code-block:: python
    
    import edgel3
    import soundfile as sf
    model = edgel3.models.load_embedding_model(retrain_type='ft', sparsity=53.5)
    emb, ts = edgel3.get_embedding(audio, sr, model=model)

Since the model is provided, keyword arguments `ft` and `sparsity` for the function `get_embedding()` will be ignored. To compute embeddings for an audio file from a given model
and save them locally, you can use snippet similar to the following:

.. code-block:: python

    import edgel3
    import numpy as np
    
    model = edgel3.models.load_embedding_model(retrain_type='ft', sparsity=53.5)	
    audio_filepath = '/path/to/file.wav'
    # Saves the file to '/path/to/file.npz'
    edgel3.process_file(audio_filepath)
    # Saves the file to `/different/dir/file.npz`
    edgel3.process_file(audio_filepath, output_dir='/different/dir', suffix='suffix')
    # Saves the file to '/path/to/file_suffix.npz'
    edgel3.process_file(audio_filepath, suffix='suffix', model=model)

    data = np.load('/path/to/file.npz')
    emb, ts = data['embedding'], data['timestamps']

