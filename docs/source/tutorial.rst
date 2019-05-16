.. _tutorial:

Tutorial
========

Introduction
------------
With EdgeL3, you can compute audio embeddings from sparse models that can be useful for resource constrained devices.
The supported audio formats are those supported by the `pysoundfile` library, which is used for loading the audio (e.g. WAV, OGG, FLAC).

.. _using_library:

Using the Library
-----------------

In EdgeL3 paper, we found 95.45% sparse model to perform equally well for our applications so we refer to this as the EdgeL3 model.
You can compute audio embeddings out of the EdgeL3 model (95.45% pruned and fine-tuned) by:

.. code-block:: python
    
    import edgel3
    import soundfile as sf
    
    audio, sr = sf.read('/path/to/file.wav')
    emb, ts = edgel3.get_embedding(audio, sr)

``get_embedding`` returns two objects. The first object ``emb`` is a T-by-D numpy array, where T is the number of analysis frames used to compute embeddings, and D is the dimensionality of the embedding.
The second object ``ts`` is a length-T numpy array containing timestamps corresponding to each embedding (to the center of the analysis window, by default).

By default, EdgeL3 extracts embedding with a model that:
* Is 95.45% sparse
* Is re-trained from 'fine-tuning' (FT) mechanism

These defaults can be changed via the following optional parameters:
* sparsity:  53.5, 63.5, 72.3, 81.0, 87.0, 90.5, or 95.45 (default)
* retrain_type: "kd", "ft" (default)

For example, to get embedding out of 81.0% sparse audio model that has been trained with knowledge-distillation method, you can use:

.. code-block:: python

    import edgel3
    import soundfile as sf

    audio, sr = sf.read('/path/to/file.wav')
    emb, ts = edgel3.get_embedding(audio, sr, retrain_type='kd', sparsity=81.0)


By default EdgeL3 will pad the beginning of the input audio signal by 0.5 seconds (half of the window size) so that the
the center of the first window corresponds to the beginning of the signal ("zero centered"), and the returned timestamps
correspond to the center of each window. You can disable this centering like this:

.. code-block:: python

    emb, ts = edgel3.get_embedding(audio, sr, center=True)


The hop size used to extract the embedding is 0.1 seconds by default (i.e. an embedding frame rate of 10 Hz).
In the following example we change the hop size from 0.1 (10 frames per second) to 0.5 (2 frames per second):

.. code-block:: python
    
    emb, ts = edgel3.get_embedding(audio, sr, hop_size=0.5)

Finally, you can silence the Keras printout during inference (verbosity) by changing it from 1 (default) to 0:

.. code-block:: python
    
    emb, ts = edgel3.get_embedding(audio, sr, verbose=0)

By default, the model file is loaded from disk every time ``get_embedding`` is called. To avoid unnecessary I/O when
processing multiple files with the same model, you can load it manually and pass it to the function via the
``model`` parameter:

.. code-block:: python
    
    model = edgel3.models.load_embedding_model(retrain_type='ft', sparsity=53.5)
    emb1, ts1 = edgel3.get_embedding(audio1, sr1, model=model)
    emb2, ts2 = edgel3.get_embedding(audio2, sr2, model=model)


Since the model is provided, keyword arguments `ft` and `sparsity` for the function `get_embedding()` will be ignored. 


To compute embeddings for an audio file from a given model and save them to the disk, you can use ``process_file``:

.. code-block:: python

    import edgel3
    import numpy as np
	
    audio_filepath = '/path/to/file.wav'
    
    # Save the embedding output to '/path/to/file.npz'
    edgel3.process_file(audio_filepath)

    # Saves the embedding output to '/path/to/file_suffix.npz'
    edgel3.process_file(audio_filepath, suffix='suffix')

    # Saves the embedding output to `/different/dir/file_suffix.npz`
    edgel3.process_file(audio_filepath, output_dir='/different/dir', suffix='suffix')


The embddings can be loaded from disk using numpy:

.. code-block:: python

    import numpy as np
			    
    data = np.load('/path/to/file.npz')
    emb, ts = data['embedding'], data['timestamps']


As with ``get_embedding``, you can load the model manually and pass it to ``process_file`` to avoid loading the model multiple times:

.. code-block:: python

    import edgel3
    import numpy as np

    model = edgel3.models.load_embedding_model(retrain_type='ft', sparsity=53.5)

    audio_filepath = '/path/to/file.wav'
    
    # Save the embedding output to '/path/to/file.npz'
    edgel3.process_file(audio_filepath, model=model)

    # Saves the embedding output to '/path/to/file_suffix.npz'
    edgel3.process_file(audio_filepath, model=model, suffix='suffix')

    # Saves the embedding output to `/different/dir/file_suffix.npz`
    edgel3.process_file(audio_filepath, model=model, output_dir='/different/dir', suffix='suffix')

Using the Command Line Interface (CLI)
--------------------------------------

To compute embeddings for a single file via the command line run:

.. code-block:: shell

    $ edgel3 /path/to/file.wav

This will create an output file at ``/path/to/file.npz``.

You can change the output directory as follows:

.. code-block:: shell

    $ edgel3 /path/to/file.wav --output /different/dir

This will create an output file at ``/different/dir/file.npz``.

You can also provide multiple input files:

.. code-block:: shell

    $ edgel3 /path/to/file1.wav /path/to/file2.wav /path/to/file3.wav

which will create the output files ``/different/dir/file1.npz``, ``/different/dir/file2.npz``, and ``different/dir/file3.npz``.

You can also provide one (or more) directories to process:

.. code-block:: shell

    $ edgel3 /path/to/audio/dir

This will process all supported audio files in the directory, though it will not recursively traverse the
directory (i.e. audio files in subfolders will not be processed).

You can append a suffix to the output file as follows:

.. code-block:: shell

    $ edgel3 /path/to/file.wav --suffix somesuffix

which will create the output file ``/path/to/file_somesuffix.npz``.

To get embedding out of a different sparse model, sparsity and retrain_type arguments can be provided, for example:

.. code-block:: shell

    $ edgel3 /path/to/file.wav --model-sparsity 53.5 --retrain-type kd

The default value for --model-sparsity is 95.45 and for --retrain-type is ft.

By default, EdgeL3 will pad the beginning of the input audio signal by 0.5 seconds (half of the window size) so that the
the center of the first window corresponds to the beginning of the signal, and the timestamps correspond to the center of each window.
You can disable this centering as follows:

.. code-block:: shell

    $ edgel3 /path/to/file.wav --no-centering

The hop size used to extract the embedding is 0.1 seconds by default (i.e. an embedding frame rate of 10 Hz).
In the following example we change the hop size from 0.1 (10 frames per second) to 0.5 (2 frames per second):

.. code-block:: shell

    $ edgel3 /path/to/file.wav --hop-size 0.5

Finally, you can suppress non-error printouts by running:

.. code-block:: shell

    $ edgel3 /path/to/file.wav --quiet

A sample of full command may look like:

.. code-block:: shell

    $ edgel3 /path/to/file.wav --output /different/dir --suffix somesuffix --model-sparsity 53.5 --retrain-type kd --no-centering --hop-size 0.5 --quiet 
