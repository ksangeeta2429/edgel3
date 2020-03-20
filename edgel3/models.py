import os
import functools
import warnings
import sklearn.decomposition

with warnings.catch_warnings():
    # Suppress TF and Keras warnings when importing
    warnings.simplefilter("ignore")
    from kapre.time_frequency import Spectrogram, Melspectrogram
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import (
        Conv2D, BatchNormalization, MaxPooling2D,
        Flatten, Activation)
    import tensorflow.keras.regularizers as regularizers


def load_embedding_model(retrain_type, sparsity):
    """
    Returns a model with the given characteristics. Loads the model
    if the model has not been loaded yet.

    Parameters
    ----------
    retrain_type: 'ft' or 'kd'
        Type of retraining for the sparsified weights of L3 audio model. 'ft' chooses the fine-tuning method
        and 'kd' returns knowledge distilled model.
    sparsity: {95.45, 53.5, 63.5, 72.3, 73.5, 81.0, 87.0, 90.5}
        The desired sparsity of audio model.

    Returns
    -------
    model : tensorflow.keras.Model
        Model object.

    """
    # Construct embedding model and load model weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MODELS['sparsified']()

    m.load_weights(load_embedding_model_path(retrain_type, sparsity))

    # Pooling for final output embedding size
    pool_size = (32, 24)
    y_a = MaxPooling2D(pool_size=pool_size, padding='same')(m.output)
    y_a = Flatten()(y_a)
    m = Model(inputs=m.input, outputs=y_a)
    return m

def load_embedding_model_path(retrain_type, sparsity):
    """
    Returns the local path to the model weights file for the model
    with the given sparsity

    Parameters
    ----------
    retrain_type: 'ft' or 'kd'
        Type of retraining for the sparsified weights of L3 audio model. 'ft' chooses the fine-tuning method
        and 'kd' returns knowledge distilled model.
    sparsity : {95.45, 53.5, 63.5, 72.3, 73.5, 81.0, 87.0, 90.5}
        Desired sparsity of the audio model.

    Returns
    -------
    output_path : str
        Path to given model object

    """

    return os.path.join(os.path.dirname(__file__),
                        'edgel3_{}_audio_sparsity_{}.h5'.format(retrain_type, sparsity))


def _construct_sparsified_audio_network():
    """
    Returns an uninitialized model object for a sparsified network with a Melspectrogram input (with 256 frequency bins).

    Returns
    -------
    model : tensorflow.keras.Model
        Model object.

    """

    weight_decay = 1e-5
    n_dft = 2048
    n_mels = 256
    n_hop = 242
    asr = 48000
    audio_window_dur = 1
    filt_size = (3, 3)
    activation = 'relu'
    embedding_size = 512

    blocks = [
        {'n': 2, 'n_filter': 64, 'pool_size': (2, 2)},
        {'n': 2, 'n_filter': 128, 'pool_size': (2, 2)},
        {'n': 2, 'n_filter': 256, 'pool_size': (2, 2)},
        {'n': 1, 'n_filter': embedding_size},
    ]

    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # MELSPECTROGRAM PREPROCESSING
    y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                         sr=asr, power_melgram=1.0, htk=True, # n_win=n_win,
                         return_decibel_melgram=True, padding='same')(x_a)
    y_a = BatchNormalization()(y_a)

    conv = lambda n_filter, filt_size, **kw: (
        Conv2D(n_filter, filt_size, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay), **kw))

    for p in blocks:
        # convolution blocks
        for _ in range(p.get('n', 1)):
            y_a = conv(p['n_filter'], p.get('filt_size', filt_size))(y_a)
            y_a = BatchNormalization()(y_a)
            y_a = Activation(activation)(y_a)
        # pooling
        if 'pool_size' in p:
            y_a = MaxPooling2D(pool_size=p['pool_size'], strides=2)(y_a)

    # embedding layer
    y_a = conv(embedding_size, filt_size,
               name='audio_embedding_layer')(y_a)
    m = Model(inputs=x_a, outputs=y_a)

    return m


MODELS = {
    'sparsified': _construct_sparsified_audio_network
}
