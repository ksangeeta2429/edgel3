import os
import warnings
import sklearn.decomposition

with warnings.catch_warnings():
    # Suppress TF and Keras warnings when importing
    warnings.simplefilter("ignore")
    from kapre.time_frequency import Spectrogram, Melspectrogram
    from keras.layers import (
        Input, Conv2D, BatchNormalization, MaxPooling2D,
        Flatten, Activation, Lambda
    )
    from keras.models import Model
    import keras.regularizers as regularizers


def load_embedding_model(model_type, emb_dim, retrain_type, sparsity):
    """
    Returns a model with the given characteristics. Loads the model
    if the model has not been loaded yet.

    Parameters
    ----------
    model_type : {sea, sparse}
        Type of smaller version of L3 model.
        If 'sea' is selected, the audio model is a UST specialized (SEA) model. 'sparse' gives a sparse L3 model with the desired 'sparsity'.
    emb_dim : {512, 256, 128, 64}
        Desired embedding dimension of the UST specialized embedding approximated (SEA) models.
    retrain_type : 'ft' or 'kd'
        Type of retraining for the sparsified weights of L3 audio model. 'ft' chooses the fine-tuning method 
        and 'kd' returns knowledge distilled model. 
    sparsity : {95.45, 53.5, 63.5, 72.3, 87.0}
        The desired sparsity of audio model.

    Returns
    -------
    model : keras.models.Model
        Model object.

    """
    # Construct embedding model and load model weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = {'emb_dim': emb_dim, 'sparsity': sparsity}
        m = MODELS[model_type](**kwargs)

    m.load_weights(load_embedding_model_path(model_type, emb_dim, retrain_type, sparsity))
    return m

def load_embedding_model_path(model_type, emb_dim, retrain_type, sparsity):
    """
    Returns the local path to the model weights file for the model
    with the given sparsity

    Parameters
    ----------
    model_type : {sea, sparse}
        Type of smaller version of L3 model.
        If 'sea' is selected, the audio model is a UST specialized (SEA) model. 'sparse' gives a sparse L3 model with the desired 'sparsity'.
    emb_dim : {512, 256, 128, 64}
        Desired embedding dimension of the UST specialized embedding approximated (SEA) models.
    retrain_type : 'ft' or 'kd'
        Type of retraining for the sparsified weights of L3 audio model. 'ft' chooses the fine-tuning method 
        and 'kd' returns knowledge distilled model. 
    sparsity : {95.45, 53.5, 63.5, 72.3, 87.0}
        Desired sparsity of the audio model.

    Returns
    -------
    output_path : str
        Path to given model object

    """
    if model_type == 'sea':
        return os.path.join(os.path.dirname(__file__), 'edgel3_sea_ust_audio_emb_{}.h5'.format(emb_dim))
    else:
        return os.path.join(os.path.dirname(__file__),
                            'edgel3_{}_audio_sparsity_{}.h5'.format(retrain_type, sparsity))

def _construct_sparsified_audio_network(**kwargs):
    """
    Returns an uninitialized model object for a sparsified network with a Melspectrogram input (with 256 frequency bins).

    Returns
    -------
    model : keras.models.Model
        Model object.

    """

    weight_decay = 1e-5
    n_dft = 2048
    n_mels = 256
    n_hop = 242
    asr = 48000
    audio_window_dur = 1

    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # MELSPECTROGRAM PREPROCESSING
    y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                      sr=asr, power_melgram=1.0, htk=True, # n_win=n_win,
                      return_decibel_melgram=True, padding='same')(x_a)
    y_a = BatchNormalization()(y_a)

    # CONV BLOCK 1
    n_filter_a_1 = 64
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    pool_size_a_4 = (32, 24)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='audio_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    pool_size_a_4 = tuple(y_a.get_shape().as_list()[1:3]) #(32, 24)
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)
    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)

    return m

def _construct_ust_specialized_audio_network(emb_dim=128, **kwargs):
    """
    Returns an uninitialized model object for a UST specialized audio network with a Melspectrogram input (with 64 frequency bins).

    Returns
    -------
    model : keras.models.Model
        Model object.

    """

    weight_decay = 1e-5
    n_dft = 1024   # original L3 has 2048
    n_mels = 64    # original L3 has 256
    n_hop = 160    # original L3 has 242
    asr = 8000     # original L3 has 48000
    audio_window_dur = 1

    # reduce the number of conv filters in each conv block according to the emb_dim given
    reduction_factor = {
                512: [1, 1, 1, 1],
                256: [2, 2, 2, 2],
                128: [2, 2, 2, 4],
                64: [2, 2, 2, 8]
            }

    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # MELSPECTROGRAM PREPROCESSING
    y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                      sr=asr, power_melgram=1.0, htk=True, # n_win=n_win,
                      return_decibel_melgram=True, padding='same')(x_a)
    y_a = BatchNormalization()(y_a)

    # CONV BLOCK 1
    n_filter_a_1 = 64//reduction_factor[emb_dim][0]
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128//reduction_factor[emb_dim][1]
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256//reduction_factor[emb_dim][2]
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512//reduction_factor[emb_dim][3]
    filt_size_a_4 = (3, 3)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='audio_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    pool_size_a_4 = tuple(y_a.get_shape().as_list()[1:3]) #(32, 24)
    y_a = MaxPooling2D(pool_size=pool_size_a_4)(y_a)
    y_a = Flatten()(y_a)

    m = Model(inputs=x_a, outputs=y_a)

    return m


MODELS = {
    'sparse': _construct_sparsified_audio_network,
    'sea': _construct_ust_specialized_audio_network
}

