from edgel3.models import load_embedding_model, load_embedding_model_path

def test_load_embedding_model_path():
    embedding_model_path = load_embedding_model_path(53.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'audio_model/edgel3_audio_sparsity_53.5.h5'

    embedding_model_path = load_embedding_model_path(63.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'audio_model/edgel3_audio_sparsity_63.5.h5'

    embedding_model_path = load_embedding_model_path(73.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'audio_model/edgel3_audio_sparsity_73.5.h5'

    embedding_model_path = load_embedding_model_path(87.0)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'audio_model/edgel3_audio_sparsity_87.0.h5'

    embedding_model_path = load_embedding_model_path(95.45)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'audio_model/edgel3_audio_sparsity_95.45.h5'


def test_load_embedding_model():
    m = load_embedding_model(53.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model(63.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model(73.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model(87.0)
    assert m.output_shape[1] == 512

    m = load_embedding_model(95.45)
    assert m.output_shape[1] == 512
