from edgel3.models import load_embedding_model, load_embedding_model_path

def test_load_embedding_model_path():

    #check the output paths of ft models
    embedding_model_path = load_embedding_model_path('ft', 53.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_53.5.h5'

    embedding_model_path = load_embedding_model_path('ft', 63.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_63.5.h5'

    embedding_model_path = load_embedding_model_path('ft', 72.3)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_72.3.h5'

    embedding_model_path = load_embedding_model_path('ft', 73.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_73.5.h5'

    embedding_model_path = load_embedding_model_path('ft', 81.0)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_81.0.h5'

    embedding_model_path = load_embedding_model_path('ft', 87.0)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_87.0.h5'

    embedding_model_path = load_embedding_model_path('ft', 90.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_90.5.h5'

    embedding_model_path = load_embedding_model_path('ft', 95.45)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_95.45.h5'

    #Check the output paths for kd models
    embedding_model_path = load_embedding_model_path('kd', 53.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_53.5.h5'

    embedding_model_path = load_embedding_model_path('kd', 63.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_63.5.h5'

    embedding_model_path = load_embedding_model_path('kd', 72.3)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_72.3.h5'

    embedding_model_path = load_embedding_model_path('kd', 73.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_73.5.h5'

    embedding_model_path = load_embedding_model_path('kd', 81.0)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_81.0.h5'

    embedding_model_path = load_embedding_model_path('kd', 87.0)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_87.0.h5'

    embedding_model_path = load_embedding_model_path('kd', 90.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_90.5.h5'

    embedding_model_path = load_embedding_model_path('kd', 95.45)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_95.45.h5'


def test_load_embedding_model():
    m = load_embedding_model('ft', 53.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('ft', 63.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('ft', 72.3)
    assert m.output_shape[1] == 512

    m = load_embedding_model('ft', 73.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('ft', 81.0)
    assert m.output_shape[1] == 512

    m = load_embedding_model('ft', 87.0)
    assert m.output_shape[1] == 512

    m = load_embedding_model('ft', 90.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('ft', 95.45)
    assert m.output_shape[1] == 512

    #Check for knowledge distilled models
    m = load_embedding_model('kd', 53.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('kd', 63.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('kd', 72.3)
    assert m.output_shape[1] == 512

    m = load_embedding_model('kd', 73.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('kd', 81.0)
    assert m.output_shape[1] == 512

    m = load_embedding_model('kd', 87.0)
    assert m.output_shape[1] == 512

    m = load_embedding_model('kd', 90.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('kd', 95.45)
    assert m.output_shape[1] == 512
