from edgel3.models import load_embedding_model, load_embedding_model_path

def test_load_embedding_model_path():

    # Check for output paths for UST specialized embedding approximated L3 models
    embedding_model_path = load_embedding_model_path('sea', 512, 'ft', 95.45)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_sea_ust_audio_emb_512.h5'

    embedding_model_path = load_embedding_model_path('sea', 512, 'kd', 95.45)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_sea_ust_audio_emb_512.h5'

    embedding_model_path = load_embedding_model_path('sea', 256, 'ft', 95.45)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_sea_ust_audio_emb_256.h5'

    embedding_model_path = load_embedding_model_path('sea', 128, 'ft', 95.45)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_sea_ust_audio_emb_128.h5'

    embedding_model_path = load_embedding_model_path('sea', 64, 'ft', 95.45)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_sea_ust_audio_emb_64.h5'

    # Check the output paths of fine-tuned sparse L3 models
    embedding_model_path = load_embedding_model_path('sparse', 128, 'ft', 53.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_53.5.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'ft', 63.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_63.5.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'ft', 72.3)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_72.3.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'ft', 73.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_73.5.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'ft', 81.0)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_81.0.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'ft', 87.0)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_87.0.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'ft', 90.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_90.5.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'ft', 95.45)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_ft_audio_sparsity_95.45.h5'

    # Check the output paths for knowledge distilled sparse L3 models
    embedding_model_path = load_embedding_model_path('sparse', 128, 'kd', 53.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_53.5.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'kd', 63.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_63.5.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'kd', 72.3)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_72.3.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'kd', 73.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_73.5.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'kd', 81.0)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_81.0.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'kd', 87.0)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_87.0.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'kd', 90.5)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_90.5.h5'

    embedding_model_path = load_embedding_model_path('sparse', 128, 'kd', 95.45)
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'edgel3/edgel3_kd_audio_sparsity_95.45.h5'

def test_load_embedding_model():
    # Check for fine-tuned sparse L3 models
    m = load_embedding_model('sparse', 128, 'ft', 53.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('sparse', 128, 'ft', 63.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('sparse', 128, 'ft', 72.3)
    assert m.output_shape[1] == 512

    m = load_embedding_model('sparse', 128, 'ft', 87.0)
    assert m.output_shape[1] == 512

    m = load_embedding_model('sparse', 128, 'ft', 95.45)
    assert m.output_shape[1] == 512

    # Check for knowledge distilled sparse L3 models
    m = load_embedding_model('sparse', 128, 'kd', 53.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('sparse', 128, 'kd', 63.5)
    assert m.output_shape[1] == 512

    m = load_embedding_model('sparse', 128, 'kd', 72.3)
    assert m.output_shape[1] == 512

    m = load_embedding_model('sparse', 128, 'kd', 87.0)
    assert m.output_shape[1] == 512

    m = load_embedding_model('sparse', 128, 'kd', 95.45)
    assert m.output_shape[1] == 512

    # Check for UST specialized embedding approximated L3 models
    m = load_embedding_model('sea', 512, 'ft', 95.45)
    assert m.output_shape[1] == 512

    m = load_embedding_model('sea', 256, 'ft', 95.45)
    assert m.output_shape[1] == 256

    m = load_embedding_model('sea', 128, 'kd', 95.45)
    assert m.output_shape[1] == 128

    m = load_embedding_model('sea', 64, 'kd', 95.45)
    assert m.output_shape[1] == 64
