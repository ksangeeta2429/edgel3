import pytest
import tempfile
import numpy as np
import os
import shutil
import soundfile as sf
from edgel3.edgel3_exceptions import EdgeL3Error
from edgel3.edgel3_warnings import EdgeL3Warning
from edgel3.models import load_embedding_model
from edgel3.core import get_embedding, get_output_path, process_file, _center_audio, _pad_audio

TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data', 'audio')

# Test audio file paths
CHIRP_MONO_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_mono.wav')
CHIRP_STEREO_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_stereo.wav')
CHIRP_44K_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_44k.wav')
CHIRP_1S_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_1s.wav')
EMPTY_PATH = os.path.join(TEST_AUDIO_DIR, 'empty.wav')
SHORT_PATH = os.path.join(TEST_AUDIO_DIR, 'short.wav')
SILENCE_PATH = os.path.join(TEST_AUDIO_DIR, 'silence.wav')

def test_get_embedding():
    hop_size = 0.1
    tol = 1e-5

    # Make sure all finetuned pruned models work fine
    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='ft', sparsity=53.5, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='ft', sparsity=63.5, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='ft', sparsity=72.3, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='ft', sparsity=73.5, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='ft', sparsity=81.0, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='ft', sparsity=87.0, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='ft', sparsity=90.5, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1_ft, ts1_ft = get_embedding(audio, sr, retrain_type='ft', sparsity=95.45, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1_ft) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1_ft))


    # Make sure all knowledge distilled pruned models work fine
    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='kd', sparsity=53.5, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='kd', sparsity=63.5, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='kd', sparsity=72.3, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='kd', sparsity=73.5, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='kd', sparsity=81.0, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='kd', sparsity=87.0, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='kd', sparsity=90.5, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = get_embedding(audio, sr, retrain_type='kd', sparsity=95.45, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    #assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))


    # Make sure we can load a model and pass it in
    model = load_embedding_model('ft', 95.45)
    emb1load, ts1load = get_embedding(audio, sr, model=model, retrain_type='ft', sparsity=95.45, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(emb1load - emb1_ft) < tol)
    assert np.all(np.abs(ts1load - ts1_ft) < tol)

    # Make sure that the embeddings are approximately the same with mono and stereo
    audio, sr = sf.read(CHIRP_STEREO_PATH)
    emb2, ts2 = get_embedding(audio, sr, retrain_type='ft', sparsity=95.45, center=True, hop_size=0.1, verbose=1)
    assert not np.any(np.isnan(emb2))

    # Make sure that the embeddings are approximately the same if we resample the audio
    audio, sr = sf.read(CHIRP_44K_PATH)
    emb3, ts3 = get_embedding(audio, sr, retrain_type='ft', sparsity=95.45, center=True, hop_size=0.1, verbose=1)
    assert not np.any(np.isnan(emb3))

    # Make sure empty audio is handled
    audio, sr = sf.read(EMPTY_PATH)
    pytest.raises(EdgeL3Error, get_embedding, audio, sr, retrain_type='ft', sparsity=95.45, center=True, hop_size=0.1, verbose=1)

    # Make sure user is warned when audio is too short
    audio, sr = sf.read(SHORT_PATH)
    pytest.warns(EdgeL3Warning, get_embedding, audio, sr, retrain_type='ft', sparsity=95.45, center=False, hop_size=0.1, verbose=1)

    # Make sure short audio can be handled
    emb4, ts4 = get_embedding(audio, sr, retrain_type='ft', sparsity=95.45, center=False, hop_size=0.1, verbose=1)

    assert emb4.shape[0] == 1
    assert emb4.shape[1] == 512
    assert len(ts4) == 1
    assert ts4[0] == 0
    assert not np.any(np.isnan(emb4))

    # Make sure silence is handled
    audio, sr = sf.read(SILENCE_PATH)
    pytest.warns(EdgeL3Warning, get_embedding, audio, sr, retrain_type='ft', sparsity=95.45, center=True, hop_size=0.1, verbose=1)

    emb5, ts5 = get_embedding(audio, sr, retrain_type='ft', sparsity=95.45, center=True, hop_size=0.1, verbose=1)
    assert emb5.shape[1] == 512
    assert not np.any(np.isnan(emb5))

    # Check for centering
    audio, sr = sf.read(CHIRP_1S_PATH)
    emb6, ts6 = get_embedding(audio, sr, retrain_type='ft', sparsity=95.45, center=True, hop_size=hop_size, verbose=1)
    n_frames = 1 + int((audio.shape[0] + sr//2 - sr) / float(int(hop_size*sr)))
    assert emb6.shape[0] == n_frames

    emb7, ts7 = get_embedding(audio, sr, retrain_type='ft', sparsity=95.45, center=False, hop_size=hop_size, verbose=1)
    n_frames = 1 + int((audio.shape[0] - sr) / float(int(hop_size*sr)))
    assert emb7.shape[0] == n_frames

    # Check for different hop size
    hop_size = 0.2
    emb8, ts8 = get_embedding(audio, sr, retrain_type='ft', sparsity=95.45, center=False, hop_size=hop_size, verbose=1)
    n_frames = 1 + int((audio.shape[0] - sr) / float(int(hop_size*sr)))
    assert emb8.shape[0] == n_frames

    # Make sure changing verbosity doesn't break
    get_embedding(audio, sr, retrain_type='ft', sparsity=95.45, center=True, hop_size=hop_size, verbose=0)

    # Make sure invalid arguments don't work
    pytest.raises(EdgeL3Error, get_embedding, audio, sr, model='invalid', \
                  retrain_type='ft', sparsity=95.45, center=False, hop_size=0.1, verbose=1)

    pytest.raises(EdgeL3Error, get_embedding, audio, sr, retrain_type='invalid', \
                  sparsity=95.45, center=False, hop_size=0.1, verbose=1)

    pytest.raises(EdgeL3Error, get_embedding, audio, sr, retrain_type='ft', \
                  sparsity=95.45, center='invalid', hop_size=0.1, verbose=1)

    pytest.raises(EdgeL3Error, get_embedding, audio, sr, retrain_type='ft', \
                  sparsity=40, center=False, hop_size=0.1, verbose=1)

    pytest.raises(EdgeL3Error, get_embedding, audio, sr, retrain_type='ft', \
                  sparsity=-95.45, center=False, hop_size=0.1, verbose=1)

    pytest.raises(EdgeL3Error, get_embedding, audio, sr, retrain_type='ft', \
                  sparsity='invalid', center=False, hop_size=0.1, verbose=1)

    pytest.raises(EdgeL3Error, get_embedding, audio, sr, retrain_type='ft', \
                  sparsity=95.45, center=False, hop_size=-1, verbose=1)

    pytest.raises(EdgeL3Error, get_embedding, audio, sr, retrain_type='ft', \
                  sparsity=95.45, center=False, hop_size=0, verbose=1)

    pytest.raises(EdgeL3Error, get_embedding, audio, sr, retrain_type='ft', \
                  sparsity=95.45, center=False, hop_size=0.1, verbose=-1)

    pytest.raises(EdgeL3Error, get_embedding, audio, sr, retrain_type=1, \
                  sparsity=95.45, center=False, hop_size=0.1, verbose=1)

    pytest.raises(EdgeL3Error, get_embedding, np.ones((10,10,10)), sr,\
                  retrain_type='ft', sparsity=95.45, center=True, hop_size=0.1, verbose=1)

def test_get_output_path():
    test_filepath = '/path/to/the/test/file/audio.wav'
    suffix = 'embedding.npz'
    test_output_dir = '/tmp/test/output/dir'
    exp_output_path = '/tmp/test/output/dir/audio_embedding.npz'
    output_path = get_output_path(test_filepath, suffix, test_output_dir)
    assert output_path == exp_output_path

    # No output directory
    exp_output_path = '/path/to/the/test/file/audio_embedding.npz'
    output_path = get_output_path(test_filepath, suffix)
    assert output_path == exp_output_path

    # No suffix
    exp_output_path = '/path/to/the/test/file/audio.npz'
    output_path = get_output_path(test_filepath, '.npz')
    assert output_path == exp_output_path


def test_process_file():
    test_output_dir = tempfile.mkdtemp()
    test_subdir = os.path.join(test_output_dir, "subdir")
    os.makedirs(test_subdir)

    # Make a copy of the file so we can test the case where we save to the same directory
    input_path_alt = os.path.join(test_subdir, "chirp_mono.wav")
    shutil.copy(CHIRP_MONO_PATH, test_subdir)

    invalid_file_path = os.path.join(test_subdir, "invalid.wav")
    with open(invalid_file_path, 'w') as f:
        f.write('This is not an audio file.')

    exp_output_path1 = os.path.join(test_output_dir, "chirp_mono.npz")
    exp_output_path2 = os.path.join(test_output_dir, "chirp_mono_suffix.npz")
    exp_output_path3 = os.path.join(test_subdir, "chirp_mono.npz")
    try:
        process_file(CHIRP_MONO_PATH, output_dir=test_output_dir)
        process_file(CHIRP_MONO_PATH, output_dir=test_output_dir, suffix='suffix')
        process_file(input_path_alt)

        # Make sure we fail when invalid files are provided
        pytest.raises(EdgeL3Error, process_file, invalid_file_path)

        # Make sure paths all exist
        assert os.path.exists(exp_output_path1)
        assert os.path.exists(exp_output_path2)
        assert os.path.exists(exp_output_path3)

        data = np.load(exp_output_path1)
        assert 'embedding' in data
        assert 'timestamps' in data

        embedding = data['embedding']
        timestamps = data['timestamps']

        # Quick sanity check on data
        assert embedding.ndim == 2
        assert timestamps.ndim == 1

        # Make sure that suffices work
    finally:
        shutil.rmtree(test_output_dir)

    # Make sure we fail when file cannot be opened
    pytest.raises(EdgeL3Error, process_file, '/fake/directory/asdf.wav')


def test_center_audio():
    audio_len = 100
    audio = np.ones((audio_len,))

    # Test even window size
    frame_len = 50
    centered = _center_audio(audio, frame_len)
    assert centered.size == 125
    assert np.all(centered[:25] == 0)
    assert np.array_equal(audio, centered[25:])

    # Test odd window size
    frame_len = 49
    centered = _center_audio(audio, frame_len)
    assert centered.size == 124
    assert np.all(centered[:24] == 0)
    assert np.array_equal(audio, centered[24:])


def test_pad_audio():
    frame_len = 50
    hop_len = 25

    # Test short case
    audio_len = 10
    audio = np.ones((audio_len,))
    padded = _pad_audio(audio, frame_len, hop_len)
    assert padded.size == 50
    assert np.array_equal(padded[:10], audio)
    assert np.all(padded[10:] == 0)

    # Test case when audio needs to be padded so all samples are processed
    audio_len = 90
    audio = np.ones((audio_len,))
    padded = _pad_audio(audio, frame_len, hop_len)
    assert padded.size == 100
    assert np.array_equal(padded[:90], audio)
    assert np.all(padded[90:] == 0)

    # Test case when audio does not need padding
    audio_len = 100
    audio = np.ones((audio_len,))
    padded = _pad_audio(audio, frame_len, hop_len)
    assert padded.size == 100
    assert np.array_equal(padded, audio)
