import pytest
import os
import tempfile
import numpy as np
import shutil
from edgel3.cli import positive_float, positive_int, get_file_list, parse_args, run, main
from argparse import ArgumentTypeError
from edgel3.edgel3_exceptions import EdgeL3Error
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data', 'audio')

# Test audio file paths
CHIRP_MONO_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_mono.wav')
CHIRP_STEREO_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_stereo.wav')
CHIRP_8K_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_8k.wav')
CHIRP_44K_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_44k.wav')
CHIRP_1S_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_1s.wav')
EMPTY_PATH = os.path.join(TEST_AUDIO_DIR, 'empty.wav')
SHORT_PATH = os.path.join(TEST_AUDIO_DIR, 'short.wav')
SILENCE_PATH = os.path.join(TEST_AUDIO_DIR, 'silence.wav')

def test_positive_float():

    # test that returned value is float
    f = positive_float(5)
    assert f == 5.0
    assert type(f) is float

    # test it works for valid strings
    f = positive_float('1.3')
    assert f == 1.3
    assert type(f) is float

    # make sure error raised for all invalid values:
    invalid = [-5, -1.0, None, 'hello']
    for i in invalid:
        pytest.raises(ArgumentTypeError, positive_float, i)

def test_positive_int():

    # test that returned value is float
    f = positive_int(5.0)
    assert f == 5
    assert type(f) is int

    # test it works for valid strings
    f = positive_int('1')
    assert f == 1
    assert type(f) is int

    # make sure error raised for all invalid values:
    invalid = [-5, -1.0, None, 'hello']
    for i in invalid:
        pytest.raises(ArgumentTypeError, positive_int, i)


def test_get_file_list():

    # test for invalid input (must be iterable, e.g. list)
    pytest.raises(ArgumentTypeError, get_file_list, CHIRP_44K_PATH)

    # test for valid list of file paths
    flist = get_file_list([CHIRP_44K_PATH, CHIRP_1S_PATH])
    assert len(flist) == 2
    assert flist[0] == CHIRP_44K_PATH and flist[1] == CHIRP_1S_PATH

    # test for valid folder
    flist = get_file_list([TEST_AUDIO_DIR])
    assert len(flist) == 8

    flist = sorted(flist)
    assert flist[0] == CHIRP_1S_PATH
    assert flist[1] == CHIRP_44K_PATH
    assert flist[2] == CHIRP_8K_PATH
    assert flist[3] == CHIRP_MONO_PATH
    assert flist[4] == CHIRP_STEREO_PATH
    assert flist[5] == EMPTY_PATH
    assert flist[6] == SHORT_PATH
    assert flist[7] == SILENCE_PATH

    # combine list of files and folders
    flist = get_file_list([TEST_AUDIO_DIR, CHIRP_44K_PATH])
    assert len(flist) == 9

    # nonexistent path
    pytest.raises(EdgeL3Error, get_file_list, ['/fake/path/to/file'])


def test_parse_args():

    # test for all the defaults
    args = [CHIRP_44K_PATH]
    args = parse_args(args)
    assert args.inputs == [CHIRP_44K_PATH]
    assert args.output_dir is None
    assert args.suffix is None
    assert args.model_type == 'sparse'
    assert args.emb_dim == 128
    assert args.retrain_type == 'ft'
    assert args.model_sparsity == 95.45
    assert args.no_centering is False
    assert args.hop_size == 0.1
    assert args.quiet is False

    # test when setting 'sparse' values
    args = [CHIRP_44K_PATH, '-o', '/output/dir', 
            '--suffix', 'suffix',
            '--model-type', 'sparse',
            '--retrain-type', 'kd', 
            '--model-sparsity', '53.5', 
            '--no-centering', 
            '--hop-size', '0.5',
            '--quiet'
        ]
    args = parse_args(args)
    assert args.inputs == [CHIRP_44K_PATH]
    assert args.output_dir == '/output/dir'
    assert args.suffix == 'suffix'
    assert args.model_type == 'sparse'
    assert args.emb_dim == 128
    assert args.retrain_type == 'kd'
    assert args.model_sparsity == 53.5
    assert args.no_centering is True
    assert args.hop_size == 0.5
    assert args.quiet is True

    # test when setting 'sea' values
    args = [CHIRP_44K_PATH, '-o', '/output/dir', 
            '--suffix', 'suffix',
            '--model-type', 'sea',
            '--emb-dim', '256', 
            '--hop-size', '0.5'
        ]
    args = parse_args(args)
    assert args.inputs == [CHIRP_44K_PATH]
    assert args.output_dir == '/output/dir'
    assert args.suffix == 'suffix'
    assert args.model_type == 'sea'
    assert args.emb_dim == 256
    assert args.retrain_type == 'ft'
    assert args.model_sparsity == 95.45
    assert args.no_centering is False
    assert args.hop_size == 0.5
    assert args.quiet is False

def test_run(capsys):

    # test invalid input
    invalid = [None, 5, 1.0]
    for i in invalid:
        pytest.raises(EdgeL3Error, run, i)

    # test empty input folder
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        tempdir = tempfile.mkdtemp()
        run([tempdir])

    # make sure it exited
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == -1

    # make sure it printed a message
    captured = capsys.readouterr()
    expected_message = 'Edgel3: No WAV files found in {}. Aborting.\n'.format(str([tempdir]))
    assert captured.out == expected_message

    # detele tempdir
    os.rmdir(tempdir)

    # test correct execution on test file (regression)
    tempdir = tempfile.mkdtemp()
    run(CHIRP_44K_PATH, output_dir=tempdir, verbose=True)

    # check output file created
    outfile = os.path.join(tempdir, 'chirp_44k.npz')
    assert os.path.isfile(outfile)

def test_main():

    # Duplicate regression test from test_run just to hit coverage
    tempdir = tempfile.mkdtemp()
    with patch('sys.argv', ['edgel3', CHIRP_44K_PATH, '--output-dir', tempdir]):
        main()

    # check output file created
    outfile = os.path.join(tempdir, 'chirp_44k.npz')
    assert os.path.isfile(outfile)

def test_script_main():

    # Duplicate regression test from test_run just to hit coverage
    tempdir = tempfile.mkdtemp()
    with patch('sys.argv', ['edgel3', CHIRP_44K_PATH, '--output-dir', tempdir]):
        import edgel3.__main__

    # check output file created
    outfile = os.path.join(tempdir, 'chirp_44k.npz')
    assert os.path.isfile(outfile)
