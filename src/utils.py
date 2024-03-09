import os
from subprocess import check_call
from IPython.display import Image

def display_fst(f):
    f.draw('tmp.dot', portrait=True)
    check_call(['dot','-Tpng','-Gdpi=200','tmp.dot','-o','tmp.png'])
    return Image(filename='tmp.png')

def read_transcription(wav_file):
    """
    Get the transcription corresponding to wav_file.
    """

    transcription_file = os.path.splitext(wav_file)[0] + '.txt'

    with open(transcription_file, 'r') as f:
        transcription = f.readline().strip()

    return transcription
