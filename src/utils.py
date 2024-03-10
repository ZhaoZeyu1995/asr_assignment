import os
import glob
from subprocess import check_call
from IPython.display import Image

def display_fst(f, filename="tmp.png"):
    f.draw('tmp.dot', portrait=True)
    check_call(['dot','-Tpng','-Gdpi=200','tmp.dot','-o', filename])
    return Image(filename=filename)

def compute_unigram_probs():
    words = "peter piper picked a peck of pickled peppers where's the peck of pickled peppers peter piper picked"
    total_words = 0
    unigram_counts = {w: 0 for w in words.split(' ')}
    for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
        transcription = read_transcription(wav_file).split(' ')

        for word in transcription:
            total_words += 1
            unigram_counts[word] += 1

    unigram_probs = {w: (float(unigram_counts[w]) / total_words) for w in words.split()}
    return unigram_probs


def read_transcription(wav_file):
    """
    Get the transcription corresponding to wav_file.
    """

    transcription_file = os.path.splitext(wav_file)[0] + '.txt'

    with open(transcription_file, 'r') as f:
        transcription = f.readline().strip()

    return transcription
