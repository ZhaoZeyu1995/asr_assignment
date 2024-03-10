#!/usr/bin/env python3

import argparse
import sys
import json
import glob
import timeit
import os
import wer
import observation_model
import openfst_python as fst
from helper_functions import parse_lexicon, generate_symbol_tables
from fst import *
from decoder import *
from utils import *

def get_num_arcs(f):
    return sum(1 + f.num_arcs(s) for s in f.states())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--self-arc-prob', type=float)
    parser.add_argument('--final-prob', type=float)
    parser.add_argument('--use-unigram', type=bool)
    parser.add_argument('--use-silence-state', type=bool)
    parser.add_argument('--silence-state-num', type=int)
    parser.add_argument('--pruning-threshold', type=float)
    parser.add_argument('--pruning-strategy', choices=["normal"], type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

    args = parser.parse_args()

    if args.config:
        f = open(args.config)
        config = json.load(f)
        f.close()

        for k, v in config.items():
            if k not in args.__dict__ or args.__dict__[k] is None:
                args.__dict__[k] = v

    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print()

    return args

if __name__ == '__main__':
    print(' '.join(sys.argv))
    args = parse_args()

    unigram_probs = compute_unigram_probs() if args.use_unigram else None

    # Create the lexicon
    string = "peter piper picked a peck of pickled peppers"
    lex = parse_lexicon('lexicon.txt')
    word_table, phone_table, state_table = generate_symbol_tables(lex)

    # Create L FST
    L = generate_L_wfst(lex)
    L.arcsort()
    display_fst(L, "l.png")

    print("Generated L")

    # Create G FST
    G = generate_G_wfst(string, unigram_probs)
    G.arcsort()
    display_fst(G, "g.png")

    print("Generated G")

    # Create H FST
    H = generate_H_wfst(args.self_arc_prob)
    H.arcsort()
    display_fst(H, "h.png")

    print("Generated H")
    
    new_f = fst.compose(H, fst.compose(L, G))
    display_fst(new_f)
    # Compose FSTs
    LG = fst.determinize(fst.compose(L, G).rmepsilon())
    LG.arcsort()
    print("Generated LG")
    LG = LG.minimize()
    print("Minimized LG")
    compose = fst.compose(H, LG).rmepsilon()
    print("Composed LG and H")
    f = fst.determinize(compose)
    print("Determinized Composition")
    f.arcsort()
    print("Determinized")
    # f = f.minimize()
    # print("Minimized")

    print("Generated Composition")

    # Train and Report Metrics
    wav_files = 0
    total_errors, total_words = 0, 0
    decode_times = []
    backtrace_times = []
    for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
        wav_files += 1

        decoder = MyViterbiDecoder(f, wav_file, args.pruning_threshold)
        
        decode_time = timeit.timeit(lambda: decoder.decode(), number=1)
        backtrace_time = timeit.timeit(lambda: decoder.backtrace(), number=1)
        decode_times.append(decode_time)
        backtrace_times.append(backtrace_time)
        (state_path, words) = decoder.backtrace()

        transcription = read_transcription(wav_file)
        error_counts = wer.compute_alignment_errors(transcription, words)
        word_count = len(transcription.split())

        total_errors += sum(error_counts)
        total_words += word_count

        print(wav_files, error_counts, word_count)

    print(f"Total WER: {total_errors / total_words}")
    print(f"Average decode() Time: {sum(decode_times) / wav_files}")
    print(f"Average backtrace() Time: {sum(backtrace_times) / wav_files}")
    print(f"FST Number of States: {f.num_states()}")
    print(f"FST Number of Arcs: {get_num_arcs(f)}")
    # print(f"FST Size: {f.size()} byte bytes")  #TODO: determine whether this works
