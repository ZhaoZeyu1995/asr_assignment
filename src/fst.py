import openfst_python as fst
import math
from helper_functions import parse_lexicon, generate_symbol_tables

lex = parse_lexicon('lexicon.txt')
word_table, phone_table, state_table = generate_symbol_tables(lex)

def generate_phone_wfst(f, start_state, phone, n):
    """
    Generate a WFST representing an n-state left-to-right phone HMM.

    Args:
        f (fst.Fst()): an FST object, assumed to exist already
        start_state (int): the index of the first state, assumed to exist already
        phone (str): the phone label
        n (int): number of states of the HMM excluding start and end

    Returns:
        the final state of the FST
    """

    current_state = start_state

    for i in range(1, n+1):

        in_label = state_table.find('{}_{}'.format(phone, i))

        # self-loop back to current state
        f.add_arc(current_state, fst.Arc(in_label, 0, None, current_state))

        # transition to next state

        # we want to output the phone label on the final state
        # note: if outputting words instead this code should be modified
        if i == n:
            out_label = phone_table.find(phone)
        else:
            out_label = 0   # output empty <eps> label

        next_state = f.add_state()
        f.add_arc(current_state, fst.Arc(in_label, out_label, None, next_state))

        current_state = next_state
    return current_state

def generate_linear_phone_wfst(phone_list):

    P = fst.Fst()

    current_state = P.add_state()
    P.set_start(current_state)

    for p in phone_list:

        next_state = P.add_state()
        P.add_arc(current_state, fst.Arc(phone_table.find(p), phone_table.find(p), None, next_state))
        current_state = next_state

    P.set_final(current_state)
    P.set_input_symbols(phone_table)
    P.set_output_symbols(phone_table)
    return P

def generate_linear_word_wfst(word_list):

    W = fst.Fst()

    current_state = W.add_state()
    W.set_start(current_state)

    for w in word_list:

        next_state = W.add_state()
        W.add_arc(current_state, fst.Arc(word_table.find(w), word_table.find(w), None, next_state))
        current_state = next_state

    W.set_final(current_state)
    W.set_input_symbols(word_table)
    W.set_output_symbols(word_table)
    return W

def generate_L_wfst(lex):
    """ Express the lexicon in WFST form
    
    Args:
        lexicon (dict): lexicon to use, created from the parse_lexicon() function
    
    Returns:
        the constructed lexicon WFST
    
    """
    L = fst.Fst()
    
    # create a single start state
    start_state = L.add_state()
    L.set_start(start_state)
    
    for (word, pron) in lex.items():
        
        current_state = start_state
        for (i,phone) in enumerate(pron):
            next_state = L.add_state()
            
            if i == len(pron)-1:
                # add word output symbol on the final arc
                L.add_arc(current_state, fst.Arc(phone_table.find(phone), \
                                                 word_table.find(word), None, next_state))
            else:
                L.add_arc(current_state, fst.Arc(phone_table.find(phone),0, None, next_state))
            
            current_state = next_state
                          
        L.set_final(current_state)
        L.add_arc(current_state, fst.Arc(0, 0, None, start_state))
        
    L.set_input_symbols(phone_table)
    L.set_output_symbols(word_table)                      
    
    return L

def generate_G_wfst(wseq):
    """ Generate a grammar WFST that accepts any sequence of words for words in a sentence.
        The bigrams not present in the sentence have a cost of 1, while those present have a cost of 0.
        Args:
            wseq (str): the sentence to use
        Returns:
            W (fst.Fst()): the grammar WFST """

    G = fst.Fst()
    start_state = G.add_state()
    G.set_start(start_state)

    prev_state = None
    word_start_states = dict()
    word_end_states = dict()

    bigrams = set(zip(wseq.split()[:-1], wseq.split()[1:]))
    for w in set(wseq.split()):
        current_state = G.add_state()
        word_start_states[w] = current_state

        weight = None if w == wseq.split()[0] else fst.Weight("tropical", 1.0)
        G.add_arc(start_state, fst.Arc(word_table.find("<eps>"), word_table.find("<eps>"), weight, current_state))

        prev_state = current_state
        current_state = G.add_state()

        G.add_arc(prev_state, fst.Arc(word_table.find(w), word_table.find(w), None, current_state))
        G.set_final(current_state)
        word_end_states[w] = current_state

        for w2, w2_state in word_start_states.items():
            weight = None if (w, w2) in bigrams else fst.Weight('tropical', 1.0)
            G.add_arc(current_state, fst.Arc(word_table.find("<eps>"), word_table.find("<eps>"), weight, w2_state))

            if w != w2:
                weight = None if (w2, w) in bigrams else fst.Weight('tropical', 1.0)
                G.add_arc(
                    word_end_states[w2],
                    fst.Arc(
                        word_table.find("<eps>"),
                        word_table.find("<eps>"),
                        weight,
                        word_start_states[w]
                    )
                )

    G.set_final(start_state)

    G.set_input_symbols(word_table)
    G.set_output_symbols(word_table)

    return G

def generate_H_wfst():
    with open("phonelist.txt", "r") as f:
        phones = [phone.strip() for phone in f.readlines()]
    H = fst.Fst()

    # create a single start state
    start_state = H.add_state()
    H.set_start(start_state)

    for _, phone in phone_table:
        if phone == "<eps>":
            continue

        current_state = H.add_state()
        H.add_arc(start_state, fst.Arc(0, 0, None, current_state))
        current_state = generate_phone_wfst(H, current_state, phone, 3)

        H.add_arc(current_state, fst.Arc(0, 0, None, start_state))
        H.set_final(current_state)

    H.set_input_symbols(state_table)
    H.set_output_symbols(phone_table)

    return H
