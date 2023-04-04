"""Provides a user interface for exploring implied chords model

Run as:

$ streamlit run st_impliedchords.py -- -krnpath <path_to_kern_files> -jsonpath <path_to_json_files>
"""

import argparse
import json
from fractions import Fraction
from PIL import Image
import os
from dataclasses import asdict
import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from IPython import display
import numpy as np
from pitchcontext import Song, PitchContext
from pitchcontext.visualize import unharmonicity2colordict, plotArray
from pitchcontext.models import computeDissonance, computeConsonance, computeUnharmonicity, ImpliedHarmony
from pitchcontext.base40 import base40naturalslist, base40

parser = argparse.ArgumentParser(description='Generate a chord sequence for a given melody.')
parser.add_argument(
    '-krnpath',
    dest='krnpath',
    help='Path to **kern files.',
    default="/Users/krane108/data/MELFeatures/mtcfsinst2.0/krn",
)
parser.add_argument(
    '-jsonpath',
    dest='jsonpath',
    help='Path to json files (in MTCFeatures format).',
    default="/Users/krane108/data/MELFeatures/mtcfsinst2.0/mtcjson",
)
args = parser.parse_args()
krnpath = args.krnpath
jsonpath = args.jsonpath

st.set_page_config(layout="wide")
col1, col2 = st.columns([2,1])

#select first file from the krnpath
krnfiles = os.listdir(krnpath)
for krnfile in krnfiles:
    if krnfile.endswith('.krn'):
        break
else:
    krnfile = 'none'

firstid = krnfile.rstrip(".krn")

with st.sidebar:
    songid = st.text_input(
        label="Song ID",
        value=firstid
    )

    #we need to load the song here, because the song is needed to set pre_c_slider and post_c_slider max
    krnfilename = os.path.join(krnpath, songid+'.krn')
    jsonfilename = os.path.join(jsonpath, songid+'.json')
    with open(jsonfilename,'r') as f:
        mtcsong = json.load(f)

    song = Song(mtcsong, krnfilename)
    songlength_beat = float(sum([Fraction(length) for length in song.mtcsong['features']['beatfraction']]))

    same_root_slider = st.slider(
        'Multiplier same root, different quality',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.1
    )

    lowweight_check = st.checkbox(
        "No chord change on low metric weight",
        value=True
    )

    root_third_final_check = st.checkbox(
        "Chord-root or third in melody at final note",
        value=True
    )

    use_scalemask_check = st.checkbox(
        "Use scale when choosing chords",
        value=True
    )
    no_fourth_fifth_slider = st.slider(
        'Multiplier root movement other than 4th or 5th',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.8
    )

    final_v_i_slider = st.slider(
        'Multiplier NO V-I final',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.5
    )

    dom_fourth_slider = st.slider(
        'Multiplier NO 4th up after dominant',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.1
    )

    dim_m2_slider = st.slider(
        'Multiplier NO minor second after dim',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.1
    )

    fourth_dom_slider = st.slider(
        'Multiplier NO major or dominant before 4th up',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.8
    )

    pre_c_slider = st.slider(
        'Length of preceding context (beats)',
        min_value=0.0,
        max_value=songlength_beat,
        step=0.5,
        value=1.0
    )
    post_c_slider = st.slider(
        'Length of following context (beats)',
        min_value=0.0,
        max_value=songlength_beat,
        step=0.5,
        value=1.0
    )
    preauto_check = st.checkbox(
        "Determine preceding context automatically.",
        value=False
    )
    postauto_check = st.checkbox(
        "Determine following context automatically",
        value=False
    )
    partialnotes_check = st.checkbox(
        "Include partial notes in preceding context.",
        value=True
    )
    removerep_check = st.checkbox(
        "Merge repeated notes.",
        value=False
    )
    accweight_check = st.checkbox(
        "Accumulate Weight.",
        value=True
    )
    include_focus_pre_check = st.checkbox(
        "Include Focus note in preceding context:",
        value=True,
    )
    include_focus_post_check = st.checkbox(
        "Include Focus note in following context:",
        value=True,
    )
    pre_usemw_check = st.checkbox(
        "Use metric weight for preceding context",
        value=True
    )
    post_usemw_check = st.checkbox(
        "Use metric weight for following context",
        value=True
    )
    pre_usedw_check = st.checkbox(
        "Use distance weight for preceding context",
        value=True
    )
    post_usedw_check = st.checkbox(
        "Use distance weight for following context",
        value=True
    )
    mindistw_pre_slider = st.slider(
        'Minimal distance weight preceding context',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.0
    )
    mindistw_post_slider = st.slider(
        'Minimal distance weight following context',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.0
    )


len_context_pre = 'auto' if preauto_check else pre_c_slider
len_context_post = 'auto' if postauto_check else post_c_slider

syncopes=True
if accweight_check:
    syncopes=False

wpc = PitchContext(
    song,
    syncopes=syncopes,
    remove_repeats=removerep_check,
    accumulate_weight=accweight_check,
    partial_notes=partialnotes_check,
    context_type='beats',
    len_context_pre=len_context_pre,
    len_context_post=len_context_post,
    use_metric_weights_pre=pre_usemw_check,
    use_metric_weights_post=post_usemw_check,
    include_focus_pre=include_focus_pre_check,
    include_focus_post=include_focus_post_check,
    use_distance_weights_pre=pre_usedw_check,
    use_distance_weights_post=post_usedw_check,
    min_distance_weight_pre=mindistw_pre_slider,
    min_distance_weight_post=mindistw_post_slider,
)

def myChordTransitionScore(chords, chord1_ixs, chord2_ixs, scalemask=np.ones(40, dtype=bool), song=None, wpc=None):

    #scoring scheme.
    pitch1 = chord1_ixs[1] % 40
    pitch2 = chord2_ixs[1] % 40
    songlength = song.songlength
    shift = (pitch2 - pitch1) % 40 #interval of roots in base40

    #no score if root of chord tones is not in the scalemask)
    if use_scalemask_check:
        if not scalemask[pitch1] or not scalemask[pitch2]:
            return 0.0

    #else compute score step by step

    # 1. start with score for 'next' chord
    score = chords[chord2_ixs]

    # Discourage same root, different quality #except maj -> dom
    if pitch1 == pitch2:
        if chord1_ixs[2] != chord2_ixs[2]:
            if  not ( chord1_ixs[2] == 2 and chord2_ixs[2] == 3 ):
                score = score * same_root_slider

    # discourage root change on note with low metric weight
    if lowweight_check:
        if song.mtcsong['features']['beatstrength'][chord2_ixs[0]] < 0.5:
            if pitch1 != pitch2 or chord1_ixs[2] != chord2_ixs[2]:
                score = -100.

    # penalty for harmonically distant
    # HOW TO DO THIS?
    # e.g. we do not want 

    # prefer root movement of fourth and fifth
    if shift != 17 and shift != 23 and shift !=0:
        score = score * no_fourth_fifth_slider

    # prefer V-I relation for final note
    if chord2_ixs[0] == songlength - 1:
        if shift != 17:
            score = score * final_v_i_slider

    # If previous is dom. Then root must be fourth up
    if chord1_ixs[2] == 3:
        if shift != 17:
            score = score * dom_fourth_slider

    # 5. If previous is dim. Then root must be semitone up
    if chord1_ixs[2] == 0:
        if shift != 5:
            score = score * dim_m2_slider

    # if root is fourth up: prefer maj or dom for first chord
    if shift == 17:
        if chord1_ixs[2] == 0 or chord1_ixs[2] == 1:
            score = score * fourth_dom_slider

    # prefer root or third in melody for last note
    if root_third_final_check:
        if chord2_ixs[0] == songlength - 1:
            melp40 = song.mtcsong['features']['pitch40'][songlength-1] - 1
            root_int = (melp40 - pitch2) % 40
            if not root_int in [0, 11, 12]:
                score = -100
    return score

ih = ImpliedHarmony(wpc)

trace, trace_score, score, traceback = ih.getOptimalChordSequence(chordTransitionScoreFunction=myChordTransitionScore)
strtrace = ih.trace2str(trace)

#replace same chord
toremove = []
for ix in range(1, len(strtrace)):
    if strtrace[ix] == strtrace[ix-1]:
        toremove.append(ix)
for ix in toremove:
    strtrace[ix] = ' '

#replace - with b
for ix in range(len(strtrace)):
    strtrace[ix] = strtrace[ix].replace('-','b')

with col1:
    pngfn_chords = song.createPNG(
        '/tmp',
        showfilename=False,
        lyrics=strtrace
    )
    image = Image.open(pngfn_chords)
    st.image(image, output_format='PNG', use_column_width=True)

    pngfn_orig = song.createPNG(
        '/Users/krane108/tmp/',
        showfilename=True
    )
    image = Image.open(pngfn_orig)
    st.image(image, output_format='PNG', use_column_width=True)

#st.write(asdict(wpc.params))

with col2:
    report = wpc.printReport(
        chordscore = [tr[1] for tr in trace_score]
    )
    components.html(f"<pre>{report}</pre>", height=650, scrolling=True)

