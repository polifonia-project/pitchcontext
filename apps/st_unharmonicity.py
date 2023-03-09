"""Provides a user interface for exploring unharmonic notes

Run as:

$ streamlit run st_unharmonicity.py -- -krnpath <path_to_kern_files> -jsonpath <path_to_json_files>
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

from pitchcontext import Song, PitchContext
from pitchcontext.visualize import unharmonicity2colordict, plotArray
from pitchcontext.models import computeDissonance, computeUnharmonicity, computeConsonance
from pitchcontext.base40 import base40naturalslist

parser = argparse.ArgumentParser(description='Visualize the dissonance of the focus note within its context.')
parser.add_argument(
    '-krnpath',
    dest='krnpath',
    help='Path to **kern files.',
    default="/Users/krane108/data/MELFeatures/eyck/krn",
)
parser.add_argument(
    '-jsonpath',
    dest='jsonpath',
    help='Path to json files (in MTCFeatures format).',
    default="/Users/krane108/data/MELFeatures/eyck/mtcjson",
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

    threshold_slider = st.slider(
        'Threshold for unharmonicity.',
        min_value=0.,
        max_value=2.,
        step=0.01,
        value=0.1
    )
    def beatstrength_threshold_format_func(x):
        if x == 1.0:
            return '1.0'
        if x == 0.5:
            return '1.0 and 0.5'
        return 'unknown'
    beatstrength_threshold = st.radio(    
        "Accept all notes with beatstr",
        (1.0, 0.5),
        index=1,
        format_func=beatstrength_threshold_format_func
    )
    def context_border_format_func(x):
        return str(x)
    context_border = st.radio(
        "Extend context till (including) note with beatstr >=",
        (1.0, 0.5),
        index=1,
        format_func=context_border_format_func
    )
    context_rel_focus = st.checkbox(
        "Never extend focus beyond note with higher weight than focus note",
        value=True,
    )
    #normalize_context_for_dissonance = context_rel_focus
    normalize_context_for_dissonance = False
    accweight_check = st.checkbox(
        "Accumulate Weight.",
        value=True
    )
    pre_usemw_check = st.checkbox(
        "Use metric weight for preceding context",
        value=True
    )
    post_usemw_check = st.checkbox(
        "Use metric weight for following context",
        value=True
    )

syncopes=True
if accweight_check:
    syncopes=False

wpc = PitchContext(
    song,
    syncopes=syncopes,
    remove_repeats=False,
    accumulate_weight=accweight_check,
    partial_notes=False,
    context_type='beats',
    len_context_pre='auto',
    len_context_post='auto',
    len_context_params={'threshold':context_border, 'not_heigher_than_focus':context_rel_focus},
    use_metric_weights_pre=pre_usemw_check,
    use_metric_weights_post=post_usemw_check,
    include_focus_pre=False,
    include_focus_post=False,
    use_distance_weights_pre=False,
    use_distance_weights_post=False,
    min_distance_weight_pre=0.0,
    min_distance_weight_post=0.0,
)

def combiner(x, y):
    res = np.zeros(len(x))
    for ix in range(len(x)):
        if np.isnan(y[ix]):
            res[ix] = x[ix]
        elif np.isnan(x[ix]):
            res[ix] = y[ix]
        else:
            res[ix] = (x[ix]+y[ix]) * 0.5
    return res

dissonance_pre, dissonance_post, dissonance_context  = computeDissonance(
    song,
    wpc,
    combiner=combiner,
    normalizecontexts=normalize_context_for_dissonance
)
consonance_pre, consonance_post, consonance_context  = computeConsonance(
    song,
    wpc,
    combiner=combiner,
    normalizecontexts=normalize_context_for_dissonance
)
# dissonance_pre, dissonance_post, dissonance_context  = computeDissonance(
#     song,
#     wpc,
#     combiner=lambda x, y: np.minimum(np.nan_to_num(x), np.nan_to_num(y)),
#     normalizecontexts=normalize_context_for_dissonance
# )
unharmonicity = computeUnharmonicity(
    song,
    wpc,
    dissonance_pre,
    consonance_pre,
    beatstrength_threshold
)

with col1:
    cons_threshold = threshold_slider
    fig_cons, ax_cons = plotArray(unharmonicity, wpc.ixs, '', '')
    plt.axhline(y=cons_threshold, color='r', linestyle=':')
    plt.title('Unharmoncity of the focus note within its context')
    plt.xlabel('Note index')
    plt.ylabel('Unharmonicity')
    plt.ylim(-0.05, 1.05)
    st.write(fig_cons)

    cdict = unharmonicity2colordict(unharmonicity, wpc.ixs, threshold_slider, song.getSongLength())
    pngfn = song.createColoredPNG(cdict, '/tmp', showfilename=False)
    try:
        image = Image.open(pngfn)
    except FileNotFoundError as e: #maybe multiple pages. Try page 1
        image = Image.open(pngfn.replace('.png','-0.png'))
    st.image(image, output_format='PNG', use_column_width=True)

    fig_pre, ax_pre = plt.subplots(figsize=(10,2))
    sns.heatmap(wpc.pitchcontext[:,:40].T, ax=ax_pre, xticklabels=wpc.ixs, yticklabels=base40naturalslist)
    ax_pre.invert_yaxis()
    plt.title('Pitchcontext vectors preceding context')
    plt.xlabel('Note index')
    plt.ylabel('Pitch')
    plt.yticks(rotation=0)
    st.write(fig_pre)

    fig_post, ax_post = plt.subplots(figsize=(10,2))
    sns.heatmap(wpc.pitchcontext[:,40:].T, ax=ax_post, xticklabels=wpc.ixs, yticklabels=base40naturalslist)
    ax_post.invert_yaxis()
    plt.title('Pitchcontext vectors following context')
    plt.xlabel('Note index')
    plt.ylabel('Pitch')
    plt.yticks(rotation=0)
    st.write(fig_post)


#st.write(asdict(wpc.params))

with col2:
    report = wpc.printReport(
        dissonance_context=dissonance_context,
        dissonance_pre=dissonance_pre,
        dissonance_post=dissonance_post,
        consonance_context=consonance_context,
        consonance_pre=consonance_pre,
        consonance_post=consonance_post,
        unharmonicity=unharmonicity,
        maxbeatstrength=[song.mtcsong['features']['maxbeatstrength'][ix] for ix in wpc.ixs]
    )
    components.html(f"<pre>{report}</pre>", height=650, scrolling=True)

