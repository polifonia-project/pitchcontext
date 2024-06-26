"""Provides a user interface for exploring dissonance model

Run as:

$ streamlit run st_dissonance.py -- -krnpath <path_to_kern_files> -jsonpath <path_to_json_files>
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
from pitchcontext.visualize import dissonance2colordict, plotArray
from pitchcontext.models import computeDissonance
from pitchcontext.base40 import base40naturalslist

parser = argparse.ArgumentParser(description='Visualize the dissonance of the focus note within its context.')
parser.add_argument(
    '-krnpath',
    dest='krnpath',
    help='Path to **kern files.',
    default="/Users/Krane108/data/MELFeatures/mtcfsinst2.0/krn",
)
parser.add_argument(
    '-jsonpath',
    dest='jsonpath',
    help='Path to json files (in MTCFeatures format).',
    default="/Users/Krane108/data/MELFeatures/mtcfsinst2.0/mtcjson",
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

    preauto_check = st.checkbox(
        "Determine preceding context automatically.",
        value=True
    )
    postauto_check = st.checkbox(
        "Determine following context automatically",
        value=True
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
    partialnotes_check = st.checkbox(
        "Include partial notes in preceding context.",
        value=True
    )
    removerep_check = st.checkbox(
        "Merge repeated notes.",
        value=True
    )
    accweight_check = st.checkbox(
        "Accumulate Weight.",
        value=True
    )
    include_focus_pre_check = st.checkbox(
        "Include Focus note in preceding context:",
        value=False,
    )
    include_focus_post_check = st.checkbox(
        "Include Focus note in following context:",
        value=False,
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
        value=False
    )
    post_usedw_check = st.checkbox(
        "Use distance weight for following context",
        value=False
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
    percentile_slider = st.slider(
        'Percentile threshold for dissonance.',
        min_value=0,
        max_value=100,
        step=1,
        value=80
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

dissonance_pre, dissonance_post, dissonance_context  = computeDissonance(
    song,
    wpc,
    combiner=lambda x, y: (x+y)*0.5,
    normalizecontexts=True
)
# dissonance_pre, dissonance_post, dissonance_context  = computeDissonance(
#     song,
#     wpc,
#     combiner=lambda x, y: np.maximum(np.nan_to_num(x), np.nan_to_num(y)),
#     normalizecontexts=True
# )

with col1:
    cons_threshold = np.nanpercentile(dissonance_context,percentile_slider)
    fig_cons, ax_cons = plotArray(dissonance_context, wpc.ixs, '', '')
    plt.axhline(y=cons_threshold, color='r', linestyle=':')
    plt.title('Dissonance of the focus note within its context')
    plt.xlabel('Note index')
    plt.ylabel('Dissonance')
    plt.ylim(-0.05, 1.05)
    st.write(fig_cons)

    cdict = dissonance2colordict(dissonance_context, wpc.ixs, percentile_slider, song.getSongLength())
    pngfn = song.createColoredPNG(cdict, '/tmp', showfilename=False)
    image = Image.open(pngfn)
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
        maxbeatstrength=[song.mtcsong['features']['maxbeatstrength'][ix] for ix in wpc.ixs]
    )
    components.html(f"<pre>{report}</pre>", height=650, scrolling=True)

