"""Provides a user interface for exploring novelty model

Run as:

$ streamlit run st_novelty.py -- -krnpath <path_to_kern_files> -jsonpath <path_to_json_files>
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
from pitchcontext.visualize import novelty2colordict, plotArray
from pitchcontext.models import computeNovelty
from pitchcontext.base40 import base40naturalslist

parser = argparse.ArgumentParser(description='Visualize the novelty of following context given the preceding context.')
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

#select a file from the krnpath
initialid = random.choice(os.listdir(krnpath))
initialid = initialid.rstrip(".krn")

with st.sidebar:
    songid = st.text_input(
        label="Song ID",
        value=initialid
    )

    #Load song here. The song is needed to compute the song length.
    krnfilename = os.path.join(krnpath, songid+'.krn')
    jsonfilename = os.path.join(jsonpath, songid+'.json')
    with open(jsonfilename,'r') as f:
        mtcsong = json.load(f)

    song = Song(mtcsong, krnfilename)
    songlength_beat = float(sum([Fraction(length) for length in song.mtcsong['features']['beatfraction']]))
    
    preauto_check = st.checkbox(
        "Determine preceding context automatically.",
        value=False
    )
    postauto_check = st.checkbox(
        "Determine following context automatically",
        value=False
    )
    pre_c_slider = st.slider(
        'Length of preceding context (beats)',
        min_value=0.0,
        max_value=songlength_beat,
        step=0.5,
        value=songlength_beat
    )
    post_c_slider = st.slider(
        'Length of following context (beats)',
        min_value=0.0,
        max_value=songlength_beat,
        step=0.5,
        value=0.0
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
        'Percentile threshold for novelty.',
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

novelty = computeNovelty(song, wpc)

with col1:    
    nov_threshold = np.nanpercentile(novelty,percentile_slider)
    fig_nov, ax_nov = plotArray(novelty, wpc.ixs, '', '')
    plt.axhline(y=nov_threshold, color='r', linestyle=':')
    plt.title('Novelty of the following context with respect to the preceding context')
    plt.xlabel('Note index')
    plt.ylabel('Novelty')
    st.write(fig_nov)

    cdict = novelty2colordict(novelty, wpc.ixs, percentile_slider, song.getSongLength())
    pngfn = song.createColoredPNG(cdict, '/Users/krane108/tmp/', showfilename=False)
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

with col2:
    report = wpc.printReport(
        novelty=novelty,
        maxbeatstrength=[song.mtcsong['features']['maxbeatstrength'][ix] for ix in wpc.ixs]
    )
    components.html(f"<pre>{report}</pre>", height=650, scrolling=True)

