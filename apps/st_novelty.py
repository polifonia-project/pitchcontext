import json
from fractions import Fraction
from PIL import Image
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from pitchcontext import Song, PitchContext
from pitchcontext.visualize import novelty2colordict, consonance2colordict, plotArray
from pitchcontext.models import computeConsonance, computeNovelty

from pitchcontext.base40 import base40naturalslist

path_to_krn = 'NLB147059_01.krn'
with open('NLB147059_01.json','r') as f:
    mtcsong = json.load(f)

song = Song(mtcsong, path_to_krn)
songlength_beat = float(sum([Fraction(length) for length in song.mtcsong['features']['beatfraction']]))

st.title("Novelty")

with st.sidebar:
    songid = st.text_input(
        label="Song ID",
        value="NLB147059_01"
    )
    krnpath = st.text_input(
        label="Path to **kern files",
        value="/Users/krane108/data/MTC/MTC-FS-INST-2.0/krn"
    )
    jsonpath = st.text_input(
        label="Path to MTCFeatures .json files",
        value="/Users/krane108/data/MTCFeatures/MTC-FS-inst-2.0/json"
    )

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

novelty = computeNovelty(song, wpc)

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
st.image(image)

#wpc.printReport(novelty=novelty, note_ix=33)
