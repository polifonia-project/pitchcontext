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

st.title("Consonance")

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
        value=1.0
    )
    post_c_slider = st.slider(
        'Length of following context (beats)',
        min_value=0.0,
        max_value=songlength_beat,
        step=0.5,
        value=1.0
    )
    removerep_check = st.checkbox(
        "Merge repeated notes.",
        value=True
    )
    accweight_check = st.checkbox(
        "Accumulate Weight.",
        value=True
    )
    includeFocus_rad = st.radio(
        "Include Focus note in context:",
        ('none', 'pre', 'post', 'both'),
        index=0
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
        'Percentile threshold for consonance.',
        min_value=0,
        max_value=100,
        step=1,
        value=40
    )

len_tuple = [None,None]
if preauto_check:
    len_tuple[0] = 'auto'
else:
    len_tuple[0] = pre_c_slider
if postauto_check:
    len_tuple[1] = 'auto'
else:
    len_tuple[1] = post_c_slider

wpc = PitchContext(
    song,
    removeRepeats=removerep_check,
    accumulateWeight=accweight_check,
    len_context_beat=len_tuple,
    use_metric_weights=(pre_usemw_check, post_usemw_check),
    includeFocus=includeFocus_rad,
    use_distance_weights=(pre_usedw_check, post_usedw_check),
    min_distance_weight=(mindistw_pre_slider, mindistw_post_slider)
)

fig_pre, ax_pre = plt.subplots(figsize=(10,2))
sns.heatmap(wpc.pitchcontext[:,:40].T, ax=ax_pre, xticklabels=wpc.ixs)
st.write(fig_pre)

fig_post, ax_post = plt.subplots(figsize=(10,2))
sns.heatmap(wpc.pitchcontext[:,40:].T, ax=ax_post, xticklabels=wpc.ixs)
st.write(fig_post)

consonance_pre, consonance_post, consonance_context  = computeConsonance(song, wpc, combiner=lambda x, y: (x+y)*0.5, normalizecontexts=True)
#consonance_pre, consonance_post, consonance_context  = computeConsonance(song, wpc, combiner=np.minimum)

cons_threshold = np.nanpercentile(consonance_context,percentile_slider)
fig_cons, ax_cons = plotArray(consonance_context, wpc.ixs, '', '')
plt.axhline(y=cons_threshold, color='r', linestyle=':')
st.write(fig_cons)

# @st.cache
# def createScore(outputdir):
#     pngfn = song.createPNG(outputdir)
#     return pngfn

# pngfn = createScore('/Users/krane108/tmp/')
# image = Image.open(pngfn)
# st.image(image)

cdict = consonance2colordict(consonance_context, wpc.ixs, percentile_slider, song.getSongLength())
pngfn = song.createColoredPNG(cdict, '/tmp', showfilename=False)
image = Image.open(pngfn)
st.image(image, output_format='PNG')

#wpc.printReport(novelty=novelty, note_ix=33)

st.write(wpc.params)

wpc.printReport(
    consonance_context=consonance_context,
    consonance_pre=consonance_pre,
    consonance_post=consonance_post,
    maxbeatstrength=[song.mtcsong['features']['maxbeatstrength'][ix] for ix in wpc.ixs]
)
