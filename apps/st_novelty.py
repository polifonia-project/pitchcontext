import json
from fractions import Fraction
from PIL import Image
import tempfile

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from pitchcontext import Song, PitchContext
from pitchcontext.visualize import novelty2colordict, consonance2colordict, plotArray
from pitchcontext.models import computeConsonance, computeNovelty

path_to_krn = 'NLB147059_01.krn'
with open('NLB147059_01.json','r') as f:
    mtcsong = json.load(f)

song = Song(mtcsong, path_to_krn)
songlength_beat = float(sum([Fraction(length) for length in song.mtcsong['features']['beatfraction']]))

st.title("Novelty")

with st.sidebar:
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
        index=2
    )
    pre_usemw_check = st.checkbox(
        "Use metric weight for preceding context",
        value=True
    )
    post_usemw_check = st.checkbox(
        "Use metric weight for following context",
        value=False
    )
    pre_usedw_check = st.checkbox(
        "Use distance weight for preceding context",
        value=True
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

wpc = PitchContext(
    song,
    removeRepeats=removerep_check,
    accumulateWeight=accweight_check,
    len_context_beat=(pre_c_slider,post_c_slider),
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

novelty = computeNovelty(song, wpc)

nov_threshold = np.nanpercentile(novelty,percentile_slider)
fig_nov, ax_nov = plotArray(novelty, wpc.ixs, '', '')
plt.axhline(y=nov_threshold, color='r', linestyle=':')
st.write(fig_nov)

# @st.cache
# def createScore(outputdir):
#     pngfn = song.createPNG(outputdir)
#     return pngfn

# pngfn = createScore('/Users/krane108/tmp/')
# image = Image.open(pngfn)
# st.image(image)

# st.write(wpc.params)

cdict = novelty2colordict(novelty, wpc.ixs, percentile_slider, song.getSongLength())
pngfn = song.createColoredPNG(cdict, '/Users/krane108/tmp/', showfilename=False)
image = Image.open(pngfn)
st.image(image)

#wpc.printReport(novelty=novelty, note_ix=33)
