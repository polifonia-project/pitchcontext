"""Provides methods for visualialization of data and results."""

import numpy as np
from matplotlib import pyplot as plt
from .base40 import base40

#from datetime import datetime
#print(__file__, datetime.now().strftime("%H:%M:%S"))

#for repeated pitches:
#spans would be better for ixs [(start,end),(start,end),...]
def getColoredIxs(target_ixs, ixs, songlength):
    colorixs = []
    for ix in target_ixs:
        colorixs.append(ix)
        temp_ix = ix+1
        while (not temp_ix in ixs) and (temp_ix < songlength):
            colorixs.append(temp_ix)
            temp_ix += 1
    return colorixs

def array2colordict(array, ixs, criterion, songlength, color='red', greynan=True):
    ixsnp = np.array(ixs)
    target_ixs = ixsnp[np.where(criterion(array))]
    nan_ixs = []
    if greynan:
        nan_ixs = ixsnp[np.where(np.isnan(array))]        
    color_ixs = getColoredIxs(target_ixs, ixs, songlength)
    return {color:color_ixs, 'grey':nan_ixs}

#color all notes with high novelty
def novelty2colordict(novelty, ixs, percentile, songlength, color='red', greynan=True):
    criterion = lambda x : x >= np.nanpercentile(novelty,percentile)
    return array2colordict(
        novelty,
        ixs,
        criterion,
        songlength,
        color,
        greynan
    )

#color all notes with low consonance
def dissonance2colordict(dissonance, ixs, percentile, songlength, color='red', greynan=True):
    criterion = lambda x : x >= np.nanpercentile(dissonance,percentile)
    return array2colordict(
        dissonance,
        ixs,
        criterion,
        songlength,
        color,
        greynan
    )

#color all unharmonic notes
def unharmonicity2colordict(unharmonicity, ixs, threshold, songlength, color='red', greynan=True):
    criterion = lambda x : x >= threshold
    return array2colordict(
        unharmonicity,
        ixs,
        criterion,
        songlength,
        color,
        greynan
    )


def plotArray(array, ixs, xlabel : str, ylabel : str, figsize=(10,3)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.ylim(-0.05,np.max(np.nan_to_num(array)) * 1.05)
    plt.plot(array)
    plt.xlabel(xlabel, fontsize='large')
    plt.ylabel(ylabel, fontsize='large')
    plt.xticks(np.arange(0, len(ixs), 1), [str(i) for i in ixs])
    plt.xticks(rotation = 90)
    return fig, ax

def printPitchContextVector(
    wpc, #PitchContext object
    note_ix=None, #report single note. IX in original song, not in ixs
    **features, #any other values to report. key: name, value: array size len(ixs)
):
    """Returns a textual report with for each note the values of several features.

    For each note print
    - pitch and (metric) weight as computed by `wpc.computeWeightedPitch`
    - indices (in `wpc.ixs`) of notes in the preceding context
    - indices (in the MTC features) of notes in the preceding context
    - indices (in `wpc.ixs`) of notes in the following context
    - indices (in the MTC features) of notes in the following context
    - pitches and corresponding weights in the precedings context
    - pitches and corresponding wieghts in the following context
    - any other feature provided as keyword argument (see below)

    Parameters
    ----------
    note_ix : int, default None
        Only print the values the note at index `note_ix` in the original melody (not in `wpc.ixs`).
    **features  : keyword arguments
        any other feature to report. The keyword is the name of the feature, the value is a 1D array
        with the same length as `wpc.ixs`.

    Returns
    ----------
    str
        String containing the report.
    """
    output = []
    for ix in range(len(wpc.ixs)):
        if note_ix:
            if note_ix != wpc.ixs[ix]: continue
        pre_pitches = []
        post_pitches = []
        for p in range(40):
            if wpc.pitchcontext[ix,p] > 0.0:
                pre_pitches.append((base40[p],wpc.pitchcontext[ix,p]))
        for p in range(40):
            if wpc.pitchcontext[ix,p+40] > 0.0:
                post_pitches.append((base40[p], wpc.pitchcontext[ix,p+40]))
        pre_pitches = [str(p) for p in sorted(pre_pitches, key=lambda x: x[1], reverse=True)]
        post_pitches = [str(p) for p in sorted(post_pitches, key=lambda x: x[1], reverse=True)]
        output.append(f"note {wpc.ixs[ix]}, ix: {ix}")
        output.append(f"  pitch, weight: {wpc.song.mtcsong['features']['pitch'][wpc.ixs[ix]]}, {wpc.song.mtcsong['features']['weights'][wpc.ixs[ix]]}")
        if len(wpc.contexts_pre[ix]) > 0:
            output.append(f"  context_pre (ixs): {wpc.contexts_pre[ix][0]}-{wpc.contexts_pre[ix][-1]}")
            output.append(f"  context_pre (notes): {np.array(wpc.ixs)[wpc.contexts_pre[ix][0]]}-{np.array(wpc.ixs)[wpc.contexts_pre[ix][-1]]}")
        else:
            output.append(f"  context_pre (ixs): []")
            output.append(f"  context_pre (notes): []")
        if len(wpc.contexts_post[ix]) > 0:
            output.append(f"  context_post (ixs): {wpc.contexts_post[ix][0]}-{wpc.contexts_post[ix][-1]}")
            output.append(f"  context_post (notes): {np.array(wpc.ixs)[wpc.contexts_post[ix]][0]}-{np.array(wpc.ixs)[wpc.contexts_post[ix]][-1]}")
        else:
            output.append(f"  context_post (ixs): []")
            output.append(f"  context_post (notes): []")
        output.append( "  pre:" + "\n       ".join(pre_pitches))
        output.append( "  post:"+ "\n        ".join(post_pitches))
        for name in features.keys():
            output.append(f"  {name}: {features[name][ix]}")
        output.append("")
    return '\n'.join(output)
