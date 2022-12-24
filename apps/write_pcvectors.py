import argparse
import numpy as np
import os
import json
import sys, traceback

from pitchcontext import Song, PitchContext
from pitchcontext.visualize import novelty2colordict, plotArray
from pitchcontext.models import computeNovelty
from pitchcontext.base40 import base40naturalslist
from pitchcontext.song import OnsetMismatchError

parser = argparse.ArgumentParser(description='Write pitchcontext vectors to file for all files in <krnpath>.')
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
parser.add_argument(
    '-outpath',
    dest='outpath',
    help='Path to put output files.',
    default="/Users/krane108/data/MELFeatures/eyck/pitchvectors",
)
parser.add_argument(
    '-startat',
    type=str,
    help='ID of the melody to start with. Skips all preceeding ones.',
    default=''
)
parser.add_argument(
    '-only',
    type=str,
    help='ID of the melody to process. Skips all others.',
    default=''
)
args = parser.parse_args()
krnpath = args.krnpath
jsonpath = args.jsonpath
outpath = args.outpath
startat = args.startat
only = args.only

#Exception: No Meter
class NoMeterError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

#make sure outpuat exists
os.makedirs(outpath, exist_ok=True)

def pc2string(krndirname, krnfilename):
    songid = krnfilename.strip('.krn')
    fullkrnfilename = os.path.join(krndirname, krnfilename)
    fulljsonfilename = os.path.join(jsonpath, songid+'.json')
    with open(fulljsonfilename,'r') as f:
        mtcsong = json.load(f)

    if mtcsong['freemeter']:
        raise NoMeterError(krnfilename)

    song = Song(mtcsong, fullkrnfilename)

    wpc = PitchContext(
        song,
        syncopes=False, #Because of accumulate_weight=True
        remove_repeats=True,
        accumulate_weight=True,
        partial_notes=True,
        context_type='beats',
        len_context_pre='auto',
        len_context_post='auto',
        use_metric_weights_pre=True,
        use_metric_weights_post=True,
        include_focus_pre=True,
        include_focus_post=True,
        use_distance_weights_pre=False,
        use_distance_weights_post=False,
    )

    songlength = len(wpc.ixs)
    #create string representation
    lines = []
    #first write length of the pitch context vector
    lines.append(str(songlength))
    #write all weighted pitch vectors
    for i in range(songlength):
        lines.append( str(song.mtcsong['features']['pitch40'][wpc.ixs[i]]-1) + ' ' + str(np.max(wpc.weightedpitch[i])))
    #write all pre and post contexts. One line for pre, one line for post
    for i in range(songlength):
        lines.append(' '.join(map(str, wpc.pitchcontext[i][:40])))
        lines.append(' '.join(map(str, wpc.pitchcontext[i][40:])))
    return '\n'.join(lines)

skip=False
if len(startat)>0: skip=True

for krndirname, dirs, krnfilenames in os.walk(krnpath):
    for krnfilename in krnfilenames:
        if len(only) > 0:
            if krnfilename != only+'.krn':
                continue
        if len(startat) > 0:
            if krnfilename == startat+'.krn':
                skip = False
        if skip:
            continue
        print(krnfilename)
        sys.stdout.flush()
        if krnfilename.endswith(".krn"):
            songid = krnfilename.strip('.krn')
            outfilename = os.path.join(outpath, songid+'.txt')
            try:
                pcstr = pc2string(krndirname, krnfilename)
            except NoMeterError as e:
                print("No Meter in ", e)
                continue
            except OnsetMismatchError as e:
                print(e)
                continue
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                continue
            with open(outfilename,'w') as f_out:
                f_out.write(pcstr)


