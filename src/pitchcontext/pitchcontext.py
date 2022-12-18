from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
from fractions import Fraction

import numpy as np

from .song import Song
from .base40 import base40

@dataclass()
class PCParameters():
    #For computing note weights
    remove_repeats: bool = True
    syncopes: bool = True
    metric_weights: str = 'beatstrength'
    accumulate_weight: bool = False

    #For computing weighted pitch context vectors
    context_type: str = 'beats'

    len_context: 'typing.Any' = None
    len_context_pre: 'typing.Any' = 'auto'
    len_context_pre_auto: bool = True
    len_context_post: 'typing.Any' = 'auto'
    len_context_post_auto: bool = True

    use_metric_weights: bool = None
    use_metric_weights_pre: bool = True
    use_metric_weights_post: bool = True

    use_distance_weights: bool = None
    use_distance_weights_pre: bool = True
    use_distance_weights_post: bool = True

    min_distance_weight: float = None
    min_distance_weight_pre: float = 0.0
    min_distance_weight_post: float = 0.0

    include_focus: bool = None
    include_focus_pre: bool = True
    include_focus_post: bool = True

    partial_notes: bool = True
    epsilon: float = 1.0e-4

    def __post_init__(self):
        #make sure these exist
        self.len_context_pre_auto = False
        self.len_context_post_auto = False

        #split into pre an post contexts       
        if self.len_context != None:
            self.len_context_pre = self.len_context
            self.len_context_post = self.len_context
            self.len_context = None
       
        if self.len_context_pre == 'auto':
            self.len_context_pre_auto = True
            self.len_context_pre = None
       
        if self.len_context_post == 'auto':
            self.len_context_post_auto = True
            self.len_context_post = None
       
        if self.use_metric_weights != None:
            self.use_metric_weights_pre = self.use_distance_weights
            self.use_metric_weights_post = self.use_distance_weights
            self.use_metric_weights = None
       
        if self.use_distance_weights != None:
            self.use_distance_weights_pre = self.use_distance_weights
            self.use_distance_weights_post = self.use_distance_weights
            self.use_distance_weights = None
       
        if self.min_distance_weight != None:
            self.min_distance_weight_pre = self.min_distance_weight
            self.min_distance_weight_post = self.min_distance_weight
            self.min_distance_weight = None
       
        if self.include_focus != None:
            self.include_focus_pre = self.include_focus
            self.include_focus_post = self.include_focus
            self.include_focus = None

class ComputePitchContext(ABC):
    def __init__(self, wpc: PitchContext):
        super().__init__()
        self.wpc = wpc
        self.song = wpc.song
        self.params = wpc.params        

    def computePreContext(self, focus_ix):
        if self.params.len_context_pre_auto:
            return self.computePreContextAuto(focus_ix)
        else:
            return self.computePreContextFixed(focus_ix)

    def computePostContext(self, focus_ix):
        if self.params.len_context_post_auto:
            return self.computePostContextAuto(focus_ix)
        else:
            return self.computePostContextFixed(focus_ix)

    @abstractmethod
    def computePreContextFixed(self, focus_ix):
        pass

    @abstractmethod
    def computePostContextFixed(self, focus_ix):
        pass

    @abstractmethod
    def computePreContextAuto(self, focus_ix):
        pass

    @abstractmethod
    def computePostContextAuto(self, focus_ix):
        pass
    
    @abstractmethod
    def computeDistanceWeightsPre(self, context_pre_ixs, focus_ix):
        pass

    @abstractmethod
    def computeDistanceWeightsPost(self, context_post_ixs, focus_ix):
        pass

    @abstractmethod
    def computeContextLengthPre(self, context_ixs):
        pass

    @abstractmethod
    def computeContextLengthPost(self, context_ixs):
        pass

class ComputePitchContextSoretime(ComputePitchContext):
    pass

class ComputePitchContextNotes(ComputePitchContext):
    pass

class ComputePitchContextBeats(ComputePitchContext):
    def __init__(self, wpc: PitchContext):
        super().__init__(wpc)
        #compute some extra features. LENGTH: wpc.ixs
        self.songlength_beat = float(sum([Fraction(length) for length in self.song.mtcsong['features']['beatfraction']])) #length of the song in beats
        self.beatinsong = np.array([self.song.mtcsong['features']['beatinsong_float'][ix] for ix in wpc.ixs])
        self.beatinsong_next = np.append(self.beatinsong[1:],self.songlength_beat+self.beatinsong[0]) #first beatinsong might be negative (upbeat)

    def computePreContextFixed(self, focus_ix):
        beatoffset = self.beatinsong - self.beatinsong[focus_ix]
        len_context_pre = self.params.len_context_pre
        epsilon = self.params.epsilon
        #N.B. for some reason, np.where returns a tuple e.g: (array([], dtype=int64),)
        if self.params.include_focus_pre:
            context_pre_ixs = np.where(np.logical_and(beatoffset>=-(len_context_pre + epsilon), beatoffset<=0))[0]
        else:
            context_pre_ixs = np.where(np.logical_and(beatoffset>=-(len_context_pre + epsilon), beatoffset<0))[0]
        
        if self.params.partial_notes:
            if focus_ix>0: #skip first, has no context_pre
                #check wether context start at beginning of a note. If not, add previous note
                #print(context_pre[0][0],beatoffset[context_pre[0][0]],len_context)
                if context_pre_ixs.shape[0]>0:
                    if np.abs( beatoffset[context_pre_ixs[0]] + len_context_pre ) > epsilon:
                        if context_pre_ixs[0]-1 >= 0:
                            context_pre_ixs = np.insert(context_pre_ixs, 0, context_pre_ixs[0]-1)
                else:
                    context_pre_ixs = np.insert(context_pre_ixs, 0, focus_ix-1) #if context was empty, add previous note anyway        
        
        return context_pre_ixs

    def computePostContextFixed(self, focus_ix):
        beatoffset = self.beatinsong - self.beatinsong[focus_ix]
        slicelength = self.beatinsong_next[focus_ix] - self.beatinsong[focus_ix]
        beatoffset_next = beatoffset - slicelength #set onset of next note to 0.0
        len_context_post = self.params.len_context_post
        epsilon = self.params.epsilon
        #N.B. for some reason, np.where returns a tuple e.g: (array([], dtype=int64),)
        if self.params.include_focus_post:
            context_post_ixs = np.where(np.logical_and(beatoffset>=0, beatoffset_next<(len_context_post - epsilon)))[0]   
        else: # ['both', 'post']
            #start context at END of note
            #do not include the note that starts AT the end of the context
            context_post_ixs = np.where(np.logical_and(beatoffset_next>=0, beatoffset_next<(len_context_post - epsilon)))[0]
        return context_post_ixs

    def computePreContextAuto(self, focus_ix):
        context_pre_ixs = []
        if self.params.include_focus_pre:
            context_pre_ixs.append(focus_ix)
        ixadd = focus_ix - 1
        while True:
            if ixadd < 0:
                break
            context_pre_ixs.append(ixadd)
            if np.sum(self.wpc.weightedpitch[ixadd]) >= 1.0-self.params.epsilon or np.sum(self.wpc.weightedpitch[ixadd]) > np.sum(self.wpc.weightedpitch[focus_ix]):
                break
            ixadd = ixadd - 1
        context_pre_ixs.reverse()
        context_pre_ixs = np.array(context_pre_ixs, dtype=int)
        return context_pre_ixs

    def computePostContextAuto(self, focus_ix):
        context_post_ixs = []
        if self.params.include_focus_post:
            context_post_ixs.append(focus_ix)
        ixadd = focus_ix + 1
        while True:
            if ixadd >= len(self.wpc.ixs):
                break
            context_post_ixs.append(ixadd)
            if np.sum(self.wpc.weightedpitch[ixadd]) >= 1.0-self.params.epsilon or np.sum(self.wpc.weightedpitch[ixadd]) > np.sum(self.wpc.weightedpitch[focus_ix]):
                break
            ixadd = ixadd + 1
        context_post_ixs = np.array(context_post_ixs, dtype=int)
        return context_post_ixs

    def computeDistanceWeightsPre(self, context_pre_ixs, focus_ix):
        beatoffset_previous = self.beatinsong - self.beatinsong[focus_ix]
        mindist = self.params.min_distance_weight_pre
        len_context_pre = self.computeContextLengthPre(context_pre_ixs)
        distance_weights_pre  = beatoffset_previous[context_pre_ixs] * (1.0-mindist)/len_context_pre + 1.0
        #set negative weights to zero:
        distance_weights_pre[distance_weights_pre<0.0] = 0.0
        return distance_weights_pre

    def computeDistanceWeightsPost(self, context_post_ixs, focus_ix):
        beatoffset = self.beatinsong - self.beatinsong[focus_ix]
        slicelength = self.beatinsong_next[focus_ix] - self.beatinsong[focus_ix]
        beatoffset_next = beatoffset - slicelength #set onset of next note to 0.0
        mindist = self.params.min_distance_weight_post
        len_context_post = self.computeContextLengthPost(context_post_ixs)
        distance_weights_post = beatoffset_next[context_post_ixs] * -(1.0-mindist)/len_context_post + 1.0
        #set negative weights to zero:
        distance_weights_post[distance_weights_post<0.0] = 0.0
        #set max weight to one (if focus note in post context, weight of focus note > 1.0)
        distance_weights_post[distance_weights_post>1.0] = 1.0
        return distance_weights_post

    def computeContextLength(self, context_ixs):
        if len(context_ixs) > 0:
            len_context = self.beatinsong_next[context_ixs[-1]] - self.beatinsong[context_ixs[0]]
        else:
            len_context = 0.0
        return len_context

    def computeContextLengthPre(self, context_ixs):
        if self.params.len_context_pre_auto:
            return self.computeContextLength(context_ixs)
        else:
            return self.params.len_context_pre

    def computeContextLengthPost(self, context_ixs):
        if self.params.len_context_post_auto:
            return self.computeContextLength(context_ixs)
        else:
            return self.params.len_context_post


#weighted pitch contect
class PitchContext:
    """Class for computing a weighted pitch context vector and keeping track of the parameters.
    
    Parameters
    ----------
    song : Song
        instance of the class Song, representing the song as MTCFeatures, with some
        additional features. By instanciating an object of this class, the following
        parameters have to be provided. Parameters with a default value can be ommitted.
    removeRepeats : boolean, default=True
        If True, skip all notes with repeated pitches.
    syncopes : boolean, default=False
        If True, take the highest metric weight DURING the span of the note.
        If False, take the metric weight at the onset time of the note.
    metric_weights : one of 'beatstrength', 'ima', 'imaspect', default='beatstrength'
        `beatstrength` : use beatstrength as computed by music21
        `ima` : use inner metric analysis weights (not yet implemented)
        `imaspect` : use inner metric analysis spectral weights (not yet implemented)
    accumulateWeight : boolean, default=False
        If true, represent the metric weight of a note by the sum of all metric weights
        in the beatstrength grid in the span of the note.
    context_type : one of 'scoretime', 'beats', 'notes', default='beats'
        scoretime: len_context is a float in units of quarterLength (length of one quarter note)
        beats: len_context is a float in units of beat (length of beat as computed by music21)
        notes: len_context is an integer, number of notes
    len_context: float or (float, float), int or (int, int), 'auto' or (...,'auto') or ('auto',...)
        length of the context. Value depends on context_type. See context_type.
        If a tuple is given, first in the tuple refers to preceding context and
        second in the tuple to the following context.
        'auto': the length of the context is determined automatically.
    use_metric_weights : boolean or (boolean, boolean), default=True
        Whether to weight the pitches in the conext by their metric weight.
        If a tuple is given, first in the tuple refers to preceding context and
        second in the tuple to the following context.
    use_distance_weights : boolean or (boolean, boolean), default=True
        If True, weight pithces in the context by their distance to the focus note.
        The weight is a linear function of the score distance to the focus note.
        The weight at the onset of the focus note is 1.0.
        The weight at the end of the context is set by `min_distance_weight`.
        If a tuple is given, first in the tuple refers to preceding context and
        second in the tuple to the following context.
    min_distance_weight : float of (float, float), default=0.0
        Distance weight at the border of the context.
        If a tuple is given, first in the tuple refers to preceding context and
        second in the tuple to the following context.
    include_focus : boolean or (boolean, boolean), default=True
        Whether to include the focus note in the context.
        If a tuple is given, first in the tuple refers to preceding context and
        second in the tuple to the following context.
    partial_notes : boolean, default=True
        If True, extend the PRE conext to the START of the first note within the context.
        This has consequences if the pre context starts IN a note.

    Attributes
    ----------
    params : dict
        Parameters for computing the WPC.
    ixs : list
        List of the indices of the notes that can be part of a context. This list is the single
        point of conversion to actual note indices within the melody.
    weightedpitch : numpy array
        Dimension is (length of ixs, 40). The first dimension corresponds to the note indices in
        `ixs`. The second dimension contains the metric weight of the corresponding note for the
        appropriate pitch in base 40 encoding.
    pitchcontext : numpy array
        Dimension is (length of ixs, 80). The first dimension corresponds to the note indices in
        `ixs`. The second dimension correpsonds to 40 pitches in the preceding context [:40] and
        40 pitches in the following context [40:]. Pitches are in base40 encoding.
    contexts_pre : list of lists
        Length is length of isx. For each note in `ixs`, `context_pre` contains a list of the
        indices of pitches in ixs that are part of the preceding context of the note.
    contexts_post : list of lists
        Length is length of isx. For each note in `ixs`, `context_post` contains a list of the
        indices of pitches in ixs that are part of the following context of the note.
    """

    def __init__(self, song, **inparams):
        self.song = song
        #contains params for actual contents of weightedpitch vector and weightedpitch context vector
        self.params = PCParameters(**inparams)
        #store for quick use
        self.epsilon = self.params.epsilon
        #do the computations
        #First compute a weight for each note
        self.weightedpitch, self.ixs = self.computeWeightedPitch()
        #create the proper computePitchContext object
        self.cpc = self.createComputePitchContext()
        #compute the pitch context
        self.pitchcontext, self.contexts_pre, self.contexts_post = self.computePitchContext()

    def createComputePitchContext(self):
        if self.params.context_type == 'beats':
            return ComputePitchContextBeats(self)
        elif self.params.context_type == 'notes':
            return ComputePitchContextNotes(self)
        elif self.params.context_type == 'scoretime':
            return ComputePitchContextSoretime(self)
        else:
            print("Error: Unsupported context type: " + self.params.context_type)

    def computeWeightedPitch(self):
        """Computes for every note a pitchvector (base40) with the (metric) weight of the note in the corresponding pitch bin.

        Returns
        -------
        numpy array
            Dimension is (length of ixs, 40). The first dimension corresponds to the note indices in
            `ixs`. The second dimension contains the metric weight of the corresponding note for the
            appropriate pitch in base 40 encoding.
        """
        #put param values in local variables for readibility
        removeRepeats = self.params.remove_repeats
        syncopes = self.params.syncopes
        metric_weights = self.params.metric_weights
        accumulateWeight = self.params.accumulate_weight

        songinstance = self.song
        song = self.song.mtcsong

        if metric_weights in ['ima', 'imaspect']:
            raise Exception(f'{metric_weights} not yet implemented.')
        
        onsettick = song['features']['onsettick']
        pitch40 = song['features']['pitch40']
        beatstrengthgrid = np.array(song['features']['beatstrengthgrid'])
        beatstrength = song['features']['beatstrength']

        song_length = songinstance.getSongLength()
        ixs = []
        if removeRepeats:
            p_prev=-1
            for ix, p40 in enumerate(song['features']['pitch40']):
                if p40 != p_prev:
                    ixs.append(ix)
                p_prev = p40
        else:
            ixs = list(range(song_length))

        weights = [0]*len(ixs)

        if accumulateWeight:
            if syncopes:
                syncopes=False
                print("Warning: setting accumulateWeight implies syncopes=False.")
            max_onset = len(beatstrengthgrid)-1
            #for each note make span of onsets:
            start_onsets = [onsettick[ix] for ix in ixs]
            stop_onsets = [onsettick[ix] for ix in ixs[1:]]+[max_onset] #add end of last note
            for ix, span in enumerate(zip(start_onsets, stop_onsets)):
                weights[ix] = sum(beatstrengthgrid[span[0]:span[1]])
        else:
            weights = [beatstrength[ix] for ix in ixs]
        
        if syncopes:
            for ix, span in enumerate(zip(ixs, ixs[1:])):
                maxbeatstrength = np.max(beatstrengthgrid[onsettick[span[0]]:onsettick[span[1]]])
                weights[ix] = maxbeatstrength

        song['features']['weights'] = [0.0] * len(pitch40)
        for ix, songix in enumerate(ixs):
            song['features']['weights'][songix] = weights[ix]

        weightedpitch = np.zeros( (len(ixs), 40) )
        for ix, songix in enumerate(ixs):
            p = pitch40[songix]
            w = weights[ix]
            weightedpitch[ix, (p-1)%40] = w
        return weightedpitch, ixs

    def getBeatinsongFloat(self):
        """Convert `beatinsong` from Fraction to float

        Returns
        -------
        numpy vector
            Length is length of `ixs`. numpy vector with beatinsong as float
        """
        song = self.song.mtcsong
        beatinsong_float = np.zeros( len(self.ixs) )
        for ix, song_ix in enumerate(self.ixs):
            beatinsong_float[ix] = float(Fraction(song['features']['beatinsong'][song_ix]))
        return beatinsong_float

    def computePitchContext(self):   
        """Compute for each note a pitchcontext vector

        Returns
        -------
        numpy array
            Dimension is (length of `ixs`, 80). The first dimension corresponds to the note indices in
            `ixs`. The second dimension correpsonds to 40 pitches in the preceding context [:40] and
            40 pitches in the following context [40:]. Pitches are in base40 encoding.
        """
        #put param values in local variables for readibility
        use_metric_weights_pre = self.params.use_metric_weights_pre
        use_metric_weights_post = self.params.use_metric_weights_post
        use_distance_weights_pre = self.params.use_distance_weights_pre
        use_distance_weights_post = self.params.use_distance_weights_post
        
        #array to store the result
        pitchcontext = np.zeros( (len(self.ixs), 40 * 2) )

        #Lists for the indices of the contexts for each note
        contexts_pre = []
        contexts_post = []
        
        for ix, songix in enumerate(self.ixs):
            #get context for the note (list of note indices)
            context_pre_ixs = self.cpc.computePreContext(ix)
            context_post_ixs = self.cpc.computePostContext(ix)

            # print('context_pre', context_pre_ixs)
            # print('context_post', context_post_ixs)

            contexts_pre.append(context_pre_ixs)
            contexts_post.append(context_post_ixs)

            #compute distance-weights
            distance_weights_pre = np.ones(context_pre_ixs.shape)
            if use_distance_weights_pre:
                distance_weights_pre = self.cpc.computeDistanceWeightsPre(context_pre_ixs, ix)

            distance_weights_post = np.ones(context_post_ixs.shape)
            if use_distance_weights_post:
                distance_weights_post = self.cpc.computeDistanceWeightsPost(context_post_ixs, ix)

            #set metric weights to 1 if not use_metric_weights
            metric_weights_pre = self.weightedpitch[context_pre_ixs]
            if not use_metric_weights_pre:
                metric_weights_pre[metric_weights_pre>0] = 1.0

            metric_weights_post = self.weightedpitch[context_post_ixs]
            if not use_metric_weights_post:
                metric_weights_post[metric_weights_post>0] = 1.0

            # print("ix", ix, self.ixs[ix])
            # print("context_pre_ixs", context_pre_ixs)
            # print("distance_weights_pre", distance_weights_pre)
            # print("metric_weights_pre", metric_weights_pre)
            # print("self.weightedpitch[context_pre_ixs]", self.weightedpitch[context_pre_ixs])
            # print("length_context_post", length_context_post)
            # print("distance_weights_pre", distance_weights_pre)
            # print("distance_weights_post", distance_weights_post)
            #combine context into one vector

            pitchcontext_pre  = np.dot(distance_weights_pre, metric_weights_pre)
            pitchcontext_post = np.dot(distance_weights_post, metric_weights_post)
            
            #store result
            pitchcontext[ix,:40] = pitchcontext_pre
            pitchcontext[ix,40:] = pitchcontext_post

        return pitchcontext, contexts_pre, contexts_post

    def printReport(
        self,
        note_ix=None, #report single note. IX in original song, not in ixs
        **features, #any other values to report. key: name, value: array size len(ixs)
    ):
        """Print for each note the values of several features to stdout.

        For each note print
        - pitch and (metric) weight as computed by `self.computeWeightedPitch`
        - indices (in `self.ixs`) of notes in the preceding context
        - indices (in the MTC features) of notes in the preceding context
        - indices (in `self.ixs`) of notes in the following context
        - indices (in the MTC features) of notes in the following context
        - pitches and corresponding weights in the precedings context
        - pitches and corresponding wieghts in the following context
        - any other feature provided as keyword argument (see below)

        Parameters
        ----------
        note_ix : int, default None
            Only print the values the note at index `note_ix` in the original melody (not in `self.ixs`).
        **features  : keyword arguments
            any other feature to report. The keyword is the name of the feature, the value is a 1D array
            with the same lenght as `self.ixs`.
        """
        for ix in range(len(self.ixs)):
            if note_ix:
                if note_ix != self.ixs[ix]: continue
            pre_pitches = []
            post_pitches = []
            for p in range(40):
                if self.pitchcontext[ix,p] > 0.0:
                    pre_pitches.append((base40[p],self.pitchcontext[ix,p]))
            for p in range(40):
                if self.pitchcontext[ix,p+40] > 0.0:
                    post_pitches.append((base40[p], self.pitchcontext[ix,p+40]))
            pre_pitches = [str(p) for p in sorted(pre_pitches, key=lambda x: x[1], reverse=True)]
            post_pitches = [str(p) for p in sorted(post_pitches, key=lambda x: x[1], reverse=True)]
            print("note", self.ixs[ix], "ix:", ix)
            print("  pitch:", self.song.mtcsong['features']['pitch'][self.ixs[ix]], self.song.mtcsong['features']['weights'][self.ixs[ix]])
            print("  context_pre (ixs):  ", self.contexts_pre[ix])
            print("  context_pre (notes):", np.array(self.ixs)[self.contexts_pre[ix]])
            print("  context_post (ixs):  ", self.contexts_post[ix])
            print("  context_post (notes):", np.array(self.ixs)[self.contexts_post[ix]])
            print("  pre:", "\n       ".join(pre_pitches))
            print("  post:", "\n        ".join(post_pitches))
            for name in features.keys():
                print(f"  {name}: {features[name][ix]}")
            print()

