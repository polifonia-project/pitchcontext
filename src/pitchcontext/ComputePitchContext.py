from abc import ABC, abstractmethod
from fractions import Fraction

import numpy as np

from . import pitchcontext

class ComputePitchContext(ABC):
    def __init__(self, wpc: pitchcontext.PitchContext):
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
    def __init__(self, wpc: pitchcontext.PitchContext):
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
                    if len_context_pre>epsilon:
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
