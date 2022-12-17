from fractions import Fraction
import numpy as np

from .song import Song
from .base40 import base40

#from datetime import datetime
#print(__file__, datetime.now().strftime("%H:%M:%S"))

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
    context_type : one of 'scoretime', 'beats', 'notes', or 'auto', default='beats'
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
    normalize : boolean, default=False
        If True, normalize the weighted pitch context vector such that the values add up to 1.0.
        epsilon : float, default=1e-4
        Used for floating point comparisons.

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
        self.params = self.createDefaultParams()
        self.setparams(inparams)
        #store for quick use
        self.epsilon = self.params['epsilon']
        self.weightedpitch, self.ixs = self.computeWeightedPitch()
        #compute some extra features. LENGTH: self.ixs
        self.songlength_beat = float(sum([Fraction(length) for length in self.song.mtcsong['features']['beatfraction']])) #length of the song in beats
        self.beatinsong = np.array([self.song.mtcsong['features']['beatinsong_float'][ix] for ix in self.ixs])
        self.beatinsong_next = np.append(self.beatinsong[1:],self.songlength_beat+self.beatinsong[0]) #first beatinsong might be negative (upbeat)
        self.songlength_scoretime = float(sum([Fraction(length) for length in self.song.mtcsong['features']['duration_frac']])) #length of the song in beats        
        self.scoretimeinsong_next = np.cumsum(self.song.mtcsong['features']['duration'])
        self.scoretimeinsong = np.append([0], self.scoretimeinsong_next[:-1])
        #compute the pitch context
        self.pitchcontext, self.contexts_pre, self.contexts_post = self.computePitchContext()

    def createDefaultParams(self):
        """Return a dictionary with default parameters.
        
        Returns
        ------
        dictionary
            A Dictionary with all parameters and default values:
            ```{
                'removeRepeats' : True,
                'syncopes' : False,
                'metric_weights' : 'beatstrength',
                'accumulateWeight' : False,
                'context_type' : 'beats',
                'len_context' : None,
                'use_metric_weights' : True,
                'use_distance_weights' : True,
                'min_distance_weight' : 0.0,
                'include_focus' : True,
                'partial_notes' : True,
                'normalize' : False,
                'epsilon' : 1e-4,
            }```
        """

        params = {
            'removeRepeats' : True,
            'syncopes' : False,
            'metric_weights' : 'beatstrength',
            'accumulateWeight' : False,
            'context_type' : 'beats',
            'len_context' : None,
            'use_metric_weights' : True,
            'use_distance_weights' : True,
            'min_distance_weight' : 0.0,
            'include_focus' : True,
            'partial_notes' : True,
            'normalize' : False,
            'epsilon' : 1e-4,
        }
        return params
    
    def setparams(self, params):
        """Set parameters in `params`, and split pre and post context values.

        Parameters
        ----------
        params : dict
            key value pairs for the parameters to change
        """
        for key in params.keys():
            if key not in self.params:
                print(f"Warning: Unused parameter: {key}")
            else:
                self.params[key] = params[key]

        #helper functions for quicker access to params
        def _setp(key,value):
            self.params[key] = value
        def _getp(key):
            return self.params[key]

        #split params that possibly refer to preceding and foloing context

        #how to determine length of context
        if type(_getp('len_context')) == tuple or type(_getp('len_context')) == list:
            if _getp('len_context')[0] == 'auto':
                _setp('len_context_pre_auto', True)
                _setp('len_context_pre', None)
            else: # pre not auto
                _setp('len_context_pre', _getp('len_context')[0])
                _setp('len_context_pre_auto', False)
            if _getp('len_context')[1] == 'auto':
                _setp('len_context_post_auto', True)
                _setp('len_context_post', None)
            else: # post not auto
                _setp('len_context_post', _getp('len_context')[1])
                _setp('len_context_post_auto', False)
        elif _getp('len_context') == 'auto': #not a tuple, both auto
            _setp('len_context_pre_auto', True)
            _setp('len_context_post_auto', True)
            _setp('len_context_pre', None)
            _setp('len_context_post', None)
        else: #not a tuple, beat values
            _setp('len_context_pre', _getp('len_context'))
            _setp('len_context_post', _getp('len_context'))
            _setp('len_context_pre_auto', False)
            _setp('len_context_post_auto', False)
        _setp('len_context', None)

        #use metric weights
        if type(_getp('use_metric_weights')) == tuple or type(_getp('use_metric_weights')) == list:
            _setp('use_metric_weights_pre', _getp('use_metric_weights')[0])
            _setp('use_metric_weights_post', _getp('use_metric_weights')[1])
        else:
            _setp('use_metric_weights_pre', _getp('use_metric_weights'))
            _setp('use_metric_weights_post', _getp('use_metric_weights'))
        _setp('use_metric_weights', None)

        #use distance weights
        if type(_getp('use_distance_weights')) == tuple or type(_getp('use_distance_weights')) == list:
            _setp('use_distance_weights_pre', _getp('use_distance_weights')[0])
            _setp('use_distance_weights_post', _getp('use_distance_weights')[1])
        else:
            _setp('use_distance_weights_pre', _getp('use_distance_weights'))
            _setp('use_distance_weights_post', _getp('use_distance_weights'))
        _setp('use_distance_weights', None)

        #minimal distance weight (at context boundary)
        if type(_getp('min_distance_weight')) == tuple or type(_getp('min_distance_weight')) == list:
            _setp('min_distance_weight_pre', _getp('min_distance_weight')[0])
            _setp('min_distance_weight_post', _getp('min_distance_weight')[1])
        else:
            _setp('min_distance_weight_pre', _getp('min_distance_weight'))
            _setp('min_distance_weight_post', _getp('min_distance_weight'))
        _setp('min_distance_weight', None)

        if type(_getp('include_focus')) == tuple or type(_getp('include_focus')) == list:
            _setp('include_focus_pre', _getp('include_focus')[0])
            _setp('include_focus_post', _getp('include_focus')[1])
        else:
            _setp('include_focus_pre', _getp('include_focus'))
            _setp('include_focus_post', _getp('include_focus'))
        _setp('include_focus', None)

        print(self.params)

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
        removeRepeats = self.params['removeRepeats']
        syncopes = self.params['syncopes']
        metric_weights = self.params['metric_weights']
        accumulateWeight = self.params['accumulateWeight']

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

    def _computePreContextBeats(self, focus_ix):
        beatoffset = self.beatinsong - self.beatinsong[focus_ix]
        len_context_pre = self.params['len_context_pre']
        epsilon = self.params['epsilon']
        #N.B. for some reason, np.where returns a tuple e.g: (array([], dtype=int64),)
        if self.params['include_focus_pre']:
            context_pre_ixs = np.where(np.logical_and(beatoffset>=-(len_context_pre + epsilon), beatoffset<=0))[0]
        else:
            context_pre_ixs = np.where(np.logical_and(beatoffset>=-(len_context_pre + epsilon), beatoffset<0))[0]
        
        if self.params['partial_notes']:
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

    def _computePostContextBeats(self, focus_ix):
        beatoffset = self.beatinsong - self.beatinsong[focus_ix]
        slicelength = self.beatinsong_next[focus_ix] - self.beatinsong[focus_ix]
        beatoffset_next = beatoffset - slicelength #set onset of next note to 0.0
        len_context_post = self.params['len_context_post']
        epsilon = self.params['epsilon']
        #N.B. for some reason, np.where returns a tuple e.g: (array([], dtype=int64),)
        if self.params['include_focus_post']:
            context_post_ixs = np.where(np.logical_and(beatoffset>=0, beatoffset_next<(len_context_post - epsilon)))[0]   
        else: # ['both', 'post']
            #start context at END of note
            #do not include the note that starts AT the end of the context
            context_post_ixs = np.where(np.logical_and(beatoffset_next>=0, beatoffset_next<(len_context_post - epsilon)))[0]
        return context_post_ixs

    def _computePreContextBeatsAuto(self, focus_ix):
        context_pre_ixs = []
        if self.params['include_focus_pre']:
            context_pre_ixs.append(focus_ix)
        ixadd = focus_ix - 1
        while True:
            if ixadd < 0:
                break
            context_pre_ixs.append(ixadd)
            if np.sum(self.weightedpitch[ixadd]) >= 1.0-self.epsilon or np.sum(self.weightedpitch[ixadd]) > np.sum(self.weightedpitch[focus_ix]):
                break
            ixadd = ixadd - 1
        context_pre_ixs.reverse()
        context_pre_ixs = np.array(context_pre_ixs, dtype=int)
        return context_pre_ixs

    def _computePostContextBeatsAuto(self, focus_ix):
        context_post_ixs = []
        if self.params['include_focus_post']:
            context_post_ixs.append(focus_ix)
        ixadd = focus_ix + 1
        while True:
            if ixadd >= len(self.ixs):
                break
            context_post_ixs.append(ixadd)
            if np.sum(self.weightedpitch[ixadd]) >= 1.0-self.epsilon or np.sum(self.weightedpitch[ixadd]) > np.sum(self.weightedpitch[focus_ix]):
                break
            ixadd = ixadd + 1
        context_post_ixs = np.array(context_post_ixs, dtype=int)
        return context_post_ixs

    def _computePreContextScoretime(self, focus_ix):
        return []

    def _computePostContextScoretime(self, focus_ix):
        return []

    def _computePreContextScoretimeAuto(self, focus_ix):
        return []

    def _computePostContextScoretimeAuto(self, focus_ix):
        return []

    def _computePreContextNotes(self, focus_ix):
        return []

    def _computePostContextNotes(self, focus_ix):
        return []

    def _computePreContextNotesAuto(self, focus_ix):
        return []

    def _computePostContextNotesAuto(self, focus_ix):
        return []

    def _computePreContext(self, focus_ix): #selector
        ctype = self.params['context_type']
        auto = self.params['len_context_pre_auto']
        if ctype == 'scoretime':
            if auto:
                return self._computePreContextScoretimeAuto(focus_ix)
            else:
                return self._computePreContextScoretime(focus_ix)
        elif ctype == 'beats':
            if auto:
                return self._computePreContextBeatsAuto(focus_ix)
            else:
                return self._computePreContextBeats(focus_ix)
        elif ctype == 'notes':
            if auto:
                return self._computePreContextNotesAuto(focus_ix)
            else:
                return self._computePreContextNotes(focus_ix)
        else:
            print("Warning: unknown context type for preceding context: "+ctype)
            return []

    def _computePostContext(self, focus_ix): #selector
        ctype = self.params['context_type']
        auto = self.params['len_context_post_auto']
        if ctype == 'scoretime':
            if auto:
                return self._computePostContextScoretimeAuto(focus_ix)
            else:
                return self._computePostContextScoretime(focus_ix)
        elif ctype == 'beats':
            if auto:
                return self._computePostContextBeatsAuto(focus_ix)
            else:
                return self._computePostContextBeats(focus_ix)
        elif ctype == 'notes':
            if auto:
                return self._computePostContextNotesAuto(focus_ix)
            else:
                return self._computePostContextNotes(focus_ix)
        else:
            print("Warning: unknown context type for following context: "+ctype)
            return []

    def _computeDistanceWeightsPreBeats(self, context_pre_ixs, focus_ix):
        beatoffset_previous = self.beatinsong - self.beatinsong[focus_ix]
        mindist = self.params['min_distance_weight_pre']
        len_context_pre = self._computeContextLength(context_pre_ixs)
        distance_weights_pre  = beatoffset_previous[context_pre_ixs] * (1.0-mindist)/len_context_pre + 1.0
        #set negative weights to zero:
        distance_weights_pre[distance_weights_pre<0.0] = 0.0
        return distance_weights_pre

    def _computeDistanceWeightsPostBeats(self, context_post_ixs, focus_ix):
        beatoffset = self.beatinsong - self.beatinsong[focus_ix]
        slicelength = self.beatinsong_next[focus_ix] - self.beatinsong[focus_ix]
        beatoffset_next = beatoffset - slicelength #set onset of next note to 0.0
        mindist = self.params['min_distance_weight_post']
        len_context_post = self._computeContextLength(context_post_ixs)
        distance_weights_post = beatoffset_next[context_post_ixs] * -(1.0-mindist)/len_context_post + 1.0
        #set negative weights to zero:
        distance_weights_post[distance_weights_post<0.0] = 0.0
        #set max weight to one (if focus note in post context, weight of focus note > 1.0)
        distance_weights_post[distance_weights_post>1.0] = 1.0
        return distance_weights_post

    def _computeDistanceWeightsPreScoretime(self, context_pre_ixs, focus_ix):
        pass

    def _computeDistanceWeightsPostScoretime(self, context_post_ixs, focus_ix):
        pass

    def _computeDistanceWeightsPreNotes(self, context_pre_ixs, focus_ix):
        pass

    def _computeDistanceWeightsPostNotes(self, context_post_ixs, focus_ix):
        pass

    def _computeDistanceWeightsPreAuto(self, context_pre_ixs, focus_ix):
        return self._computeDistanceWeightsPreBeats(context_pre_ixs, focus_ix)

    def _computeDistanceWeightsPostAuto(self, context_post_ixs, focus_ix):
        return self._computeDistanceWeightsPostBeats(context_post_ixs, focus_ix)

    def _computeDistanceWeightsPre(self, context_pre_ixs, focus_ix):
        if self.params['use_distance_weights_pre']:
            ctype = self.params['context_type']
            if ctype == 'scoretime':
                return self._computeDistanceWeightsPreScoretime(context_pre_ixs, focus_ix)
            elif ctype == 'beats':
                return self._computeDistanceWeightsPreBeats(context_pre_ixs, focus_ix)
            elif ctype == 'notes':
                return self._computeDistanceWeightsPreNotes(context_pre_ixs, focus_ix)
            elif ctype == 'auto':
                return self._computeDistanceWeightsPreAuto(context_pre_ixs, focus_ix)
            else:
                print("Warning: unknown context type for preceding context: " + str(ctype))
                return np.ones((context_pre_ixs.shape))
        else:
            return np.ones((context_pre_ixs.shape))

    def _computeDistanceWeightsPost(self, context_post_ixs, focus_ix):
        if self.params['use_distance_weights_post']:
            ctype = self.params['context_type']
            if ctype == 'scoretime':
                return self._computeDistanceWeightsPostScoretime(context_post_ixs, focus_ix)
            elif ctype == 'beats':
                return self._computeDistanceWeightsPostBeats(context_post_ixs, focus_ix)
            elif ctype == 'notes':
                return self._computeDistanceWeightsPostNotes(context_post_ixs, focus_ix)
            elif ctype == 'auto':
                return self._computeDistanceWeightsPostAuto(context_post_ixs, focus_ix)
            else:
                print("Warning: unknown context type for preceding context: " + str(ctype))
                return np.ones((context_post_ixs.shape))
        else:
            return np.ones((context_post_ixs.shape))

    def _computeContextLengthScoretime(self, context_post_ixs):
        return 0
    
    def _computeContextLengthBeats(self, context_ixs):
        if len(context_ixs) > 0:
            len_context = self.beatinsong_next[context_ixs[-1]] - self.beatinsong[context_ixs[0]]
        else:
            len_context = 0.0
        return len_context

    def _computeContextLengthNotes(self, context_ixs):
        return 0
    
    def _computeContextLengthAuto(self, context_ixs):
        return 0

    def _computeContextLength(self, context_ixs):
        ctype = self.params['context_type']
        if ctype == 'scoretime':
            return self._computeContextLengthScoretime(context_ixs)
        elif ctype == 'beats':
            return self._computeContextLengthBeats(context_ixs)
        elif ctype == 'notes':
            return self._computeContextLengthNotes(context_ixs)
        elif ctype == 'auto':
            return self._computeContextLengthAuto(context_ixs)
        else:
            print("Warning: unknown context type: " + str(ctype))
            return 0.0

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
        len_context_pre = self.params['len_context_pre']
        len_context_post = self.params['len_context_post']
        len_context_pre_auto = self.params['len_context_pre_auto']
        len_context_post_auto = self.params['len_context_post_auto']
        use_metric_weights_pre = self.params['use_metric_weights_pre']
        use_metric_weights_post = self.params['use_metric_weights_post']
        use_distance_weights_pre = self.params['use_distance_weights_pre']
        use_distance_weights_post = self.params['use_distance_weights_post']
        min_distance_weight_pre = self.params['min_distance_weight_pre']
        min_distance_weight_post = self.params['min_distance_weight_post']
        include_focus = self.params['include_focus']
        partial_notes = self.params['partial_notes']
        normalize = self.params['normalize']
        epsilon = self.params['epsilon']
        
        song = self.song.mtcsong
        
        #array to store the result
        pitchcontext = np.zeros( (len(self.ixs), 40 * 2) )

        #Lists for the indices of the contexts for each note
        contexts_pre = []
        contexts_post = []
        
        for ix, songix in enumerate(self.ixs):
            #get context for the note (list of note indices)
            context_pre_ixs = self._computePreContext(ix)
            context_post_ixs = self._computePostContext(ix)

            # print('context_pre', context_pre_ixs)
            # print('context_post', context_post_ixs)

            contexts_pre.append(context_pre_ixs)
            contexts_post.append(context_post_ixs)

            #compute distance-weights
            distance_weights_pre = self._computeDistanceWeightsPre(context_pre_ixs, ix)
            distance_weights_post = self._computeDistanceWeightsPost(context_post_ixs, ix)

            metric_weights_pre = self.weightedpitch[context_pre_ixs]
            if not use_metric_weights_pre:
                metric_weights_pre[metric_weights_pre>0] = 1.0

            metric_weights_post = self.weightedpitch[context_post_ixs]
            if not use_metric_weights_post:
                metric_weights_post[metric_weights_post>0] = 1.0

            # print("ix", ix, ixs[ix])
            # print("length_context_pre", length_context_pre)
            # print("length_context_post", length_context_post)
            # print("distance_weights_pre", distance_weights_pre)
            # print("distance_weights_post", distance_weights_post)
            #combine context into one vector

            pitchcontext_pre  = np.dot(distance_weights_pre, metric_weights_pre)
            pitchcontext_post = np.dot(distance_weights_post, metric_weights_post)
            #normalize
            
            if normalize:
                pitchcontext_pre /= np.sum(np.abs(pitchcontext_pre),axis=0)
                pitchcontext_post /= np.sum(np.abs(pitchcontext_post),axis=0)
            
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

