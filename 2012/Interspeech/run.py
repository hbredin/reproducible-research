#!/usr/bin/env python
# encoding: utf-8

# Copyright 2012 Herve BREDIN (bredin@limsi.fr)

"""
   Unsupervised Speaker Identification using Overlaid Texts in TV Broadcast
           
              Johann Poignant, Hervé Bredin, Viet Bac Le, 
           Laurent Besacier, Claude Barras and Georges Quénot.

    This script relies on PyAnnote available at 
    http://packages.python.org/PyAnnote
    
    To reproduce the experiments described in this paper 
    and generate Tables 3, 4, 5 and 6 of "Results" section,
    simply run the following command line::
     
        >>> python run.py
    
"""

# =============================================================================
# == CHECK PYANNOTE VERSION ===================================================
# =============================================================================

DESIGNED_FOR_PYANNOTE_VERSION = "0.2.2"

# check PyAnnote is available
try:
    import pyannote
except Exception, e:
    raise ImportError("This script relies on PyAnnote %s available at "
                      "http://packages.python.org/PyAnnote" % \
                       DESIGNED_FOR_PYANNOTE_VERSION)

# check PyAnnote version
try:
    assert(pyannote.__version__ >= DESIGNED_FOR_PYANNOTE_VERSION)
except Exception, e:
    raise ImportError("This script requires PyAnnote %s "
                      "(you have: %s)." % (DESIGNED_FOR_PYANNOTE_VERSION, \
                                           pyannote.__version__))

# =============================================================================
# == IMPORTS ==================================================================
# =============================================================================

# .uem, .mdtm and .repere files parsers
from pyannote.parser import UEMParser, MDTMParser, REPEREParser

from pyannote.algorithm.tagging import HungarianTagger, \
                                       ArgMaxTagger, \
                                       ConservativeDirectTagger
from pyannote.base.matrix import Cooccurrence, CoTFIDF
from pyannote.base.annotation import Unknown

# evaluation metric 
from pyannote.metric.repere import EstimatedGlobalErrorRate

# used to display a progress bar
from progressbar import ProgressBar, Bar
# used to pretty-print Tables 3, 4, 5 and 6
from prettytable import PrettyTable

# =============================================================================
# == LOAD DATA ================================================================
# =============================================================================

# --------------------------------------------
# LOAD GROUNDTRUTH FOR TEST SET
# as described in Section "4.1 REPERE Corpus"
# --------------------------------------------

# list of test videos
f = open("data/videos.txt", "r")
videos = [line.strip() for line in f.readlines()]
f.close()

# standard condition
standard_condition = UEMParser("data/standard_condition.uem")

# annotated frames
annotated_frames = UEMParser("data/annotated_frames.uem")

# list of anchors
f = open("data/anchors.txt", "r")
anchors = [line.strip() for line in f.readlines()]
f.close()

# manual speaker identification
manual_speaker_identification = MDTMParser("data/manual_speaker.mdtm", \
                                           multitrack=True)

# --------------------------------------------------
# LOAD MONOMODAL COMPONENTS OUTPUT ON TEST SET
# as described in Section "2. Monomodal Components"
# --------------------------------------------------

# automatic speaker diarization
auto_speaker_diarization = MDTMParser("data/auto_speaker_diarization.mdtm", \
                                      multitrack=True)

# automatic speaker identification
auto_speaker_identification = \
                     REPEREParser("data/auto_speaker_identification.repere", \
                                  multitrack=True, confidence=False)

# overlaid name detection output
auto_overlaid_names = REPEREParser("data/auto_overlaid_names.repere", \
                                   multitrack=True, confidence=False)


# ----------------------------------------------
# INITIALIZE NAME PROPAGATION ALGORITHMS
# as described in Section "3. Name Propagation"
# ----------------------------------------------

# 'on' stands for overlaid name detection
# 'sd' stands for (unsupervised) speaker diarization
# 'sid' stands for (supervised) speaker identification

one_to_one = HungarianTagger(cost=Cooccurrence)
one_to_many = ArgMaxTagger(cost=CoTFIDF)
direct = ConservativeDirectTagger()

M1 = lambda on, sd, sid : one_to_one(on, sd)
M2 = lambda on, sd, sid : direct(on, one_to_one(on, sd))
M3 = lambda on, sd, sid : direct(on, one_to_many(on, sd))

SID = lambda on, sd, sid : sid
combo = lambda on, sd, sid : direct(on, one_to_many(on, sid))

propagation_algorithms = {'SID' : SID, 
                          'M1': M1, 'M2': M2, 'M3': M3, 
                          'M3 + SID' : combo}

# used to keep track of error rates
eger = {}

# =============================================================================
# == TITLE ====================================================================
# =============================================================================

print """
=============================================================================
  Unsupervised Speaker Identification using Overlaid Texts in TV Broadcast
-----------------------------------------------------------------------------
                Johann Poignant, Hervé Bredin, Viet Bac Le, 
            Laurent Besacier, Claude Barras and Georges Quénot.
=============================================================================
"""

# =============================================================================
# == TABLES 3 & 4 =============================================================
# =============================================================================

# initialize error rates
eger['Tables 3 & 4'] = {'All': {}, 'No anchor': {}}
for propagation in propagation_algorithms:
    eger['Tables 3 & 4']['All'][propagation] = EstimatedGlobalErrorRate()
    eger['Tables 3 & 4']['No anchor'][propagation] = EstimatedGlobalErrorRate() 

# initialize progress bar
pb = ProgressBar(term_width=69, \
                 maxval=len(videos)*len(propagation_algorithms), \
                 widgets=['Tables 3 & 4: ', Bar()]).start()

for v, video in enumerate(videos):
    
    # extract automatic speaker diarization for this video
    sd = auto_speaker_diarization.annotation(video, 'speaker')
    # anonymize labels (Unknown001, Unknown002, etc.)
    sd = sd.anonymize()
    
    # extract overlaid name detection for this video
    on = auto_overlaid_names.annotation(video, 'written')
    
    # extract automatic speaker identification for this video
    sid = auto_speaker_identification.annotation(video, 'speaker')
    
    # extract groundtruth for this video
    msi = manual_speaker_identification.annotation(video, 'speaker')

    for p, propagation in enumerate(propagation_algorithms):
        
        # propagate name
        s = propagation_algorithms[propagation](on, sd, sid)
        
        # evaluate on all frames
        af = annotated_frames.timeline(video)
        eger['Tables 3 & 4']['All'][propagation](msi, s, annotated=af) 
        # evaluate only on frames without anchors in groundtruth
        af = af(msi(anchors).timeline.gaps(af.extent()), mode='loose')
        eger['Tables 3 & 4']['No anchor'][propagation](msi, s, annotated=af)
        
        pb.update(v*len(propagation_algorithms)+(p+1))
    
pb.finish()

# pretty print Table 3
table3 = PrettyTable(["Speakers", "Propagation", "EGER", \
                      "Precision", "Recall", "F1-Measure"])
table3.float_format = "1.3"
for speakers in ['All', 'No anchor']:
    for propagation in ['M1', 'M2', 'M3']:
        error = eger['Tables 3 & 4'][speakers][propagation]
        table3.add_row([speakers, propagation, abs(error), \
                        error.precision, error.recall, error.f_measure])
print table3
print "Table 3: Name propagation performance, full cond."
print

# pretty print Table 4
table4 = PrettyTable(["Speakers", "Propagation", "EGER", \
                      "Precision", "Recall", "F1-Measure"])
table4.float_format = "1.3"
for speakers in ['All', 'No anchor']:
    for propagation in ['SID', 'M3', 'M3 + SID']:
        error = eger['Tables 3 & 4'][speakers][propagation]
        table4.add_row([speakers, propagation, abs(error), \
                        error.precision, error.recall, error.f_measure])
print table4
print "Table 4: Supervised (SID) vs. unsupervised (M3) speaker"
print "identification and their combination (M3+SID), full cond."
print

# =============================================================================
# == TABLE 5 ==================================================================
# =============================================================================

# initialize error rates
eger['Table 5'] = {'All': {}, 'No anchor': {}}
eger['Table 5']['All']['Standard'] = EstimatedGlobalErrorRate()
eger['Table 5']['All']['Full video'] = eger['Tables 3 & 4']['All']['M3']
eger['Table 5']['No anchor']['Standard'] = EstimatedGlobalErrorRate() 
eger['Table 5']['No anchor']['Full video'] = eger['Tables 3 & 4']['No anchor']['M3']

# initialize progress bar
pb = ProgressBar(term_width=68, maxval=len(videos), \
                 widgets=['Table 5: ', Bar()]).start()

for v, video in enumerate(videos):
    
    # extract standard condition
    sc = standard_condition.timeline(video)
    
    # extract automatic speaker diarization for this video
    sd = auto_speaker_diarization.annotation(video, 'speaker')
    # focus on standard condition
    sd = sd(sc, mode='loose')
    # anonymize labels (Unknown001, Unknown002, etc.)
    sd = sd.anonymize()
    
    # extract overlaid name detection for this video
    on = auto_overlaid_names.annotation(video, 'written')
    # focus on standard condition
    on = on(sc, mode='loose')
    
    # automatic speaker identification for this video
    sid = None # (not needed in this set of experiments)

    # extract groundtruth for this video
    msi = manual_speaker_identification.annotation(video, 'speaker')
    
    # propagate name
    s = M3(on, sd, sid)
        
    # evaluate on all frames
    af = annotated_frames.timeline(video)
    eger['Table 5']['All']['Standard'](msi, s, annotated=af) 
    # evaluate only on frames without anchors in groundtruth
    af = af(msi(anchors).timeline.gaps(af.extent()), mode='loose')
    eger['Table 5']['No anchor']['Standard'](msi, s, annotated=af)
        
    pb.update(v+1)
    
pb.finish()

# pretty print Table 5
table5 = PrettyTable(["Speakers", "Condition", "EGER", \
                      "Precision", "Recall", "F1-Measure"])
table5.float_format = "1.3"
for speakers in ['All', 'No anchor']:
    for condition in ['Standard', 'Full video']:
        error = eger['Table 5'][speakers][condition]
        table5.add_row([speakers, condition, abs(error), \
                        error.precision, error.recall, error.f_measure])
print table5
print "Table 5: Effect of condition on M3 performance."
print

# =============================================================================
# == TABLE 6 ==================================================================
# =============================================================================

# initialize error rates
eger['Table 6'] = {'Perfect': {}, 'Automatic': {}}
eger['Table 6']['Perfect']['Perfect'] = EstimatedGlobalErrorRate()
eger['Table 6']['Perfect']['M1'] = EstimatedGlobalErrorRate()
eger['Table 6']['Automatic']['M1'] = EstimatedGlobalErrorRate()
eger['Table 6']['Automatic']['M2'] = EstimatedGlobalErrorRate()
eger['Table 6']['Automatic']['M3'] = EstimatedGlobalErrorRate()

# initialize progress bar
pb = ProgressBar(term_width=69, maxval=len(videos), \
                 widgets=['Table 6: ', Bar()]).start()

for v, video in enumerate(videos):
    
    # extract standard condition
    sc = standard_condition.timeline(video)

    # extract overlaid name detection for this video
    on = auto_overlaid_names.annotation(video, 'written')
    # focus on standard condition
    on = on(sc, mode='loose')

    # automatic speaker identification for this video
    sid = None # (not needed in this set of experiments)

    # extract groundtruth for this video
    msi = manual_speaker_identification.annotation(video, 'speaker')
    # focus on standard condition
    msi = msi(sc, mode='loose')

    # evaluate only on frames without anchors in groundtruth
    af = annotated_frames.timeline(video)
    af = af(msi(anchors).timeline.gaps(af.extent()), mode='loose')

    # extract automatic speaker diarization for this video
    sd = auto_speaker_diarization.annotation(video, 'speaker')
    # focus on standard condition
    sd = sd(sc, mode='loose')
    # anonymize labels (Unknown001, Unknown002, etc.)
    sd = sd.anonymize()
    
    # --- perfect speaker diarization + perfect propagation
    # is equivalent to start from groundtruth and rename to Unknown 
    # any person whose name is not found anywhere by overlaid name detection
    translation = {label: Unknown() \
                   for label in set(msi.labels())-set(on.labels())}
    s = msi % translation
    eger['Table 6']['Perfect']['Perfect'](msi, s, annotated=af)
    
    # --- perfect speaker diarization + M1 propagation
    psd = msi.anonymize()
    s = M1(on, psd, sid)
    eger['Table 6']['Perfect']['M1'](msi, s, annotated=af)
    
    # --- automatic speaker diarization + M1/M2/M3 propagation
    for propagation in ['M1', 'M2', 'M3']:
        s = propagation_algorithms[propagation](on, sd, sid)        
        eger['Table 6']['Automatic'][propagation](msi, s, annotated=af)
    
    pb.update(v+1)
    
pb.finish()


# pretty print Table 6
table6 = PrettyTable(["SD", "Propagation", "EGER", \
                      "Precision", "Recall", "F1-Measure"])
table6.float_format = "1.3"
for propagation in ['Perfect', 'M1']:
    error = eger['Table 6']['Perfect'][propagation]
    table6.add_row(['Perfect', propagation, abs(error), \
                    error.precision, error.recall, error.f_measure])
for propagation in ['M1', 'M2', 'M3']:
    error = eger['Table 6']['Automatic'][propagation]
    table6.add_row(['Automatic', propagation, abs(error), \
                    error.precision, error.recall, error.f_measure])

print table6
print "Table 6: Effect of speaker diarization (SD) and name"
print "propagation errors (standard condition, without anchors)."
print
