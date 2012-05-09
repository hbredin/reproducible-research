#!/usr/bin/env python
# encoding: utf-8

# Copyright 2012 Herve BREDIN (bredin@limsi.fr)

# This file is part of PyAnnote.
# 
#     PyAnnote is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     PyAnnote is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with PyAnnote.  If not, see <http://www.gnu.org/licenses/>.


"""
   "Unsupervised Speaker Identification using Overlaid Texts in TV Broadcast"
           
           by Johann Poignant, Hervé Bredin, Viet Bac Le, 
           Laurent Besacier, Claude Barras and Georges Quénot.

    To reproduce the experiments described in this paper 
    and generate Tables 3, 4, 5 and 6 of "Results" section,
    simply run the following command line::
     
        >>> python run.py

"""

# =============================================================================
# == CHECK PYANNOTE VERSION ===================================================
# =============================================================================

DESIGNED_FOR_PYANNOTE_VERSION = "0.2.1"

import pyannote
try:
    assert(pyannote.__version__ >= DESIGNED_FOR_PYANNOTE_VERSION)
except Exception, e:
    raise ImportError("This script relies on PyAnnote %s "
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

# manual overlaid name detection
manual_overlaid_names = MDTMParser("data/manual_overlaid_names.mdtm", \
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

propagation_algorithms = {'SID' : SID, \
                          'M1': M1, 'M2': M2, 'M3': M3, \
                          'M3 + SID' : combo}

# used to keep track of error rates
eger = {}

# =============================================================================
# == TABLES 3 & 4 =============================================================
# =============================================================================

# initialize error rates
eger['Tables 3 & 4'] = {'All': {}, 'No anchor': {}}
for propagation in propagation_algorithms:
    eger['Tables 3 & 4']['All'][propagation] = EstimatedGlobalErrorRate()
    eger['Tables 3 & 4']['No anchor'][propagation] = EstimatedGlobalErrorRate() 

# initialize progress bar
pb = ProgressBar(maxval=len(videos)*len(propagation_algorithms), \
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
        # evaluate only on frames without anchors in groundtrut
        af = af(msi(anchors).timeline.gaps(), mode='loose')
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
pb = ProgressBar(maxval=len(videos), widgets=['Table 5: ', Bar()]).start()
                  
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
    # evaluate only on frames without anchors in groundtrut
    af = af(msi(anchors).timeline.gaps(), mode='loose')
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

# TODO
