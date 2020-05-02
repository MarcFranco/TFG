# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:44:07 2020

@author: Marc Franco Meca
"""


import numpy as np
import scipy.signal as sig
import librosa
import time
from TFG_Deconvolution import spectrumDBFS
from TFG_Deconvolution import plots
from scipy.io import wavfile
from masp import shoebox_room_sim as srs

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SETUP

# Room definition

width = 6
height = 3
depth = 10

room = np.array([depth, width, height])

# Desired RT per octave band, and time to truncate the responses
rt60 = np.array([1., 0.8, 0.7, 0.6, 0.5, 0.4])
nBands = len(rt60)

# Generate octave bands
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 125
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb-1]

# Absorption for approximately achieving the RT60 above - row per band
abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]

# Critical distance for the room
_, d_critical, _ = srs.room_stats(room, abs_wall)

# Receiver position
rec = np.array([ [9.0, 3.0, 1.5]])
nRec = rec.shape[0]

# Source positions
src = np.array([ [1.0, 2.0, 1.5],[4.0, 3.0, 1.5],[7.0, 4.0, 1.5] ])
nSrc = src.shape[0]

# Mic orientations and directivities
mic_specs = np.array([[1, 0, 0, 1]]) # Omnidirectional Microphone


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RUN SIMULATOR

# Echogram
tic = time.time()

maxlim = 1.5 # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
limits = np.minimum(rt60, maxlim)

# Compute echograms
# abs_echograms, rec_echograms, echograms = srs.compute_echograms_mic(room, src, rec, abs_wall, limits, mic_specs);
abs_echograms = srs.compute_echograms_mic(room, src, rec, abs_wall, limits, mic_specs);

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RENDERING

# In this case all the information (receiver directivity especially) is already
# encoded in the echograms, hence they are rendered directly to discrete RIRs
fs = 48000
mic_rirs = srs.render_rirs_mic(abs_echograms, band_centerfreqs, fs)

toc = time.time()
print('Elapsed time is ' + str(toc-tic) + 'seconds.')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# GENERATE SOUND SCENES
# Each source is convolved with the respective mic IR, and summed with
# the rest of the sources to create the microphone mixed signals

sourcepath = 'C:\TFG\Codi\masp-master\data/milk_cow_blues_4src.wav'
src_sigs = librosa.core.load(sourcepath, sr=None, mono=False)[0].T[:,:nSrc]

mic_sigs = srs.apply_source_signals_mic(mic_rirs, src_sigs)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Validate Method using Farina Database

fs, sinesweep = wavfile.read('C:\TFG\Codi\masp-master\sounds\Sweep.wav')

fs, inversefilter = wavfile.read('C:\TFG\Codi\masp-master\sounds\InvSweep.wav')
delta = sig.fftconvolve(sinesweep, inversefilter)
delta = delta/(np.abs(max(delta))) # normalization

fs, deconvolution = wavfile.read('C:\TFG\Codi\masp-master\sounds\Sweep(x)Invsweep.wav')
mseDeconvolution = np.mean((deconvolution - delta)**2)

freqSine, sinesweepdB = spectrumDBFS(sinesweep, fs)
plots(sinesweep,None,'Logarithmic SineSweep x(t)',fs,freqSine)

freqInv, inversefilterdB = spectrumDBFS(inversefilter, fs)
plots(inversefilter,None,'Inverse filter f(t)',fs,freqInv)

freqDelta, deltadB = spectrumDBFS(delta, fs)
plots(delta,None,'Delta d(t) = x(t) * f(t)',fs,freqDelta)

freqDec, deconvolutiondB = spectrumDBFS(deconvolution, fs)
plots(deconvolution,None,'Delta d(t) = x(t) * f(t)',fs,freqDec)

impulseresponse = mic_rirs[:,0,0] 
freqIR, impulseresponsedB = spectrumDBFS(impulseresponse, fs)
plots(impulseresponse, None, 'Impulse response h(t)',fs,freqIR)

measured = sig.fftconvolve(sinesweep,impulseresponse)
freqMeasured, measureddB = spectrumDBFS(measured, fs)
plots(measured, None,'Measured  y(t) = x(t)*h(t)',fs,freqMeasured)

estimationIR = sig.fftconvolve(measured, inversefilter)
estimationIR = estimationIR/(np.abs(max(estimationIR))) # normalization
estimationIR = estimationIR[inversefilter.size:inversefilter.size+impulseresponse.size] # adjust length because of FFT
freqEstimated, estimationIRdB = spectrumDBFS(estimationIR, fs)
plots(estimationIR, None,'Estimated IR  h(t) = y(t)*x_inv(t)',fs,freqEstimated)

mseFarina = np.mean((impulseresponse - estimationIR)**2)