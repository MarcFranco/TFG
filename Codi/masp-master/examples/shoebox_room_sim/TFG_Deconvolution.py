# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:00:49 2020

@author: Marc Franco Meca
"""

import numpy as np
from masp import shoebox_room_sim as srs
import time
import librosa
import matplotlib.pyplot as plt
import scipy.signal as sig
import soundfile as sf

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
# DECONVOLUTION

#  Logarithmic SineSweep x(t)
def log_sinesweep(finf,fsup,T,t,fs):
    w1 = 2*np.pi*finf
    w2 = 2*np.pi*fsup    
    K = T * w1 / np.log(w2/w1)
    L = T / np.log(w2/w1)
    sweep=np.sin(K*(np.exp(t/L)-1.0))  
    return sweep

# Inverse filter = Reverse Logarithmic SineSweep w/ Magnitude Modulation
def inverse_filter(finf,fsup,T,t,x):
    w1 = 2*np.pi*finf
    w2 = 2*np.pi*fsup  
    L = T / np.log(w2/w1)
    N = np.exp(t/L)   
    inv = x[::-1]/N
    return inv

def spectrumDBFS(x, fs, win=None):
    N = len(x)  
    if win is None:
        win = np.ones(x.shape)
    if len(x) != len(win):
        raise ValueError('Signal and window must be of the same length')
    x = x * win
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)
    s_mag = np.abs(sp) * 2 / np.sum(win)
    ref = s_mag.max()
    s_dbfs = 20 * np.log10(s_mag/ref)
    return freq, s_dbfs

def plots(x, xdB, title,fs,freq):
    if xdB is not None:
        fig, axs = plt.subplots(3, 1, figsize=(12,12))
        axs[0].set(xlabel='Frequency [samples]', ylabel='Amplitude')
        axs[0].set_title(title)
        axs[0].plot(x)
        axs[1].set(xlabel='Time [seconds]', ylabel='Frequency [Hz]')
        Pxx, freqs, bins, im = axs[1].specgram(x, 1024, fs, noverlap=900)
        axs[2].set(xlabel='Frequency [Hz]', ylabel='Amplitude [dBFS]')
        axs[2].semilogx(freq, xdB)
    else:
        fig, axs = plt.subplots(2, 1, figsize=(12,12))
        axs[0].set_title(title)
        axs[0].set(xlabel='Frequency [samples]', ylabel='Amplitude')
        axs[0].plot(x)
        axs[1].set(xlabel='Time [seconds]', ylabel='Frequency [Hz]')
        Pxx, freqs, bins, im = axs[1].specgram(x, 1024, fs, noverlap=900)
            
def plots_allSpectrum(x,h,y,label1,label2,label3,freqSine,freqInv,freqDelta):
    plt.figure()
    plt.grid()
    plt.semilogx(freqSine, x, label=label1)
    plt.semilogx(freqInv, h, label=label2)
    plt.semilogx(freqDelta, y, label=label3)
    plt.title('Spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dBFS]')
    plt.legend()
    plt.show()

finf = 10
fsup = 22000
T = 4
fs = 48000
t = np.arange(0,T*fs)/fs

sinesweep=log_sinesweep(finf,fsup,T,t,fs)
sf.write('sinesweep.wav', sinesweep, fs)
inversefilter=inverse_filter(finf,fsup,T,t,sinesweep)
sf.write('inversefilter.wav', inversefilter, fs)
delta = sig.fftconvolve(sinesweep, inversefilter)
delta = delta/(np.abs(max(delta))) # normalization
delta = delta[inversefilter.size-1:] # adjust length because of FFT
sf.write('deltaFarina.wav', delta, fs)


freqSine, sinesweepdB = spectrumDBFS(sinesweep, fs)
plots(sinesweep,sinesweepdB,'Logarithmic SineSweep x(t)',fs,freqSine)

freqInv, inversefilterdB = spectrumDBFS(inversefilter, fs)
plots(inversefilter,inversefilterdB,'Inverse filter f(t)',fs,freqInv)

freqDelta, deltadB = spectrumDBFS(delta, fs)
plots(delta,deltadB,'Delta d(t) = x(t) * f(t)',fs,freqDelta)

plots_allSpectrum(sinesweepdB,inversefilterdB,deltadB,'Log. SineSweep','Inverse filter','Delta',freqSine,freqInv,freqDelta)

impulseresponse = mic_rirs[:,0,0] #get an Impulse Response
sf.write('IR.wav', impulseresponse, fs)
freqIR, impulseresponsedB = spectrumDBFS(impulseresponse, fs)
plots(impulseresponse,impulseresponsedB,'Impulse response h(t)',fs,freqIR)
plots(impulseresponse, None, 'Impulse response h(t)',fs,freqIR)

measured = sig.fftconvolve(sinesweep,impulseresponse)
sf.write('measured.wav', measured, fs)
freqMeasured, measureddB = spectrumDBFS(measured, fs)
#plots(measured,measureddB,'Measured  y(t) = x(t)*h(t)',fs,freqMeasured)
plots(measured, None,'Measured  y(t) = x(t)*h(t)',fs,freqMeasured)

estimationIR = sig.fftconvolve(measured, inversefilter)
estimationIR = estimationIR/(np.abs(max(estimationIR))) # normalization
estimationIR = estimationIR[inversefilter.size:inversefilter.size+impulseresponse.size] # adjust length because of FFT
sf.write('estimatedIR.wav', estimationIR, 48000)
freqEstimated, estimationIRdB = spectrumDBFS(estimationIR, fs)
#plots(estimationIR,estimationIRdB,'Estimated IR  h(t) = y(t)*x_inv(t)')
plots(estimationIR, None,'Estimated IR  h(t) = y(t)*x_inv(t)',fs,freqEstimated)

mse = np.mean((impulseresponse - estimationIR)**2)

