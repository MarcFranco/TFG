# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2019, Eurecat / UPF
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   @file   test_script_mics.py
#   @author Andrés Pérez-López
#   @date   29/07/2019
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from masp import shoebox_room_sim as srs
import time
import librosa
import matplotlib.pyplot as plt 
import math
import statistics 
from scipy import fftpack
from scipy import signal
import scipy.signal as sig
import soundfile as sf
from scipy.signal import butter, lfilter, hilbert
from scipy import stats
from scipy.io import wavfile

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
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# RT60

def octave_lf(centerfreqs):
    return centerfreqs/ np.sqrt(2.0)

def octave_hf(centerfreqs):
    return centerfreqs * np.sqrt(2.0)

def third_lf(centerfreqs):
    return centerfreqs / 2.0**(1.0 / 6.0)
    
def third_hf(centerfreqs):
    return centerfreqs * 2.0**(1.0 / 6.0)

def check_type(band_type,centerfreqs):    
    if band_type == 'octave':
        low = octave_lf(centerfreqs)
        high = octave_hf(centerfreqs)
    elif band_type == 'third':
        low = third_lf(centerfreqs)
        high = third_hf(centerfreqs)
    return band_type, low, high;
   
def RT_estimator(rt): 
    if rt == 'edt':
        begin = 0.0
        end = -10.0
        factor = 6.0    
    elif rt == 'rt10':
        begin = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'rt20':
        begin = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 'rt30':
        begin = -5.0
        end = -35.0
        factor = 2.0
    
    return begin,end,factor;

      
def butterworth_bandpass(lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butterworth_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butterworth_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y

def envelope_plot(impulse,env,index,centerfreqs):   
    time= np.arange(0,impulse.size)/fs
    plt.figure()
    plt.title('Signal and Filtered of frequency band '+ repr(centerfreqs[index]) + ' Hz')
    plt.plot(time, impulse, label='Signal')
    plt.plot(time, env, label='Envelope')
    plt.xlabel("Time [s]")
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

def movingaverage (values, M, index):
    weights = np.repeat(1.0, M)/M
    if index == 1: #To only plot the first time
        maf_plot(weights)
    maf = np.convolve(values, weights, 'valid')       
    return maf

def maf_plot(weights):
    a = np.ones(1)  
    w, h = signal.freqz(weights,a)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Magnitude and phase response of the Moving Average filter')
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.ylabel('Magnitude [dB]')
    plt.xlabel('Frequency [rad/sample]')
    plt.subplot(2, 1, 2)
    angles = np.unwrap(np.angle(h))
    plt.plot(w, angles)
    plt.ylabel('Angle (radians)')
    plt.xlabel('Frequency [rad/sample]')
    plt.show()
    
def plot_schroeder(EdB,begin_sample,end_sample,index,centerfreqs):
    plt.figure()    
    marker_begin = [begin_sample]
    marker_end = [begin_sample, end_sample]
    plt.title('Schroeder Integral of frequency band ' + repr(centerfreqs[index])  + ' Hz')
    plt.plot(EdB,'-bo',markevery=marker_begin, label=repr(round(EdB[begin_sample]))+' dB')
    plt.plot(EdB,'-bo',markevery=marker_end, label=repr(round(EdB[end_sample]))+' dB')
    plt.plot(EdB,'-r',label='Schroeder dB')
    plt.ylabel('Level [dB]')
    plt.xlabel('Samples')
    plt.grid()
    plt.legend()
    plt.show()
    
def print_rt60(rt60,centerfreqs):   
    print('----Reverberation Time (RT60)----')
    for i in range (0,rt60.size):
        print(repr(centerfreqs[i]) +'Hz : ' + repr(rt60[i]) )
       
def revTime60(data,band_centerfreqs,window,rt_type,oct_type):    
    band_type, low, high = check_type(oct_type,band_centerfreqs) 
    begin,end,factor=RT_estimator(rt_type)
    rt60 = np.zeros(band_centerfreqs.size)  
    
    for band in range(band_centerfreqs.size):
        
        # Filtering signal w/butterworth & hilbert 
        filtered_signal = butterworth_bandpass_filter(data, low[band], high[band], fs, order=3)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))    
        amplitude_envelope = np.abs(hilbert(filtered_signal)) 
        amplitude_envelope = amplitude_envelope/np.max(amplitude_envelope)
        envelope_plot(abs_signal,amplitude_envelope,band,band_centerfreqs)
        
        # Moving Average filter
        maIR = movingaverage(amplitude_envelope, window, band)
    
        # Schroeder integration
        A = np.cumsum(maIR[::-1]**2)[::-1]
        EdB = 20.0 * np.log10(A/np.max(A))
        # Linear regression of Schroeder curve w/ L
        schroeder_begin = EdB[np.abs(EdB - begin).argmin()]
        schroeder_end = EdB[np.abs(EdB - end).argmin()]
        begin_sample = np.where(EdB == schroeder_begin)[0][0]
        end_sample = np.where(EdB == schroeder_end)[0][0] 
        
        plot_schroeder(EdB,begin_sample,end_sample,band,band_centerfreqs)
        
        L = np.arange(begin_sample, end_sample + 1) / fs
        schroeder = EdB[begin_sample:end_sample + 1]
        slope, intercept = stats.linregress(L, schroeder)[0:2]
       
        # Reverberation time
        rt60[band]=-60/slope
        
    return rt60

window=5001
rt_type='rt30'
oct_type='third'
rt60_impulseresponse = revTime60(impulseresponse,band_centerfreqs,window,rt_type,oct_type)
print_rt60(rt60_impulseresponse,band_centerfreqs)
rt60_estimationIR = revTime60(estimationIR,band_centerfreqs,window,rt_type,oct_type)
print_rt60(rt60_estimationIR,band_centerfreqs)

x = np.arange(1 * fs)
ir = (np.random.rand(x.size)*2-1) * np.exp(-5*x/fs)
rt60_andres = revTime60(ir,band_centerfreqs,window,rt_type,oct_type)
print_rt60(rt60_andres,band_centerfreqs)

# Bass Ratio (BR) Objective: 0.9 - 1
def bass_ratio(rt60): 
    BR = (rt60[0] + rt60[1])/(rt60[2]+rt60[3]) 
    print ("Bass Ratio (BR): " + str('%.3f'%BR))
    return BR

# Brightness (Br) Objective: >0.80 
def brightness(rt60): 
    Br = (rt60[4] + rt60[5])/(rt60[2]+rt60[3]) 
    print ("Brightness (Br): " + str('%.3f'%Br))
    return Br

bassRatio_impulse=bass_ratio(rt60_impulseresponse)
bassRatio_estimationIR=bass_ratio(rt60_estimationIR)
brightness_impulse=brightness(rt60_impulseresponse)
brightness_estimationIR=brightness(rt60_estimationIR)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fs, vocal = wavfile.read('C:\TFG\Codi\masp-master\sounds\A.wav')
x = np.arange(1 * fs)
ir = (np.random.rand(x.size)*2-1) * np.exp(-5*x/fs)
convolvedSignal = sig.fftconvolve(ir, vocal[:,0])
convolvedSignal=convolvedSignal/np.max(convolvedSignal)
sf.write('AReverb.wav', convolvedSignal, fs)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EVALUATION METRIC

mse = np.mean((impulseresponse - estimationIR)**2)
rmse=math.sqrt (mse)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def print_c50(c50,centerfreqs):   
    print('----Speech Clarity (C50)----')
    for i in range (0,c50.size):
        print(repr(centerfreqs[i]) +'Hz : ' + repr(c50[i]) )

 # Speech Clarity (C50) Objective: >2dB
def speechClarity50(data,band_centerfreqs,oct_type):    
    band_type, low, high = check_type(oct_type,band_centerfreqs) 
    C50 = np.zeros(band_centerfreqs.size)  

    for band in range(band_centerfreqs.size):
        filtered_signal = butterworth_bandpass_filter(data, low[band], high[band], fs, order=3)
        p2 = filtered_signal**2.0
        t = int(0.05*fs)
        C50[band] = 10.0 * np.log10((np.sum(p2[:t]) / np.sum(p2[t:])))
    return C50

oct_type='third'
c50_impulseresponse=speechClarity50(impulseresponse,band_centerfreqs,oct_type)
print_c50(c50_impulseresponse,band_centerfreqs)
c50_estimationIR=speechClarity50(estimationIR,band_centerfreqs,oct_type)
print_c50(c50_estimationIR,band_centerfreqs)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def print_d50(d50,centerfreqs):   
    print('----Definition (D50)----')
    for i in range (0,d50.size):
        print(repr(centerfreqs[i]) +'Hz : ' + repr(d50[i]) )
        
# Definition (D50) Objective: 0.4 - 0.6
def Definition(C50,band_centerfreqs):    
    D50 = np.zeros(band_centerfreqs.size)  
    for band in range(band_centerfreqs.size):
        D50[band] = 1 / (1 + 10**-(C50[band]/10)) 
    return D50

d50_impulseresponse=Definition(c50_impulseresponse,band_centerfreqs)
print_d50(d50_impulseresponse,band_centerfreqs)
d50_estimationIR=Definition(c50_estimationIR,band_centerfreqs)
print_d50(d50_estimationIR,band_centerfreqs)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def print_Smid(Smid,centerfreqs):   
    print('----Speech Sound Level (Smid)----')
    for i in range (0,Smid.size):
        print(repr(centerfreqs[i]) +'Hz : ' + repr(Smid[i]) )

# Strength (S)  Objective: (4 <= Smid(0º) <= 8) and (2 <= Smid(90º) <= 6)
def SpeechSoundLevel(data,Lw,band_centerfreqs,oct_type):
    
    band_type, low, high = check_type(oct_type,band_centerfreqs) 
    Smid = np.zeros(band_centerfreqs.size)  
    LpE10m = Lw-31

    for band in range(band_centerfreqs.size):    
        filtered_signal = butterworth_bandpass_filter(data, low[band], high[band], fs, order=3)
        p2 = filtered_signal**2.0        
        LpElistener = 10.0*np.log10(np.sum(p2))     
        Smid[band] = LpElistener - LpE10m
    return Smid

Lw = 94
oct_type='third'
Smid_impulseresponse=SpeechSoundLevel(impulseresponse,Lw,band_centerfreqs,oct_type)
print_Smid(Smid_impulseresponse,band_centerfreqs)
Smid_estimationIR=SpeechSoundLevel(estimationIR,Lw,band_centerfreqs,oct_type)
print_Smid(Smid_estimationIR,band_centerfreqs)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
# Encode Audio to Ambisonics
def encode_bFormat(audio,azimuth,elevation):
    W = audio/np.sqrt(2) #Omnidirectional microphone (pressure)
    X = audio*math.cos(azimuth)*math.cos(elevation) #Figure-of-eight microphonesne (acoustic velocity components)
    Y = audio*math.sin(azimuth)*math.cos(elevation)
    Z = audio*math.sin(elevation) 
    
    audioBFormat = np.vstack((W,X,Y,Z))
    audioBFormat = np.transpose(audioBFormat)
    audioBFormat=audioBFormat/np.max(audioBFormat)
    
    return W,X,Y,Z,audioBFormat

# Write File in B-Format
def write_bFormat(output_filename,audioBFormat,fs):    
    sf.write(output_filename, audioBFormat, fs)
    return audioBFormat
   
def compute_stft(audioBFormat,fs,win_type,win_length):    
    for x in range(0,len(audioBFormat[1])):       
        f,t,Zxx = sig.stft(audioBFormat[:,x], fs, win_type, win_length)
        if x == 0: #Initialize the variable stft
            stft = np.empty([len(audioBFormat[1]),f.size,t.size],dtype=np.complex128)    
        stft[x,:,:] = Zxx
    return f, t, stft
    
def sound_pressure(audioBFormat_stft):
    P = audioBFormat_stft[0,:,:]    
    return P

def particle_velocity(audioBFormat_stft,p0,c):
    Z0=p0*c    
    U = (-1.0/(np.sqrt(2)*Z0))*audioBFormat_stft[1:,:,:]    
    return U

def intensity_vector(audioBFormat_stft,f,t,p0,c):
    P = sound_pressure(audioBFormat_stft)
    U = particle_velocity(audioBFormat_stft,p0,c)
    I = np.empty((len(audioBFormat_stft[:,1,1])-1,f.size,t.size))
    I = (1/2)*np.real(P * np.conj(U))
    return I

def intensity_vector_bFormat(audioBFormat_stft,f,t,p0,c):
    Z0=p0*c   
    I = np.empty((len(audioBFormat_stft[:,1,1])-1,f.size,t.size))
    I = -(1/(2*np.sqrt(2)*Z0))*np.real(audioBFormat_stft[0,:,:] * np.conj(np.transpose(audioBFormat_stft[1:,:,:], (0, 1, 2))))
    return I

def direction_of_incidence(I,f,t):    
    DOA = np.empty((len(I[:,1,1]),f.size,t.size))    
    DOA[0:,:,:] = -(np.divide(I[0:,:,:], np.linalg.norm(I, ord=2, axis=0))) 
    return DOA

def direction_of_incidence_bFormat(I,f,t):    
    DOA = np.empty((len(I[:,1,1]),f.size,t.size))    
    DOA[0:,:,:] = (np.divide(np.real(audioBFormat_stft[0,:,:] * np.conj(np.transpose(audioBFormat_stft[1:,:,:], (0, 1, 2)))), np.linalg.norm(np.real(audioBFormat_stft[0,:,:] * np.conj(np.transpose(audioBFormat_stft[1:,:,:], (0, 1, 2)))), ord=2, axis=0))) 
    return DOA

def cartesian_spherical(x):
    r= np.sqrt(x[0,:,:]**2 + x[1,:,:]**2 + x[2,:,:]**2)
    azimuth=np.arctan2(x[1,:,:],x[0,:,:])
    elevation=np.arcsin(x[2,:,:]/r)
    return r,azimuth,elevation

def energy_density(audioBFormat_stft,p0,c):
    P = sound_pressure(audioBFormat_stft)
    U = particle_velocity(audioBFormat_stft,p0,c)
    energy = ((p0/4)*np.power(np.linalg.norm(U, ord=2, axis=0),2)) + ((1/(4*p0*np.power(c,2)))*np.power(abs(P),2))
    return energy

def energy_density_bFormat(audioBFormat_stft,p0,c):
    energy = (1/(4*p0*np.power(c,2)))*((np.power(np.linalg.norm(np.transpose(audioBFormat_stft[1:,:,:], (0, 1, 2)), ord=2, axis=0),2)/2)+(np.power(abs(audioBFormat_stft[0,:,:]),2)))
    return energy

def Diffuseness(I,energy,f,t,dt):
    diffuseness = np.empty((f.size, t.size))

    for k in range(f.size):
        for n in range(int(dt / 2), int(t.size - dt / 2)):
            num = np.linalg.norm(np.mean(I[:, k, n:n + dt],axis = 1),ord=2, axis=0)
            den = c * np.mean(energy[k,n:n+dt])
            diffuseness[k,n] = 1-((num)/(den)) 
            # Borders: copy neighbor values
            diffuseness[k, 0:int(dt/2)] = diffuseness[k, int(dt/2)]        
            diffuseness[k, int(t.size - dt / 2):t.size] = diffuseness[k, t.size - int(dt/2) - 1]
    return diffuseness

def plotSpectrogram(title, x, colorMap, xlabel, ylabel, barlabel, minValue, maxValue):
    plt.figure()
    plt.suptitle(title)
    plt.pcolormesh(np.abs(x), cmap = colorMap, vmin = minValue, vmax = maxValue)    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(barlabel)
    
def plotRadians(title,x,xlabel,ylabel,vmin,vmax):
    plt.figure()
    plt.suptitle(title)
    plt.plot(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(vmin,vmax)
    plt.grid()

    
def plotReflection(azimuth, elevation, title, xlabel, ylabel,xvmin,xvmax,yvmin,yvmax):
    az = np.empty(1)
    el = np.empty(1)    
    az = azimuth.flatten() 
    el = elevation.flatten()                         
    plt.figure()
    plt.suptitle(title)   
    plt.plot(az,el,'-o')    
    plt.xlabel(xlabel)
    plt.xlim(xvmin,xvmax)
    plt.ylabel(ylabel)
    plt.ylim(yvmin,yvmax)
    plt.grid()


fs, mono_audio = wavfile.read('C:\TFG\Codi\masp-master\sounds\medieval.wav')
az = -np.pi 
el = 0
W,X,Y,Z,audioBFormat = encode_bFormat(mono_audio,az,el)  
win_type = 'hann'
win_length = 256
f,t,audioBFormat_stft = compute_stft(audioBFormat,fs,win_type,win_length)

c = 346.13  # m/s
p0 = 1.1839 # kg/m3
P = sound_pressure(audioBFormat_stft)
U = particle_velocity(audioBFormat_stft,p0,c)
I = intensity_vector(audioBFormat_stft,f,t,p0,c)
IbFormat = intensity_vector_bFormat(audioBFormat_stft,f,t,p0,c)

DOA = direction_of_incidence(I,f,t)
DOAbFormat = direction_of_incidence_bFormat(I,f,t)

r,azimuth,elevation = cartesian_spherical(DOA) 
plotSpectrogram('Azimuth', azimuth, 'plasma','Time', 'Frequency', 'Azimuth', -np.pi, np.pi)
plotSpectrogram('Elevation', elevation, 'plasma','Time', 'Frequency', 'Elevation',-np.pi/2, np.pi/2)
#plotRadians('Azimuth', azimuth, 'Time', 'Angle [rad]',-4, 4)
#plotRadians('Elevation', elevation, 'Time', 'Angle [rad]',-2, 2)
plotReflection(azimuth, elevation, 'Reflection Direction','Azimuth', 'Elevation',-4,4,-2,2)

energy = energy_density(audioBFormat_stft,p0,c)
energybFormat = energy_density_bFormat(audioBFormat_stft,p0,c)

dt = 10
diffuseness = Diffuseness(I,energy,f,t,dt) 
plotSpectrogram('Diffuseness', diffuseness, 'plasma','Time', 'Energy', 'Diffuseness',0,1)

output_filename='medievalBFormat.wav'
audioBFormat=write_bFormat(output_filename,audioBFormat,fs)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Function to compute the Surface of a room
def compute_surface(width,height,depth): 
    # Assuming there are no windows and door. In that case substract the S from it
    # and sum the S to the total S. 
    s_Ceiling = width * depth
    s_Floor = width * depth
    s_FrontWall = width * height 
    s_BackWall = width * height 
    s_RightWall = depth * height
    s_LeftWall = depth * height
    surface= [s_Ceiling,s_Floor,s_FrontWall,s_BackWall,s_RightWall,s_LeftWall]
    S = s_Ceiling + s_Floor + s_FrontWall + s_BackWall + s_RightWall + s_LeftWall 
    return surface,S

# Function to compute the Sound Absorption Coefficient for each frequency band
def Absorption_Coefficient(nBands,surface,S,abs_coef):
    alpha = []
    for a in range (0,nBands):
        alpha_band = np.dot(surface,abs_coef[:,a])/S 
        alpha.append(alpha_band)
    return alpha

# Function to compute the Room Constant (R)
def Room_Constant(S,alpha,nBands):
    R=[]
    for d in range (0,nBands):
        ct_room=(S*alpha[d])/(1-alpha[d])
        R.append(ct_room)
    return R

# Function to compute the Critical Distance
def Critical_Distance(nBands,Q,S,alpha,centerfreqs):
    Dc=[]
    R=Room_Constant(S,alpha,nBands)
    print('----Critical Distance----')
    for k in range (0,nBands):
        cd=math.sqrt((Q*R[k])/(16*math.pi))
        Dc.append(cd)
        print (repr(centerfreqs[k]) + 'Hz : ' + str(Dc[k])) 
    return Dc

# Compute the distance between the source and the receiver
def Distance_sr(src,rec):
    r=[]
    for x in range (0,len(src)):
        distance = math.sqrt((src[x,0]-rec[0,0])**2 + (src[x,1]-rec[0,1])**2 + (src[x,2]-rec[0,2])**2)
        r.append(distance)
    return r

def print_ALCons(ALCons,centerfreqs):   
    print('----Percentage Articulation Loss of Consonants (%ALCons)----')
    for i in range (0,ALCons.size):
        print(repr(centerfreqs[i]) +'Hz : ' + repr(ALCons[i]) )

# Articulation Loss of Consonants (%ALCons) Objective: 0 - 7%
def ALCons(r,rt60,V,Q,Dc,band_centerfreqs):
    
    Cons = np.zeros(band_centerfreqs.size) 
    
    for band in range(band_centerfreqs.size): 
        if r<=3.16*Dc[band]:
            Cons[band] = (200*(r**2)*(rt60[band]**2))/(V*Q)
        else:
            Cons[band] = 9*rt60[band]
    return Cons

surface,S=compute_surface(width,height,depth)
abs_coef = np.matrix([[0.57,0.39,0.41,0.82,0.89,0.72],[0.2,0.15,0.12,0.1,0.1,0.07],[0.01,0.01,0.02,0.03,0.04,0.05],[0.01,0.01,0.02,0.03,0.04,0.05],[0.01,0.01,0.02,0.03,0.04,0.05],[0.01,0.01,0.02,0.03,0.04,0.05]])
alpha=Absorption_Coefficient(nBands,surface,S,abs_coef)
r = Distance_sr(src,rec)
V = room[0] * room[1] * room[2] # Volume of the class
Q = 2 # Directivity Factor for speech in a class
Dc=Critical_Distance(nBands,Q,S,alpha,band_centerfreqs)
ALCons_impulseresponse=ALCons(r[0],rt60_impulseresponse,V,Q,Dc,band_centerfreqs)
print_ALCons(ALCons_impulseresponse,band_centerfreqs)
ALCons_estimationIR=ALCons(r[0],rt60_estimationIR,V,Q,Dc,band_centerfreqs)
print_ALCons(ALCons_estimationIR,band_centerfreqs)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FUNCTIONS TO COMPUTE THE ACOUSTIC PARAMETERS

# Function to compute the Surface of a room
def compute_surface(width,height,depth): 
    # Assuming there are no windows and door. In that case substract the S from it
    # and sum the S to the total S. 
    s_Ceiling = width * depth
    s_Floor = width * depth
    s_FrontWall = width * height 
    s_BackWall = width * height 
    s_RightWall = depth * height
    s_LeftWall = depth * height
    surface= [s_Ceiling,s_Floor,s_FrontWall,s_BackWall,s_RightWall,s_LeftWall]
    S = s_Ceiling + s_Floor + s_FrontWall + s_BackWall + s_RightWall + s_LeftWall 
    return surface,S

# Function to compute the Sound Absorption Coefficient for each frequency band
def Absorption_Coefficient(nBands,surface,S,abs_coef):
    alpha = []
    for a in range (0,nBands):
        alpha_band = np.dot(surface,abs_coef[:,a])/S 
        alpha.append(alpha_band)
    return alpha

# Function to compute the Air Coefficient for each frequency band
def Air_Coefficient(nBands,humidity,band_centerfreqs):
    m=[]
    for a in range (0,nBands):
        air_coef=5.5*(10**-4)*(50/humidity)*((band_centerfreqs[a]/1000)**1.7) # Air coefficient
        m.append(air_coef)
    return m

# Function to compute the RT60
def Reverberation_Time(nBands,V,c,humidity,band_centerfreqs,abs_coef,width,height,depth):    
    RT60=[]
    surface,S=compute_surface(width,height,depth)
    alpha=Absorption_Coefficient(nBands,surface,S,abs_coef)
    m=Air_Coefficient(nBands,humidity,band_centerfreqs)
    for e in range (0,nBands):
        if V < 500 and alpha[e] < 0.2: #No air coef and Sabine
            rt = (60*V) / (1.086*c*S*alpha[e])
            RT60.append(rt)
        elif V < 500 and alpha[e] > 0.2: #No air coef and Eyring
            rt = (60*V) / (1.086*c*S*(-math.log(1-alpha[e])))
            RT60.append(rt)
        elif V > 500 and alpha[e] < 0.2:#Air coef and Sabine
            rt = (60*V) / (1.086*c*(S*alpha[e]+4*m*V))
            RT60.append(rt)
        else: #Air coef and Eyring
            rt = (60*V) / (1.086*c*S*(-math.log(1-alpha[e])+(4*m*V)/S))
            RT60.append(rt)
        print ("Frequency Band: " + str(band_centerfreqs[e]) + " --> RT60: " + str(RT60[e])) 
    return RT60,S,alpha

# Function to compute the Room Constant (R)
def Room_Constant(S,alpha,nBands):
    R=[]
    for d in range (0,nBands):
        ct_room=(S*alpha[d])/(1-alpha[d])
        R.append(ct_room)
    return R

# Function to compute the Critical Distance
def Critical_Distance(nBands,Q,S,alpha):
    Dc=[]
    R=Room_Constant(S,alpha,nBands)
    for k in range (0,nBands):
        cd=math.sqrt((Q*R[k])/(16*math.pi))
        Dc.append(cd)
        print ("Critical Distance for band frequency: " + str(band_centerfreqs[k]) + " --> Dc: " + str(Dc[k])) 
    return Dc

# Bass Ratio (BR) Objective: 0.9 - 1
def Bass_Ratio(rt60): 
    BR = (rt60[0] + rt60[1])/(rt60[2]+rt60[3]) 
    print ("Bass Ratio (BR): " + str('%.3f'%BR))
    return BR

# Brightness (Br) Objective: >0.80 
def Brightness(rt60): 
    Br = (rt60[4] + rt60[5])/(rt60[2]+rt60[3]) 
    print ("Brightness (Br): " + str('%.3f'%Br))
    return Br

 # Speech Clarity (C50) Objective: >2dB
def Speech_Clarity(rir,fs):
    samples_b50=rir[:int(0.05*fs),0,0]
    Energy_b50 = sum(map(lambda i:i*i,samples_b50)) 
    samples_a50=rir[int(0.05*fs):,0,0]
    Energy_a50 = sum(map(lambda i:i*i,samples_a50)) 
    C50=10*math.log10(Energy_b50/Energy_a50)
    print ("Speech Clarity (C50): " + str('%.3f'%C50)) 
    return C50

# Definition (D50) Objective: 0.4 - 0.6
def Definition(C50):
    D50 = 1 / (1 + 10**-(C50/10)) 
    print ("Definition (D50): " + str('%.3f'%D50)) 
    return D50

# Compute the distance between the source and the receiver
def Distance_sr(src,rec):
    r=[]
    for x in range (0,len(src)):
        distance = math.sqrt((src[x,0]-rec[0,0])**2 + (src[x,1]-rec[0,1])**2 + (src[x,2]-rec[0,2])**2)
        r.append(distance)
    return r

# Articulation Loss of Consonants (%ALCons) Objective: 0 - 7%
def ALCons(r,RT,V,Q,Dc):
    if r<=3.16*Dc:
        Cons = (200*(r**2)*(RT**2))/(V*Q)
    else:
        Cons = 9*RT
    return Cons

# Strength (S)  Objective: (4 <= Smid(0º) <= 8) and (2 <= Smid(90º) <= 6)
def Strength(Lw,Q,r):
    Lp = Lw - abs(10*math.log(Q/4*math.pi*(r**2)))
    smid = Lp - Lw + 39
    return smid

# Function to plot %ALCons vs Source-Receiver 
def plot_ALCons_sr(ALCons,r):    
    plt.plot(ALCons,r, color='r', linestyle='solid', marker='o', linewidth=1, markersize=8)
    plt.grid()
    plt.xlabel('%ALCons [%]');
    plt.ylabel('Distance Source-Receiver [meters]');
    plt.title('%ALCons depending on the distance between sender and receiver')
    plt.show()  

# Function to plot %ALCons vs RT60
def plot_ALCons_rt60(ALCons,rt60):
    plt.plot(ALCons,rt60, color='g', linestyle='solid', marker='o', linewidth=1, markersize=8)
    plt.grid()
    plt.xlabel('%ALCons [%]');
    plt.ylabel('RT60 [seconds]');
    plt.title('%ALCons depending on the RT60')
    plt.show() 
    
# Function to plot Smid vs Source-Receiver 
def plot_Smid_sr(Smid,r):
    plt.plot(Smid,r, color='b', linestyle='solid', marker='o', linewidth=1, markersize=8)
    plt.grid()
    plt.xlabel('Speech Sound Level');
    plt.ylabel('Distance Source-Receiver [meters]');
    plt.title('Strength depending on the distance between sender and receiver')
    plt.show() 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# COMPUTE THE ACOUSTIC PARAMETERS

#Init variables
temperature = 20
humidity=25
c = 331.4 + 0.6* temperature
V = room[0] * room[1] * room[2] # Volume of the class
Q = 2 # Directivity Factor for speech in a class
abs_coef = np.matrix([[0.57,0.39,0.41,0.82,0.89,0.72],[0.2,0.15,0.12,0.1,0.1,0.07],[0.01,0.01,0.02,0.03,0.04,0.05],[0.01,0.01,0.02,0.03,0.04,0.05],[0.01,0.01,0.02,0.03,0.04,0.05],[0.01,0.01,0.02,0.03,0.04,0.05]])
    
print ("-------- ACOUSTIC PARAMETERS --------" ) 
RT60,S,alpha = Reverberation_Time(nBands,V,c,humidity,band_centerfreqs,abs_coef,width,height,depth)
Dc = Critical_Distance(nBands,Q,S,alpha)
BR = Bass_Ratio(rt60)
Br = Brightness(rt60)
C50 = Speech_Clarity(mic_rirs,fs)
D50 = Definition(C50)

r = Distance_sr(src,rec)

RT = rt60 # (time given)
#RT = statistics.mean(rt60) (time averaged of the given)
#RT = RT60 (time computed)
#RT = statistics.mean(RT60) (time averaged of the computed)

print ("----- ALCONS with different distance between src-rec and fixed RT60 -----") 
ALCons_r=[]
#Compute %ALCONS with different distance between src-rec and fixed RT60 (1kHz band)
for y in range (0,len(r)):
    Cons_r = ALCons(r[y],RT[3],V,Q,Dc[3])  
    print ("%ALCons: " + str('%.3f'%Cons_r) + " at a distance: " + str('%.3f'%r[y]))
    ALCons_r.append(Cons_r)        

print ("----- ALCONS with different rt60 and fixed src-rec distance -----")
ALCons_60=[]
#Compute %ALCONS with different rt60 and fixed src-rec distance (distance r1)
for z in range (0,len(rt60)):    
    Cons_60 = ALCons(r[1],RT[z],V,Q,Dc[3])
    print ("%ALCons: " + str('%.3f'% Cons_60) + " with a RT60 of: " + str(rt60[z])) 
    ALCons_60.append(Cons_60)    

print ("----- Speech Sound Level with different src-rec distance -----") 
Smid=[] 
Lw = 94 # Source Power dbSPL
for j in range(0,len(r)):    
    smid = Strength(Lw,Q,r[j])
    print ("Strength (S): " + str('%.3f'% smid) + " at a distance: " + str(r[j])) 
    Smid.append(smid)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_ALCons_sr(ALCons_r,r)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_ALCons_sr(ALCons_60,rt60)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_Smid_sr(Smid,r)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plt.plot(mic_rirs[:,0,0]) #Impulse Response Plot


