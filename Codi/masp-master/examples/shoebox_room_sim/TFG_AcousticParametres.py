# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:26:03 2020

@author: alumne
"""

import numpy as np
import matplotlib.pyplot as plt 
import math
import time
import librosa
from scipy import signal
import scipy.signal as sig
import soundfile as sf
from masp import shoebox_room_sim as srs
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
