# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:37:58 2020

@author: Marc Franco Meca
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from scipy.io import wavfile
import sys
import scipy.signal as sig
import matplotlib.pyplot as plt 
sys.path.append("C:\TFG\Codi\masp-master\examples\shoebox_room_sim")
from functions import (revTime60,bass_ratio,brightness,speechClarity50,
Definition,compute_surface,roomModes,musicClarity80, Absorption_Coefficient,
Distance_sr,Critical_Distance,ALCons,NoiseCriteria)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# EMPTY ROOM

# Room definition

width = 2.28
height = 2.32
depth = 2.63

room = np.array([depth, width, height])

temperature = 20
c = 331.4 + 0.6* temperature

numNodes = 2
modes=roomModes(depth,width,height,numNodes,c)
nBands=6

# Generate octave bands
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 125
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb-1]

# Receiver position
rec = np.array([ [1.16, 1.2, 1.315]])
nRec = rec.shape[0]

# Source positions
src = np.array([ [0.66, 0.7, 0.81],[1.66, 0.7, 0.81]])
nSrc = src.shape[0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# EMPTY ROOM


fs, measuredEmpty = wavfile.read('C:\TFG\Codi\masp-master\sounds\Empty\IRSweep2.wav')
fs, inversefilter = wavfile.read('C:\TFG\Codi\masp-master\sounds\Empty\Inverse.wav')

impulseresponse = sig.fftconvolve(measuredEmpty[:,1], inversefilter)
impulseresponse = impulseresponse/(np.abs(max(impulseresponse))) # normalization
idx = np.argmax(np.abs(impulseresponse))
impulseresponse = impulseresponse[(idx-1000):(idx+50000)] # adjust length because of FFT
plt.plot(impulseresponse)

# RT60
window=5001
rt_type='rt30'
oct_type='third'
rt60_Empty = revTime60(impulseresponse,band_centerfreqs,window,rt_type,oct_type,fs)
bassRatio_Empty=bass_ratio(rt60_Empty)
brightness_Empty=brightness(rt60_Empty)
c50_Empty=speechClarity50(impulseresponse,band_centerfreqs,oct_type,fs)
c80_Empty=musicClarity80(impulseresponse,band_centerfreqs,oct_type,fs)
d50_Empty=Definition(c50_Empty,band_centerfreqs)


surface,S=compute_surface(width,height,depth)
abs_coef = np.matrix([[0.1,0.1,0.04,0.02,0.02,0.02],[0.1,0.1,0.04,0.02,0.02,0.02],[0.1,0.1,0.04,0.02,0.02,0.02],[0.1,0.1,0.04,0.02,0.02,0.02],[0.1,0.1,0.04,0.02,0.02,0.02],[0.1,0.1,0.04,0.02,0.02,0.02]])
alpha=Absorption_Coefficient(nBands,surface,S,abs_coef)
r = Distance_sr(src,rec)
V = room[0] * room[1] * room[2] 
Q = 2 
Dc=Critical_Distance(nBands,Q,S,alpha,band_centerfreqs)
ALCons_Empty=ALCons(r[0],rt60_Empty,V,Q,Dc,band_centerfreqs)

#oct_levels=[74.1, 76.3, 68.9, 59.6, 49.3, 42.9, 41.0, 35.8] #Measured Values
#nc=NoiseCriteria(oct_levels)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# FURNISHED ROOM

fs, measuredFull = wavfile.read('C:\TFG\Codi\masp-master\sounds\Full\IRSweep2.wav')
fs, inversefilter = wavfile.read('C:\TFG\Codi\masp-master\sounds\Empty\Inverse.wav')

impulseresponse = sig.fftconvolve(measuredFull[:,1], inversefilter)
impulseresponse = impulseresponse/(np.abs(max(impulseresponse))) # normalization
idx = np.argmax(np.abs(impulseresponse))
impulseresponse = impulseresponse[(idx-1000):(idx+50000)] # adjust length because of FFT
plt.plot(impulseresponse)

rt60_Full = revTime60(impulseresponse,band_centerfreqs,window,rt_type,oct_type,fs)
bassRatio_Full=bass_ratio(rt60_Full)
brightness_Full=brightness(rt60_Full)
c50_Full=speechClarity50(impulseresponse,band_centerfreqs,oct_type,fs)
c80_Full=musicClarity80(impulseresponse,band_centerfreqs,oct_type,fs)
d50_Full=Definition(c50_Full,band_centerfreqs)

surface,S=compute_surface(width,height,depth)
abs_coef = np.matrix([[0.1,0.1,0.04,0.02,0.02,0.02],[0.1,0.1,0.04,0.02,0.02,0.02],[0.1,0.1,0.04,0.02,0.02,0.02],[0.1,0.1,0.04,0.02,0.02,0.02],[0.1,0.1,0.04,0.02,0.02,0.02],[0.1,0.1,0.04,0.02,0.02,0.02]])
alpha=Absorption_Coefficient(nBands,surface,S,abs_coef)
r = Distance_sr(src,rec)
V = room[0] * room[1] * room[2] 
Q = 2 
Dc=Critical_Distance(nBands,Q,S,alpha,band_centerfreqs)
ALCons_Full=ALCons(r[0],rt60_Full,V,Q,Dc,band_centerfreqs)

#oct_levels=[58.7, 54.2, 51.5, 49.1, 44.0, 40.7, 38.8, 37.2] #Measured Values
#nc_Empty=NoiseCriteria(oct_levels)