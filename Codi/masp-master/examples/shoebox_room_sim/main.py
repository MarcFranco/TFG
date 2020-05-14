# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:37:58 2020

@author: Marc Franco Meca
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# IMPORT LIBRARIES
import numpy as np
from masp import shoebox_room_sim as srs
import time
import librosa
import scipy.signal as sig
import soundfile as sf
from scipy.io import wavfile
import sys
sys.path.append("C:\TFG\Codi\masp-master\examples\shoebox_room_sim")
from functions import (log_sinesweep,inverse_filter,spectrumDBFS,plots,
plots_allSpectrum,revTime60,bass_ratio,brightness,speechClarity50,
Definition, SpeechSoundLevel,compute_surface,roomModes,
Absorption_Coefficient,Distance_sr,Critical_Distance,ALCons,Parametric_Reverberation_Time,
NoiseCriteria,convolution_audio_IRambisonics,compute_stft,sound_pressure,particle_velocity,
intensity_vector,intensity_vector_bFormat,direction_of_incidence, musicClarity80,
direction_of_incidence_bFormat,cartesian_spherical,energy_density,normalisation_standard,
energy_density_bFormat,Diffuseness,write_bFormat,plotSpectrogram,plotReflection,
plotRadians,apply_normalisation,channel_ordering,convolve_IR)

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
src = np.array([ [1.0, 2.0, 1.5],[4.0, 3.0, 1.5]])
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

#Define the sine sweep parameters
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

#Plot the sine sweep
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
# VALIDATE METHOD USING FARINA DATABASE

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
window=5001
rt_type='rt30'
oct_type='third'
rt60_impulseresponse = revTime60(impulseresponse,band_centerfreqs,window,rt_type,oct_type,fs)
rt60_estimationIR = revTime60(estimationIR,band_centerfreqs,window,rt_type,oct_type,fs)

# EVALUATION METRIC
mse = np.mean((impulseresponse - estimationIR)**2)

bassRatio_impulse=bass_ratio(rt60_impulseresponse)
bassRatio_estimationIR=bass_ratio(rt60_estimationIR)
brightness_impulse=brightness(rt60_impulseresponse)
brightness_estimationIR=brightness(rt60_estimationIR)

c50_impulseresponse=speechClarity50(impulseresponse,band_centerfreqs,oct_type,fs)
c50_estimationIR=speechClarity50(estimationIR,band_centerfreqs,oct_type,fs)

c80_impulseresponse=musicClarity80(impulseresponse,band_centerfreqs,oct_type,fs)
c80_estimationIR=musicClarity80(estimationIR,band_centerfreqs,oct_type,fs)


d50_impulseresponse=Definition(c50_impulseresponse,band_centerfreqs)
d50_estimationIR=Definition(c50_estimationIR,band_centerfreqs)

#Smid_impulseresponse=SpeechSoundLevel(impulseresponse,impulseresponse10m,Lw,band_centerfreqs,oct_type,fs)

temperature = 20
humidity=25
c = 331.4 + 0.6* temperature
surface,S=compute_surface(width,height,depth)
abs_coef = np.matrix([[0.57,0.39,0.41,0.82,0.89,0.72],[0.2,0.15,0.12,0.1,0.1,0.07],[0.01,0.01,0.02,0.03,0.04,0.05],[0.01,0.01,0.02,0.03,0.04,0.05],[0.01,0.01,0.02,0.03,0.04,0.05],[0.01,0.01,0.02,0.03,0.04,0.05]])
alpha=Absorption_Coefficient(nBands,surface,S,abs_coef)
r = Distance_sr(src,rec)
V = room[0] * room[1] * room[2] # Volume of the class
Q = 2 # Directivity Factor for speech in a class
Dc=Critical_Distance(nBands,Q,S,alpha,band_centerfreqs)

rt60_parametric = Parametric_Reverberation_Time(nBands,V,c,humidity,band_centerfreqs,abs_coef,width,height,depth,alpha,S)
ALCons_impulseresponse=ALCons(r[0],rt60_impulseresponse,V,Q,Dc,band_centerfreqs)
ALCons_estimationIR=ALCons(r[0],rt60_estimationIR,V,Q,Dc,band_centerfreqs)

oct_levels=[74.1, 76.3, 68.9, 59.6, 49.3, 42.9, 41.0, 35.8] #Measured Values
nc=NoiseCriteria(oct_levels)

numNodes = 2
modes=roomModes(depth,width,height,numNodes,c)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SH orders for receivers
rec_orders = np.array([1]) # rec1: first order(4ch), rec2: 3rd order (16ch)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RUN SIMULATOR

# Echogram
tic = time.time()

maxlim = 1.5 # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
limits = np.minimum(rt60, maxlim)

# Compute echograms
# abs_echograms, rec_echograms, echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RENDERING

fs = 48000
# In this case all the information (e.g. SH directivities) are already
# encoded in the echograms, hence they are rendered directly to discrete RIRs
sh_rirs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs)

toc = time.time()
print('Elapsed time is ' + str(toc-tic) + 'seconds.')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# GENERATE SOUND SCENES
# Each source is convolved with the respective mic IR, and summed with
# the rest of the sources to create the microphone mixed signals

sourcepath = 'C:\TFG\Codi\masp-master\data/milk_cow_blues_4src.wav'
#sourcepath = 'C:\TFG\Codi\masp-master\sounds/medieval.wav'
src_sigs = librosa.core.load(sourcepath, sr=None, mono=False)[0].T[:,:nSrc]

sh_sigs = srs.apply_source_signals_sh(sh_rirs, src_sigs)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fs, mono_audio = wavfile.read('C:\TFG\Codi\masp-master\sounds\medieval.wav')
az = -np.pi 
el = 0
ambisonicsOrder = 1
fs, ir_university = wavfile.read('C:\TFG\Codi\masp-master\sounds\ir_centre_stalls.wav')

#audioBFormat=convolution_audio_IRambisonics(mono_audio,sh_rirs)
#audioBFormat=convolution_audio_IRambisonics(mono_audio,ir_university)
audioBFormat=convolve_IR(mono_audio,ir_university)
channelOrder='ACN'
audioBFormat_ordered=channel_ordering(channelOrder,audioBFormat[:,0],audioBFormat[:,1],audioBFormat[:,2],audioBFormat[:,3])
norm_type='maxN'
norm=normalisation_standard(ambisonicsOrder,norm_type)
audioBFormat_normalized=apply_normalisation(norm,audioBFormat_ordered[:,0],audioBFormat_ordered[:,1],audioBFormat_ordered[:,2],audioBFormat_ordered[:,3])
#W,X,Y,Z,audioBFormat = encode_bFormat(mono_audio,az,el,norm)  

win_type = 'hann'
win_length = 256
f,t,audioBFormat_stft = compute_stft(audioBFormat_normalized,fs,win_type,win_length)

p0 = 1.1839 # kg/m3
P = sound_pressure(audioBFormat_stft)
U = particle_velocity(audioBFormat_stft,p0,c)
I = intensity_vector(audioBFormat_stft,f,t,p0,c)
IbFormat = intensity_vector_bFormat(audioBFormat_stft,f,t,p0,c)

DOA = direction_of_incidence(I,f,t)
DOAbFormat = direction_of_incidence_bFormat(audioBFormat_stft,I,f,t)

r,azimuth,elevation = cartesian_spherical(DOA) 
plotSpectrogram('Azimuth', azimuth, 'plasma','Time', 'Frequency', 'Azimuth', -np.pi, np.pi)
plotSpectrogram('Elevation', elevation, 'plasma','Time', 'Frequency', 'Elevation',-np.pi/2, np.pi/2)
#plotRadians('Azimuth', azimuth, 'Time', 'Angle [rad]',-4, 4)
#plotRadians('Elevation', elevation, 'Time', 'Angle [rad]',-2, 2)
barlabel='Number of Samples'
plotReflection(azimuth, elevation, 'Reflection Direction','Azimuth', 'Elevation',-4,4,-2,2,barlabel)

energy = energy_density(audioBFormat_stft,p0,c)
energybFormat = energy_density_bFormat(audioBFormat_stft,p0,c)

dt = 10
diffuseness = Diffuseness(I,energy,f,t,dt,c) 
plotSpectrogram('Diffuseness', diffuseness, 'plasma','Time', 'Energy', 'Diffuseness',0,1)

output_filename='medievalBFormat.wav'
audioBFormat=write_bFormat(output_filename,audioBFormat,fs)