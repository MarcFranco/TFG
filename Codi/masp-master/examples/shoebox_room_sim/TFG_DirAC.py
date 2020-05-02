# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:35:56 2020

@author: Marc Franco Meca
"""


import numpy as np
import matplotlib.pyplot as plt 
import math
import time
import librosa
import scipy.signal as sig
import soundfile as sf
from scipy.io import wavfile
from masp import shoebox_room_sim as srs
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SETUP

# Room definition
room = np.array([10.2, 7.1, 3.2])

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
rec = np.array([ [4.5, 3.4, 1.5] ])
nRec = rec.shape[0]

# Source positions
src = np.array([ [6.2, 2.0, 1.8], [7.9, 3.3, 1.75] ])
nSrc = src.shape[0]

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

# In this case all the information (e.g. SH directivities) are already
# encoded in the echograms, hence they are rendered directly to discrete RIRs
fs = 48000
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

    
def plotReflection(azimuth, elevation, title, xlabel, ylabel,xvmin,xvmax,yvmin,yvmax,barlabel):
    az = np.empty(1)
    el = np.empty(1)    
    az = azimuth.flatten() 
    el = elevation.flatten() 
    nbins = [360, 180]
    H, xedges, yedges = np.histogram2d(az,el,bins=nbins)     
    H = np.rot90(H)
    H = np.flipud(H)     
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
                      
    plt.figure()
    plt.suptitle(title)   
    plt.pcolormesh(xedges,yedges,Hmasked)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(barlabel)
    plt.xlabel(xlabel)
    plt.xlim(xvmin,xvmax)
    plt.ylabel(ylabel)
    plt.ylim(yvmin,yvmax)
    
def convolution_audio_IRambisonics(mono_audio,sh_rirs):
    for i in range(sh_rirs[0,:,0,0].size):
        array = sig.fftconvolve(mono_audio, sh_rirs[:,i,0,0]) 
        if i == 0: #Initialize the variable stft
            convolved = np.empty((array.size,sh_rirs[0,:,0,0].size))    
        convolved[:,i] = array
    return convolved
#%%
fs, mono_audio = wavfile.read('C:\TFG\Codi\masp-master\sounds\medieval.wav')
az = -np.pi 
el = 0
#W,X,Y,Z,audioBFormat = encode_bFormat(mono_audio,az,el)  
audioBFormat= convolution_audio_IRambisonics(mono_audio,sh_rirs)


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
barlabel='Number of Samples'
plotReflection(azimuth, elevation, 'Reflection Direction','Azimuth', 'Elevation',-4,4,-2,2,barlabel)

energy = energy_density(audioBFormat_stft,p0,c)
energybFormat = energy_density_bFormat(audioBFormat_stft,p0,c)

dt = 10
diffuseness = Diffuseness(I,energy,f,t,dt) 
plotSpectrogram('Diffuseness', diffuseness, 'plasma','Time', 'Energy', 'Diffuseness',0,1)

output_filename='medievalBFormat.wav'
audioBFormat=write_bFormat(output_filename,audioBFormat,fs)