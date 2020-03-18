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
from scipy import signal
import scipy.signal as sig
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

sourcepath = '../../data/milk_cow_blues_4src.wav'
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

# Inverse filter
def inverse_filter(finf,fsup,T,t,x):
    w1 = 2*np.pi*finf
    w2 = 2*np.pi*fsup   
    #Haciendo esto sale una respuesta impulsional pero el filtro no compensa el señal original
    #kend = -6*np.log2(w2/w1)    
    #k = np.exp(t*kend/T)
    R= np.log(w2/w1)
    k = np.exp(t*R/T)
    
    inv = x[::-1]/k
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

finf = 10
fsup = 22000
T = 3
fs = 44100
t = np.arange(0,T*fs)/fs


sinesweep=log_sinesweep(finf,fsup,T,t,fs)
inversefilter=inverse_filter(finf,fsup,T,t,sinesweep)
measuredsignal = sig.fftconvolve(log_sinesweep,inversefilter,mode='same')

impulseresponse = sig.fftconvolve(sinesweep, inversefilter, mode='same')

# Get spectra of all signals
freq, sinesweepdB = spectrumDBFS(sinesweep, fs)
freq, inversefilterdB = spectrumDBFS(inversefilter, fs)
freq, impulseresponsedB = spectrumDBFS(impulseresponse, fs)

fig, axs = plt.subplots(3, 1, figsize=(7,7))
axs[0].set_title('Logarithmic SineSweep x(t)')
axs[0].plot(t, sinesweep)
Pxx, freqs, bins, im = axs[1].specgram(sinesweep, 1024, fs, noverlap=900)
axs[2].semilogx(freq, sinesweepdB)

fig, axs = plt.subplots(3, 1, figsize=(7,7))
axs[0].set_title('Inverse filter f(t)')
axs[0].plot(t, inversefilter)
Pxx, freqs, bins, im = axs[1].specgram(inversefilter, 1024, fs, noverlap=900)
axs[2].semilogx(freq, inversefilterdB)

fig, axs = plt.subplots(2, 1, figsize=(7,7))
axs[0].set_title('Impulse Response')
axs[0].plot(t, impulseresponse)
Pxx, freqs, bins, im = axs[1].specgram(impulseresponse, 1024, fs, noverlap=900)
axs[2].semilogx(freq, impulseresponsedB)

plt.figure()
plt.grid()
plt.semilogx(freq, sinesweepdB, label='Log. SineSweep')
plt.semilogx(freq, inversefilterdB, label='Inverse filter')
plt.semilogx(freq, impulseresponsedB, label='Impulse Response')
plt.title('Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dBFS]')
plt.legend()
plt.show()
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


