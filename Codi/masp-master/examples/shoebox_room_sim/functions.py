# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:38:25 2020

@author: Marc Franco Meca
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# # # # # # # # # # # # # # # # # # # # # # # 
#                                           #
#               Import Libraries            #
#                                           #
# # # # # # # # # # # # # # # # # # # # # # # 

import numpy as np
import matplotlib.pyplot as plt 
import math
from scipy import signal
import scipy.signal as sig
import soundfile as sf
from scipy.signal import butter, lfilter, hilbert
from scipy import stats


# # # # # # # # # # # # # # # # # # # # # # # 
#                                           #
#               Deconvolution               #
#                                           #
# # # # # # # # # # # # # # # # # # # # # # # 

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

# # # # # # # # # # # # # # # # # # # # # # # 
#                                           #
#            Acoustics Parameters           #
#                                           #
# # # # # # # # # # # # # # # # # # # # # # # 
    
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

def envelope_plot(impulse,env,index,centerfreqs,fs):   
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
       
def revTime60(data,band_centerfreqs,window,rt_type,oct_type,fs):    
    band_type, low, high = check_type(oct_type,band_centerfreqs) 
    begin,end,factor=RT_estimator(rt_type)
    rt60 = np.zeros(band_centerfreqs.size)  
    
    for band in range(band_centerfreqs.size):
        
        # Filtering signal w/butterworth & hilbert 
        filtered_signal = butterworth_bandpass_filter(data, low[band], high[band], fs, order=3)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))    
        amplitude_envelope = np.abs(hilbert(filtered_signal)) 
        amplitude_envelope = amplitude_envelope/np.max(amplitude_envelope)
        envelope_plot(abs_signal,amplitude_envelope,band,band_centerfreqs,fs)
        
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
    
    print_rt60(rt60,band_centerfreqs)    
    return rt60

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

def print_c50(c50,centerfreqs):   
    print('----Speech Clarity (C50)----')
    for i in range (0,c50.size):
        print(repr(centerfreqs[i]) +'Hz : ' + repr(c50[i]) )

 # Speech Clarity (C50) Objective: >2dB
def speechClarity50(data,band_centerfreqs,oct_type,fs):    
    band_type, low, high = check_type(oct_type,band_centerfreqs) 
    C50 = np.zeros(band_centerfreqs.size)  

    for band in range(band_centerfreqs.size):
        filtered_signal = butterworth_bandpass_filter(data, low[band], high[band], fs, order=3)
        p2 = filtered_signal**2.0
        t = int(0.05*fs)
        C50[band] = 10.0 * np.log10((np.sum(p2[:t]) / np.sum(p2[t:])))
    print_c50(C50,band_centerfreqs)
    return C50

def print_d50(d50,centerfreqs):   
    print('----Definition (D50)----')
    for i in range (0,d50.size):
        print(repr(centerfreqs[i]) +'Hz : ' + repr(d50[i]) )
        
# Definition (D50) Objective: 0.4 - 0.6
def Definition(C50,band_centerfreqs):    
    D50 = np.zeros(band_centerfreqs.size)  
    for band in range(band_centerfreqs.size):
        D50[band] = 1 / (1 + 10**-(C50[band]/10)) 
    print_d50(D50,band_centerfreqs)
    return D50

def print_Smid(Smid,centerfreqs):   
    print('----Speech Sound Level (Smid)----')
    for i in range (0,Smid.size):
        print(repr(centerfreqs[i]) +'Hz : ' + repr(Smid[i]) )

# Strength (S)  Objective: (4 <= Smid(0º) <= 8) and (2 <= Smid(90º) <= 6)
def SpeechSoundLevel(IR, IR10m,Lw,band_centerfreqs,oct_type,fs):
    
    band_type, low, high = check_type(oct_type,band_centerfreqs) 
    Smid = np.zeros(band_centerfreqs.size)  
    
    for band in range(band_centerfreqs.size):    
        filtered_signal_IR = butterworth_bandpass_filter(IR, low[band], high[band], fs, order=3)
        p2_IR = filtered_signal_IR**2.0  
        filtered_signal_IR10m = butterworth_bandpass_filter(IR10m, low[band], high[band], fs, order=3)
        p2_IR10m = filtered_signal_IR10m**2.0            
        Smid[band] = 10 * np.log10(np.sum(p2_IR) / np.sum(p2_IR10m))
    print_Smid(Smid,band_centerfreqs)
    return Smid

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
    print_ALCons(Cons,band_centerfreqs)
    return Cons

def Air_Coefficient(nBands,humidity,band_centerfreqs):
    m=[]
    for a in range (0,nBands):
        air_coef=5.5*(10**-4)*(50/humidity)*((band_centerfreqs[a]/1000)**1.7) # Air coefficient
        m.append(air_coef)
    return m

def Parametric_Reverberation_Time(nBands,V,c,humidity,band_centerfreqs,abs_coef,width,height,depth,alpha,S):    
    RT60=[]
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
    return RT60

def NoiseCriteria(oct_levels):    
    ncCURVES = {
    15: np.array([47.0, 36.0, 29.0, 22.0, 17.0, 14.0, 12.0, 11.0]),
    20: np.array([51.0, 40.0, 33.0, 26.0, 22.0, 19.0, 17.0, 16.0]),
    25: np.array([54.0, 44.0, 37.0, 31.0, 27.0, 24.0, 22.0, 21.0]),
    30: np.array([57.0, 48.0, 41.0, 35.0, 31.0, 29.0, 28.0, 27.0]),
    35: np.array([60.0, 52.0, 45.0, 40.0, 36.0, 34.0, 33.0, 32.0]),
    40: np.array([64.0, 56.0, 50.0, 45.0, 41.0, 39.0, 38.0, 37.0]),
    45: np.array([67.0, 60.0, 54.0, 49.0, 46.0, 44.0, 43.0, 42.0]),
    50: np.array([71.0, 64.0, 58.0, 54.0, 51.0, 49.0, 48.0, 47.0]),
    55: np.array([74.0, 67.0, 62.0, 58.0, 56.0, 54.0, 53.0, 52.0]),
    60: np.array([77.0, 71.0, 67.0, 63.0, 61.0, 59.0, 58.0, 57.0]),
    65: np.array([80.0, 75.0, 71.0, 68.0, 66.0, 64.0, 63.0, 62.0]),
    70: np.array([83.0, 79.0, 75.0, 72.0, 71.0, 70.0, 69.0, 68.0])
    }
    ncRange=np.arange(15, 71, 5)
    for nc in ncRange:
        curve = ncCURVES.get(nc)
        if (round(max(oct_levels))<= curve[oct_levels.index(max(oct_levels))]):
            break
    print('----Noise Criteria (NC) Curves----')
    print ("NC "+str(nc))
    return nc

# # # # # # # # # # # # # # # # # # # # # # # 
#                                           #
#     Directional Audio Coding (DirAC)      #
#                                           #
# # # # # # # # # # # # # # # # # # # # # # # 
 
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

def direction_of_incidence_bFormat(audioBFormat_stft,I,f,t):    
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

def Diffuseness(I,energy,f,t,dt,c):
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
