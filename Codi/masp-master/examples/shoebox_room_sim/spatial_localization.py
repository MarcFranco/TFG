# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:39:40 2020

@author: Marc Franco Meca
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from scipy.io import wavfile
import sys
sys.path.append("C:\TFG\Codi\masp-master\examples\shoebox_room_sim")
from functions import (compute_stft, direction_of_incidence,
                       Diffuseness,plot_spectrogram,plot_reflection)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
fs, IR = wavfile.read('C:\TFG\Codi\masp-master\sounds\ir_row_3l_centre_mid.wav')

temperature = 20
c = 331.4 + 0.6* temperature
p0 = 1.1839 # kg/m3
win_type = 'hann'
win_length = 256

f,t,audioBFormat_stft = compute_stft(IR,fs,win_type,win_length)

DOA, r, azimuth, elevation =direction_of_incidence(audioBFormat_stft,f,t,p0,c)

dt=10
diffuseness = Diffuseness(audioBFormat_stft,f,t,dt,p0,c)
plot_spectrogram('Diffuseness', diffuseness, 'plasma','Time [samples]', 'Energy [dB]', 'Diffuseness',0,1)

threshold=0.5  
plot_reflection(azimuth, elevation, diffuseness,threshold, 'Reflection Direction', 'Azimuth [rad]', 'Elevation [rad]', 'Number Samples')
