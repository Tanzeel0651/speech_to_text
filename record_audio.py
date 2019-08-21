from sys import byteorder
from array import array
from struct import pack
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from pysndfx import AudioEffectsChain
import python_speech_features

import pyaudio
import wave

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

def reduce_noise_mfcc_up(y, sr):

    hop_length = 512
    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(frequency=min_hz*(-1), gain=12.0, slope=0.5)#.highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.5)#.limiter(gain=8.0)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 10000        #16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def trim_silence(y):
    y_trimmed, index = librosa.effects.trim(y, top_db=20, frame_length=2, hop_length=500)
    trimmed_length = librosa.get_duration(y) - librosa.get_duration(y_trimmed)

    return y_trimmed, trimmed_length


def write_file(file_name):
    file_name = file_name+'.wav'
    data, sr = librosa.load('audio/'+file_name)
    y = reduce_noise_mfcc_up(data, sr)
    y_reduced_mfcc_up, time_trimmed = trim_silence(y)

    destination = 'audio/'+file_name
    librosa.output.write_wav(destination, y_reduced_mfcc_up, RATE)

def check(file_name):
    for line in open('transcribed.txt'):
        if line.strip() == file_name: return True
    return False
    
if __name__ == '__main__':
    for line in open('transcribe.txt'):
        #print(line)
        line = line.strip()
        if check(line):
            print('reached')
            continue
        print("Speak \'{}\'".format(line))
        record_to_file('{}.wav'.format('audio/'+line))
        #write_file('{}'.format(line))
        print("done - result written to {}.wav".format(line))
        with open('transcribed.txt', 'a') as f:
            f.write(line+'\n')
# if __name__ == '__main__':
#     line = input('Enter file name')
#     record_to_file('{}.wav'.format(line))


# import pyaudio
# import math
# import struct
# import wave
# import os
# import sys
# #Assuming Energy threshold upper than 30 dB
# Threshold = 30

# SHORT_NORMALIZE = (1.0/32768.0)
# chunk = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# swidth = 2
# Max_Seconds = 10
# TimeoutSignal=((RATE / chunk * Max_Seconds) + 2)
# silence = True
# path = 'audio/'
# Time=0
# all =[]

# def GetStream(chunk):
#     return stream.read(chunk)
# def rms(frame):
#     count = len(frame)/swidth
#     format = "%dh"%(count)
#     shorts = struct.unpack( format, frame )

#     sum_squares = 0.0
#     for sample in shorts:
#         n = sample * SHORT_NORMALIZE
#         sum_squares += n*n
#         rms = math.pow(sum_squares/count,0.5);

#         return rms * 1000

# def WriteSpeech(WriteData, file_name):
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     wf = wave.open(path+file_name, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(WriteData)
#     wf.close()

# def KeepRecord(TimeoutSignal, LastBlock, file_name):
#     all.append(LastBlock)
#     for i in range(0, TimeoutSignal):
#         try:
#             data = GetStream(chunk)
#         except:
#             continue
#         #I chage here (new Ident)
#         all.append(data)

#     print("end record after timeout");
#     data = ''.join(all)
#     print ("writing to file {}".format(file_name));
#     WriteSpeech(data, file_name)
#     silence = True
#     Time=0    

# def listen(silence,Time, file_name):
#     print("Speak {}".format(file_name))
#     while silence:
#         try:
#             input = GetStream(chunk)
#         except:
#             continue
#         rms_value = rms(input)
#         if (rms_value > Threshold):
#             silence=False
#             LastBlock=input
#             print ("Recording....")
#             KeepRecord(TimeoutSignal, LastBlock, file_name)
#         Time = Time + 1
#         if (Time > TimeoutSignal):
#             print("Time Out No Speech Detected")
#             sys.exit()

# p = pyaudio.PyAudio()

# for line in open('transcribe.txt'):
#     file_name = line.strip()
#     stream = p.open(format = FORMAT,
#         channels = CHANNELS,
#         rate = RATE,
#         input = True,
#         output = True,
#         frames_per_buffer = chunk)
#     listen(silence,Time,file_name)

