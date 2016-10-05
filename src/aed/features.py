'''
Copyright (C) 2015 Axel Plinge and Rene Grzeszick. 
This file made available under the terms of the MIT license (see the LICENSE file).
For any publication please cite the corresponding paper (see the README file).
'''

import soundfile
import numpy as np
from numpy import log, exp, sqrt, cos
from math import pow, log10

class MFCCCalculator(object):
    '''    
    Adapted from the Vampy example plugin "PyMFCC" by Gyorgy Fazekas
    http://code.soundsoftware.ac.uk/projects/vampy/repository/entry/Example%20VamPy%20plugins/PyMFCC.py
    '''

    def __init__(self, sample_rate, frame_length):
        ''' 
        Initialize MFCC Calculator.
        @param sample_rate: audio sample rate
        @param input_size: length of magnitude spectrum (half of FFT size assumed)            
        '''
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0
        self.min_hz = 0
        self.max_hz = self.nyquist
        self.input_size = frame_length / 2
        self.num_bands = 40
        self.window = np.hamming(frame_length)
        self.filter_matrix = None
        return

    def __precalculate(self):
        '''
        '''
        self.filter_matrix = self.__get_filter_matrix()
        self.DCT_matrix = self.__get_DCT_matrix(self.num_bands)
        return

    def __get_filter_centres(self):
        '''
        Calculate Mel filter centers around FFT bins.
        '''
        max_mel = 1000 * log(1 + self.max_hz / 700.0) / log(1 + 1000.0 / 700.0)
        min_mel = 1000 * log(1 + self.min_hz / 700.0) / log(1 + 1000.0 / 700.0)
        centers_mel = np.array(range(self.num_bands + 2)) * (max_mel - min_mel) / (self.num_bands + 1) + min_mel
        centers_bin = np.floor(0.5 + 700.0 * self.input_size * (exp(centers_mel * log(1 + 1000.0 / 700.0) / 1000.0) - 1) / self.nyquist)
        return np.array(centers_bin, int)

    def __get_filter_matrix(self):
        '''
        Compose the Mel scaling matrix.
        '''
        filter_matrix = np.zeros((self.num_bands, self.input_size))
        self.filter_centres = self.__get_filter_centres()
        for i in range(self.num_bands):
            start, centre, end = self.filter_centres[i:i + 3]
            self._set_filter(filter_matrix[i], start, centre, end)
        return filter_matrix.transpose()

    def _set_filter(self, filt, filter_start, filter_centre, filter_end):
        '''
        Calculate a single Mel filter.
        '''
        filt[filter_start:filter_centre] = (np.array(range(filter_start, filter_centre)) - filter_start) / np.float(filter_centre - filter_start)
        filt[filter_centre:filter_end] = (filter_end - np.array(range(filter_centre, filter_end))) / np.float(filter_end - filter_centre)
        return

    def __warp_spectrum(self, magnitude_spectrum):
        '''
        Compute the Mel scaled spectrum.
        '''
        return np.dot(magnitude_spectrum, self.filter_matrix)

    def __get_DCT_matrix(self, size):
        '''
        Calculate the square DCT transform matrix. 
        '''
        DCTmx = np.array(range(size), np.float).repeat(size).reshape(size, size)
        DCTmxT = np.pi * (DCTmx.transpose() + 0.5) / size
        DCTmxT = (1.0 / sqrt(size / 2.0)) * cos(DCTmx * DCTmxT)
        DCTmxT[0] = DCTmxT[0] * (sqrt(2.0) / 2.0)
        return DCTmxT

    def __dct(self, data_matrix):
        '''
        Compute DCT of input matrix.
        '''
        return np.dot(self.DCT_matrix, data_matrix)

    def get_features(self, chunk):
        ''' 
        Compute MFCCs from audio data 
        '''
        if self.filter_matrix is None:
            self.__precalculate()

        framelen = 2 * self.input_size
        framespectrum = np.fft.fft(self.window * chunk)
        magspec = abs(framespectrum[:framelen / 2])
        melspec = self.__warp_spectrum(magspec)

        # empty frames
        # most likely caused by an xrun
        # will be filled with minimum dc
        if np.min(melspec) < 1e-96:
            melspec[melspec < 1e-96] = 1e-96

        melceps = self.__dct(np.log(melspec))
        return melceps[1:14]


class GFCCCaclulator(MFCCCalculator):
    '''        
    Gammatone Frequency Cepstral Coefficients
    
    cp. also
    
    - B. Glasberg und B. Moore
     Derivation of auditory filter shapes from notched-noise data.
     Hearing Research, 47(1-2):103-138, August 1990. 
    - R. Patterson, I. Nimmo-Smith, J. Holdsworth und P. Rice
     An efficient auditory filterbank based on the gammatone functions. 
     Tech.Rep. APU Report 2341, MRC, Applied Psychology Unit, Cambridge U.K, 1988.
    - M. Slaney: An efficient implementation of the Patterson-Holdsworth auditory filter bank.
     Tech.Rep. 35, Apple Computer, Inc., 1993. 
    - M. Unoki and M. Akagi
     A method of signal extraction from noisy signal based on auditory scene analysis
     Speech Commun., vol. 27, no. 3, pp. 261-279, 1999.    
    '''
    _ERB_C1 = 24.673
    _ERB_C2 = 4.368
    _ERB_C3 = 21.366

    def _hz2erb(self, freq):
        ''' 
        Convert Hz to ERB
        '''
        return (self._ERB_C3 * log10((self._ERB_C2 * freq / 1000.0) + 1.0))

    def _erb2hz(self, erb):
        ''' 
        Convert ERB to Hz 
        '''
        return 1000.0 * (pow(10.0, (erb / self._ERB_C3)) - 1.0) / self._ERB_C2

    def get_filter_centres(self, input_size, num_bands=128):
        centers = []
        erb_min = self._hz2erb(300)
        erb_range = self._hz2erb(9000) - erb_min
        for i in range(-1, num_bands + 1):
            f_i = self._erb2hz(erb_min + (erb_range * i) / float(num_bands - 1))
            centers.append((f_i * self.inputSize) / self.sample_rate)
        return np.array(centers, int)

    def amplitude(self, f, fc, b):
        d = (fc - f)
        n1 = pow(b, 4.0)
        d1 = pow(b, 4.0) - 6.0 * pow(b, 2.0) * pow(d, 2.0) + pow(d, 4.0)
        d2 = 4.0 * (b * pow(d, 3.0) - pow(b, 3.0) * d)
        a = n1 * pow((pow(d1, 2.0) + pow(d2, 2.0)), -0.5)
        return a

    def bandwidthGlasbergMoore(self, fc):
        return  24.673 * ((4.368 * 1e-3 * fc) + 1.0)

    def _set_filter(self, filt, filter_start, filter_centre, filter_end):
        '''
        Calculate a single Gammatone filter.
        '''
        fc = filter_centre * self.sample_rate / self.input_size
        wc = self.bandwidthGlasbergMoore(fc)
        for tap in  range(self.input_size):
            f = tap * self.sample_rate / self.input_size
            filt[tap] = self.amplitude(f, fc, wc)
        return


class LoudnessCalculator(object):
    '''
    Perceptual Loudness
    '''

    def __init__(self, samplerate, framelen):
        '''
        Initialize perceptual loudness using samplerate and framelength
        '''
        self.framelen = framelen
        self.window = np.hamming(framelen)
        self.filter = np.zeros(framelen / 2)
        for i in xrange(framelen / 2):
            self.filter[i] = self.__weightA(float(i * samplerate) / framelen)
        return


    def __weightA(self, f):
        '''
        A - weighting 
        '''
        return  (12200 * 12200 * f * f * f * f) / ((f * f + 20.62 * 20.62) * (f * f + 12200 * 12200) * np.sqrt(f * f + 107.7 * 107.7) * np.sqrt(f * f + 737.9 * 737.9))


    def loudness(self, frame):
        '''
        Computes loundness feature
        '''
        framespectrum = np.fft.fft(self.window * frame)
        magspec = abs(framespectrum[:self.framelen / 2])
        # apply A weighting
        filterd = np.dot(magspec, self.filter)
        return np.sum(filterd)


    def energy(self, frame):
        '''
        Computes energy feature
        '''
        return np.log10(np.mean(np.abs(frame)) + 1e-6)


class FeatureNormalizer(object):
    ''' 
    Basic whitening 
    '''

    def __init__(self):
        self.means = None
        self.invstds = None
        return


    def setup(self, data_mat):
        '''
        Estimated the mean and stddev of the given data
        These will later on be used for normalization.
        '''
        self.means = np.mean(data_mat, 0)
        self.invstds = np.std(data_mat, 0)
        for i, val in enumerate(self.invstds):
            if val == 0.0:
                self.invstds[i] = 1.0
            else:
                self.invstds[i] = 1.0 / val
        return


    def normalize(self, data):
        '''
        Normalizes data using the mean and stdev of the training data.
        '''
        return (data - self.means) * self.invstds


class FeatureCalculator(object):
    '''
    Computes Loudness, MFCC and GFCC for an audio file. 
    '''

    def __read(self, filename):
        '''
        Read audio data.
        '''
        data, samplerate = soundfile.read(filename)
        return data, samplerate

    def compute(self, filename, channels=None):
        '''
        Computes features (Loudness, MFCC and GFCC) for the given filename
        @param filename: Path to the audiofile
        @param channels: 1D numpy array of channel indices to use.
                         If None, all channels are used. (default=None)
        @return: features, framerate
        '''
        self.framelength = 1024
        data, samplerate = self.__read(filename)
        if len(data.shape) == 1:
            # Add second axis to one-channel data
            # data should be in shape [samples, channels]
            data = data[np.newaxis].T

        if channels is None:
            channels = np.arange(data.shape[1])

        mfcc = MFCCCalculator(samplerate, self.framelength)
        gfcc = GFCCCaclulator(samplerate, self.framelength)
        loud = LoudnessCalculator(samplerate, self.framelength)

        numsamples = data.shape[0]
        slices = np.arange(0, numsamples - self.framelength, self.framelength / 2)
        result = np.zeros((channels.shape[0], len(slices), 1 + 13 + 13), dtype=np.float)
        for i, channel in enumerate(channels):
            for frameindex, sampleindex in enumerate(slices):
                chunk = data[sampleindex:sampleindex + self.framelength, channel]
                loudn = loud.loudness(chunk)
                mfccs = mfcc.get_features(chunk)
                gfccs = gfcc.get_features(chunk)
                result[i, frameindex, 0] = loudn
                result[i, frameindex, 1:14] = mfccs
                result[i, frameindex, 14:] = gfccs
        return result, float(samplerate) / (self.framelength / 2), samplerate

