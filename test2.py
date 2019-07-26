#!/usr/bin/env python
# encoding: utf-8

## Module infomation ###
# Python (3.4.4)
# numpy (1.10.2)
# PyAudio (0.2.9)
# matplotlib (1.5.1)
# All 32bit edition
########################

import numpy as np
import pyaudio
import cv2
import matplotlib.pyplot as plt

class SpectrumAnalyzer:
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    CHUNK = 256
    START = 0
    N = CHUNK
    thresh = 7
    wave_x = 0
    wave_y = 0
    spec_x = 0
    spec_y = 0
    data = []
    spectr = np.zeros((N, N))
    # qweqweqwe = []

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format = self.FORMAT,
            channels = self.CHANNELS, 
            rate = self.RATE, 
            input = True,
            output = False,
            frames_per_buffer = self.CHUNK)
        # self.file = open("test.csv", "w")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('We_Are_The_Champions_16000_72_out_3123.avi',fourcc, 72.0, (self.N, self.N), False)
        # Main loop
        self.loop()

    def loop(self):
        try:
            while True :
                self.data = self.audioinput()
                self.fft()
                # self.shift_arr()
                # self.file.write(str(np.round(self.spec_y, decimals=3)))
                # now_r = np.round(self.spec_y, decimals=)
                now_r = self.spec_y
                self.spectr = np.roll(self.spectr, 1, axis=0)
                self.spectr[0] = now_r
                img = (np.clip(self.spectr, 0, self.thresh)*(255/self.thresh)).astype('uint8')
                self.out.write(img)
                cv2.imshow("img", cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA) )
                cv2.waitKey(1)
                # self.qweqweqwe.append(max(now_r))
                # print(max( self.qweqweqwe))
                # print(np.round(self.spec_y, decimals=3))
                # self.graphplot()

        except KeyboardInterrupt:
            self.pa.close(self.stream)

        print("End...")
        self.out.release()

    def audioinput(self):
        ret = self.stream.read(self.CHUNK)
        ret = np.fromstring(ret, np.float32)
        return ret

    def fft(self):
        self.wave_x = range(self.START, self.START + self.N)
        self.wave_y = self.data[self.START:self.START + self.N]
        self.spec_x = np.fft.fftfreq(self.N, d = 1.0 / self.RATE)  
        y = np.fft.fft(self.data[self.START:self.START + self.N])    
        # self.spec_y = y.real
        self.spec_y = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in y]

    def graphplot(self):
        plt.clf()
        # wave
        plt.subplot(311)
        plt.plot(self.wave_x, self.wave_y)
        plt.axis([self.START, self.START + self.N, -0.5, 0.5])
        plt.xlabel("time [sample]")
        plt.ylabel("amplitude")
        #Spectrum
        plt.subplot(312)
        plt.plot(self.spec_x, self.spec_y, marker= 'o', linestyle='-')
        plt.axis([0, self.RATE / 2, 0, 50])
        plt.xlabel("frequency [Hz]")
        plt.ylabel("amplitude spectrum")
        #Pause
        plt.pause(.01)

if __name__ == "__main__":
    spec = SpectrumAnalyzer()
