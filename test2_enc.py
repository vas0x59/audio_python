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
# from keras import   
from keras.models import load_model
import catboost
import threading

class SpectrumAnalyzer:
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 12000
    CHUNK = 150
    START = 0
    N = CHUNK
    thresh = 6
    wave_x = 0
    wave_y = 0
    spec_x = 0
    spec_y = 0
    data = []
    spectr = np.zeros((N, N//2))
    now_r = np.zeros(N//2)
    hist = np.zeros((500, 4))
    # qweqweqwe = []
    def predict(self, img, now_r):
        image = cv2.resize(img, (64, 64))/255.0
        image = np.expand_dims(image, -1)
        pred_x = self.encoder.predict(np.expand_dims(image,axis=0))[0]
        # now = self.now_r
        # print(pred_x, now_r)
        # data_exp = np.concatenate([pred_x, now_r], axis=0)
        pred = self.clf.predict(pred_x)[0][0]
        ## The_Show_Must_Go_On We_Are_The_Champions We_Will_Rock_You none
        # print()
        # pred = self.clf.predict_proba(pred_x)[0]
        return pred
        # to_nn = np.expand_dims(spectr[:, :]
    def load_models(self):
        # self.encoder = load_model('./encoder_model_12000_out2.h5')
        # self.clf = catboost.CatBoostClassifier()
        # self.clf.load_model('./catboost_decoder_12000_out2_exp_5.catboost')
        self.encoder = load_model('./encoder_model_12000_out2.h5')
        self.clf = catboost.CatBoostClassifier()
        self.clf.load_model('./catboost_decoder_12000_out2_6.catboost')
        # self.encoder = load_model('./encoder_model_2.h5')
        # self.clf = catboost.CatBoostClassifier()
        # self.clf.load_model('./catboost_decoder.catboost')
    def predict_th(self):
        self.load_models()
        dd = {"none":3, "The_Show_Must_Go_On":0, "We_Will_Rock_You":2, "We_Are_The_Champions":1}
        i = 0
        
        while True:
            self.pr = self.predict(self.img, self.now_r)
            # qw+=self.pr
            # now_r = self.pr
            # self.hist = np.roll(self.hist, 1, axis=0)
            # self.hist[0] = self.pr
            
            # st = ""
            # maxx = 0
            # maxx_i = 0
            # for i in dd:
            #     # st += str(i )
            #     # st += ":"
            #     val = self.hist[:, dd[i]].mean().round(5)
            #     # print(i)
            #     if val > maxx:
            #         maxx = val 
            #         maxx_i = i
            # st += str(maxx_i)
            # st += str()
            # st += " "
            # print(st)
            
            print(self.pr)

            dd[self.pr]+=1
            print(dd)


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
        
        # self.out = cv2.VideoWriter('Radioactive_150_20000_72_2.avi',fourcc, 72.0, (self.N, self.N), False)
        # Main loop
        self.img = (np.clip(self.spectr, 0, self.thresh)*(255/self.thresh)).astype('uint8')
        t = threading.Thread(target=self.predict_th)
        t.daemon = True
        t.start()
        self.loop()

    def loop(self):
        try:
            while True :
                self.data = self.audioinput()
                self.fft()
                # self.shift_arr()
                # self.file.write(str(np.round(self.spec_y, decimals=3)))
                # now_r = np.round(self.spec_y, decimals=)
                self.now_r = self.spec_y[:self.N//2]
                self.spectr = np.roll(self.spectr, 1, axis=0)
                self.spectr[0] = self.now_r
                self.img = (np.clip(self.spectr, 0, self.thresh)*(255/self.thresh)).astype('uint8')
                # self.out.write(img)
                # print(self.spectr.shape)
                
                
                cv2.imshow("img", cv2.resize(self.img, (256, 512), interpolation = cv2.INTER_AREA) )
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
