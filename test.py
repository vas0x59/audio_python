import pyaudio
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.signal import butter, lfilter
import wave
#import cProfile
#cProfile.run('soundplot(stream)')

plt.rcParams["figure.figsize"] = 8,4

RATE = 44100
CHUNK = int(RATE/2) # RATE / number of updates per second
#Filter co-efficients 
nyq = 0.5 * RATE
low = 3000 / nyq
high = 6000 / nyq
b, a = butter(7, [low, high], btype='band')

#Figure structure
fig, (ax, ax2) =plt.subplots(nrows=2, sharex=True)
x = np.linspace(1, CHUNK, CHUNK)
extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]



def soundplot(stream):
    t1=time.time()
    data = np.array(np.fromstring(stream.read(CHUNK),dtype=np.int32))
    y1 = lfilter(b, a, data)
    ax.imshow(y1[np.newaxis, :], cmap="jet", aspect="auto")
    plt.xlim(extent[0], extent[1])
    plt.ylim(-50000000, 50000000)
    ax2.plot(x, y1)
    plt.pause(0.00001)
    plt.cla()  # which clears data but not axes
    y1 = []
    print(time.time()-t1)
import cProfile
#cProfile.run('soundplot(stream)')
if __name__=="__main__":
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paInt32,channels=1,rate=RATE,input=True,
                  frames_per_buffer=CHUNK)
    for i in range(RATE):
        cProfile.run('soundplot(stream)')
        #soundplot(stream)
    stream.stop_stream()
    stream.close()
    p.terminate()
