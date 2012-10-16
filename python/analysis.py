from numpy.fft import *
from pylab import plot, figure, xlabel, ylabel

class FFT:
    def __init__(self, data, deltaT, axis):
        self.data = data
        self.deltaT = deltaT

    def compute(self):
        self.computed = fft(self.data)[1:self.data.size/2]
        print "DeltaT " + str(self.deltaT)
        self.freqAxis = fftfreq(self.data.size, d=self.deltaT)[1:self.data.size/2]

        return self

    def display(self, fig ="7"):
        figure(fig)
        plot(self.freqAxis, self.computed,'r')
        xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')

        return self

