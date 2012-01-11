import numpy as np
import matplotlib.pyplot as plt
from numpy import exp
from numpy import sqrt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from scipy import *

from scipy.fftpack import rfft

__author__="matt"
__date__ ="$05/01/2012 11:28:23 PM$"

if __name__ == "__main__":
    print "Hello World"
    data = np.genfromtxt('../../results/run.dat', dtype=float, delimiter=':')
    h_e = []
    t = []

    n = 100000
    nyquist=1./2
    
    for item in data:
       #h_e.append( (item[4], item[5]) )
       if item[0] >2:
           t.append( item[0] )
           h_e.append( item[5] )
    
#    plt.plot(t, h_e)
#    plt.xlabel('t')
#    plt.ylabel('h_e')
    print h_e[0], h_e[1]
    as_fft = log(abs(rfft(h_e))/10)
    #as_fft[0] = 0
    plt.xlabel('f')
    plt.ylabel('H_e')
    plt.plot(np.fft.fftfreq(len(h_e)*2, d=1e-5), as_fft)
    
    plt.axis('normal')
    plt.title('Blah')
    plt.show()    