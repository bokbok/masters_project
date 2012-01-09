import numpy as np
import matplotlib.pyplot as plt
from numpy import exp
from numpy import sqrt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

from scipy.fftpack import fft

__author__="matt"
__date__ ="$05/01/2012 11:28:23 PM$"

if __name__ == "__main__":
    print "Hello World"
    data = np.genfromtxt('c:\\temp\\run.dat', dtype=None, delimiter=':')
    h_e = []
    t = []

    freq = np.linspace(0, 1e5, 1e5)
    
    for item in data:
       #h_e.append( (item[4], item[5]) )
       t.append( item[0] )
       h_e.append( item[4] )
    
#    plt.plot(t, h_e)
#    plt.xlabel('t')
#    plt.ylabel('h_e')
    
    plt.xlabel('f')
    plt.ylabel('H_e')
    plt.plot(t, fft(h_e))
    
    plt.axis('normal')
    plt.title('Blah')
    plt.show()    