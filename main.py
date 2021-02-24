%%prun -s tottime -q -T prun1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.animation import FuncAnimation
from svgpathtools import svg2paths2

N_samples = 500 # Temporal resolution
N_vectors = 500 # Spectral bandwidth

file_path = "/home/wes/Downloads/WB_black.svg"

paths, attributes, svg_attributes = svg2paths2(file_path)
total_length = sum([path.length() for path in paths])
samples_per_path = [round(N_samples * path.length()/total_length) for path in paths]
shape = [path.point(i/samples_in_path) for path, samples_in_path in zip(paths, samples_per_path) for i in range(samples_in_path)]

dt = 1 / N_samples
t = np.arange(0, 1, dt)

freqs = np.arange(N_vectors // 2, -N_vectors // 2, -1)

# Make a matrix to represent the Fourier basis functions
#        (    freqs,         t)
# shape: (N_vectors, N_samples)
Fourier = np.exp(- 2 * np.pi * 1j * np.outer(freqs, t))

# Or basically any path drawing whatever you want
forme = np.exp(- 2 * np.pi * 1j * t) + 0.7*np.exp(+ 4 * np.pi * 1j * t)
forme = np.array(shape)

Fourier_coeffs = Fourier @ forme * dt

orbits = Fourier * Fourier_coeffs.reshape(N_vectors, 1)
cum_orbits = np.cumsum(Fourier * Fourier_coeffs.reshape(N_vectors, 1), axis=0)
cum_orbits = np.vstack([np.zeros((N_samples,)), cum_orbits])

radius = np.abs(orbits[:, 0])

fig, ax = plt.subplots(figsize=(10,10))
fig_lim = np.max(np.abs(cum_orbits))*1.05
plt.xlim(np.min(cum_orbits.real),np.max(cum_orbits.real))
plt.ylim(np.min(cum_orbits.imag),np.max(cum_orbits.imag))
plt.plot(forme.real, forme.imag)
pl, = plt.plot([], [])

draw_idx = [i for i, r in enumerate(radius) if r > 0.005*fig_lim]

def animate(frame):
    ax.patches = []
    c_B = cum_orbits[0, int(frame)]
    for i in range(1, N_vectors+1):
        c_A, c_B = c_B, cum_orbits[i, int(frame)]
        r = radius[i-1]
        if r > 0.005*fig_lim:
            ax.add_patch(plt.Circle((c_A.real, c_A.imag), r, fill=False, linestyle="dotted"))
            ax.add_patch(ConnectionPatch((c_A.real, c_A.imag), (c_B.real, c_B.imag), ax.transData, arrowstyle="-"))
            #ax.add_patch(plt.Arrow(c_A.real, c_A.imag, c_C.real, c_C.imag, width=0.3))
    pl.set_data(cum_orbits[-1,:int(frame)+1].real, cum_orbits[-1,:int(frame)+1].imag)
    return ax.patches+ [pl,]
    #return pl,

ani = FuncAnimation(fig, func=animate, frames=N_samples*t, interval=1, blit=True)
plt.show()