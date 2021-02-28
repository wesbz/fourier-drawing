import re
import argparse
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.animation import FuncAnimation

from svgpathtools import Path, Document

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
if __name__=="__main__":
    
    def color_parser(color):
        match = re.match("#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$", color)
        if not match.group():
            raise ValueError(f"color {color} is invalid.")
        
        return match.group(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path",
                        help="Path of the image to use.",
                        type=str)
    
    parser.add_argument("N_points",
                        help="Number of points to sample from the image.",
                        default=100,
                        type=int)
    parser.add_argument("N_vectors",
                        help="Number of Fourier components to use (frequencies in [-N_vectors//2 + 1; N_vectors//2])",
                        default=100,
                        type=int)
    
    parser.add_argument("--threshold_path_length",
                        help="Threshold for path length below which paths are not considered (to avoid artifacts).",
                        default=1e-3,
                        type=float)
    parser.add_argument("--threshold_vectors_module",
                        help="Threshold for vectors module below which vectors are not drawn (for speed and visibility contraints).",
                        default=1e-2,
                        type=float)
    
    parser.add_argument("--plot",
                        help="Plot the animation with matplotlib.",
                        action="store_true")
    parser.add_argument("--interval",
                        help="Interval between two frames (in milliseconds).",
                        default=1,
                        type=int)

    # Video related arguments
    parser.add_argument("-v", "--video",
                        help="Generate a video output and saves it at given path.",
                        type=str)
    parser.add_argument("--fps",
                        help="Frame rate for the generated video.",
                        default=24,
                        type=int)
    parser.add_argument("--dpi",
                        help="Dots per inch in the generated video.",
                        default=100,
                        type=int)
    parser.add_argument("--width",
                        help="Nb of pixels in width in the generated video.",
                        type=int)
    parser.add_argument("--height",
                        help="Nb of pixels in height in the generated video.",
                        type=int)
    # Layout related arguments
    ##  Background
    parser.add_argument("--background_color",
                        help="Color of the background in hex format (e.g. '#FFFFFF' or None for transparent).",
                        default="#ffffff",
                        type=color_parser)
    
    ##  Line
    parser.add_argument("--line_color",
                        help="Color of the drawn line in hex format (e.g. '#FFFFFF' or '#FFFFFFFF' with alpha channel).",
                        default="#1f77b4",
                        type=color_parser)
    parser.add_argument("--line_width",
                        help="Width of the drawn line.",
                        default=0.8,
                        type=float)
    parser.add_argument("--line_decay",
                        help="Decay speed of line opacity.")
    
    ##  Circles
    parser.add_argument("--circle_color",
                        help="Color of the circles in hex format (e.g. '#FFFFFF' or '#FFFFFFFF' with alpha channel).",
                        default="#000000",
                        type=color_parser)
    parser.add_argument("--circle_width",
                        help="Width of the circles.",
                        default=0.5,
                        type=float)
    
    ##  Vectors
    parser.add_argument("--vector_color",
                        help="Color of the vectors in hex format (e.g. '#FFFFFF' or '#FFFFFFFF' with alpha channel).",
                        default="#000000",
                        type=color_parser)
    parser.add_argument("--vector_width",
                        help="Width of the vectors.",
                        default=0.5,
                        type=float)
    
    args = parser.parse_args()
    
    file_path = "/home/wes/Downloads/output_ju_2.svg"
    paths = get_paths(args.image_path)
    shape = order_paths(paths)
    orbits, cum_orbits = compute_fourier_serie(shape)
    ani = plot_fourier(orbits, cum_orbits)
    
    if (args.width == None) ^ (args.height == None):
        raise ValueError("Specify BOTH width and height for the plot.")
    
    if args.video:
        with tqdm(total=args.N_points) as pbar:
            if args.video.split(".")[-1] == "mp4":
                ani.save(args.video, fps=args.fps, progress_callback=lambda x, y: pbar.update(1), writer="ffmpeg", dpi=args.dpi, extra_args=["-threads", "0"])
        
            elif args.video.split(".")[-1] == "gif":
                ani.save(args.video, fps=args.fps, progress_callback=lambda x, y: pbar.update(1), writer="pillow", dpi=args.dpi, extra_args=["-threads", "0"])

    if args.plot:
plt.show()