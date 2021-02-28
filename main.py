import re
import argparse
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.animation import FuncAnimation

from svgpathtools import Path, Document


def get_paths(file_path):
    doc = Document(file_path)
    paths = doc.paths()
    paths = [path for path in paths if path.length()>args.threshold_path_length]
    return paths

def order_paths(paths):
    total_length = sum([path.length() for path in paths])
    samples_per_path = [round(args.N_points * path.length()/total_length) for path in paths]

    # Order the shapes in each path in paths
    for i, path in enumerate(paths):
        tmp = Path(path.pop(0))
        while len(path) > 1:
            curr_end = tmp[-1].end
            next_start = np.argmin(np.abs([next_path.start-curr_end for next_path in path]))
            tmp.append(path.pop(next_start))
        tmp.append(path[0])
        paths[i] = tmp

    # Sample points along the paths
    shape = [[path.point(i/samples_in_path) for i in range(samples_in_path) if samples_in_path > 0] for path, samples_in_path in zip(paths, samples_per_path)]
    
    # Order the paths
    tmp = shape.pop(0)
    while len(shape) > 0:
        curr_end = tmp[-1]
        next_start = np.argmin(np.abs([path[0]-curr_end for path in shape]))
        tmp.extend(shape.pop(next_start))

    # Center and normalize the points
    shape = np.conjugate(tmp)
    shape -= np.mean(shape)
    shape /= np.max(np.abs(shape))
    
    return shape


def compute_fourier_serie(shape):
    N_samples = len(shape)
    dt = 1 / N_samples
    t = np.arange(0, 1, dt)
    freqs = np.arange(args.N_vectors // 2, -args.N_vectors // 2, -1)

    # Make a matrix to represent the Fourier basis functions
    #        (    freqs,         t)
    # shape: (N_vectors, N_samples)
    Fourier = np.exp(- 2 * np.pi * 1j * np.outer(freqs, t))

    # Or basically any path drawing whatever you want
    forme = np.array(shape)

    Fourier_coeffs = Fourier @ forme * dt

    orbits = Fourier * Fourier_coeffs.reshape(args.N_vectors, 1)
    cum_orbits = np.cumsum(Fourier * Fourier_coeffs.reshape(args.N_vectors, 1), axis=0)
    cum_orbits = np.vstack([np.zeros((args.N_points,)), cum_orbits])
    
    return orbits, cum_orbits

def plot_fourier(orbits, cum_orbits):
    h, w = (np.max(cum_orbits.imag)-np.min(cum_orbits.imag)), (np.max(cum_orbits.real)-np.min(cum_orbits.real))
    r = w / h
    if args.width != None and args.height != None:
        fig, ax = plt.subplots(figsize=(args.width/args.dpi, args.height/args.dpi), dpi=args.dpi)
        W, H = args.width, args.height
    else:
        fig, ax = plt.subplots(figsize=plt.figaspect(h/w), dpi= args.dpi)
        W, H = fig.get_size_inches()
    if len(args.background_color) == 9:
        fig.patch.set_alpha(float(int(args.background_color[-2:], 16) / 256))
    fig.patch.set_facecolor(args.background_color[:7])
    R = W / H
    plt.ylim(np.min(cum_orbits.imag),np.max(cum_orbits.imag))
    plt.xlim(np.min(cum_orbits.real)-(R-r)*h/2,np.max(cum_orbits.real)+(R-r)*h/2)
    #plt.gca().set_aspect("auto")
    plt.axis("off")
    #plt.plot(forme.real, forme.imag)
    pl, = plt.plot([], [], color=args.line_color, linewidth=args.line_width)
    plt.tight_layout(pad=0)

    radius = np.abs(orbits[:, 0])
    draw_idx = [i for i, r in enumerate(radius) if r > args.threshold_vectors_module]

    for i in draw_idx:
        c_A, c_B = cum_orbits[i, 0], cum_orbits[i+1, 0]
        r = radius[i]
        ax.add_patch(plt.Circle((c_A.real, c_A.imag), r, fill=False, linestyle="dotted"))
        ax.add_patch(ConnectionPatch((c_A.real, c_A.imag), (c_B.real, c_B.imag), ax.transData, arrowstyle="-"))

    def animate(frame):
        ax.patches = []
        c_B = cum_orbits[0, int(frame)]
        #for i in range(1, N_vectors+1):
        for j, i in enumerate(draw_idx):
            c_A, c_B = cum_orbits[i, int(frame)], cum_orbits[i+1, int(frame)]
            r = radius[i]
            # ax.patches[2*j].set_center((c_A.real, c_A.imag))
            # ax.patches[2*j+1].xyA = (c_A.real, c_A.imag)
            # ax.patches[2*j+1].xyB = (c_B.real, c_B.imag)
            # ax.patches[2*j+1].stale = True
            ax.add_patch(plt.Circle((c_A.real, c_A.imag), r, fill=False, linestyle="dotted", linewidth=args.circle_width))
            ax.add_patch(ConnectionPatch((c_A.real, c_A.imag), (c_B.real, c_B.imag), ax.transData, arrowstyle="-", linewidth=args.vector_width))
            #ax.add_patch(plt.Arrow(c_A.real, c_A.imag, c_C.real, c_C.imag, width=0.3))
        x, y = cum_orbits[-1,:int(frame)+1].real, cum_orbits[-1,:int(frame)+1].imag
        pl.set_data(x, y)
        return ax.patches+ [pl,]

    ani = FuncAnimation(
        fig,
        func=animate,
        frames=range(0, args.N_points, ),
        interval=args.interval,
        blit=False
    )
    
    return ani


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