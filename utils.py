import numpy as np

def fourier_filter(f, filter_frac=0.5):
    f_fft = np.fft.rfft(f)
    _x = np.linspace(0, 1, len(f_fft))
    f_fft *= np.exp(-((_x/filter_frac)**20))
    return np.fft.irfft(f_fft, n=len(f))

def make_pngs_to_gif(save_dir, name='movie', show_gif=False):
    import imageio
    from IPython import display

    images = []
    pngfiles = glob.glob(save_dir + "*png")
    for filename in pngfiles:
        images.append(imageio.imread(filename))
    imageio.mimsave(save_dir + name + ".gif", images)
    if(show_gif):
        display.Image(save_dir + name + ".gif")