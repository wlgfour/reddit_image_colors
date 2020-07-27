import json
import os
from multiprocessing import Pool, Process

import numpy as np
import pandas as pd
from skimage import io
import urllib


def read_auth(path='config'):
    return json.load(open(path))


def apply_imread(df):
    def read_img(url, depth=0):
        try:
            img = io.imread(url)
            if img.shape[-1] == 4:
                return img[:, :, :-1]
            elif img.shape[-1] == 3:
                return img
            else:
                return None
        except ValueError:
            print(f'{url} could not be downloaded')
            return None
        except (TimeoutError, urllib.error.URLError):
            if depth < 5:
                print(f'{url} could not be downloaded')
                return None
            else:
                return read_img(url, depth=depth+1)

    df['img'] = df['url'].apply(read_img)
    return df.dropna()


def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def write_frames(subreddit, fig, ax, angles):
    for i, a in angles:
        ax.view_init(15, a)
        fname = os.path.join(f'{subreddit}_tmp', f'{int(i)}.png')
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)


def bin3d(cfg, hsv, minhist=0):
    """ returns [[x, y, z, hist_size, color], ...]
    """
    hist, (bx, by, bz) = np.histogramdd(hsv, bins=cfg.nbins, density=True)
    x = (bx[1:] + bx[:-1]) / 2
    y = (by[1:] + by[:-1]) / 2
    z = (bz[1:] + bz[:-1]) / 2
    mesh = np.meshgrid(x, y, z)
    X, Y, Z = [m.flatten() for m in mesh]
    hist = hist.flatten()  # count per bucket
    colors = np.concatenate([l[:, np.newaxis] for l in [X, Y, Z]], axis=-1)
    # Exclude zero bins
    return [[X[i], Y[i], Z[i], hist[i], colors[i]] for i in range(len(X)) if hist[i] > minhist]


class Config:
    def __init__(self, filename='cfg', atts=None):
        self.filename = filename
        if atts is not None:
            self.atts = atts
        elif os.path.isfile(filename):
            self.atts = json.load(open(filename))
        else:
            raise ValueError(
                'Must supply Config with either a filename that points'
                ' to an existing configuration or atts dict.'
            )

    def write(self):
        json.dump(self.atts, open(self.filename, 'w'))

    def __getattr__(self, name):
        if name != 'atts' and name in self.atts:
            return self.atts[name]
