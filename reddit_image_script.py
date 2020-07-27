import os
from multiprocessing import Pool, Process

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import praw
from mpl_toolkits.mplot3d import Axes3D
from skimage import io

from helpers import *

mpl.use('Agg')

pd.options.mode.chained_assignment = None

cfg_atts = {
    'subreddit': 'dataisbeautiful',
    'nposts': 100,
    'sample_rate': 1000,
    'ncores': 10,
    'plot_mode': '3DHIST',
    'nbins': 15,
    'write_mode': 'gif',
    'nframes': 200,
    'framerate': 10,
    'resume': False,
    'cleanup': True,
    'scale': 50,
    'minsize': 0.33
}


if __name__ == '__main__':
    cfg = Config(filename=cfg_atts['subreddit'], atts=cfg_atts)
    savefilename = 'test'
    # ============================= CODE =============================
    auth = read_auth('config')
    reddit = praw.Reddit(
        client_id=auth['client_id'],
        client_secret=auth['client_secret'],
        user_agent=auth['user_agent']
    )

    # Get the top posts from the subreddit and put them in a dataframe
    subreddit = reddit.subreddit(cfg.subreddit)
    data_dict = {'id': [], 'url': []}
    for submission in subreddit.top(limit=cfg.nposts):
        data_dict['id'].append(submission.id)
        data_dict['url'].append(submission.url)
    df = pd.DataFrame(data_dict)

    # Get the link endings and filter out non-images
    df['end'] = df['url'].apply(lambda s: s.split('.')[-1])
    ends = df[df['end'].str.len().le(5)]['end'].drop_duplicates()
    nends = len(ends)
    ends = set(ends)
    print(
        f'Top {cfg.nposts} urls with suffixes less than 5 characters have endings: {ends}.'
        f' {nends} fall into this category'
    )
    images = df[df['end'].str.contains(r'^jpg|png$')]

    # Download the images in parallel
    print(f'Downloading {len(images)} images...')
    images = parallelize_dataframe(images, apply_imread, cfg.ncores)
    colors = np.concatenate([a.reshape(-1, 3)[::cfg.sample_rate]
                             for a in images['img'].values])
    colors = colors / 255
    split_colors = np.array_split(colors, cfg.ncores)

    # Compute hsv values in parallel
    print(f'Computing {colors.shape[0]} hsv values...')
    pool = Pool(cfg.ncores)
    hsv = np.concatenate(pool.map(mpl.colors.rgb_to_hsv, split_colors), axis=0)
    pool.close()
    pool.join()

    # Plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    if cfg.plot_mode == 'SCATTER':
        savefilename = f'scatter__{cfg.subreddit}__{len(images)}_imgs'
        print(f'Plotting {hsv.shape[0]} points...')
        _ = ax.scatter3D(hsv[:, 0], hsv[:, 1], hsv[:, 2],
                         c=colors, s=cfg.scale, linewidth=0)
    elif cfg.plot_mode == '3DHIST':
        savefilename = f'hist__{cfg.subreddit}__{len(images)}_imgs__{cfg.nbins}_bins.gif'
        print(f'Binning {hsv.shape[0]} points into {cfg.nbins} bins...')
        data = bin3d(cfg, hsv, minhist=cfg.minsize)
        print(f'Plotting {len(data)} points...')
        # Plot each point individually because s is janky
        for x, y, z, s, c in data:
            if s == 0:
                continue
            _ = ax.scatter([x], [y], [z], linewidth=0,
                           c=[c], s=cfg.scale*np.log(1+s), alpha=0.6)

    # Axis parameters
    ax.set_axis_off()

    # Saving
    if cfg.write_mode == 'gif':
        angles = np.array([[i, a] for i, a in enumerate(
            np.linspace(0, 360, cfg.nframes))])
        # tmpdir to store images
        if cfg.resume:
            currfiles = os.listdir(f'{cfg.subreddit}_tmp')
            i_done = [int(s[:s.index('.')])
                      for s in currfiles if s[0].isdigit()]
            angles = np.array([[a, i] for a, i in angles if i not in i_done])
        else:
            if os.path.isdir(f'{cfg.subreddit}_tmp'):
                os.rmdir(f'{cfg.subreddit}_tmp')
            os.mkdir(f'{cfg.subreddit}_tmp')
        # Write frames in parallel
        print(f'Saving {len(angles)} frames...')
        angles_split = np.array_split(angles, cfg.ncores)
        procs = []
        for angs in angles_split:
            p = Process(target=write_frames, args=[cfg.subreddit, fig, ax, angs])
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        # Write the frames to a gif
        print('Writing gif...')
        filenames = os.listdir(f'{cfg.subreddit}_tmp')
        filenames = sorted(filenames, key=lambda s: int(s[:s.index('.')]))
        with imageio.get_writer(f'{savefilename}.gif', mode='I', fps=cfg.framerate) as writer:
            for filename in filenames:
                image = imageio.imread(os.path.join(
                    f'{cfg.subreddit}_tmp', filename))
                writer.append_data(image)

        if cfg.cleanup:
            # Remove stored frames
            for fname in os.listdir(f'{cfg.subreddit}_tmp'):
                os.remove(os.path.join(f'{cfg.subreddit}_tmp', fname))
            os.rmdir(f'{cfg.subreddit}_tmp')
    else:  # Write single image
        print('Saving figure...')
        fig.savefig(f'{savefilename}.png')

    print('Done.')
