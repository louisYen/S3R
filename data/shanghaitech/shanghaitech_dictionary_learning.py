
import os
import cv2
import time
import torch
import joblib
import pickle
import argparse
import numpy as np
import pandas as pd
import os.path as osp

from tqdm import tqdm
from pathlib import Path
from tap import Tap
from typing import Optional
from termcolor import colored

from sklearn.decomposition import MiniBatchDictionaryLearning

try:
    from typing import Literal # typing.Literal is only available from Python 3.8 and up
except ImportError:
    from typing_extensions import Literal

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def color(text, txt_color='green', attrs=['bold']):
    return colored(text, txt_color, attrs=attrs)

class DictArgumentParser(Tap):
    # ============= 
    # basic setting
    # -------------
    backbone: Literal['i3d', 'c3d'] = 'i3d' # default backbone
    batch_size: int = 32 # number of instances in a batch of data (default: 32)
    workers: int = 4 # number of workers in dataloader
    dataset: Literal['shanghaitech', 'ucf-crime', 'xd-violence'] = 'shanghaitech' # dataset to train
    quantize_size: int = 32 # new temporal size for training
    max_iter: int = 100 # maximum iteration for dictionary learning

    # ============ 
    # path setting
    # ------------
    root_path: Path = 'data' # Directory path of data
    dictionary_path: Path = 'dictionary' # Directory path of dictionary

    # ==== 
    # misc
    # ----
    seed: Optional[int] = 42 # random seed for dictionary learning
    debug: bool = False # debug mode

class DictionarySet(object):
    def __init__(
            self,
            root: Path = Path('data'),
            dataset: str = 'shanghaitech',
            extractor: str = 'i3d',
            quantize_size: int = 32,
            is_normal: bool = True,
            verbose: bool = True,
            debug: bool = False,
            test_mode: bool = False,
        ):

        root = root.joinpath(dataset.lower())

        self.is_normal = is_normal
        self.dataset = dataset
        self.quantize_size = quantize_size
        self.debug = debug

        self.subset = 'test' if test_mode else 'train'
        self.root = root

        # >> Load video list
        video_dict = {
            dataset: pd.read_csv(
                root.joinpath('{}.{}ing.csv'.format(
                    dataset, self.subset)))['video-id'].values[:]
        }
        self.video_dict = video_dict

        self.test_mode = test_mode
        self._prepare_data(video_dict, verbose)
        self._load_feature(verbose)

    def _prepare_data(self, video_dict: dict, verbose: bool = True):
        num_videos = 0
        self.video_list = []
        for dataset, video_list in video_dict.items():
            if self.test_mode is False:
                if dataset == 'shanghaitech': index = 63
                elif dataset == 'ucf-crime': index = 810

                template = '{video_id}_{backbone}.npy'
                regular_videos, anomaly_videos = video_list[index:], video_list[:index]

                videos = regular_videos if self.is_normal else anomaly_videos
            else:
                videos = video_list

            num_videos += len(videos)
            if self.debug:
                videos = videos[:2]

            self.video_list.append((dataset, template, videos))

        self.data_info = """
    >> Dictionary Learning
    Dataset description: [{state}] mode.

        - summary:
            {summary}
        - subset: {subset}
        """.format(
            state = color('Regular') if self.is_normal else color('Anomaly', 'magenta'),
            vnum = num_videos,
            summary = '\n\t    '.join([
                f'- there are {len(videos):4d} videos in {dataset}.'
            ]),
            subset = self.subset.capitalize(),
        )

        if verbose: print(self.data_info)

    def _load_feature(self, verbose: bool = False):
        video_features = None
        dataset = self.dataset
        for videos in tqdm(self.video_list, desc=f'Load video features ({self.dataset})', ncols=100):
            dataset, template, video_list = videos
            data_file = f'dictionary/{dataset}/{dataset}_regular_features-2048dim.training.pickle'
            with open(data_file, 'rb') as f:
                feature_dict = pickle.load(f)['feature']
            data_features = np.stack([f for v, f in feature_dict.items()], axis = 0)

            if video_features is None: video_features = data_features
            else: video_features = np.vstack((video_features, data_features))

        self.video_features = video_features

def dictionary_learning(
        video_features: np.array, # normal video features of shape VxNxTxC (all V videos, N crops)
        dataset: str = 'shanghaitech',
        n_slots: int = 16, # number of dictionary elements to extract.
        random_seed: int = 42, # random seed for dictionary learning (default: 42)
        max_iter: int = 100, # maximum number of iterations to perform.
        batch_size: int = 32, # batch size for dictionary learning (default: 32)
        debug: bool = False, # debug mode (default: False)
        n_jobs: int = 4, # number of workers (default: 4)
        algo: str = 'omp', # algorithm used to transform the data (default: omp)
        transform_n_nonzero_coefs: int = 2, # Number of nonzero coefficients to target in each column of the solution.(default: 2)
        ratio: int = 90, # the ratio for keeping the number of slots (default: 90)
        output_path: Path = 'dictionary' # output path
    ):

        V, N, T, C = video_features.shape # V videos in total, N crops, T quantized snippets, C channels

        video_features = video_features.mean(axis=(1, 2)) # VxC

        if debug:
            video_features = video_features[:2*batch_size,]
            n_slots = 2
            max_iter = 1

        R = ratio
        n_slots = int(np.around(V * R / 100.))

        filename = f'{dataset}_dictionaries.taskaware.{algo}.{max_iter}iters.{R}pct.npy'
        output_file = output_path.joinpath(dataset, filename)
        mkdir(output_path.joinpath(dataset))

        if batch_size < 0:
            batch_size = V

        dict_learner = MiniBatchDictionaryLearning(
            n_components = n_slots,
            transform_algorithm = algo,
            random_state = random_seed,
            n_iter = max_iter,
            verbose = 2,
            batch_size = batch_size,
            n_jobs = n_jobs,
            transform_n_nonzero_coefs = transform_n_nonzero_coefs)

        """ Examples provided from ``sklearn.decomposition.MiniBatchDictionaryLearning''
        reference:
        https://scikit-learn.org/0.24/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html?highlight=minibatchdictionary#sklearn.decomposition.MiniBatchDictionaryLearning

        >>> import numpy as np
        >>> from sklearn.datasets import make_sparse_coded_signal
        >>> from sklearn.decomposition import MiniBatchDictionaryLearning
        >>> X, dictionary, code = make_sparse_coded_signal(
        ...     n_samples=100, n_components=15, n_features=20, n_nonzero_coefs=10,
        ...     random_state=42)
        >>> dict_learner = MiniBatchDictionaryLearning(
        ...     n_components=15, transform_algorithm='lasso_lars', random_state=42,
        ... )
        >>> X_transformed = dict_learner.fit_transform(X)

        please note that the shape of X is n_features * n_samples
        """

        x = video_features.T # C x V

        x_transformed = dict_learner.fit_transform(x) # C x num_slots
        x_transformed = x_transformed.T # num_slots x C

        np.save(output_file, x_transformed)

def main():
    global args
    envrows, envcols = list(map(int, os.popen('stty size', 'r').read().split()))

    args = DictArgumentParser().parse_args()

    dataset_cfg = dict(
        root=args.root_path,
        dataset=args.dataset, # list of dataset to construct global dictionary
        quantize_size=args.quantize_size,
        is_normal=True, # only use normal videos
        debug=args.debug,
    )

    dictionarySet = DictionarySet(**dataset_cfg)

    from timeit import default_timer as timer
    from datetime import timedelta
    st = timer()

    # for more details, please refer to ``sklearn.decomposition.MiniBatchDictionaryLearning''
    dictionary_learning(
        dictionarySet.video_features,
        dataset = args.dataset,
        batch_size = args.batch_size,
        max_iter = args.max_iter,
        debug = args.debug,
        random_seed = args.seed,
        n_jobs = args.workers,
        output_path = args.dictionary_path)

    et = timer()
    print('\n\nElapsed Time: ', timedelta(seconds=et-st))

if __name__ == "__main__":
    main()
