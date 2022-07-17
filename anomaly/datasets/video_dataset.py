
import cv2
import torch
import joblib
import pickle
import numpy as np
import pandas as pd
import torch.utils.data as data

from tqdm import tqdm
from pathlib import Path
from utils import process_feat
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def softmax(scores, axis):
    es = np.exp(scores - scores.max(axis=axis)[..., None])
    return es / es.sum(axis=axis)[..., None]

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

class Dataset(data.Dataset):
    def __init__(
            self,
            data_root: Path = Path('data'),
            dataset: str='shanghaitech',
            backbone: str='i3d',
            quantize_size: int=32,
            is_normal: bool=True,
            transform = None,
            test_mode: bool=False,
            verbose: bool=False,
            dictionary = None,
            dictionary_root: Path=Path('.dictionary'),
            data_file: str=None,
            ann_file: str=None,
            univ_dict_file: str=None,
            task_dict_file: str=None,
            regular_file: str=None,
            tmp_dict_file: str=None,
            use_dictionary: bool=True,
            modality: str='universal'
        ):

        assert modality in ['taskaware', 'universal', 'univ-task']

        self.is_normal = is_normal
        self.dataset = dataset
        self.backbone = backbone
        self.quantize_size = quantize_size
        self.ann_file = ann_file

        self.subset = 'test' if test_mode else 'train'
        self.data_root = data_root
        self.dictionary_root = dictionary_root
        self.data_dir = data_root.joinpath(dataset).joinpath(self.subset)

        # >> Load video list
        video_list = pd.read_csv(data_file)
        video_list = video_list['video-id'].values[:]

        self.transform = transform
        self.test_mode = test_mode
        self._prepare_data(video_list, verbose)

        self.num_frame = 0
        self.labels = None

        self.data_path_formatter = data_root.joinpath(
                dataset).joinpath(
                    backbone).joinpath(
                        self.subset).joinpath(
                            '{video_id}_{backbone}.npy')

        # >> Obtain global video statistics
        if dictionary is None:
            if 'universal' in modality:
                memory = self._get_dictionary(univ_dict_file)
                self.dictionary = self._get_video_statistics(
                    self.video_list, memory, regular_file, tmp_dict_file)
            elif 'taskaware' in modality:
                self.dictionary = self._get_dictionary(task_dict_file)
        else:
            self.dictionary = dictionary

    def _prepare_frame_level_labels(self, video_list):
        import json
        ann_file = self.ann_file
        with open(ann_file, 'r') as fin:
            db = json.load(fin)

        ground_truths = list()
        for video_id in video_list:
            labels = db[video_id]['labels']
            ground_truths.append(labels)
        ground_truths = np.concatenate(ground_truths)
        return ground_truths

    def _prepare_data(self, video_list: list, verbose: bool=True):
        if self.test_mode is False:
            if 'shanghaitech' in self.dataset: index = 63
            elif 'ucf-crime' in self.dataset: index = 810

            self.video_list = video_list[index:] if self.is_normal else video_list[:index]
        else:
            self.video_list = video_list
            self.ground_truths = self._prepare_frame_level_labels(video_list)

        self.data_info = """
    Dataset description: [{state}] mode.

        - there are {vnum} videos in {dataset}.
        - subset: {subset}
        """.format(
            state = 'Regular' if self.is_normal else 'Anomaly',
            vnum = len(self.video_list),
            dataset = self.dataset,
            subset = self.subset.capitalize(),
        )

        if verbose: print(self.data_info)

    def _get_dictionary(self, dict_file):

        memory = np.load(dict_file)

        return memory.astype(np.float32)

    def _get_video_statistics(self, video_list, memory, regular_file=None, tmp_dict_file=None):

        def universal_feature(
            regular_features, # all normal video features of shape  [M, n, t', c]
            memory, # universal memory of shape [n_memory, c]
            use_l2_norm: bool = True,
        ):
            M, n, t, c = regular_features.shape

            if c != memory.shape[-1]:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=c, svd_solver='full') # map the data to c dimensions
                data_features = pca.fit_transform(memory)
                memory = data_features.reshape(*memory.shape[:-1], -1) # [nv, nc, t, c]

            x = regular_features.mean(axis=1).reshape(-1, regular_features.shape[-1]) # [M*t', c]

            cache = memory.copy() # [n_memory, c]
            n_slots = cache.shape[0]

            x = normalize(x, norm='l2', axis=1) # [M*t', c]
            memory = normalize(memory, norm='l2', axis=1) # [n_memory, c]

            attn = x @ memory.T # [M*t', n_memory]

            # >> remove unselected atoms
            attn = attn.mean(axis=0, keepdims=True) # [n_memory]
            attn = softmax(attn, axis=-1) # [1, n_memory]
            attn = attn.squeeze() # [n_memory]

            topk = np.flip(np.argsort(attn))[:n_slots//2] # topk related atoms

            out = cache[topk] # [k, c]

            out = np.vstack((out, x.mean(axis=0, keepdims=True))) # [k+1, c]

            return out

        if Path(tmp_dict_file).exists():
            video_features = np.load(tmp_dict_file).astype(np.float32)
        else:
            with open(regular_file, 'rb') as fin:
                regular_dict = pickle.load(fin)['feature'] # {v: [nc=num_crops, t=32, c], ...}
            regular_features = np.stack([f for v, f in regular_dict.items()], axis=0)

            video_features = universal_feature(regular_features, memory)
            with open(tmp_dict_file, 'wb') as f:
                np.save(f, video_features)

        return video_features.astype(np.float32)

    def __getitem__(self, index):

        video_id = self.video_list[index]
        dictionary = self.dictionary

        data_path_formatter = str(self.data_path_formatter)

        label = self.get_label()  # get video level label 0/1

        data_path = data_path_formatter.format(
            video_id=video_id,
            backbone=self.backbone)

        features = np.load(data_path, allow_pickle=True) # tau x N x C, N=10
        features = np.array(features, dtype=np.float32) # tau x N x C

        if self.transform is not None:
            features = self.transform(features)


        if self.test_mode:
            return features, dictionary
        else:
            t, n_group, channels = features.shape

            features = np.transpose(features, (2, 0, 1)) # C x tau x N

            # quantize each video to 32-snippet-length video
            width, height = self.quantize_size, channels
            features = cv2.resize(features, (width, height),
                interpolation=cv2.INTER_LINEAR) # CxTxN

            video = np.transpose(features, (2, 1, 0)) # NxTxC

            # global video statistics
            regular_labels = torch.tensor(0.0) # being normal video

            return video, label, dictionary, regular_labels

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.video_list)

    def get_num_frames(self):
        return self.num_frame
