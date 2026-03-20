from .base import AbstractDataset
from .utils import *
import gzip 
import pickle
import os
from pathlib import Path

import json
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class ToysDataset(AbstractDataset):
    @property
    def num_items(self):
        return len(self.load_meta_dict())
        
    @classmethod
    def code(cls):
        return 'toys'

    @classmethod
    def url(cls):
        return []

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['toys_with_category.json', 'toys_category_meta.json']  

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if not all(folder_path.joinpath(f).is_file() for f in self.all_raw_file_names()):
            raise FileNotFoundError("Raw files not found. Please place the files manually in the expected folder.")

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        meta_raw = self.load_meta_dict()
        df = df[df['sid'].isin(meta_raw)]  # filter items without meta info
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        meta = {smap[k]: v for k, v in meta_raw.items() if k in smap}
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'meta': meta,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[0])
        df = pd.read_json(file_path,lines=True) ##lines  추가
        ##
        # df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
        # df.columns = ['uid', 'sid', 'rating', 'timestamp']
        df = df[['uid', 'sid', 'rating', 'timestamp']]  # 필요 컬럼만 필터링
        
        return df
    
    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[1])

        meta_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 전체 파일을 리스트로 로드
            for item in data:
                if 'title' in item and len(item['title']) > 0:
                    meta_dict[item['asin'].strip()] = item['title'].strip()
        
        return meta_dict

