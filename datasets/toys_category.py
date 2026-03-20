from .base import AbstractDataset
from .utils import *

import pickle
import os
from pathlib import Path

import json
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class ToysCategoryDataset(AbstractDataset):
    @property
    def num_items(self):
        return len(self.load_meta_dict())

    @classmethod
    def code(cls):
        return 'toys_category'

    @classmethod
    def url(cls):
        return []

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['toys_with_category.json']

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
        df = df[df['sid'].isin(meta_raw)]
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        meta = {smap[k]: v for k, v in meta_raw.items() if k in smap}
        dataset = {
            'train': train,
            'val': val,
            'test': test,
            'meta': meta,
            'umap': umap,
            'smap': smap
        }
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[0])
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    try:
                        item = json.loads(line)
                        # 필수 필드 체크
                        if all(k in item for k in ['uid', 'sid', 'rating', 'timestamp', 'category_label']):
                            rows.append([
                                item['uid'],
                                item['sid'],
                                item['rating'],
                                item['timestamp'],
                                str(item['category_label'])
                            ])
                        else:
                            print(f"필드 누락 row (idx={idx}):", item)
                    except Exception as e:
                        print(f"JSON 파싱 에러 (idx={idx}):", e, line)
        df = pd.DataFrame(rows, columns=['uid', 'sid', 'rating', 'timestamp', 'category_label'])
        df['sid'] = df['category_label']
        return df[['uid', 'sid', 'rating', 'timestamp']]

    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[1])  # meta 파일 (json)
        meta_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                label = str(item['category_label'])
                cats = item.get('categories', [])
                if isinstance(cats, list) and len(cats) > 0:
                    cat_path = cats[0]
                    if len(cat_path) == 1:
                        cat_str = cat_path[0]
                    elif len(cat_path) == 2:
                        cat_str = cat_path[1]
                    elif len(cat_path) >= 3:
                        cat_str = cat_path[2]
                    else:
                        cat_str = ""
                else:
                    cat_str = ""
                meta_dict[label] = cat_str
        return meta_dict