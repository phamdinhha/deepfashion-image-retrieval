import numpy as np
import faiss
import pandas as pd
from collections import defaultdict
import os
import torch


def load_npy_file(fn):
    return np.load(fn)


class ImageSearchEngine:
    def __init__(
            self,
            path=None,
            features: np.ndarray = None,
            image_index2info: dict = None,
            train=False,
            normalize=True,
    ):
        assert path is not None or features is not None, 'path or features is required'
        self.index = self.build_index(path, features, train=train, normalize=normalize)
        self.dim = self.index.d
        self.image_index2info = image_index2info

    @staticmethod
    def build_index(path, features: np.ndarray, train, normalize):
        if path is None:
            if normalize:
                features = features / ((features ** 2).sum(axis=1, keepdims=True) ** 0.5)
            dim = features.shape[1]
            if not train:
                index = faiss.IndexFlatIP(dim)
                index.add(features)
            else:
                quantizer = faiss.IndexFlatIP(dim)
                num_clusters = 100
                index = faiss.IndexIVFFlat(quantizer, dim, num_clusters, faiss.METRIC_INNER_PRODUCT)
                print('Training...')
                index.train(features)
                print('Training done!!!')
                index.add(features)
        else:
            assert os.path.exists(path), f"{path} is not existed!"
            index = faiss.read_index(path)
        return index

    def image_indices2posts(self, image_indices, sim_scores):
        posts = defaultdict(list)
        for i, s in zip(image_indices, sim_scores):
            info = self.image_index2info[i]
            posts[info['image_id']].append({'path': info['path'], 'score': s})

        return posts

    def save_index(self, fn):
        faiss.write_index(self.index, fn)

    def search(self, q: np.ndarray, num_results=10):
        if len(q.shape) == 1:
            q = q[None]
        assert q.shape[1] == self.dim, f'query vectors must have dim = {self.dim}'
        similarity_scores, indices = self.index.search(q, num_results)

        results = [self.image_indices2posts(indices[i], similarity_scores[i]) for i in range(indices.shape[0])]
        return results


def load_inverted_index(train_csv_path):
    data_df = []
    train_df = pd.read_csv(train_csv_path, sep='\t')
    for index, row in train_df.iterrows():
        image_path = row['image_name']
        file_name = image_path.split('/')[-1].split('.')[0]
        data_df.append([index, file_name, image_path])
    data_df = pd.DataFrame(data_df, columns=['index', 'image_id', 'image_path'])
    image_index2info = dict((data_df.apply(lambda x: (x[0], {'image_id': x[1], 'path': x[2], }), axis=1)).values)
    return image_index2info
    


if __name__ == '__main__':
    index_path = 'dump/dump_index_resnet.pkl'
    image_index2info = load_invert_indexed(image_ids_fn='data/all_feat.list', image_info_fn='data/all_images_path.csv')

    features = load_npy_file('features_resnet50.npy')
    path = None

    test_features = load_npy_file('features_resnet50.npy')
    test_features = test_features / ((test_features ** 2).sum(axis=1, keepdims=True) ** 0.5)

    search_engine = ImageSearchEngine(
        path=path,
        features=features,
        image_index2info=image_index2info,
        train=True,
    )

    k = 10
    results = search_engine.search(test_features[[13134]], k)
    print(results[0])

    


