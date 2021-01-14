import glob 
import pandas as pd
from search_engine import ImageSearchEngine, load_inverted_index, load_npy_file
from feature_extractor import *
from fastai import * 
from fastai.callbacks import * 
from fastai.vision import *

MODEL_PATH = './pretrained_model'
MODEL_NAME = 'resnet50_model.pkl'

image_index2info = load_inverted_index('./csv/train_eval.csv')
features = load_npy_file('./pretrained_model/deep_fashion_features_2.npy')
path = None

search_engine = ImageSearchEngine(
    path=path,
    features=features,
    image_index2info=image_index2info,
    train=True,
)


learner = load_learner(path=MODEL_PATH, file=MODEL_NAME)
feature_extractor = FeatureExtractor(learner)

img = './test_images/2.jpg'
open_image(img)

image_feature = feature_extractor.extract_feature(img, base_64=False)

results = search_engine.search(image_feature[None], num_results=5)
print(results)