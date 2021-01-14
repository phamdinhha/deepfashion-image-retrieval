from fastai import * 
from fastai.callbacks import * 
from fastai.vision import *
import base64
import warnings
import torch
import torch.nn as nn

from search_engine import ImageSearchEngine, load_inverted_index, load_npy_file

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


class FeatureHook:
    def __init__(self, module: nn.Module, hook_func: HookFunc, forward: bool = True, detach: bool = True):
        self.hook_func, self.forward, self.detach = hook_func, forward, detach
        f = module.register_forward_hook if forward else module.register_backward_hook
        self.hook = f(self._hook)
        self.removed = False
        self.stored = None

    def _hook(self, module: nn.Module, input: Tensors, output: Tensors):
        if self.detach:
            input = (i.detach() for i in input) if is_listy(input) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        hook_out = self.hook_func(module, input, output)
        self.stored = hook_out if self.stored is None else np.row_stack((self.stored, hook_out))

    def remove(self):
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


class FeatureExtractor:
    def __init__(self, learner: Learner, module_name='1.4', ):
        assert learner is not None, f"learner is required"
        self.learner = learner
        self.module = self._get_module_by_name(model=self.learner.model, name=module_name)

    @staticmethod
    def _get_module_by_name(model: nn.Module, name):
        return dict(model.named_modules()).get(name)

    def extract_feature(self, img, base_64=False):
        assert type(img) in [str, bytes]
        if base_64:
            img = open_image(io.BytesIO(base64.b64decode(img))).resize(150)
        else:
            # if type(img) is str:
            #     with open(img, mode='rb') as f:
            #         img = open_image(f.read()).resize(150)
            # else:
            img = open_image(img).resize(150)

        with FeatureHook(self.module, self._get_feature, forward=True, detach=True) as hook:
            self.learner.predict(img)
            output = hook.stored

        output = output / (((output ** 2).sum(axis=1, keepdims=True)) ** 0.5)
        return output.squeeze()

    @staticmethod
    def _get_feature(module: nn.Module, input: Tensors, output: Tensors):
        return output.flatten(1).cpu().numpy()

