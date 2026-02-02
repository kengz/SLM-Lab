"""ML-dependent utilities (torch, numpy, cv2).

These functions require ML dependencies and are only available when the full
ML environment is installed. In minimal install mode (dstack orchestration only),
these won't be available.
"""
from collections import deque

import cv2
import json
import numpy as np
import operator
import pandas as pd
import pydash as ps
import time
import torch
import torch.multiprocessing as mp

NUM_CPUS = mp.cpu_count()


class LabJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        else:
            return str(obj)


def batch_get(arr, idxs):
    '''Get multi-idxs from an array depending if it's a python list or np.array'''
    if isinstance(arr, (list, deque)):
        return np.array(operator.itemgetter(*idxs)(arr))
    else:
        return arr[idxs]


def concat_batches(batches):
    '''
    Concat batch objects from agent.memory.sample() into one batch, when all agents experience similar envs
    Also concat any nested epi sub-batches into flat batch
    {k: arr1} + {k: arr2} = {k: arr1 + arr2}
    '''
    # if is nested, then is episodic
    is_episodic = isinstance(batches[0]['dones'][0], (list, np.ndarray))
    concat_batch = {}
    for k in batches[0]:
        datas = []
        for batch in batches:
            data = batch[k]
            if is_episodic:  # make into plain batch instead of nested
                data = np.concatenate(data)
            datas.append(data)
        concat_batch[k] = np.concatenate(datas)
    return concat_batch


def epi_done(done):
    '''
    General method to check if episode is done for both single and vectorized env
    Vector environments handle their own resets automatically via gymnasium,
    so only single environments need explicit reset in control loop.
    '''
    return np.isscalar(done) and done


def get_class_attr(obj):
    '''Get the class attr of an object as dict'''
    attr_dict = {}
    for k, v in obj.__dict__.items():
        if isinstance(v, torch.nn.Module):
            val = f'(device:{v.device}) {v}'
        elif hasattr(v, '__dict__') or ps.is_tuple(v):
            val = str(v)
        else:
            val = v
        attr_dict[k] = val
    return attr_dict


def parallelize(fn, args, num_cpus=NUM_CPUS):
    '''
    Parallelize a method fn, args and return results with order preserved per args.
    args should be a list of tuples.
    @returns {list} results Order preserved output from fn.
    '''
    with mp.Pool(num_cpus, maxtasksperchild=1) as pool:
        results = pool.starmap(fn, args)
    return results


def use_gpu(spec_gpu: str | bool | None) -> bool:
    '''Check if GPU should be used based on gpu setting: auto, true, false, or legacy boolean'''
    if spec_gpu in ('auto', None):
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    return spec_gpu not in ('false', False)


def set_cuda_id(spec):
    '''Use trial and session id to hash and modulo cuda device count for a cuda_id to maximize device usage. Sets the net_spec for the base Net class to pick up.'''
    # Don't trigger any cuda call if not using GPU. Otherwise will break multiprocessing on machines with CUDA.
    # see issues https://github.com/pytorch/pytorch/issues/334 https://github.com/pytorch/pytorch/issues/3491 https://github.com/pytorch/pytorch/issues/9996
    if not use_gpu(spec['agent']['net'].get('gpu')):
        return
    meta_spec = spec['meta']
    trial_idx = meta_spec['trial'] or 0
    session_idx = meta_spec['session'] or 0
    if meta_spec['distributed'] == 'shared':  # shared hogwild uses only global networks, offset them to idx 0
        session_idx = 0
    job_idx = trial_idx * meta_spec['max_session'] + session_idx
    job_idx += meta_spec['cuda_offset']
    device_count = torch.cuda.device_count()
    cuda_id = job_idx % device_count if torch.cuda.is_available() else None

    spec['agent']['net']['cuda_id'] = cuda_id


def set_random_seed(spec):
    '''Generate and set random seed for relevant modules, and record it in spec.meta.random_seed'''
    trial = spec['meta']['trial']
    session = spec['meta']['session']
    random_seed = int(1e5 * (trial or 0) + 1e3 * (session or 0) + time.time())
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    spec['meta']['random_seed'] = random_seed
    return random_seed


def split_minibatch(batch, mb_size):
    '''Split a batch into minibatches of mb_size or smaller, without replacement'''
    size = len(batch['rewards'])
    # If minibatch size >= batch size, just return the whole batch
    if mb_size >= size:
        return [batch]
    idxs = np.arange(size)
    np.random.shuffle(idxs)
    chunks = int(size / mb_size)
    nested_idxs = np.array_split(idxs[:chunks * mb_size], chunks)
    if size % mb_size != 0:  # append leftover from split
        nested_idxs += [idxs[chunks * mb_size:]]
    mini_batches = []
    for minibatch_idxs in nested_idxs:
        minibatch = {k: v[minibatch_idxs] for k, v in batch.items()}
        mini_batches.append(minibatch)
    return mini_batches


def to_json(d, indent=2):
    '''Shorthand method for stringify JSON with indent'''
    return json.dumps(d, indent=indent, cls=LabJsonEncoder)


def to_torch_batch(batch, device, is_episodic):
    '''Mutate a batch (dict) to make its values from numpy into PyTorch tensor'''
    for k in batch:
        if is_episodic:  # for episodic format
            batch[k] = np.concatenate(batch[k])
        elif ps.is_list(batch[k]):
            batch[k] = np.array(batch[k])
        # Optimize tensor creation - direct device placement avoids intermediate CPU tensor
        if batch[k].dtype != np.float32:
            batch[k] = batch[k].astype(np.float32)
        batch[k] = torch.from_numpy(batch[k]).to(device, non_blocking=True)
    return batch


# Atari image preprocessing

def to_opencv_image(im):
    '''Convert to OpenCV image shape h,w,c'''
    shape = im.shape
    if len(shape) == 3 and shape[0] < shape[-1]:
        return im.transpose(1, 2, 0)
    else:
        return im


def to_pytorch_image(im):
    '''Convert to PyTorch image shape c,h,w'''
    shape = im.shape
    if len(shape) == 3 and shape[-1] < shape[0]:
        return im.transpose(2, 0, 1)
    else:
        return im


def grayscale_image(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def resize_image(im, w_h):
    return cv2.resize(im, w_h, interpolation=cv2.INTER_AREA)


def normalize_image(im):
    '''Normalizing image by dividing max value 255'''
    # NOTE: beware in its application, may cause loss to be 255 times lower due to smaller input values
    return np.divide(im, 255.0)


def preprocess_image(im, w_h=(84, 84)):
    '''
    Image preprocessing using OpenAI Baselines method: grayscale, resize
    This resize uses stretching instead of cropping
    '''
    im = to_opencv_image(im)
    im = grayscale_image(im)
    im = resize_image(im, w_h)
    im = np.expand_dims(im, 0)
    return im


def debug_image(im):
    '''
    Renders an image for debugging; pauses process until key press
    Handles tensor/numpy and conventions among libraries
    '''
    if torch.is_tensor(im):  # if PyTorch tensor, get numpy
        im = im.cpu().numpy()
    im = to_opencv_image(im)
    im = im.astype(np.uint8)  # typecast guard
    if im.shape[0] == 3:  # RGB image
        # accommodate from RGB (numpy) to BGR (cv2)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow('debug image', im)
    cv2.waitKey(0)
