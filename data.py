import cv2 as cv
import numpy as np
import os
from palmtree import img_dir
from palmtree.model import make_features
from palmtree.cascade import extract_all_images
import pylab as pl


def _append_features(X, img, extract_features):
        if extract_features:
            X.append(make_features(img))
        else:
            X.append(img)            
    

def load_data(extract_features=True):
    X = []
    y = []
    files = []

    all_images = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    branches = []
    no_branches = []
    for image in all_images:
        if image.startswith('Branch'):
            branches.append(image)
        if image.startswith('NoBranch'):
            no_branches.append(image)
    
    # Collect images with a branch
    for branch in branches:
        filename = os.path.join(img_dir, branch)
        files.append(filename)
#         imgn = cv.imread(filename)
#         pl.figure()
#         pl.imshow(imgn)

        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        assert(img is not None)
        _append_features(X, img, extract_features)
        y.append(1)
        
    # Collect images with no branch
    for no_branch in no_branches:
        filename = os.path.join(img_dir, no_branch) 
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        assert(img is not None)
        images = []
        extract_all_images(images, img, 500)
        for i, images in enumerate(images):
            _append_features(X, images, extract_features)            
            y.append(0)
            files.append(filename + "_%d" % i)

    print("got %d samples" % len(y))
    return np.array(X), np.array(y), files
