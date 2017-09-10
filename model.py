from sklearn.pipeline import Pipeline
from sklearn.decomposition.pca import PCA
from skimage.feature.texture import greycomatrix, greycoprops
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from sklearn.preprocessing.data import StandardScaler
import pickle
from palmtree import model_dir
import os
from skimage.restoration._denoise import denoise_wavelet


def make_model(classifier, **params):
    pipeline = Pipeline([
                        ('feature_extractor', FeatureExtractor()),
                        ('scaler', StandardScaler()),
                        ('model', classifier(**params)),
                        ])
    return pipeline


def save_model(mod, filename):
    with open(os.path.join(model_dir, filename), "wb") as f:
        pickle.dump(mod, f)


def load_model(filename):
    with open(os.path.join(model_dir, filename), "rb") as f:
        return pickle.load(f)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, **_):
        self.active = False
    
    def transform(self, X, *_):
        if self.active:
            features = []
            for image in X:
                features.append(make_features(image))
            return np.array(features, dtype=float)
        else:
            return X
    
    def fit(self, *_):
        return self    


def make_features(img):
    row = []           
    imgt = _threshold(img) 
    row.extend(_glcm_measures(imgt))
    _ensure_variance(img)
    pca = _pca(img)
    row.extend(pca.explained_variance_[1:])
    return row


def _threshold(img):
    img_th = img / 32
    img_th = np.clip(img_th, 0, 7)
    img_th = np.array(img_th, dtype='uint8')
    return img_th    


def _denoise(img):
    img_red = denoise_wavelet(img)
    return np.array(img_red, dtype='uint8')


def _ensure_variance(img):
    img += np.array(np.random.random_integers(0, 1, img.shape), dtype='uint8')


def _fft_bands(X):
    n_bands = 10
    limit = 100
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    half = int(n_rows / 2 - 1)
    freq = np.fft.fftfreq(n_rows) * n_rows
    freq = freq[:half]
    avg = np.array([0] * half, dtype=float)
    for vec in np.transpose(X):
        sp_complex = np.fft.fft(vec, norm="ortho")
        sp_abs = np.abs(sp_complex)[:half]
        avg += sp_abs
    avg /= n_cols
    avg /= n_rows
    delta = limit / n_bands
    bands = np.array([0] * n_bands, dtype=float)
    counts = np.array([0] * n_bands)
    for f, a in zip(freq, avg):
        for i in range(n_bands):
            if f>=i*delta and f<(i+1)*delta:
                bands[i] += a
                counts[i] += 1
    for i in range(n_bands):
        if counts[i] > 0:
            bands[i] /= counts[i]
    return bands


def _pca(X):
    pca = PCA(n_components=10)
    pca.fit(X)
    return pca


def _glcm_measures(X):
    measures = []
    glcm = greycomatrix(X, [1], [0], levels=8, normed=True)
    for p in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'):
        res = greycoprops(glcm, p)
        measures.extend(list(res.reshape(res.size)))
    return measures
