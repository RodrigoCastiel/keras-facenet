import shutil
import os
# Filter only ERROR messages in TensorFlow logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import argparse
import numpy as np

import sklearn.model_selection
import sklearn.utils
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics

from imageio import imsave
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from utils.dataset_loader import DatasetLoader
from code.trained_keras_facenet import TrainedKerasFacenet


cached_embeddings_filepath = ".cached_files/embeddings"
cached_labels_filepath = ".cached_files/labels"
facenet_model_filepath = 'model/keras/model/facenet_keras.h5'


def main():
  # Parse command line arguments.
  parser = argparse.ArgumentParser(description="")
  parser.add_argument(
    "--compute_embeddings",
    help="If true, it loads the image dataset and recomputes all the embeddings."
  )
  parser.add_argument(
    "--perform_tsne",
    help="If true, it shows a 2D data visualization with TNSE."
  )
  args = parser.parse_args()

  if args.compute_embeddings:
    X, w_true = load_images_and_compute_embeddings()
  else:
    X, w_true = load_embeddings_and_labels()

  print("Loading complete. %d data points extracted." % w_true.shape[0])
  evaluate_kmeans(X, w_true)

  if args.perform_tsne:
    visualize_tsne(X, w_true)


def load_images_and_compute_embeddings():
  # Load image dataset.
  dataset_path = 'data/lfw/raw'
  print("Loading pictures under '%s'." % dataset_path)
  loader = DatasetLoader()
  faces, labels = loader.load_test_dataset(dataset_path)
  print(">> %d face pictures extracted." % faces.shape[0])

  # Load pre-trained facenet model, and compute embeddings.
  print("Loading pre-trained facenet model at '%s'." % facenet_model_filepath)
  facenet = TrainedKerasFacenet(filepath=facenet_model_filepath)
  embeddings = facenet.compute_embbedings(faces)

  # Cache data.
  print("Caching data..")
  np.save(cached_labels_filepath, labels)
  np.save(cached_embeddings_filepath, embeddings)

  return (embeddings, labels)


def load_embeddings_and_labels():
  print("Loading cached embeddings and ground-truth labels.")
  X = np.load(cached_embeddings_filepath + ".npy")
  w_true = np.load(cached_labels_filepath + ".npy")
  return (X, w_true)


def visualize_principal_components(X, w_train):
  pca = PCA(n_components=3)
  X_transformed = pca.fit_transform(X)
  x = X_transformed[:, 0]
  y = X_transformed[:, 1]
  z = X_transformed[:, 2]

  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x, y, z, c='r', marker='o')

  plt.title('Embedding Vectors')
  ax.legend(loc='upper right')
  plt.show()


def visualize_tsne(X, w_train):
  tsne = TSNE(n_components=2)
  X_transformed = tsne.fit_transform(X)
  x = X_transformed[:, 0]
  y = X_transformed[:, 1]
  plt.scatter(x, y, c='r', marker='o')
  plt.title('Embedding Vectors')
  plt.show()


def evaluate_kmeans(X_data, w_true):
  K = len(np.unique(w_true))

  print("\nPerforming cross-validation on K-means with K = %d." % K)
  scores = perform_cross_validation(
    KMeans(n_clusters=K, random_state=0),
    X_data,
    w_true,
    num_folds_cv=30,
  )

  avg_score = np.mean(scores)
  error_margin = 2*np.std(scores)
  print("Adjusted Rand Scores: ")
  print(scores)
  print("Average Score: %f (+/- %f). " % (avg_score, error_margin))


def perform_cross_validation(classifier, X_data, w_true, num_folds_cv):
  scorer = sklearn.metrics.make_scorer(metrics.adjusted_rand_score)
  skf = sklearn.model_selection.KFold(n_splits=num_folds_cv)
  scores = sklearn.model_selection.cross_val_score(
    classifier, X_data, w_true, scoring=scorer, cv=skf, n_jobs=-1, verbose=False,
  )
  return scores


if __name__ == "__main__":
  main()
