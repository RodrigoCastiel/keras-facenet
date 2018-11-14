import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import argparse
import os
import numpy as np
import shutil
import sklearn.cluster
import sklearn.manifold
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils

from utils.dataset_loader import DatasetLoader
from code.trained_keras_facenet import TrainedKerasFacenet


# Filter only ERROR messages in TensorFlow logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

cached_embeddings_filepath = ".cached_files/embeddings-"
cached_labels_filepath = ".cached_files/labels-"
facenet_model_filepath = 'model/keras/model/facenet_keras.h5'


def main():
  # Parse command line arguments.
  args = parse_command_line_args()

  print("+--------------------------------------+")
  print("|   Digital Image Processing Project   |")
  print("+--------------------------------------+")
  print("|      Author: Rodrigo Castiel         |")
  print("+--------------------------------------+")

  dataset_name = args.dataset
  if args.compute_embeddings:
    # Load dataset, load facenet model and compute embeddings.
    X, w_true = load_images_and_compute_embeddings(dataset_name, args.use_raw)
  else:
    # Load cached embeddings and labels.
    X, w_true = load_embeddings_and_labels(dataset_name)

  K = len(np.unique(w_true))
  print("Loading complete.")
  print("%d data points extracted from %d clusters." % (w_true.shape[0], K))

  # Evaluate different clustering methods using cross-validation.
  kmeans = sklearn.cluster.KMeans(n_clusters=K, random_state=0)
  agglomerative = sklearn.cluster.AgglomerativeClustering(n_clusters=K)
  spectral = sklearn.cluster.SpectralClustering(n_clusters=K, random_state=0)
  methods = [spectral, kmeans, agglomerative]
  evaluate_clustering_methods(methods, X, w_true)

  # Optionally, visualize dataset using TSNE.
  if args.show_tsne:
    visualize_tsne(X, w_true)


def load_images_and_compute_embeddings(dataset_name, use_raw):
  # Load image dataset (default).
  dataset_path = "data/" + dataset_name + "/"
  print("Loading pictures under '%s'." % dataset_path)
  loader = DatasetLoader()
  faces, labels = loader.load_test_dataset(dataset_path, use_raw=use_raw)
  print(">> %d face pictures extracted." % faces.shape[0])

  # Load pre-trained facenet model, and compute embeddings.
  print("Loading pre-trained facenet model at '%s'." % facenet_model_filepath)
  facenet = TrainedKerasFacenet(filepath=facenet_model_filepath)
  embeddings = facenet.compute_embbedings(faces)

  # Cache data.
  print("Caching data.. ", end='')
  np.save(cached_labels_filepath + dataset_name, labels)
  np.save(cached_embeddings_filepath + dataset_name, embeddings)
  print("DONE")

  return (embeddings, labels)


def load_embeddings_and_labels(dataset_name):
  print("Loading cached embeddings and ground-truth labels.")
  X = np.load(cached_embeddings_filepath + dataset_name + ".npy")
  w_true = np.load(cached_labels_filepath + dataset_name + ".npy")
  return (X, w_true)


def visualize_tsne(X, w_train):
  tsne = sklearn.manifold.TSNE(n_components=2)
  X_transformed = tsne.fit_transform(X)
  x = X_transformed[:, 0]
  y = X_transformed[:, 1]
  plt.scatter(x, y, c='r', marker='o')
  plt.title('Embedding Vectors')
  plt.show()


def evaluate_clustering_methods(methods, X_data, w_true, n_times=30):
  scores = []
  max_len = 44

  for method in methods:
    clustering_name = type(method).__name__
    scores = np.zeros(n_times)
    for i in range(n_times):
      print("  Running clustering %d/%d..." % (i+1, n_times), end='\r')
      # Shuffle dataset.
      indices = np.arange(len(w_true))
      np.random.shuffle(indices)

      # Fit and predict clustering method on shuffled (X_data, w_true).
      clustering = sklearn.base.clone(method)
      w_pred = clustering.fit_predict(X_data[indices])
      scores[i] = sklearn.metrics.adjusted_rand_score(w_true[indices], w_pred)

    avg_score = np.mean(scores)
    error_margin = 2*np.std(scores)

    # Print out the results.
    num_dots = max_len - len(clustering_name)
    print(
      "+ %s %s %lf%% (+/- %lf)"
      % (clustering_name, num_dots*".", avg_score, error_margin)
    )


def perform_cross_validation(classifier, X_data, w_true, num_folds_cv):
  scorer = sklearn.metrics.make_scorer(sklearn.metrics.adjusted_rand_score)
  skf = sklearn.model_selection.KFold(n_splits=num_folds_cv)
  scores = sklearn.model_selection.cross_val_score(
    classifier,
    X_data,
    w_true,
    scoring=scorer,
    cv=skf,
    n_jobs=-1,
    verbose=True,
  )

  return scores


def parse_command_line_args():
  parser = argparse.ArgumentParser(description="")
  parser.add_argument(
    "--dataset",
    help="'lfw', 'muct' or 'personal_faces'", 
    type=str,
    default="personal_faces",
  )
  parser.add_argument(
    "--compute_embeddings",
    help="If true, it loads the image dataset and recomputes all the embeddings.",
    type=bool,
    default=False,
  )
  parser.add_argument(
    "--show_tsne",
    help="If true, it shows a 2D data visualization with TNSE.",
    type=bool,
    default=False,
  )
  parser.add_argument(
    "--use_raw",
    help="If true, it will not align dataset pictures.",
    type=bool,
    default=False,
  )
  args = parser.parse_args()
  print(args)
  return args


if __name__ == "__main__":
  main()
