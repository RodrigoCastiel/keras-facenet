import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import argparse
import logging as log
import os
import numpy as np
import shutil
import sklearn.cluster
import sklearn.manifold
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils
import sys

from utils.dataset_loader import DatasetLoader
from code.trained_keras_facenet import TrainedKerasFacenet


# Filter only ERROR messages in TensorFlow logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

cached_embeddings_filepath = ".cached_files/embeddings-"
cached_labels_filepath = ".cached_files/labels-"
facenet_model_filepath = 'model/keras/model/facenet_keras.h5'

validation_set_fraction = 0.8
n_times = 30

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
    X, w_true = load_images_and_compute_embeddings(
      dataset_name,
      args.use_raw,
      args.min_threshold,
    )
  else:
    # Load cached embeddings and labels.
    X, w_true = load_embeddings_and_labels(dataset_name)

  K = len(np.unique(w_true))
  print("Loading complete.")
  print("%d data points extracted from %d clusters." % (w_true.shape[0], K))

  # Optionally, visualize dataset using TSNE.
  if args.show_tsne:
    visualize_tsne(X, w_true, dataset_name)

  # Evaluate different clustering methods.
  dbscan_methods = [("dbscan", n/10.0, 1) for n in range(1, 13)]
  meanshift_methods = [("mean-shift", n/10.0) for n in range(1, 13)]
  methods = [
    ("k-means", "opt"),
    ("agglomerative", "opt"),
    ("spectral", "opt"),
  ] + dbscan_methods + meanshift_methods

  evaluate_clustering_methods(
    methods,
    X,
    w_true,
    experiment_mode=args.experiment_mode,
  )


def load_images_and_compute_embeddings(dataset_name, use_raw, min_threshold):
  # Load image dataset (default).
  dataset_path = "data/" + dataset_name + "/"
  print("Loading pictures under '%s'." % dataset_path)
  loader = DatasetLoader()
  faces, labels = loader.load_test_dataset(dataset_path, use_raw, min_threshold)
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


def visualize_tsne(X, w_train, dataset_name):
  tsne = sklearn.manifold.TSNE(n_components=2)
  X_transformed = tsne.fit_transform(X)
  x = X_transformed[:, 0]
  y = X_transformed[:, 1]

  num_clusters = np.unique(w_train).shape[0]
  color_map = plt.cm.get_cmap('nipy_spectral', num_clusters)
  plt.scatter(x, y, lw=0.1, c=w_train, cmap=color_map, edgecolors='black')
  plt.colorbar(label='cluster index')
  plt.title('TSNE Feature-Space Visualization on *%s*' % dataset_name)
  plt.xlabel("X")
  plt.ylabel("Y")
  plt.savefig("tsne_view_%s.pdf" % dataset_name)
  plt.show()


def evaluate_clustering_methods(
  methods,
  X_data,
  w_true,
  experiment_mode=False,
):
  scores = []
  loading = [
    "|=      |", "|==     |", "|===    |", "| ===   |", "|  ===  |",
    "|   === |", "|    ===|", "|     ==|", "|      =|", "|       |",
  ]

  print("\n----------------- Clustering Evaluation -----------------")
  scores_table = []
  max_len = 30
  for method in methods:
    scores = np.zeros(n_times)
    for i in range(n_times):
      if not experiment_mode:
        print(
          "  %s Running %s clustering %d/%d..." 
            % (loading[i % len(loading)], method[0], i+1, n_times),
          end='\r',
        )
      # Sample for multistep validation.
      validation_set_size = int(X_data.shape[0] * validation_set_fraction)
      indices = np.random.choice(X_data.shape[0], validation_set_size)

      # Fit and predict clustering method on shuffled (X_data, w_true).
      n_clusters = len(np.unique(w_true[indices]))
      clustering = build_clustering_method(method[0], method[1:], n_clusters)
      w_pred = clustering.fit_predict(X_data[indices])
      scores[i] = sklearn.metrics.adjusted_rand_score(w_true[indices], w_pred)

    avg_score = np.mean(scores)
    error_margin = 2*np.std(scores)
    scores_table.append(scores)

    # Print adjusted rand index.
    clustering_name = str(method)
    num_dots = max_len - len(clustering_name)
    print(
      "+ %s %s %lf (+/- %lf)"
      % (clustering_name, num_dots*".", avg_score, error_margin)
    )
  print("---------------------------------------------------------\n\n")
  print("validation_set_fraction =", validation_set_fraction)
  print("n _times =", n_times)
  print()
  print("Clustering Scores (adjusted_rand_index)")
  scores_table = np.array(scores_table)
  for i, method in enumerate(methods):
    num_spaces = max_len - len(str(method))
    print(str(method) + (" " * num_spaces), end="")
    for j in range(scores_table.shape[1]):
      print(" %1.6f" % scores_table[i][j], end="")
    print()
  print()


def build_clustering_method(method_name, parameters, n_clusters):
  if method_name == "k-means":
    K = n_clusters if parameters[0] == "opt" else parameters[0]
    return sklearn.cluster.KMeans(n_clusters=K)
  elif method_name == "agglomerative":
    K = n_clusters if parameters[0] == "opt" else parameters[0]
    return sklearn.cluster.AgglomerativeClustering(n_clusters=K)
  elif method_name == "spectral":
    K = n_clusters if parameters[0] == "opt" else parameters[0]
    return sklearn.cluster.SpectralClustering(n_clusters=K)
  elif method_name == "mean-shift":
    bandwidth = parameters[0]
    return sklearn.cluster.MeanShift(bandwidth=bandwidth)
  elif method_name == "dbscan":
    eps = parameters[0]
    min_samples = parameters[1]
    return sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
  else:
    return None


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
  parser.add_argument(
    "--min_threshold",
    help="Minimum number of pictures per folder.",
    type=int,
    default=1,
  )
  parser.add_argument(
    "--experiment_mode",
    help="True iff experiment mode (redirects stdout).",
    type=bool,
    default=False,
  )
  args = parser.parse_args()
  print(args)

  if args.experiment_mode:
    log_filepath = "%s.txt" % args.dataset
    print("Redirecting stdout to '%s'." % log_filepath)
    sys.stdout = open(log_filepath, 'w')
    print(args)

  return args


if __name__ == "__main__":
  main()
