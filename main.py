import shutil
import os
# Filter only ERROR messages in TensorFlow logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import argparse
import numpy as np
import sklearn.cluster
from imageio import imsave

from utils.dataset_loader import DatasetLoader
from code.trained_keras_facenet import TrainedKerasFacenet

def main():
  args = parse_command_line_args()

  # Load dataset of images.
  dataset_path = 'data/personal_faces'
  print("Loading pictures under '%s'." % dataset_path)
  loader = DatasetLoader()
  faces, _ = loader.load_from_folder_recursive(dataset_path, True)
  print(">> %d face pictures extracted." % faces.shape[0])

  # Load pre-trained facenet model, and compute embeddings.
  print("Computing embeddings.. ", end='')
  facenet = TrainedKerasFacenet(filepath='model/keras/model/facenet_keras.h5')
  embeddings = facenet.compute_embbedings(faces)
  print("DONE")

  if args.method == "mean-shift":
    bandwidths = [0.55, 0.60, 0.65]
    run_meanshift_clustering(bandwidths, faces, embeddings)
  else:
    K_values = np.arange(15, 21)
    run_clustering_with_k(K_values, args.method, faces, embeddings)


def run_clustering_with_k(K_values, method, faces, embeddings):
  for K in K_values:
    # Perform clustering with hyper-parameter K.
    clustering = build_clustering_method(method, [K])
    print("Performing %s clustering with K = %d." % (method, K))
    labels = clustering.fit_predict(embeddings)
    output_dir = "output/%s/K=%d/" % (method, K)
    save_clusters(faces, labels, output_dir)


def run_meanshift_clustering(bandwidths, faces, embeddings):
  method = "mean-shift"
  for bandwidth in bandwidths:
    # Perform clustering with hyper-parameter 'bandwidth'.
    clustering = build_clustering_method(method, [bandwidth])
    print(
      "Performing %s clustering with bandwidth = %f." 
      % (method, bandwidth)
    )
    labels = clustering.fit_predict(embeddings)
    output_dir = "output/%s/bw=%f/" % (method, bandwidth)
    save_clusters(faces, labels, output_dir)


def save_clusters(faces, predicted_labels, output_dir):
  # Save grouped pictures.
  if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)

  for k, face in enumerate(faces):
    cluster = predicted_labels[k]
    cluster_path = output_dir + "cluster_%d/" % (cluster)

    if not os.path.isdir(cluster_path):
      os.makedirs(cluster_path)

    out_img_path = (cluster_path + "%d.png") % (k)
    imsave(out_img_path, face)


def build_clustering_method(method_name, parameters):
  if method_name == "k-means":
    K = parameters[0]
    return sklearn.cluster.KMeans(n_clusters=K)
  elif method_name == "agglomerative":
    K = parameters[0]
    return sklearn.cluster.AgglomerativeClustering(n_clusters=K)
  elif method_name == "spectral":
    K = parameters[0]
    return sklearn.cluster.SpectralClustering(n_clusters=K)
  elif method_name == "mean-shift":
    bandwidth = parameters[0]
    return sklearn.cluster.MeanShift(bandwidth=bandwidth)
  else:
    return None


def parse_command_line_args():
  parser = argparse.ArgumentParser(description="")
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
    "--method",
    help= "k-means, agglomerative, mean-shift or spectral",
    type=str,
    default="k-means"
  )
  args = parser.parse_args()
  print(args)
  return args


if __name__ == "__main__":
  main()
