import shutil
import os
# Filter only ERROR messages in TensorFlow logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np
import sklearn.cluster
from imageio import imsave

from utils.dataset_loader import DatasetLoader
from code.trained_keras_facenet import TrainedKerasFacenet


def main():
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

  K_values = list(range(15, 21))
  for K in K_values:
    # Perform K-means clustering.
    print("Performing clustering with K = %d." % K)
    clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=K)
    labels = clustering.fit_predict(embeddings)
    # clustering = sklearn.cluster.KMeans(n_clusters=K, random_state=0)
    # labels = clustering.fit_predict(embeddings)

    # Save grouped pictures.
    output_dir = "output/K=%d/" % K
    if os.path.isdir(output_dir):
      shutil.rmtree(output_dir)

    for k, face in enumerate(faces):
      cluster = labels[k]
      cluster_path = output_dir + "cluster_%d/" % (cluster)

      if not os.path.isdir(cluster_path):
        os.makedirs(cluster_path)

      out_img_path = (cluster_path + "%d.png") % (k)
      imsave(out_img_path, face)

if __name__ == "__main__":
  main()
