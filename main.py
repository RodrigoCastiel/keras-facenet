import shutil
import os
# Filter only ERROR messages in TensorFlow logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np
from sklearn.cluster import KMeans
from imageio import imsave

from utils.dataset_loader import DatasetLoader
from code.trained_keras_facenet import TrainedKerasFacenet


def main():
  # Load dataset of images.
  dataset_path = 'data/images/personal'
  print("Loading pictures under '%s'." % dataset_path)
  loader = DatasetLoader()
  faces, _ = loader.load_from_folder_recursive(dataset_path)

  # Load pre-trained facenet model, and compute embeddings.
  facenet = TrainedKerasFacenet(filepath='model/keras/model/facenet_keras.h5')
  embeddings = facenet.compute_embbedings(faces)
  print(">> %d face pictures extracted." % embeddings.shape[0])

  K_values = list(range(21, 26))
  for K in K_values:
    # Perform K-means clustering.
    print("\nPerforming K-means clustering with K = %d." % K)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(embeddings)

    # Save grouped pictures.
    output_dir = "output/K=%d/" % K
    if os.path.isdir(output_dir):
      shutil.rmtree(output_dir)

    for k, face in enumerate(faces):
      cluster =  kmeans.labels_[k]
      cluster_path = output_dir + "cluster_%d/" % (cluster)

      if not os.path.isdir(cluster_path):
        os.makedirs(cluster_path)

      out_img_path = (cluster_path + "%d.png") % (k)
      imsave(out_img_path, face)

if __name__ == "__main__":
  main()
