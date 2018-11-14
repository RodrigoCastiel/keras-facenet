import glob
import matplotlib as plt
import numpy as np
import os

from imageio import imread
from skimage.transform import resize

import cv2


class DatasetLoader:
  def __init__(self, align=True, margin=10, face_img_size=160):
    self.align = align
    self.margin = margin
    self.face_img_size = face_img_size
    self.cascade_path = 'model/cv2/haarcascade_frontalface_alt2.xml'
    self.cascade = cv2.CascadeClassifier(self.cascade_path)

  def load_from_folder(self, folder_path, use_raw=False):
    jpg_filepaths = glob.glob(folder_path + "*.jpg")
    png_filepaths = glob.glob(folder_path + "*.png")
    filepaths = jpg_filepaths + png_filepaths
    return self.load_and_align_images(filepaths, use_raw=use_raw)

  def load_from_folder_recursive(self, base_folder_path, use_raw=False):
    jpg_filepaths = glob.glob(base_folder_path + "/**/*.jpg", recursive=True)
    png_filepaths = glob.glob(base_folder_path + "/**/*.png", recursive=True)
    filepaths = jpg_filepaths + png_filepaths
    print("Loading %d images.." % len(filepaths))
    return self.load_and_align_images(filepaths, use_raw=use_raw)

  def load_test_dataset(self, data_path, use_raw=False):
    """
    Test dataset should contain single-face pictures only.
    They must be grouped by similarity into different folders under 'data_path'.
    Each folder is considered to be a cluster.
    It returns two arrays: (images, labels).
    """
    if data_path[-1] != "/":
      data_path += "/"

    # List all paths in 'data_path', and filter in folders only.
    folders = [(data_path + path) for path in os.listdir(data_path)
                                  if os.path.isdir(data_path + path)]

    # Load images.
    images = []
    labels = []
    num_clusters = len(folders)
    cluster = 0
    for (i, folder) in enumerate(folders):
      print("  Processing folder %d/%d." % (i, num_clusters), end="\r")
      filepaths = glob.glob(folder + "/*.jpg") + glob.glob(folder + "/*.png")

      # Skip folders containing only one picture.
      if len(filepaths) < 3:
        continue

      # Load image, and crop the first detected face.
      cluster_faces, _ = self.load_and_align_images(filepaths, False, use_raw)
      for face in cluster_faces:
        images.append(face)
        labels.append(cluster)
      cluster += 1

    return (np.array(images), np.array(labels))

  def load_and_align_images(self, filepaths, all_faces=True, use_raw=False):
    aligned_faces = []
    face_metadata = []

    for filepath in filepaths:
      face_pics, metadata = self.load_and_align_img(filepath, all_faces, use_raw)
      aligned_faces.extend(face_pics)
      face_metadata.extend(metadata)

    return np.array(aligned_faces), face_metadata

  def load_and_align_img(self, filepath, all_faces=True, use_raw=False):
    L = self.face_img_size
    img = imread(filepath)

    if use_raw:
      return ([img], [{
        "filepath": filepath,
        "rect": (0, 0, img.shape[1], img.shape[0]),
      }])

    faces = self.cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(faces) and not all_faces:
      faces = [faces[0]]

    face_pics = []
    face_metadata = []
    for (x, y, w, h) in faces:
      if x < self.margin or y < self.margin:
        continue
      cropped = img[y-self.margin//2:y+h+self.margin//2,
                    x-self.margin//2:x+w+self.margin//2, :]
      face_pics.append(
        resize(cropped, (L, L), mode='reflect', anti_aliasing=True)
      )
      face_metadata.append({
        "filepath": filepath,
        "rect": (x, y, w, h),
      })

    return (face_pics, face_metadata)
