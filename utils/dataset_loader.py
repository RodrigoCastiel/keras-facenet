import glob
import numpy as np
import matplotlib as plt
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

  def load_from_folder(self, folder_path):
    jpg_filepaths = glob.glob(folder_path + "*.jpg")
    png_filepaths = glob.glob(folder_path + "*.png")
    filepaths = jpg_filepaths + png_filepaths
    return self.load_and_align_images(filepaths)

  def load_from_folder_recursive(self, base_folder_path):
    jpg_filepaths = glob.glob(base_folder_path + "/**/*.jpg", recursive=True)
    png_filepaths = glob.glob(base_folder_path + "/**/*.png", recursive=True)
    filepaths = jpg_filepaths + png_filepaths
    return self.load_and_align_images(filepaths)

  def load_and_align_images(self, filepaths):
    aligned_faces = []
    for filepath in filepaths:
      aligned_faces.extend(self.load_and_align_img(filepath))
    return np.array(aligned_faces)

  def load_and_align_img(self, filepath):
    L = self.face_img_size
    img = imread(filepath)
    faces = self.cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    face_pics = []
    for (x, y, w, h) in faces:
      cropped = img[y-self.margin//2:y+h+self.margin//2,
                    x-self.margin//2:x+w+self.margin//2, :]
      face_pics.append(
        resize(cropped, (L, L), mode='reflect', anti_aliasing=True)
      )
    return face_pics
