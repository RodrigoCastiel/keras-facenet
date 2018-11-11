import numpy as np
from imageio import imread
from scipy.spatial import distance
from keras.models import load_model

class TrainedKerasFacenet:
  def __init__(self, filepath):
    """
    Loads a pre-trained Keras-Facenet model, such as Inception-ResNet V1, from
    filepath.
    """
    self.model = load_model(filepath)

  def compute_embbedings(self, images, batch_size=1):
    """
    Compute the embeddings (i.e., feature vector) for each input image.
    Returns a [N x d] np.array, where each of N rows is an embedding.
    """
    processed_imgs = TrainedKerasFacenet.preprocess_image(images)
    pd = []
    for img in processed_imgs:
      pd.append(self.model.predict_on_batch(np.array([img])))
    embs = TrainedKerasFacenet.l2_normalize(np.concatenate(pd))
    return embs

  @staticmethod
  def l2_normalize(img, axis=-1, epsilon=1e-10):
    """
    Normalizes an image using the L2 norm.
    """
    output = img / np.sqrt(
      np.maximum(np.sum(np.square(img), axis=axis, keepdims=True), epsilon)
    )
    return output

  @staticmethod
  def preprocess_image(img):
    if img.ndim == 4:
      axis = (1, 2, 3)
      size = img[0].size
    elif img.ndim == 3:
      axis = (0, 1, 2)
      size = img.size
    else:
      raise ValueError('Number of dimensions should be 3 or 4.')

    mean = np.mean(img, axis=axis, keepdims=True)
    std = np.std(img, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    processed = (img - mean) / std_adj
    return processed
