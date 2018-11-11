from utils.dataset_loader import DatasetLoader
from code.trained_keras_facenet import TrainedKerasFacenet

def main():
  # Load dataset of images.
  dataset_path = 'data/images'
  loader = DatasetLoader()
  images = loader.load_from_folder_recursive(dataset_path)

  # Load pre-trained facenet model, and compute embeddings.
  facenet = TrainedKerasFacenet(filepath='model/keras/model/facenet_keras.h5')
  embeddings = facenet.compute_embbedings(images)
  print(embeddings.shape)

if __name__ == "__main__":
  main()
