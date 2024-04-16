"""
Фунция для создания даталоадеров из изображений для мультиклассвой классификации
"""
import os

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as transforms

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers = NUM_WORKERS
  ):
  """
  Создает обучающий и тестовый даталоадеры

  Args:
    train_dir: Путь к обучающей директории
    test_dir: Путь к тестовой директории
    transform: Torchvision transforms для преобразования изображений
    batch_size: Количество сэмплов на батч в даталоадере
    num_workers: Целое число процессов, передающих данные на GPU
  
  Returns:
    Кортеж элементов (train_dataloader, test_dataloader, class_names)

    Пример использования: 
    train_dataloader, test_dataloader, class_names = create_dataloaders(
                                  train_dir=path/to/train_dir,
                                  test_dir=path/to/test_dir,
                                  transform=transforms.Compose(['some transforms']),
                                  batch_size=32,
                                  num_workers=4
                                )
  """
  # Создадим датасеты из изображений
  train_data = datasets.ImageFolder(
      root=train_dir,
      transform=transform
  )

  test_data = datasets.ImageFolder(
      root=test_dir,
      transform=transform
  )

  # Получим список имен классов
  class_names = train_data.classes

  # Создадим даталоадеры
  train_dataloader = DataLoader(
      dataset=train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=NUM_WORKERS
  )

  test_dataloader = DataLoader(
      dataset=test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=NUM_WORKERS
  )

  return train_dataloader, test_dataloader, class_names
