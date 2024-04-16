"""
Содержит вспомогательные фукнции для работы с моделью
"""
import torch
from pathlib import Path

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str
  ):
  """
  Производит сохранение модели ы выбранную директорию

  Args:
    model: Модель, которую необходимо сохранить.
    target_dir: Путь к директории для сохранения.
    model_name: Название модели, которое будет использоваться при сохранении. 
    Должно также содержать формат (".pt" или  ".pth")
  """
  # Создадим директорию для сохранения
  target_dir_path = Path(target_dir)
  if target_dir_path.is_dir():
    print("Целевая директория уже существует")
  else:
    target_dir_path.mkdir(parents=True)

  # Создадим путь сохранения модели
  assert model_name.endswith('.pt') or model_name.endswith('.pth'), 'Название модели должно содержать формат: ".pt" или ".pth"'
  model_save_path = target_dir_path / model_name

  # Сохраним state_dict модели
  print(f'[INFO] Сохранение модели в {target_dir_path}')
  torch.save(
      obj=model.state_dict(),
      f=model_save_path
  )
