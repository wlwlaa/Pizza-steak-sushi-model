"""
Содержит вспомогательные фукнции для работы с моделью
"""
import torch
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torchvision.transforms import v2 as transforms

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


# 1. Получим данные
def predAndPlotImage(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device
  ) -> None:
  # 2. Откроем изображение с помощью PIL
  img = Image.open(image_path)

  # 3. Создадим преобразования, если они не заданы
  if not transform:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

  ### Классификация ###
  # 4. Убедимся, что модель на нужном девайсе
  model.to(device)

  # 5. Включим режим получения результатов
  model.eval()
  with torch.inference_mode():
    # 6. Преобразуем изображение
    transformed_img = transform(img).unsqueeze(dim=0).to(device) # [batch_size, color_channels, height, width]

    # 7. Получим вероятности
    preds = model(transformed_img).softmax(dim=1).squeeze(dim=0)

  # 8. Получим метку
  label = preds.argmax(dim=0).item()

  # 9. Выведем результат
  plt.figure()
  plt.imshow(img)
  plt.axis(False)
  plt.title(f'Это {class_names[label]} с вероятностью {(preds[label] * 100):.2f}%');


def predCustomImage(
    model: nn.Module,
    url: str,
    class_names: List[str],
    data_path: str
  ) -> None:
  data_path = Path(data_path)

  if not data_path.is_dir():
    data_path.mkdir(parents=True)
    print(f'[INFO] Создаем папку {data_path}')

  custom_path = data_path / url.split('/')[-1]
  print(f'[INFO] Загрузим {custom_path}...')
  responce = requests.get(url)

  with open(custom_path, 'wb') as f:
    f.write(responce.content)

  predAndPlotImage(
      model=model,
      image_path=custom_path,
      class_names=class_names
  )
