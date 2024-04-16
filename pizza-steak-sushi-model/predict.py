import torch
import torchvision
from torchvision.transforms import v2 as transforms
import argparse

import model_builder


# Создадим парсер и аргументы
parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    help='Путь к изображению для распознавания'
)

parser.add_argument(
    '--model_path',
    type=str,
    default='models/tinyvgg_model.pth',
    help='Путь к модели'
)

# Получим путь к изображению
args = parser.parse_args()
IMAGE_PATH = args.image
print(f'[INFO] Анализируем {IMAGE_PATH}')

# Зададим путь к модели
MODEL_PATH = args.model_path
print(f'[INFO] С помощью {MODEL_PATH}')

# Определим аппаратно-независимый код
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Определим названия классов
class_names = ['pizza', 'steak', 'sushi']

# Функция для загрузки модели
def load_model(path) -> torch.nn.Module:
  # Инициализируем такую же модель, что при обучении
  model = model_builder.TinyVGG(
      in_features=3,
      hidden_features=128,
      out_features=3
  ).to(device)
  # Загрузим сохраненный state_dict
  model.load_state_dict(torch.load(path))
  return model


def predict(image, filepath):
  # Загрузим модель
  model = load_model(filepath)

  # Загрузим изображение
  image = torchvision.io.read_image(str(IMAGE_PATH)).type(torch.float) / 255.

  # Создадим конвейер обработки фото
  transform = transforms.Compose([
      transforms.Resize((64, 64)),
      transforms.ToImage()
  ])
  image = transform(image)

  # Получим результат
  model.eval()
  with torch.inference_mode():
    # Отправим на нужный девайс
    image = image.to(device)

    # Получим предсказания
    pred_probs = model(image.unsqueeze(dim=0)).softmax(dim=1).squeeze()

    pred_label = pred_probs.argmax(dim=0).item()
    pred_class = class_names[pred_label]

  print(f'[INFO] Класс: {pred_class}, вероятность: {(pred_probs[pred_label]*100):.3f}%')

if __name__ == '__main__':
  predict(IMAGE_PATH, MODEL_PATH)
