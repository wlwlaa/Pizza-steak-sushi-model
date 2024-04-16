import os
from pathlib import Path
import zipfile
import requests

# Настроим пути
data_path = Path('data')
image_path = data_path / 'pizza_steak_sushi'

# Создадим директорию с изображениями
if image_path.is_dir():
  print(f'{image_path} уже существует...')
else:
  print(f'Создадим {image_path}')
  image_path.mkdir(parents=True)

# Загрузим датасет
with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
  print(f"Загружаем датасет...")
  response = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
  f.write(response.content)

# Распакуем файлы
with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as f:
  print('Распаковываем датасет...')
  f.extractall(image_path)

# Удалим архив
print('Очищаем мусор...')
os.remove(data_path / 'pizza_steak_sushi.zip')
print("Данные успешно загружены!")
