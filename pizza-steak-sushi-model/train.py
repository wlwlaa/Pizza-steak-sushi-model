"""
Обучает мультиклассификационную модель используя аппаратно-независимый код
"""
import os
import argparse

import torch
from torchvision.transforms import v2 as transforms

import data_setup, engine, model_builder, utils


# Настроим парсер
parser = argparse.ArgumentParser(description='Введите гиперпараметры')

# Зададим аргументы для каждого параметра
parser.add_argument(
    '--num_epochs',
    default=10,
    type=int,
    help='Количество эпох обучения'
)

parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='Количество элементов батча'
)

parser.add_argument(
    '--hidden_units',
    default=10,
    type=int,
    help='Количество элементов скрытого слоя'
)

parser.add_argument(
    '--learning_rate',
    default=0.001,
    type=float,
    help='Темп обучения'
)

parser.add_argument(
    '--train_dir',
    default='data/pizza_steak_sushi/train',
    type=str,
    help='Путь к директории обучения'
)

parser.add_argument(
    '--test_dir',
    default='data/pizza_steak_sushi/test',
    type=str,
    help='Путь к директории тестирования'
)

# Получим аргументы из парсера
args = parser.parse_args()

# Зададим гиперпараметры
EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
print(f'[INFO] Обучение модели с {HIDDEN_UNITS} элементов в скрытом слое в {EPOCHS} эпох, размер батча {BATCH_SIZE}, темп обучения {LEARNING_RATE}')

# Настроим пути
train_dir = args.train_dir
test_dir = args.test_dir
print(f'[INFO] Путь обучения: {train_dir}')
print(f'[INFO] Путь тестирования: {test_dir}')

# Зададим устройство
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Создадим конвейер преобразований
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Создадим даталоадеры и получим имена классов
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=transform,
    batch_size=BATCH_SIZE
)

# Создадим модель с помощью model_builder.py
model = model_builder.TinyVGG(
    in_features=3,
    hidden_features=HIDDEN_UNITS,
    out_features=len(class_names)
).to(device)

# Настроим loss и optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Обучим модель с помощью engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=EPOCHS,
    device=device
)

# Сохраним модель с помощью utils.py
utils.save_model(
    model=model,
    target_dir='models',
    model_name='tinyvgg_model.pth'
)
