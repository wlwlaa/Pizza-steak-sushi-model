from tqdm.auto import tqdm
import torch
from typing import Dict, Tuple, List


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
  ) -> Tuple[float, float]:
  """
  Одноэпохальное обучение модели

  Функция производит проход по даталоадеру с выполнением всех
  шагов обучения модели.

  Args:
    model: Обучаемая модель.
    dataloader: Тренировочный даталоадер.
    loss_fn: Функция для нахождения обратной ошибки.
    optimizer: Оптимайзер для минимизации функции ошибки.
    device: Девайс, на котором будет происходить обучение.
  """

  # Переведем модель в режим обучения
  model.train()

  # Установим точность и ошибку
  train_loss, train_acc = 0, 0

  # Выполним проход по даталоадеру
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # 1. Прямой проход
    preds = model(X)

    # 2. Вычислим функцию ошибки
    loss = loss_fn(preds, y)
    train_loss += loss.item()

    # 2.1 Вычислим точность
    pred_labels = torch.argmax(torch.softmax(preds, dim=1), dim=1)
    train_acc += (pred_labels == y).sum().item() / len(y)

    # 3. Занулим градиенты
    optimizer.zero_grad()

    # 4. Высчитаем градиент
    loss.backward()

    # 5. Обновим коэффициенты
    optimizer.step()

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
  ) -> Tuple[float, float]:
  """
  Одноэпохальное тестирование модели

  Функция производит проход по даталоадеру с выполнением всех
  шагов обучения модели.

  Args:
    model: Тестируемая модель.
    dataloader: Тестовый даталоадер.
    loss_fn: Функция для нахождения обратной ошибки.
    device: Девайс, на котором будет происходить обучение.
  """
  # Установим точность и ошибку
  test_loss, test_acc = 0, 0

  # Переведем модель в режим тестирования
  model.eval()
  with torch.inference_mode():
  # Выполним проход по даталоадеру
    for X_test, y_test in dataloader:
      X_test, y_test = X_test.to(device), y_test.to(device)

      # 1. Прямой проход
      test_preds = model(X_test)

      # 2. Вычислим функцию ошибки
      test_loss += loss_fn(test_preds, y_test).item()

      # 2.1 Вычислим точность
      test_labels = torch.argmax(torch.softmax(test_preds, dim=1), dim=1)
      test_acc += (test_labels == y_test).sum().item() / len(y_test)

  test_loss /= len(dataloader)
  test_acc /= len(dataloader)

  return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device
  ) -> Dict[str, List[float]]:
  """
  Полный цикл обучения и тестирования модели

  Производит train_step() и test_step() заданное количество эпох. 
  Высчитывает, сохраняет и возвращает метрики процесса обучения модели.

  Args:
    model: Модель, которую необходимо обучить.
    train_dataloader: Даталоадер с данными для обучения.
    test_dataloader: Даталоадер с данными для тестирования.
    loss_fn: Функция для нахождения обратной ошибки.
    optimizer: Оптимайзер для обновления коэффициентов модели.
    epochs: Целое число, количество процедур обучения и тестирования модели.
    device: Устройство, на которм будет происходить обучение

  Returns:
    Словарь с основными метриками процессов обучения и тестирования модели. 
  """
  # Создадим словарь для отслеживания метрик
  results = {
      'train_loss': [],
      'train_acc': [],
      'test_loss': [],
      'test_acc': []
  }

  # Пройдем этапы тестирования и обучения заданное кол-во эпох
  for epoch in tqdm(range(epochs)):
    # Произведем цикл обучения модели
    train_loss, train_acc = train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )
    # Произведем цикл тестирования модели
    test_loss, test_acc = test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device
    )

    # Выведем метрики
    print(
        f"""
        Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train accuracy: {(train_acc*100):.2f}% | Test loss: {test_loss:.4f} | Test accuracy: {(test_acc*100):.2f}%
        """
    )

    # Зафиксируем данные
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

  return results
