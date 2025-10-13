# Model-Based Attacks (MBA) Implementation

## Обзор

Реализованы Model-Based Attacks (MBA) - атаки, использующие предобученные суррогатные модели для генерации adversarial примеров без доступа к градиентам жертвенной модели.

## Архитектура

### Суррогатные модели
- **AttackLSTM**: LSTM-based attacker для генерации возмущений
- **AttackResidualCNN**: CNN-based attacker с ResCNN архитектурой
- **AttackPatchTST**: Transformer-based attacker с PatchTST архитектурой

### MBA атаки
- **ModelBasedAttack**: Базовая MBA атака
- **IterativeModelBasedAttack**: Итеративная MBA с расширенными возможностями
- **MBALSTMAttack**, **MBACNNAttack**, **MBAPatchTSTAttack**: Специфичные MBA атаки
- **IterativeMBALSTMAttack**, **IterativeMBACNNAttack**, **IterativeMBAPatchTSTAttack**: Итеративные версии

## Использование

### 1. Тренировка суррогатной модели

```bash
python train_attacker.py \
    attacker_model=AttackLSTM \
    dataset=PowerCons \
    model_id=100 \
    epochs=50 \
    lr=1e-4 \
    alpha_l2=1e-3 \
    eps=0.5
```

### 2. Запуск MBA атаки

```bash
python attack_run.py \
    attack=MBA_LSTM \
    dataset=PowerCons \
    model_id_attack=100 \
    attack_model=ResidualCNN
```

### 3. Итеративная MBA атака

```bash
python attack_run.py \
    attack=MBA_Iterative \
    dataset=PowerCons \
    model_id_attack=100 \
    attack_model=ResidualCNN
```

## Конфигурация

### Параметры суррогатных моделей

#### AttackLSTM
- `hidden_dim`: Размер скрытого слоя (по умолчанию: 64)
- `x_dim`: Размерность входа (по умолчанию: 1)
- `activation_type`: Тип активации ("identity", "tanh", "relu")
- `dropout`: Dropout rate (по умолчанию: 0.25)
- `num_layers`: Количество слоев LSTM (по умолчанию: 3)

#### AttackResidualCNN
- `hidden_dim`: Размер скрытого слоя (по умолчанию: 128)
- `x_dim`: Размерность входа (по умолчанию: 1)
- `activation_type`: Тип активации ("identity", "tanh", "relu")
- `rescnn_kwargs`: Параметры ResCNN (nf, kss, bn, do)

#### AttackPatchTST
- `hidden_dim`: Размер скрытого слоя (по умолчанию: 128)
- `x_dim`: Размерность входа (по умолчанию: 1)
- `activation_type`: Тип активации ("identity", "tanh", "relu")
- `patch_kwargs`: Параметры PatchTST (seq_len, n_layers, d_model, etc.)

### Параметры MBA атак

#### Базовая MBA
- `eps`: Максимальная величина возмущения (по умолчанию: 0.5)
- `n_steps`: Количество шагов (по умолчанию: 1)
- `clamp`: Ограничения на данные (по умолчанию: null)

#### Итеративная MBA
- `eps`: Максимальная величина возмущения
- `n_steps`: Количество итераций (по умолчанию: 10)
- `alpha`: Размер шага (по умолчанию: eps/n_steps)
- `rand_init`: Случайная инициализация (по умолчанию: true)
- `use_sign`: Использовать знак градиента (по умолчанию: false)
- `bpda`: Backward Pass Differentiable Approximation (по умолчанию: true)
- `proj`: Тип проекции ("none", "linf", "l2")
- `momentum_mu`: Momentum для MI-FGSM (по умолчанию: 0.0)
- `step_normalize`: Нормализация шага ("meanabs", "l2", "linf", null)
- `step_noise_std`: Стандартное отклонение шума (по умолчанию: 0.0)

## Примеры конфигов

### Тренировка суррогата
```yaml
# config/train_attacker_config.yaml
defaults:
  - dataset: PowerCons
  - attacker_model: AttackLSTM

epochs: 50
lr: 1e-4
alpha_l2: 1e-3
eps: 0.5
patience: 4
```

### MBA атака
```yaml
# config/attack/MBA_LSTM.yaml
name: MBALSTMAttack
short_name: mba_lstm_attack

attack_params:
  eps: 0.5
  n_steps: 1

attacker_model_params:
  name: AttackLSTM
  params:
    hidden_dim: 64
    x_dim: 1
    activation_type: "identity"
  path: "checkpoints/attackers/attacker_AttackLSTM_100_PowerCons.pth"
```

## Интеграция с существующей системой

MBA атаки полностью интегрированы в существующую архитектуру:
- Поддержка Hydra конфигов
- Интеграция с ClearML логированием
- Совместимость с существующими estimator'ами
- Поддержка Optuna для гиперпараметр-тюнинга

## Файловая структура

```
src/
├── models/
│   ├── AttackLSTM.py
│   ├── AttackResidualCNN.py
│   └── AttackPatchTST.py
├── attacks/
│   └── mba.py
├── training/
│   └── train_attacker.py
└── config/
    └── config.py (обновлен)

config/
├── attacker_model/
│   ├── AttackLSTM.yaml
│   ├── AttackResidualCNN.yaml
│   └── AttackPatchTST.yaml
├── attack/
│   ├── MBA_LSTM.yaml
│   ├── MBA_CNN.yaml
│   ├── MBA_PatchTST.yaml
│   └── MBA_Iterative.yaml
└── train_attacker_config.yaml

train_attacker.py
```

## Полный алгоритм использования MBA атак

### Шаг 1: Тренировка классификатора (жертвенной модели)

**Обязательно!** Перед обучением атакера необходимо обучить классификатор, который будет атаковаться.

```bash
python train_classifier.py model=LSTM dataset=PowerCons model_id=100
```

**Что происходит:**
- Обучается классификатор LSTM на датасете PowerCons
- Сохраняется в `checkpoints/model_LSTM_100_PowerCons.pt`
- Этот файл будет использоваться как жертвенная модель

**Конфиг для классификатора:**
```yaml
# config/model/LSTM.yaml
name: LSTM
params:
  hidden_dim: 64
  x_dim: 1
  activation_type: "sigmoid"
  dropout: 0.25
  num_layers: 2
```

### Шаг 2: Тренировка атакера (суррогатной модели)

Теперь обучаем атакер, который будет генерировать возмущения для атаки на классификатор.

```bash
python train_attacker.py attacker_model=AttackPatchTST dataset=PowerCons model_id=100
```

**Что происходит:**
- Загружается обученный классификатор LSTM (жертва)
- Обучается атакер AttackPatchTST для генерации возмущений
- Атакер учится максимизировать потери жертвенной модели
- Сохраняется в `checkpoints/attackers/attacker_AttackPatchTST_100_PowerCons.pth`

**Конфиг для атакера:**
```yaml
# config/attacker_model/AttackPatchTST.yaml
name: AttackPatchTST
params:
  hidden_dim: 128
  x_dim: 1
  activation_type: "identity"
  dropout: 0.25
  num_layers: 3
```

**Конфиг тренировки атакера:**
```yaml
# config/train_attacker_config.yaml
victim_model:
  name: "LSTM"  # Жертвенная модель
  params:
    n_classes: 2
    x_dim: 1
  attack_train_mode: false

attacker_model:
  name: "AttackPatchTST"  # Атакер
  params:
    hidden_dim: 128
    x_dim: 1
    activation_type: "identity"
    dropout: 0.25
    num_layers: 3

epochs: 50
lr: 1e-4
alpha_l2: 1e-3
eps: 0.5
patience: 4
```

### Шаг 3: Запуск MBA атаки

Теперь запускаем атаку, используя обученный атакер против жертвенной модели.

```bash
python attack_run.py attack=MBA_PatchTST dataset=PowerCons model_id_attack=100 attack_model=LSTM
```

**Что происходит:**
- Загружается жертвенная модель LSTM
- Загружается обученный атакер AttackPatchTST
- Атакер генерирует возмущения для тестовых данных
- Применяются возмущения к данным
- Измеряется эффективность атаки

**Конфиг для MBA атаки:**
```yaml
# config/attack/MBA_PatchTST.yaml
name: MBAPatchTSTAttack
short_name: mba_patchtst_attack

attack_params:
  eps: 0.5
  n_steps: 1

attacker_model_params:
  name: AttackPatchTST
  params:
    hidden_dim: 128
    x_dim: 1
    activation_type: "identity"
  path: "checkpoints/attackers/attacker_AttackPatchTST_100_PowerCons.pth"
```

## Примеры полных пайплайнов

### Пример 1: LSTM → AttackPatchTST → LSTM

```bash
# 1. Обучаем жертвенную модель LSTM
python train_classifier.py model=LSTM dataset=PowerCons model_id=100

# 2. Обучаем атакер PatchTST для атаки на LSTM
python train_attacker.py attacker_model=AttackPatchTST dataset=PowerCons model_id=100

# 3. Запускаем MBA атаку PatchTST на LSTM
python attack_run.py attack=MBA_PatchTST dataset=PowerCons model_id_attack=100 attack_model=LSTM
```

### Пример 2: ResidualCNN → AttackLSTM → ResidualCNN

```bash
# 1. Обучаем жертвенную модель ResidualCNN
python train_classifier.py model=ResidualCNN dataset=PowerCons model_id=200

# 2. Обучаем атакер LSTM для атаки на ResidualCNN
python train_attacker.py attacker_model=AttackLSTM dataset=PowerCons model_id=200

# 3. Запускаем MBA атаку LSTM на ResidualCNN
python attack_run.py attack=MBA_LSTM dataset=PowerCons model_id_attack=200 attack_model=ResidualCNN
```

### Пример 3: PatchTST → AttackResidualCNN → PatchTST

```bash
# 1. Обучаем жертвенную модель PatchTST
python train_classifier.py model=PatchTST dataset=PowerCons model_id=300

# 2. Обучаем атакер ResidualCNN для атаки на PatchTST
python train_attacker.py attacker_model=AttackResidualCNN dataset=PowerCons model_id=300

# 3. Запускаем MBA атаку ResidualCNN на PatchTST
python attack_run.py attack=MBA_CNN dataset=PowerCons model_id_attack=300 attack_model=PatchTST
```

## Важные замечания

### Порядок выполнения:
1. **Сначала** `train_classifier.py` - создает жертвенную модель
2. **Потом** `train_attacker.py` - создает атакер для этой жертвы
3. **Наконец** `attack_run.py` - запускает атаку

### Совместимость моделей:
- Атакер и жертвенная модель могут иметь **разные архитектуры**
- Главное - совместимость входных/выходных размеров
- Все модели должны быть обучены на **одном датасете**

### Файловая структура результатов:
```
checkpoints/
├── model_LSTM_100_PowerCons.pt                    # Жертвенная модель
├── model_ResidualCNN_200_PowerCons.pt             # Другая жертвенная модель
├── model_PatchTST_300_PowerCons.pt                # Еще одна жертвенная модель
└── attackers/
    ├── attacker_AttackPatchTST_100_PowerCons.pth  # Атакер для LSTM
    ├── attacker_AttackLSTM_200_PowerCons.pth      # Атакер для ResidualCNN
    └── attacker_AttackResidualCNN_300_PowerCons.pth # Атакер для PatchTST
```

### Итеративные MBA атаки:
Для более сложных атак используйте итеративные версии:

```bash
python attack_run.py attack=MBA_Iterative dataset=PowerCons model_id_attack=100 attack_model=LSTM
```

Итеративные атаки поддерживают:
- Множественные шаги атаки
- Momentum (MI-FGSM)
- Проекции (L2, L∞)
- BPDA (Backward Pass Differentiable Approximation)
- Случайную инициализацию

## Примечания

- Суррогатные модели должны быть предобучены перед использованием в MBA атаках
- Итеративные MBA атаки поддерживают расширенные техники (momentum, проекции, BPDA)
- Все MBA атаки совместимы с существующими метриками и логированием
- Поддерживается загрузка суррогатных моделей из ClearML
