# ASR-DeepSpeech2

## Общее

Репозиторий с проектом создания, обучения и инференса модели ASR (Automatic Speech Recognition) с архитектурой DeepSpeech2

## Обучение

Обучение проивзводилось полностью на Kaggle. Инструкция для обучения на датасете train-clean-360:

1. Заходим на страницу [датасета](https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus) и создаем ноутбук с ним (кнопка New Notebook)

2. Клонируем репозиторий

- Так как репозиторий закрытый, необходимо создать персональный токен на github. Далее в ноубуке добавляем переменную GIT_TOKEN через кнопку Add-ons в верхней панели и копируем импорт этой переменной. Далее клонируем репозиторий. Должно выглядеть примерно так:

```bash
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GIT_TOKEN")

git clone https://{secret_value_0}@github.com/yuri-pavar/ASR-DeepSpeech2.git
```

- Если репозиторий открыт, клонируем стандартно

```bash
git clone https://github.com/yuri-pavar/ASR-DeepSpeech2.git
```

3. Далее в терминале переходим в папку проекта и загружаем необходимые библиотеки

```bash
cd ASR-DeepSpeech2

pip install -r requirements.txt
```

4. Добавляем переменную окружения для мониторинга в CometML
```bash
import os
os.environ['COMET_API_KEY'] = 'your_cometml_api_key'
```

5. Далее запускаем скрипт обучения:
```bash
python3 train.py -cn=ds2_kaggle_INP_360_new_mlr_l200.yaml
```