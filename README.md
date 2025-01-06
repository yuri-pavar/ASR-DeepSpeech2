# ASR-DeepSpeech2

## Общее

Репозиторий с проектом создания, обучения и инференса модели ASR (Automatic Speech Recognition) с архитектурой DeepSpeech2

Обучение производилось полностью на Kaggle. В ходе работы было сделано:
1. тесты и обучение на датасете train-clean-100
2. обучение на датасете train-clean-360
3. дообучение модели (обученной на train-clean-360) на датасете train-other-500
4. обучение на датасете train-other-500

## Обучение

### 1. Обучение train-clean-360

Время: 07:04 ч = 2550 итераций = 17 эпох
Экперимент CometML: ds2_train360


Инструкция:

1. Заходим на страницу [датасета](https://www.kaggle.com/datasets/a24998667/librispeech) и создаем ноутбук с ним (кнопка New Notebook)

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

### 2. Дообучение train-other-500

Время: 07:04 ч + ~ 6 ч = 2550 итераций  + х итераций = 17 эпох + 14 эпох
Экперимент CometML: ds2_train360

Инструкция для выполнения на kaggle:

1. Заходим на страницу [датасета](https://www.kaggle.com/datasets/a24998667/librispeech) и создаем ноутбук с ним (кнопка New Notebook)

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

5. Далее запускаем скрипт для дообучения и получаем ошибку. Для запуска с resume_from нужна директория /kaggle/working/ASR-DeepSpeech2/saved/testing/, которая создается при запуске:
```bash
python3 train.py -cn=ds2_kaggle_INP_360_new_mlr_l200_augs_ft.yaml
```

6. Скачиваем обученную ранее модель в директорию /kaggle/working/ASR-DeepSpeech2/saved/testing/
```bash
import gdown

url = 'https://drive.google.com/uc?export=download&id=1Kx07D3t3f1QAADRMwQMmNHeTsmsTmNiK'

output = '/kaggle/working/ASR-DeepSpeech2/saved/testing/checkpoint-epoch18.pth'

gdown.download(url, output, quiet=False)
```

7. Далее запускаем скрипт для дообучения:
```bash
python3 train.py -cn=ds2_kaggle_INP_360_new_mlr_l200_augs_ft.yaml
```

### 3. Обучение train-other-500

Время: 05:40 ч = 1950 итераций = 13 эпох
Экперимент CometML: ds2_train500


Инструкция для выполнения на kaggle:

1. Заходим на страницу [датасета](https://www.kaggle.com/datasets/a24998667/librispeech) и создаем ноутбук с ним (кнопка New Notebook)

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
python3 train.py -cn=ds2_kaggle_INP_360_new_mlr_l200_augs_o500.yaml
```

## Инференс

Инструкция для выполнения на kaggle:

1. Заходим на страницу [датасета](https://www.kaggle.com/datasets/a24998667/librispeech) и создаем ноутбук с ним (кнопка New Notebook)

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

4. Скачиваем обученную модель
```bash
import gdown

# url = 'https://drive.google.com/uc?export=download&id=1Kx07D3t3f1QAADRMwQMmNHeTsmsTmNiK' #model2
url = 'https://drive.google.com/uc?export=download&id=17-f--5-k-7NpvpWepVQ2N8S1bXdvNUtu' #model3
output = '/kaggle/working/ASR-DeepSpeech2/bmodel2.pth'
gdown.download(url, output, quiet=False)
```

5. Далее запускаем скрипт инференса:
```bash
python3 inference.py -cn=inference_bmodel2_kaggle.yaml
```

## Результаты

Первые 2 из наиболее успешных экспериментов показали схожие результаты CER около 0.35 и WER около 0.62 .

Во время третьего эксперимента kaggle остановился и среда была полностью потеряна, поэтому инференса модели нет.

[Отчет в CometML](https://www.comet.com/yuripavar/pytorch-template-asr-example/view/new/panels)

| metric                   | 360    | 360+500 | 500    |
|--------------------------|--------|---------|--------|
| test_CER_(Argmax)        | 0.3783 | 0.3597  |   -    |
| test_WER_(Argmax)        | 0.8080 | 0.7869  |   -    |
| test_CER_(BeamSearch)    | 0.9851 | 0.9846  |   -    |
| test_WER_(BeamSearch)    | 1.0    | 1.0     |   -    |
| est_CER_(LM_BeamSearch)  | 0.3668 | 0.3460  |   -    |
| test_WER_(LM_BeamSearch) | 0.6303 | 0.6181  |   -    |