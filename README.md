[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pre-commit/pre-commit-hooks/main.svg)](https://results.pre-commit.ci/latest/github/pre-commit/pre-commit-hooks/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

## Как использовать

### Первый запуск \[Linux\]

Предполагается, что у вас уже установлена [`conda`][link_miniconda_install].

В первый раз после скачивания нужно создать `virtual environment (venv)` и установить в
него все пакеты:

```bash
conda create --name cat_env --file conda-linux-64.lock
conda activate cat_env
poetry install
```

### Использование

Подключите созданный на первом шаге `venv`

```bash
conda activate cat_env
```

## Задачи

### CV задачи

- [ ] Реализовать обучение через `pytorch lightning`
- [ ] Сделать систему буквально запускаемой от одной кнопки
- [ ] Перенести данные на `Google Drive` или `Yandex Drive`
- [ ] Исправить проблему того, что `pre-commit` не проверяет директорию `src`

### MLOps задачи

- [x] **I** этап: чистый код, `venv` и `pre-commit` \
  - [x] Объединить работу [`conda`][link_conda_env] и [`poetry`][link_poetry] как в
        [этом][link_conda_poetry_together] примере \
  - [x] Добавить файлы:
    - [x] environment.yml [\[link\]][link_conda_env]
    - [x] virtual-packages.yml [\[link 1, ][link_conda_lock_1] [link
          2\]][link_conda_lock_2]
    - [x] conda-linux-64.lock
  - [x] Добавить поддержку `GPU`
  - [x] Set env. variable [PYTHONNOUSERSIT][link_pythonnousersit] to `True`
  - [x] Разобраться с `black`, `isort`, `flake8`, `bandit`, `nbQA`, `mirrors-prettier` и
        при необходимости их добавить в `pre-commit` (возможно они будут конфликтовать,
        с этим нужно разобраться и оставить только стабильные вещи)
  - [x] добавить `pre-commit`
- [ ] **II**-ой этап: `dvc`, `hydra`, `logging` и `onnx`
  - [x] Установить [DVC][link_dvc_get_started] через подгрузку с дисков (не git lfs)
    - [x] Внедрить эффективное хранение большого количества маленьких фото
      - [Пример][link_dvc_zip_or_array] из жизни: как компания столкнуласть с долгим
        скачаиванием большого количества файлов и выбирала между `zip` и `parquet`
      - High performance [`parquet-cpp`][link_parquet_cpp_habr]
    - [x] Добавить тесты для системы хранения фото
    - [ ] Добавить хранение моделей после обучения
  - [x] Переписать обучение модели через `pytorch lighning`
  - [ ] Добавить логирование обучения
  - [ ] Перенести гиперпараметры в yaml конфиги hydra
  - [ ] Добавить экспорт модели в `onnx` и запускать инференс через неё
- [ ] **III**-ий этап: `Triton`
  - [ ] [`TODO`][link_todo]
- [ ] **IV**-ый этап: `OS`
  - [ ] Сделать совместимость проекта с `Windows`
  - [ ] Добавить [Sphinx][link_habr_sphinx]
  - [ ] Подключить аугаментацию через `DALI`

### Задачи на эстетику

- [ ] Посмотреть [лучшие](https://github.com/matiassingers/awesome-readme) `readme.md` и
      перенять полезные идеи
- [ ] Добавить `emoji`
- [ ] Добавить список используемых технологий и его поддерживать

[link_miniconda_install]:
  https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
[link_conda_env]:
  https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment
[link_poetry]: https://python-poetry.org/docs/#installing-with-the-official-installer
[link_pythonnousersit]:
  https://docs.python.org/3/using/cmdline.html#envvar-PYTHONNOUSERSITE
[link_conda_poetry_together]:
  https://stackoverflow.com/questions/70851048/does-it-make-sense-to-use-conda-poetry
[link_todo]: https://en.wikipedia.org/wiki/Comment_(computer_programming)#Tags
[link_conda_lock_1]: https://github.com/conda/conda-lock
[link_conda_lock_2]:
  https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html
[link_habr_sphinx]: https://habr.com/ru/companies/otus/articles/713992/
[link_dvc_get_started]: https://dvc.org/doc/start/data-management/data-versioning
[link_dvc_zip_or_array]:
  https://fizzylogic.nl/2023/01/13/did-you-know-dvc-doesn-t-handle-large-datasets-neither-did-we-and-here-s-how-we-fixed-it
[link_parquet_cpp_habr]: https://habr.com/ru/companies/otus/articles/503132/

### Как скачать и сериализовать сырые картинки

```bash
python3 src/init_serialized_data.py
```
