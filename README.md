### CV задачи
- [ ] Реализовать обучение через `pytorch lightning`
- [ ] Сделать систему буквально запускаемой от одной кнопки
- [ ] Перенести данные на `Google Drive` или `Yandex Drive`
- [ ] Исправить проблему того, что `pre-commit` не проверяет директорию `src`

### MLOps задачи
- [ ] **I** этап: чистый код, `venv` и `pre-commit`
    - [ ] Объединить работу [`conda`][link_conda_env] и [`poetry`][link_poetry] как в [этом][link_conda_poetry_together] примере
      - [ ] Добавить файлы:
        - environment.yml
        - virtual-packages.yml
        - conda-linux-64.lock
      - [ ] Set env. variable [PYTHONNOUSERSIT][link_pythonnousersit] to `True`
    - [ ] Разобраться с `black`, `isort`, `flake8` и при необходимости их добавить (возможно они будут конфликтовать, с этим нужно разобраться и оставить только стабильные вещи)
    - [ ] добавить `pre-commit`
- [ ] **II**-ой этап: `dvc`, `hydra`, `logging` и `onnx`
    - [ ] Установить DVC через подгрузку с дисков (не git lfs)
    - [ ] Перенести гиперпараметры в yaml конфиги hydra
    - [ ] Добавить логирование обучения (после устновки `pytorch lighning`)
    - [ ] Добавить экспорт модели в `onnx` и запускать инференс через неё
- [ ] **III**-ий этап: `Triron`
    - [ ] `TODO`

### Задачи на эстетику
- [ ] Посмотреть [лучшие](https://github.com/matiassingers/awesome-readme) `readme.md` и перенять полезные идеи
- [ ] Добавить `emoji`
- [ ] Добавить список используемых технологий и его поддерживать


[link_conda_env]: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment
[link_poetry]: https://python-poetry.org/docs/#installing-with-the-official-installer
[link_pythonnousersit]: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONNOUSERSITE
[link_conda_poetry_together]: https://stackoverflow.com/questions/70851048/does-it-make-sense-to-use-conda-poetry