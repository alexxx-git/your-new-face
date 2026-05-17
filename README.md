# Your New Face

Веб-сервис для генерации возрастного портрета по загруженному фото.

Пользователь загружает изображение, выбирает возраст, задача уходит в очередь,
один GPU worker выполняет inference, результат сохраняется в MinIO и
показывается в веб-интерфейсе.

## Состав

- `nginx` — единая входная точка на `http://localhost/`.
- `static` — веб-интерфейс из `api/static`.
- `api` — FastAPI: загрузка фото, статусы задач, выдача изображений и метрик.
- `celery_worker` — GPU worker, выполняет генерацию через `new_face/test_single_pass.py`.
- `rmq` — RabbitMQ, очередь задач.
- `postgres` — хранит задачи и метрики генерации.
- `minio` — хранит входные и выходные изображения.
- `flower` — мониторинг Celery на `http://localhost:5555`.
- `prometheus`, `grafana`, `cadvisor`, `nvidia-gpu-exporter`, `loki`, `promtail` — мониторинг.

## Подготовка

Скопируйте пример окружения:

```powershell
copy .env.example .env
```

Проверьте значения в `.env`, особенно пароли и порты.

ML-веса не хранятся в репозитории и не копируются в Docker image. По умолчанию
они монтируются из `D:/Projects/models`:

```yaml
D:/Projects/models:/yournewface/new_face/models:ro
```

Ожидаемая структура:

```text
D:/Projects/models/
  CLIP-ViT-H-14-laion2B-s32B-b79K/
  RealVisXL_V5.0/
  Adapter/
  Lora/
  buffalo_l/               # InsightFace, если уже скачан
```

Если путь другой, поменяйте volume у `celery_worker` в `docker-compose.yml`.

## Запуск

Собрать образы:

```powershell
docker compose build api celery_worker flower
```

Запустить сервисы:

```powershell
docker compose up -d
```

Открыть интерфейс:

```text
http://localhost/
```

Первый inference может быть медленным: worker загружает модель в VRAM.
После первой задачи pipeline остаётся в памяти процесса worker и переиспользуется.

## Мониторинг

Grafana:

```text
http://localhost:3000
```

Логин и пароль задаются в `.env`:

```text
GRAFANA_ADMIN_USER
GRAFANA_ADMIN_PASSWORD
```

Дашборд:

```text
Dashboards -> Your New Face -> Container Basics
```

Прямая ссылка:

```text
http://localhost:3000/d/your-new-face-overview/container-basics?orgId=1&refresh=10s
```

На дашборде есть:

- CPU/RAM/network контейнеров проекта.
- GPU VRAM.
- Время inference.
- Метрики качества: composite score, face similarity, LPIPS.
- Age metrics: target age, predicted age, age MAE.
- Processing queue: сколько задач ожидает обработки.

Prometheus endpoint с ML-метриками:

```text
http://localhost/api/generation-metrics
```

## Полезные команды

Статус контейнеров:

```powershell
docker compose ps
```

Логи GPU worker:

```powershell
docker compose logs -f celery_worker
```

Пересобрать API после изменений в интерфейсе/API:

```powershell
docker compose build api
docker compose up -d --force-recreate api
docker compose restart nginx prometheus grafana
```

Пересобрать worker после изменений ML-кода или зависимостей:

```powershell
docker compose build celery_worker
docker compose up -d --force-recreate celery_worker
```

Проверить GPU внутри worker:

```powershell
docker compose exec -T celery_worker python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Смотреть VRAM:

```powershell
nvidia-smi -l 1
```

## Ограничения

- Обрабатывается одна задача за раз: `celery_worker` запущен с `--pool=solo --concurrency=1`.
- Ползунок возраста ограничен диапазоном `20-60`.
- Максимальный размер файла после клиентского сжатия: `5 MB`.
- Репозиторий не должен хранить модели, датасеты, временные результаты, тестовые изображения и локальные эксперименты.

## Разработка

Зависимости управляются через `pyproject.toml` и `uv.lock`.

Сгенерировать `requirements.txt` с ML-зависимостями:

```powershell
uv export --extra ml --format requirements.txt --no-hashes --output-file requirements.txt
```

Проверить compose:

```powershell
docker compose config --quiet
```