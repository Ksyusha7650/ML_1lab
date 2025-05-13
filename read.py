from tensorboard.backend.event_processing import event_accumulator

# Укажите путь к файлу с логами
log_file = 'samples/logs/INCEPTION_V3/events.out.tfevents.1746921804.MarinaHP.15572.0'

# Создаём накопитель и задаём, что нас интересуют все scalar-значения
ea = event_accumulator.EventAccumulator(
    log_file,
    size_guidance={  # 0 = загружать без ограничений
        'scalars': 0,
        'histograms': 0,
        'images': 0,
        'compressedHistograms': 0,
        'tensors': 0,
    }
)

# Загружаем все записи
ea.Reload()

# Посмотрим, какие теги scalar-метрик есть в логе
scalars_tags = ea.Tags().get('scalars', [])
print("Найденные scalar-теги:", scalars_tags)

# Для каждого тега выведем все события
for tag in scalars_tags:
    print(f"\n=== {tag} ===")
    for event in ea.Scalars(tag):
        # event.step — номер шага, event.wall_time — время (сек), event.value — значение метрики
        print(f"Step {event.step}, Time {event.wall_time:.1f}, Value {event.value:.4f}")
