# Windows-first research implementation notes

Основное ТЗ находится в `deep-research-report.md`. Этот файл фиксирует, как требования переложены на репозиторий.

## Архитектурные контракты

- Данные хранятся как `manifest JSONL + NPZ shards`.
- Manifest содержит обязательные поля из ТЗ и валидируется через `tools/validate_manifest.py`.
- Tensor shard содержит `landmarks [T,21,3]`, `sequence_mask`, `frame_confidence`, `handedness_score`, `coord_space`.
- Preprocessing разделен на `pose` stream и `motion` stream, чтобы не потерять глобальную траекторию для `swipe_left/right`.
- Mirroring меняет только направленные классы `swipe_left <-> swipe_right`.

## Экспериментальные варианты

- `C0`: `research_pipeline.models.rule_based.RuleBasedRecognizer`.
- `C1`: random forest на engineered clip summaries.
- `C1-T`: compact temporal path. Сейчас есть dependency-free temporal prototype для smoke и `models/tcn.py` как Torch TCN extension point.
- `C2`: `ContextAwarePolicy` поверх предсказаний.
- `C2+Local`: отдельный local adaptation config и общий manifest merge stage.

## Windows-first ограничения

Windows остается основным контуром для подготовки данных, обучения, benchmark и desktop demo. Core ML export вынесен в отдельный portability stage и на Windows пишет contract report вместо попытки выполнить неподдерживаемую конвертацию.

