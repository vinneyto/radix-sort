Работаем в существующем репозитории:

```text
https://github.com/vinneyto/radix-sort
```

В репозитории уже есть корректная WebGPU-реализация stable indirect LSD radix sort и skill-файлы, которые описывают алгоритм.

Нужно добавить в этот же репозиторий отдельный пример реализации radix sort для PyTorch MPS, используя встроенную динамическую компиляцию Metal-шейдеров:

```python
torch.mps.compile_shader(...)
```

Не писать C++, Objective-C++, `setup.py`, `torch.utils.cpp_extension` или отдельную нативную обвязку.

## Сначала изучи существующую реализацию

Обязательно прочитай:

```text
src/RadixSort.ts
SKILLS/radix-impl/SKILL.md
SKILLS/three-webgpu-radix-sort/SKILL.md
README.md
```

PyTorch/MPS-реализация должна сначала воспроизвести именно текущий WebGPU baseline, а не сразу полную оптимизированную версию общего skill.

Сохрани тот же алгоритмический контракт и максимально похожую структуру Metal-ядер:

```text
radix_histogram
radix_scan
radix_scatter
```

## Новая папка

Добавь в корень репозитория отдельную папку:

```text
pytorch-mps/
```

Предлагаемая структура:

```text
pytorch-mps/
├── README.md
├── mps_radix_sort.py
├── test_mps_radix_sort.py
├── benchmark_mps_radix_sort.py
└── requirements.txt
```

Не изменяй поведение существующего WebGPU-примера.

Изменения в существующих файлах допускаются только для:

```text
корневого README.md;
.gitignore;
общей документации;
ссылки на новый PyTorch MPS example.
```

## Запуск

Все команды должны запускаться из корня репозитория:

```bash
python pytorch-mps/test_mps_radix_sort.py
python pytorch-mps/benchmark_mps_radix_sort.py
```

Допускается также запуск из самой папки:

```bash
cd pytorch-mps
python test_mps_radix_sort.py
python benchmark_mps_radix_sort.py
```

Код не должен зависеть от установки TypeScript/WebGPU-части репозитория.

## Конфигурация baseline

Используй ту же конфигурацию, что в текущем `src/RadixSort.ts`:

```text
KEY_BITS = 32
RADIX_BITS = 4
RADIX_SIZE = 16
PASSES = 8
WORKGROUP_SIZE = 256
ITEMS_PER_THREAD = 1
```

Не добавляй в первой версии:

```text
несколько items на thread;
несколько tiles на threadgroup;
processedEarlierTiles;
hierarchical scan;
SIMD-group ballot;
оптимизированный stable rank;
8-bit radix;
64-bit keys;
autograd/backward.
```

Первая цель — максимально точный перенос уже проверенной WebGPU-реализации.

## Контракт сортировки

Реализуй стабильную indirect LSD radix sort:

```python
sorted_indices = radix_sort_mps(keys, indices)
```

Вход:

```text
keys:
    одномерный contiguous MPS tensor;
    32-битные целые ключи.

indices:
    одномерный contiguous MPS tensor;
    32-битные индексы в keys.
```

Ключ для позиции читается косвенно:

```text
source_index = indices[position]
key = keys[source_index]
```

Сортируются только индексы.

Массив `keys`:

```text
не переставляется;
не копируется между radix-проходами;
остаётся immutable.
```

Результат должен удовлетворять:

```text
keys[sorted_indices[i]]
    <= keys[sorted_indices[i + 1]]
```

Сортировка должна быть стабильной:

```text
если два элемента имеют одинаковые ключи,
их относительный порядок из входного indices сохраняется.
```

Предпочтительно использовать `torch.uint32`, если эта комбинация операций полностью поддерживается установленной версией PyTorch MPS.

Если `torch.uint32` создаёт проблемы, используй `torch.int32` как контейнер 32-битных значений, а в Metal интерпретируй буфер как:

```metal
device const uint*
```

Документируй выбранный вариант.

## Компиляция Metal

Metal source должен компилироваться непосредственно из Python:

```python
METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// kernels
"""

LIBRARY = torch.mps.compile_shader(METAL_SOURCE)
```

Компилируй библиотеку один раз при импорте модуля или лениво при первом использовании.

Не компилируй Metal source заново при каждом вызове `sort()`.

Перед основной реализацией проверь API с простым smoke test.

Также выведи или проверь доступные свойства ядра:

```python
kernel.max_threads_per_threadgroup
kernel.thread_execution_width
kernel.static_thread_group_memory_length
```

Ядра запускай с явными параметрами:

```python
threads=...
group_size=...
```

Не полагайся на автоматическое определение количества потоков из первого тензора.

## Архитектура одного radix-прохода

Каждый из восьми проходов должен состоять из трёх отдельных dispatch:

```text
1. radix_histogram
2. radix_scan
3. radix_scatter
```

Оркестрация:

```python
for pass_index in range(PASSES):
    shift = pass_index * RADIX_BITS

    radix_histogram(...)
    radix_scan(...)
    radix_scatter(...)

    swap(input_indices, output_indices)
```

Не выполнять между dispatch:

```python
.cpu()
.item()
.numpy()
torch.mps.synchronize()
```

Синхронизацию использовать только:

```text
в тестах перед CPU readback;
в benchmark перед началом и после конца измеряемого участка;
при отладке ошибок.
```

## Ядро 1: radix_histogram

Один threadgroup обрабатывает 256 входных элементов.

Используй локальную атомарную histogram:

```metal
threadgroup atomic_uint local_histogram[16];
```

Алгоритм:

```text
1. Первые 16 потоков обнуляют bins.
2. Threadgroup barrier.
3. Каждый валидный поток читает один input index.
4. Через index читает соответствующий key.
5. Извлекает текущий четырёхбитный digit.
6. Делает atomic increment локального bin.
7. Threadgroup barrier.
8. Первые 16 потоков записывают histogram группы
   в глобальный histograms buffer.
```

Digit:

```metal
uint digit = (key >> shift) & 15u;
```

Используй relaxed atomics, поскольку порядок increment не влияет на итоговый count:

```metal
atomic_store_explicit(..., memory_order_relaxed);
atomic_fetch_add_explicit(..., 1u, memory_order_relaxed);
atomic_load_explicit(..., memory_order_relaxed);
```

Metal built-ins:

```metal
uint position [[thread_position_in_grid]]
uint lid      [[thread_position_in_threadgroup]]
uint group    [[threadgroup_position_in_grid]]
```

## Ядро 2: radix_scan

Сначала реализуй простой scan, соответствующий текущему WebGPU baseline.

Запускается одна threadgroup из 256 потоков.

Первые 16 потоков параллельно обрабатывают bins.

Для каждого bin:

```text
total = 0

for group in 0 .. group_count:
    offsets[group, bin] = total
    total += histograms[group, bin]

bin_totals[bin] = total
```

`bin_totals` хранить в:

```metal
threadgroup uint bin_totals[16];
```

После barrier поток `lid == 0` вычисляет глобальную базу каждого bin:

```text
base = 0

for digit in 0 .. 15:
    for group in 0 .. group_count:
        offsets[group, digit] += base

    base += bin_totals[digit]
```

После выполнения:

```text
offsets[group, digit]
=
bin_base[digit]
+
exclusive_group_prefix[group, digit]
```

Не реализовывать hierarchical scan в первой версии.

## Ядро 3: radix_scatter

Одна threadgroup обрабатывает 256 элементов.

Используй:

```metal
threadgroup uint tile_digits[256];
```

Каждый поток вычисляет:

```text
valid = position < count
```

Если поток валиден:

```text
source_index = input_indices[position]
key = keys[source_index]
digit = (key >> shift) & 15
```

Если невалиден:

```text
digit = 16
```

Значение `16` — sentinel, поскольку допустимые bins находятся в диапазоне `0...15`.

Все потоки записывают digit:

```metal
tile_digits[lid] = digit;
```

Затем выполняется threadgroup barrier.

Для каждого валидного элемента вычисли стабильный локальный rank:

```text
rank =
    количество previous positions,
    для которых:

    previous < lid
    и tile_digits[previous] == digit
```

Реализация baseline:

```metal
uint rank = 0;

for (uint previous = 0; previous < lid; ++previous) {
    if (tile_digits[previous] == digit) {
        ++rank;
    }
}
```

Не использовать атомарный increment для вычисления rank.

Порядок выполнения атомиков не обязан соответствовать порядку локальных потоков, поэтому atomic rank нарушит стабильность LSD radix sort.

Итоговая позиция:

```text
offset_address =
    group * RADIX_SIZE + digit

output_position =
    offsets[offset_address] + rank
```

Запись:

```metal
output_indices[output_position] = source_index;
```

## Аргументы Metal-ядер

При передаче scalar arguments из Python учти, что Python `int` может по умолчанию передаваться как 64-битное значение.

Для Metal-параметров вида:

```metal
constant uint& count
constant uint& shift
constant uint& group_count
```

используй `arg_casts` с `"int32"` для соответствующих позиций.

Например:

```python
kernel(
    tensor_a,
    tensor_b,
    count,
    shift,
    threads=dispatch_size,
    group_size=WORKGROUP_SIZE,
    arg_casts={
        scalar_argument_index: "int32",
    },
)
```

Не копируй номера аргументов вслепую.

Проверь фактическую сигнатуру каждого ядра и укажи корректные позиции в `arg_casts`.

## Буферы

Нужны следующие MPS tensors:

```text
original indices;
internal scratch indices;
caller-visible output indices;
histograms;
offsets.
```

Размеры:

```python
group_count = (count + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE

histograms.shape == (group_count * RADIX_SIZE,)
offsets.shape == (group_count * RADIX_SIZE,)
```

Все буферы должны быть:

```text
device.type == "mps";
contiguous;
32-битными.
```

Организуй ping-pong так же, как в WebGPU-реализации:

```text
pass 0: original → scratch
pass 1: scratch  → output
pass 2: output   → scratch
pass 3: scratch  → output
pass 4: output   → scratch
pass 5: scratch  → output
pass 6: output   → scratch
pass 7: scratch  → output
```

После восьмого прохода возвращаемый результат должен находиться в `output`.

Обработай `count == 0` без запуска Metal-ядер.

## Python API

В `pytorch-mps/mps_radix_sort.py` реализуй:

```python
class MPSRadixSort:
    def __init__(self, capacity: int):
        ...

    def sort(
        self,
        keys: torch.Tensor,
        indices: torch.Tensor,
        length: int | None = None,
    ) -> torch.Tensor:
        ...
```

Проверяй:

```text
MPS доступен;
keys и indices одномерные;
оба тензора находятся на MPS;
оба contiguous;
оба имеют поддерживаемый 32-битный dtype;
length не превышает capacity;
length не превышает indices.numel();
indices ссылаются на допустимые элементы keys.
```

Не проверяй каждый индекс через CPU внутри основного производительного пути.

Полную проверку допустимости индексов можно делать в тестах или в отдельном debug-режиме.

Переиспользуй временные буферы между вызовами `sort()`, пока capacity не меняется.

Также добавь функцию:

```python
def radix_sort_mps(
    keys: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    ...
```

## Тесты

В `pytorch-mps/test_mps_radix_sort.py` создай автономный запускаемый test script.

Не обязательно добавлять `pytest`.

Скрипт должен запускаться обычной командой:

```bash
python pytorch-mps/test_mps_radix_sort.py
```

Если MPS недоступен, скрипт должен:

```text
вывести понятное сообщение;
завершиться без ложного сообщения об успехе;
не падать с непонятным stack trace.
```

CPU oracle должен сортировать именно входную последовательность `indices` по косвенно прочитанным ключам:

```python
expected = sorted(
    indices_cpu.tolist(),
    key=lambda index: int(keys_cpu[index]),
)
```

Python sort стабилен, поэтому подходит как эталон.

Проверяй полное совпадение списка индексов.

Недостаточно проверить только монотонность ключей.

Размеры тестов:

```text
0
1
2
15
16
17
255
256
257
511
512
513
1_000
10_000
100_000
1_000_000
```

Наборы данных:

```text
полностью случайные 32-битные ключи;
duplicate-heavy ключи;
все ключи одинаковые;
уже отсортированные ключи;
ключи в обратном порядке;
случайно перемешанные indices;
повторяющиеся indices;
частичная сортировка через length;
неполная последняя threadgroup.
```

После запуска GPU sort:

```python
torch.mps.synchronize()
actual = result.cpu()
```

При несовпадении выведи:

```text
название теста;
размер;
первую позицию несовпадения;
expected index;
actual index;
expected key;
actual key.
```

В конце выведи:

```text
все тесты пройдены;
количество пройденных случаев.
```

## Benchmark

В `pytorch-mps/benchmark_mps_radix_sort.py` создай автономный benchmark script:

```bash
python pytorch-mps/benchmark_mps_radix_sort.py
```

Сначала выполни несколько warmup-сортировок.

Основное GPU-время измеряй так:

```python
torch.mps.synchronize()
started = time.perf_counter()

for _ in range(repeats):
    sorter.sort(keys, indices)

torch.mps.synchronize()
elapsed = time.perf_counter() - started
```

Не включай в основное GPU-время:

```text
создание тестовых данных;
CPU → MPS transfer;
компиляцию Metal source;
создание MPSRadixSort;
GPU → CPU readback;
CPU validation.
```

Покажи результаты минимум для:

```text
10_000
100_000
1_000_000
```

Для каждого размера выведи:

```text
число элементов;
число повторов;
среднее время одной GPU-сортировки;
минимальное время, если измерения выполняются отдельно;
CPU reference time;
GPU speedup;
статус полной проверки результата.
```

Сравни с CPU stable sort.

Если используешь `torch.argsort`, отдельно укажи:

```text
устройство выполнения;
стабильная ли сортировка;
включён ли перенос данных;
является ли сравнение алгоритмически эквивалентным.
```

## requirements.txt

Добавь минимальный:

```text
torch
```

Не фиксируй старую версию PyTorch, в которой отсутствует `torch.mps.compile_shader`.

В `pytorch-mps/README.md` укажи:

```text
требуемую версию macOS;
Apple Silicon или поддерживаемый MPS Mac;
MPS-enabled PyTorch;
способ проверить torch.backends.mps.is_available();
команды запуска тестов и benchmark;
текущие ограничения baseline.
```

Не выдумывай точную минимальную версию PyTorch.

Определи её по реально доступному API или сформулируй как:

```text
PyTorch build containing torch.mps.compile_shader
```

## Обновление корневого README

Добавь короткий раздел со ссылкой:

```text
PyTorch MPS Metal implementation
```

Укажи, что это отдельный пример, который:

```text
повторяет WebGPU baseline;
компилирует Metal kernels напрямую из Python;
не требует C++ extension.
```

Не переписывай остальные части README без необходимости.

## Проверка результата

После реализации выполни из корня репозитория:

```bash
python pytorch-mps/test_mps_radix_sort.py
python pytorch-mps/benchmark_mps_radix_sort.py
```

Также не сломай существующий проект:

```bash
npm test
npm run typecheck
npm run format:check
npm run build
```

Если текущая среда не является macOS с доступным MPS:

```text
всё равно реализуй файлы;
выполни доступные статические проверки;
честно укажи, что Metal kernels не были запущены;
не заявляй, что GPU-тесты прошли.
```

## Финальный отчёт

В конце сообщи:

```text
1. Какие файлы добавлены.
2. Какие существующие файлы изменены.
3. Как устроены три Metal kernel.
4. Как реализован ping-pong.
5. Какие команды запуска использовать.
6. Какие тесты реально выполнены.
7. Какие benchmark-результаты реально получены.
8. Что не удалось проверить в текущей среде.
9. Какие оптимизации можно реализовать следующим этапом.
```

Возможные следующие этапы только перечисли, но пока не реализуй:

```text
SIMD-group stable rank;
ITEMS_PER_THREAD = 2/4/8;
несколько tiles на threadgroup;
processedEarlierTiles;
hierarchical scan;
RADIX_BITS = 8;
пропуск неиспользуемых старших проходов.
```
