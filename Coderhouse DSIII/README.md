# Pipeline base — Preentrega DSIII

Infraestructura inicial de un proyecto de clasificación en PyTorch: detección de dispositivo, un MLP (`nn.Module`), loop de entrenamiento con Adam y evaluación de pérdida y accuracy en un conjunto de validación separado.

## Estructura

```
data/synthetic_classification.csv   dataset sintético (4 features, 3 clases)
src/device.py                       detección CUDA / MPS / CPU
src/model.py                        MLPClassifier
src/dataset.py                      carga, split estratificado y estandarización
src/train.py                        entrenamiento y validación
requirements.txt
```

## Cómo ejecutar

Desde la raíz del proyecto:

```bash
pip install -r requirements.txt
python -m src.train
```

Si `data/synthetic_classification.csv` no existe, el script lo genera con una semilla fija y lo vuelve a usar en las corridas siguientes. El 20 % de cada clase queda en validación. La media y el desvío se calculan solo con entrenamiento, para no filtrar información del conjunto de validación.

## Checkpoint de esta corrida

| Dato | Valor |
| --- | --- |
| PyTorch | 2.6.0+cu124 |
| Dispositivo detectado | CUDA (GPU) |
| `learning_rate` | `0.001` |
| Optimizador | Adam |
| Pérdida | `CrossEntropyLoss` |
| Épocas | 30 |
| Arquitectura | MLP 4 → 16 (ReLU) → 3 logits |

El gradiente se calcula con `loss.backward()` dentro del loop de entrenamiento (`torch.autograd`). La evaluación corre en `model.eval()` y bajo `torch.no_grad()`.

### Curva de pérdida

| Época | Pérdida train | Pérdida validación | Accuracy validación |
| ---: | ---: | ---: | ---: |
| 1 | 1.0584 | 1.0064 | 0.5833 |
| 10 | 0.3445 | 0.3191 | 0.9389 |
| 20 | 0.1495 | 0.1759 | 0.9556 |
| 30 | 0.1076 | 0.1459 | 0.9556 |

La pérdida bajó durante las épocas, tanto en entrenamiento (1.0584 → 0.1076) como en validación (1.0064 → 0.1459). El descenso fue sostenido y las dos curvas se movieron juntas, sin un rebote de la pérdida de validación. El accuracy de validación pasó de 0.58 a 0.96: el MLP aprendió a separar las tres clases del dataset sintético.
