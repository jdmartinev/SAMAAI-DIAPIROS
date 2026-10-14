# Prototipo: SeisLM para detectar segmentos anómalos de Raspberry Shake

## Objetivo

Asignar un score a cada registro de cinco minutos y priorizar los segmentos cuyo patrón de vibración difiere del comportamiento habitual de la estación.

El pipeline implementado es:

1. Cargar el registro y verificar calidad, frecuencia de muestreo y componentes.
2. Dividirlo en ventanas con solapamiento.
3. Normalizar cada componente de cada ventana siguiendo el ejemplo oficial.
4. Extraer embeddings con **SeisLM preentrenado y congelado**.
5. Ajustar **Isolation Forest** usando únicamente embeddings de segmentos normales de referencia.
6. Agregar los scores de ventanas para obtener un score por segmento.
7. Calibrar un umbral con otros segmentos normales independientes.
8. Ordenar y marcar los segmentos de evaluación.

No se entrena ni se ajusta el encoder. Sí se ajusta el detector de anomalías para aprender la normalidad local. Un score alto significa candidato para revisión, no identificación automática de un sismo. Este paquete reemplaza el prototipo anterior: TimesFM y las características manuales ya no son el pipeline principal ni forman parte del script.

## Implementación de SeisLM verificada

Se revisaron el código y el notebook de características del repositorio oficial. El ejemplo carga:

```python
from seisLM.model.foundation.pretrained_models import LitMultiDimWav2Vec2
model = LitMultiDimWav2Vec2.load_from_checkpoint(checkpoint).model
model.eval()
output = model.wav2vec2(input_values=batch, output_hidden_states=True)
embedding = output.hidden_states[-1].mean(dim=1)
```

`SeisLMEncoder` sigue esa ruta, añade selección de dispositivo, `inference_mode`, desactivación de gradientes y procesamiento por lotes. Usa el último estado oculto y promedio temporal: un vector por ventana. No utiliza logits de un clasificador ni t-SNE para construir el detector.

El demo oficial usa 100 Hz, orden ZNE, tensores `(batch, 3, muestras)` y ventanas de **3001 muestras**. El prototipo mantiene exactamente esa longitud (30.01 s), con un paso de 1500 muestras (15 s). Añade una última ventana completa desplazada para cubrir el final del registro. Un registro de 30000 muestras genera 19 ventanas, sin rellenar con ceros.

La normalización resta la media y divide por la desviación estándar de cada componente/ventana. Esto reduce la información de amplitud absoluta: el score busca principalmente diferencias de forma. Una futura comparación puede conservar energía como score adicional, sin sustituir los embeddings.

Fuente revisada: árbol del repositorio `ae1d27ac498ec7ae106e56eeaac386a0a5dd124f`. Este identificador corresponde al árbol recuperado, no se presenta como commit de instalación.

## Instalar

Usa un entorno separado, Python 3.10 o posterior. El entorno de dependencias del repositorio puede requerir ajustes según PyTorch, torchtune y torchao; no se ha resuelto y ejecutado completo aquí.

Desde la carpeta del prototipo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

git clone https://github.com/liutianlin0121/seisLM.git upstream_seislm
pip install -e upstream_seislm
```

Si prefieres Colab, empieza con el [notebook oficial de extracción de características](https://colab.research.google.com/github/liutianlin0121/seisLM/blob/main/examples/demo_seislm_visualized_pretrained_features.ipynb), que contiene las instrucciones de instalación del proyecto.

Descarga el checkpoint base que referencia ese mismo notebook:

```bash
mkdir -p checkpoints
gdown 12L2Js1DIXCIxacQJ4K0hRKJ6qHFrQ7K- -O checkpoints/seislm_base.ckpt
```

El ID se obtuvo del notebook oficial, no fue supuesto. Si Google Drive restringe la descarga, usa la [carpeta de checkpoints indicada por los autores](https://drive.google.com/drive/folders/194vColWfn0utQbHP04PAt2efKO40Z3Os?usp=sharing) y selecciona el checkpoint base mostrado en el demo (`epoch=39-step=1203000.ckpt`). El archivo se puede renombrar a `seislm_base.ckpt`.

La carga de Lightning emplea `weights_only=False` porque el checkpoint incorpora metadatos de configuración serializados. Carga solamente el checkpoint oficial o uno de confianza. El programa calcula su SHA-256 y lo registra para reproducibilidad.

## Datos y componentes

Por ahora cada archivo debe representar exactamente 300 s a 100 Hz. No hay remuestreo automático: si tus datos tienen otra tasa, aplica un remuestreo con filtro antialias y verifícalo antes de correr este prototipo.

### MiniSEED

Se admiten archivos que contienen:

- Un único componente vertical con sufijo de canal `Z`.
- Tres componentes alineados con sufijos `Z`, `N` y `E`.

El script lee los metadatos de ObsPy, ordena ZNE y exige la misma estación, ubicación, familia de instrumento, hora inicial y cantidad de muestras. No asume que EHZ sea el canal de tu equipo. Los componentes 1/2 requieren una conversión de orientación verificada antes de usarlos.

### NumPy

Se admiten `.npy` sin pickle:

- `(30000,)` o `(1,30000)`: vertical.
- `(3,30000)`: ZNE, en ese orden.

Para NPY debes indicar `--sample-rate 100` y verificar que sea la tasa real.

### Raspberry Shake de un solo canal

El checkpoint espera tres componentes. La adaptación implementada es copiar el canal vertical en los tres slots, únicamente al indicar:

```bash
--missing-components copy
```

Esto coincide con la estrategia de copia empleada en el demo y con el comportamiento de completar componentes disponibles en el código de preentrenamiento cuando solo hay uno variable. **No equivale a disponer de tres mediciones físicas y no demuestra desempeño en un Shake vertical.** Se documenta como una adaptación experimental y debe evaluarse con registros revisados de tu estación. Por defecto se rechazan los archivos de un canal para evitar una adaptación silenciosa.

Las señales completamente planas, ventanas casi planas, NaN, valores enmascarados, gaps y overlaps se rechazan. Un archivo inválido detiene el experimento con un error: revisa la calidad por separado. Falta añadir chequeos específicos de clipping, cambios de ganancia y discontinuidades sin gaps.

## Separar referencia, calibración y evaluación

Prepara `manifest.csv` con rutas relativas al propio manifest:

```csv
path,split
recordings/normal_fit_001.mseed,fit
recordings/normal_fit_002.mseed,fit
recordings/normal_cal_001.mseed,calibration
recordings/normal_cal_002.mseed,calibration
recordings/unknown_001.mseed,test
```

El ejemplo solo muestra el formato. Usa muchas más grabaciones normales representativas, con horarios, días y condiciones habituales diferentes.

| Split | Uso | Debe ser normal |
|---|---|---|
| `fit` | Ajustar StandardScaler e Isolation Forest | Sí |
| `calibration` | Calibrar umbral de segmentos | Sí |
| `test` | Ordenar y marcar registros nuevos | No se presupone |

Divide por día de adquisición o evento **antes** de generar ventanas. No distribuyas ventanas del mismo segmento entre splits. Mantén los archivos adyacentes de un mismo evento en el mismo split. El script rechaza archivos de bytes idénticos; no puede identificar solapamientos físicos entre archivos diferentes ni decidir qué registros son normales.

Mantén una estación/familia de sensores por experimento. La normalización no elimina todas las diferencias entre instrumentos. Corregir respuesta instrumental o filtrar no forma parte del script; cualquier procesamiento adicional debe ser consistente y evaluado frente al procedimiento del checkpoint.

## Ejecutar

MiniSEED con tres componentes:

```bash
python detect_segments.py \
  --manifest manifest.csv \
  --checkpoint checkpoints/seislm_base.ckpt \
  --device cpu \
  --output results_seislm
```

MiniSEED vertical:

```bash
python detect_segments.py \
  --manifest manifest.csv \
  --checkpoint checkpoints/seislm_base.ckpt \
  --missing-components copy \
  --device cpu \
  --output results_seislm
```

Para NPY añade `--sample-rate 100`. Para GPU cambia a `--device cuda`. Si falta memoria, reduce `--batch-size 4` a 1. Comienza en un computador/servidor; no se ha medido la viabilidad de ejecutar SeisLM en la Raspberry Pi.

Las grabaciones se procesan localmente. No se envían a un servicio de inferencia; el acceso a internet se necesita para obtener código, dependencias y checkpoint.

## Score y umbral

1. StandardScaler se ajusta únicamente con embeddings de `fit`.
2. Isolation Forest se ajusta sobre esos embeddings escalados.
3. Score de ventana: `-detector.score_samples(embedding_escalado)`. Un valor mayor indica mayor rareza.
4. Score de segmento: media de las tres ventanas con mayor score (`--top-k 3`). El solapamiento significa que esas tres ventanas pueden representar un mismo evento; no son tres eventos independientes.
5. Umbral: percentil 99 de los scores **por segmento** de `calibration` (`--quantile 0.99`).
6. Solo los registros `test` reciben una decisión: `score > umbral`.

No se interpreta el score como probabilidad. Calibrar sobre segmentos conserva la unidad final de decisión; un umbral de ventanas no se traslada directamente al segmento agregado. El percentil 99 requiere bastantes segmentos normales independientes para estimar la cola; con unos pocos archivos es inestable. No garantiza 1% de falsas alarmas futuras.

La media temporal del encoder puede diluir eventos muy cortos. Si eso ocurre, evalúa otra capa/pooling o ventanas compatibles alternativas como una comparación explícita; no cambies múltiples decisiones a la vez.

## Resultados

- `segments.csv`: score y ranking de segmentos, split, umbral y marca de anomalía solo para evaluación.
- `windows.csv`: score y tiempo inicial/final de cada ventana para ubicar la actividad.
- `embeddings_00000.npz`, etc.: embeddings, offsets, ruta y hash de la grabación.
- `detector.joblib`: scaler, detector y parámetros del umbral. Carga este archivo solo si confías en quien lo generó.
- `config.json`: checkpoint y hash, parámetros de extracción, adaptación de canales y configuración del experimento.

Los directorios de resultados corresponden a una ejecución; usa una carpeta nueva para cada experimento. El script vuelve a extraer embeddings en cada ejecución: conservarlos permite diseñar luego una etapa de scoring independiente, pero no se implementa todavía un CLI de reutilización ni de inferencia con un detector ya guardado.

## Validación y estado real

Se ejecutaron tres pruebas automatizadas:

- Canal vertical rechazado salvo copia explícita; ventanas normalizadas y cobertura hasta el final del registro.
- Rechazo de señales planas y no finitas.
- Ajuste/calibración independientes de los datos `test`; escritura de resultados y artefacto del detector.

```bash
python test_pipeline.py
```

La prueba de scoring usa vectores sintéticos como dobles de embeddings. **No usa ni sustituye SeisLM en el programa principal.** El programa principal exige un checkpoint y no tiene un fallback a modelos aleatorios, características manuales o TimesFM.

El loader, forma de entrada, normalización y llamada al encoder se verificaron leyendo el ejemplo y código oficiales. No se ha descargado ni ejecutado el checkpoint aquí; tampoco se ha ejecutado el lector sobre tus MiniSEED. Por tanto, el estado es **prototipo implementado con extracción pendiente de verificación de ejecución**, no pipeline validado sobre Raspberry Shake. Sin un registro real no puede afirmarse precisión ni sensibilidad.

## Próximo experimento con tus datos

1. Confirmar modelo de Shake, canal(es), formato y tasa de muestreo.
2. Ejecutar una grabación por el encoder y comprobar forma/valores del embedding.
3. Construir splits con grabaciones normales revisadas y eventos conocidos.
4. Inspeccionar waveform y espectrograma de los segmentos con mayor score.
5. Medir precisión entre los primeros segmentos del ranking, recall de eventos conocidos y falsas alarmas por día.
6. Comparar con RMS, STA/LTA y un detector de características espectrales; decidir si SeisLM añade valor.

Para distinguir terremotos de maquinaria/impactos, añade etiquetas de revisión o un detector sísmico especializado como una salida separada. Un detector general de rareza no asigna por sí solo la causa física.

## Fuentes

- [Repositorio oficial de SeisLM](https://github.com/liutianlin0121/seisLM)
- [Notebook oficial de embeddings](https://github.com/liutianlin0121/seisLM/blob/main/examples/demo_seislm_visualized_pretrained_features.ipynb)
- [Loader oficial](https://github.com/liutianlin0121/seisLM/blob/main/seisLM/model/foundation/pretrained_models.py)
- [Completar componentes y normalización](https://github.com/liutianlin0121/seisLM/blob/main/seisLM/data_pipeline/augmentations.py)
- [Artículo de SeisLM](https://arxiv.org/abs/2410.15765)
