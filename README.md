# Clasificación No Supervisada de Ruido Sísmico Ambiental en Contexto de Diapiros de Lodo

**Proyect** | Raspberry Shake · ObsPy · Deep Embedded Clustering  
**Fecha:** Junio 2026

---

## 1. Motivación y contexto

Los **diapiros de lodo** son estructuras geológicas formadas por la intrusión de sedimentos plásticos saturados en fluidos que ascienden a través de capas más densas. Generan un ambiente sísmico de baja frecuencia (0.1–5 Hz) producto de actividad interna: migración de fluidos, gases y deformación plástica del sustrato.

El reto central del proyecto es que ese ambiente es **sismológicamente suave** — dominado por ruido antropogénico superficial — exactamente como los entornos urbanos estudiados en la literatura reciente de *urban seismology*. Esto abre la posibilidad de adaptar métodos de clasificación de fuentes urbanas (personas, vehículos, ambiente general) al monitoreo pasivo de diapiros.

La herramienta instrumental es el **Raspberry Shake**, un sismógrafo de bajo costo basado en Raspberry Pi con presencia en una red global de sensores ciudadanos que publican datos libremente vía protocolo FDSN.

---

## 2. Estado del arte

### 2.1 Clasificación de fuentes sísmicas urbanas

La literatura reciente muestra que es posible discriminar automáticamente fuentes de ruido sísmico antropogénico sin etiquetado manual:

- **USED — Urban Seismic Event Detection** (Hasabnis et al., 2024): framework end-to-end con deep learning para detectar y localizar vehículos y peatones en señales sísmicas urbanas. Genera un dataset etiquetado específicamente diseñado para ML, con resolución temporal de 0.25 s.

- **CNN espectral para peatones** (Andajani et al., 2026, Univ. de Tokio): usando extracción espectral de features con CNN, logra >80% de accuracy en clasificación de grupos de peatones por tamaño, incluso con ruido ambiental significativo, validado con cámara como *ground truth*.

- **Clustering no supervisado de ruido urbano** (Saadia & Fotopoulos, 2023): desarrolla un workflow semi-automatizado para clasificar ruido sísmico ambiental adquirido en entorno urbano, extrayendo eventos como picos de amplitud en el espacio tiempo-frecuencia y aplicando clustering sobre features multivariados.

- **Deep Clustering de ruido urbano** (Cheng et al., 2025, GJI): framework multietapa basado en autoencoder convolucional + DEC para ruido inducido por trenes, demostrando que los algoritmos de clustering profundo optimizan simultáneamente la extracción de features y el agrupamiento.

### 2.2 Métodos y arquitecturas dominantes

| Enfoque | Features de entrada | Accuracy típico |
|---|---|---|
| CNN 1D/2D sobre espectrograma | STFT, Mel, CWT | 80–95% |
| LSTM + CNN híbrido | Series temporales + espectro | ~90% |
| Deep Embedded Clustering (DEC) | Embeddings de autoencoder convolucional | Sin etiquetas |
| Random Forest (baseline) | RMS, frecuencia dominante, kurtosis | 70–85% |

### 2.3 Raspberry Shake y acceso a datos

- La red global de Raspberry Shake expone datos históricos vía servidor **FDSN** (`data.raspberryshake.org`), accesible directamente con ObsPy (`Client('RASPISHAKE')`).
- Los datos se sirven con un desfase mínimo de **30 minutos** respecto al tiempo real.
- Cada solicitud permite descargar hasta **24 horas** de datos; para periodos más largos se divide en bloques.
- Los canales relevantes son: `EHZ` (geófono vertical, 100 sps) y `ENZ/ENE/ENN` (acelerómetro, 100 sps).

### 2.4 Estaciones proxy para diapiros de lodo

Para encontrar una estación con patrones de ruido similares a una zona de diapiros se recomienda buscar sensores en:

| Región | Justificación |
|---|---|
| Azerbaiyán (Península de Absheron) | Mayor concentración de diapiros de lodo del mundo |
| Trinidad y Tobago | Diapiros activos, registro sísmico continuo |
| Indonesia (Sidoarjo, LUSI) | Diapiro más estudiado instrumentalmente |
| Rumania (Berca-Arbănași) | Diapiros europeos con actividad documentada |

El criterio de similitud se valida comparando las **Probability Density Functions de PSD** (Peterson Noise Model) entre el sensor local y los candidatos remotos usando `obspy.signal.PPSD`.

---

## 3. Paper de referencia para replicación

### Mousavi et al. (2019) — IEEE Geoscience and Remote Sensing Letters

> **"Unsupervised Clustering of Seismic Signals Using Deep Convolutional Autoencoders"**  
> S. M. Mousavi, W. Zhu, W. Ellsworth, G. Beroza  
> DOI: [10.1109/LGRS.2019.2909218](https://doi.org/10.1109/LGRS.2019.2909218)

**Recursos disponibles públicamente:**
- Código: https://github.com/smousavi05/Unsupervised_Deep_Learning
- Dataset: https://drive.google.com/file/d/16itT_IZpM8w8KyFN8eL8iEfYX66Hk6Xb/view
- Notebook: `unsupervised_deep_learning.ipynb` (incluido en el repo)

**Tarea original del paper:**
1. Discriminar sismogramas registrados a distintas distancias hipocentrales (local vs. telesísmico)
2. Discriminar polaridad de primera llegada (P-wave first motion)

**Adaptación propuesta:**
Reemplazar el dataset de terremotos por ventanas de ruido ambiental urbano, manteniendo la arquitectura intacta, para clasificar en `{ambiente, persona, moto}`.

---

## 4. Arquitectura del modelo

### 4.1 Autoencoder convolucional (pre-entrenamiento)

```
Input: espectrograma 2D  →  (64, 64, 1)
  ↓
Conv2D(32, 3×3) → ReLU → MaxPool2D
Conv2D(64, 3×3) → ReLU → MaxPool2D
Conv2D(128, 3×3) → ReLU → Flatten
  ↓
Dense(16)  ←── Espacio latente Z (16 dimensiones)
  ↑
Dense(reshape) → ConvTranspose2D×3 → UpSampling2D×2
  ↓
Output reconstruido: (64, 64, 1)

Pérdida: MSE (reconstrucción)
```

### 4.2 Deep Embedded Clustering (DEC)

Sobre el encoder pre-entrenado se añade una **capa de clustering** que optimiza simultáneamente la representación latente y la asignación a clusters:

```
Z (16-dim)
  ↓
ClusteringLayer  →  Q (soft assignment, distribución t de Student)
  ↓
KL(P || Q)  ←── P = target distribution (distribución auxiliar agudizada)
```

La distribución auxiliar P se calcula como:

$$P_{ij} = \frac{Q_{ij}^2 / \sum_i Q_{ij}}{\sum_j (Q_{ij}^2 / \sum_i Q_{ij})}$$

### 4.3 Flujo de entrenamiento

```
1. Pre-entrenar autoencoder  →  minimizar MSE de reconstrucción
2. Inicializar centroides     →  K-Means sobre Z del encoder
3. Fine-tuning DEC            →  minimizar KL(P || Q) iterativamente
4. Asignación final           →  argmax(Q) por muestra
```

---

## 5. Pipeline de adquisición, preprocesamiento y visualización

### 5.1 Adquisición (ObsPy + FDSN Raspberry Shake)

```python
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

client = Client('RASPISHAKE')

# Buscar estaciones en zona de interés
inventory = client.get_stations(
    network='AM', station='*', level='station',
    latitude=40.3, longitude=49.8,  # Azerbaiyán
    maxradius=2.0
)

# Descargar 1 hora de datos
stn   = 'RXXXX'
start = UTCDateTime("2024-06-01T08:00:00")
end   = UTCDateTime("2024-06-01T09:00:00")

inv    = client.get_stations(network='AM', station=stn, level='RESP')
stream = client.get_waveforms('AM', stn, '00', 'EHZ', start, end)
stream.attach_response(inv)
```

### 5.2 Preprocesamiento

```python
def preprocess(stream, freqmin=1.0, freqmax=45.0):
    st = stream.copy()
    st.detrend('demean')
    st.detrend('linear')
    st.taper(max_percentage=0.05)
    st.remove_response(output='VEL', pre_filt=(0.5, 1.0, 45.0, 50.0))
    st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4)
    return st

st_clean = preprocess(stream)
```

### 5.3 Construcción del dataset de espectrogramas

```python
import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import zoom

def build_spectrogram_dataset(stream, window_sec=4.0, n_freq=64, n_time=64):
    sr   = stream[0].stats.sampling_rate
    data = stream[0].data
    win  = int(window_sec * sr)
    step = win // 2
    specs = []

    for i in range(0, len(data) - win, step):
        seg = data[i:i+win]
        f, t, Sxx = spectrogram(seg, fs=sr, nperseg=win//8, noverlap=win//16)
        mask_f    = (f >= 1) & (f <= 50)
        Sxx_log   = np.log1p(Sxx[mask_f])
        scale     = (n_freq / Sxx_log.shape[0], n_time / Sxx_log.shape[1])
        Sxx_r     = zoom(Sxx_log, scale)
        Sxx_norm  = (Sxx_r - Sxx_r.mean()) / (Sxx_r.std() + 1e-8)
        specs.append(Sxx_norm)

    return np.array(specs)[..., np.newaxis]  # (N, 64, 64, 1)

X = build_spectrogram_dataset(st_clean)
```

### 5.4 Autoencoder + DEC

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.cluster import KMeans

def build_autoencoder(input_shape=(64, 64, 1), latent_dim=16):
    inp = layers.Input(shape=input_shape)
    x   = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x   = layers.MaxPooling2D()(x)
    x   = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x   = layers.MaxPooling2D()(x)
    x   = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    shape_pre = x.shape[1:]
    x   = layers.Flatten()(x)
    z   = layers.Dense(latent_dim, name='latent')(x)
    x   = layers.Dense(np.prod(shape_pre))(z)
    x   = layers.Reshape(shape_pre)(x)
    x   = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
    x   = layers.UpSampling2D()(x)
    x   = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
    x   = layers.UpSampling2D()(x)
    out = layers.Conv2DTranspose(1, 3, activation='linear', padding='same')(x)
    return Model(inp, out, name='autoencoder'), Model(inp, z, name='encoder')

class ClusteringLayer(layers.Layer):
    def __init__(self, n_clusters, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters

    def build(self, input_shape):
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_shape[-1]),
            initializer='glorot_uniform', name='clusters'
        )

    def call(self, inputs):
        dist = tf.reduce_sum(
            tf.square(tf.expand_dims(inputs, 1) - self.clusters), axis=2)
        q = 1.0 / (1.0 + dist)
        return q / tf.reduce_sum(q, axis=1, keepdims=True)

def target_distribution(q):
    weight = q ** 2 / q.sum(axis=0)
    return (weight.T / weight.sum(axis=1)).T

def train_dec(X, n_clusters=3, pretrain_epochs=50, dec_epochs=100):
    ae, encoder = build_autoencoder(input_shape=X.shape[1:])
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X, X, epochs=pretrain_epochs, batch_size=32, verbose=1)

    Z  = encoder.predict(X)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    km.fit(Z)

    inp = encoder.input
    q   = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    dec = Model(inp, q, name='DEC')
    dec.get_layer('clustering').set_weights([km.cluster_centers_])
    dec.compile(optimizer='adam', loss='kld')

    for epoch in range(dec_epochs):
        q_vals = dec.predict(X)
        p_vals = target_distribution(q_vals)
        loss   = dec.train_on_batch(X, p_vals)
        if epoch % 10 == 0:
            labels = q_vals.argmax(axis=1)
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Clusters: {np.bincount(labels)}")

    labels = dec.predict(X).argmax(axis=1)
    return dec, encoder, labels
```

### 5.5 Visualización

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_clusters(encoder, X, labels, class_names=None):
    Z    = encoder.predict(X)
    Z_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(Z)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(Z_2d[:,0], Z_2d[:,1], c=labels, cmap='tab10', s=5, alpha=0.6)
    axes[0].set_title('Espacio latente — t-SNE')
    plt.colorbar(sc, ax=axes[0])

    n_k = len(np.unique(labels))
    for k in range(n_k):
        mean_spec = X[labels == k].mean(axis=0).squeeze()
        lbl = class_names[k] if class_names else f'Cluster {k}'
        axes[1].plot(mean_spec.mean(axis=1), label=lbl)
    axes[1].set_title('Espectro medio por cluster')
    axes[1].set_xlabel('Frecuencia (bin)')
    axes[1].set_ylabel('Energía media (log)')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
```

---

## 6. Roadmap de trabajo

| Fase | Tarea | Estado |
|------|-------|--------|
| 1 | Correr notebook original (datos del paper) | ⬜ Pendiente |
| 2 | Adquirir datos FDSN de estación proxy | ⬜ Pendiente |
| 3 | Adaptar pipeline de espectrogramas al nuevo dataset | ⬜ Pendiente |
| 4 | Entrenar DEC y evaluar clusters | ⬜ Pendiente |
| 5 | Interpretar clusters vs fuentes (persona/moto/ambiente) | ⬜ Pendiente |
| 6 | Comparar PSD entre estación proxy y sensor local | ⬜ Pendiente |

---

## 7. Referencias

- Mousavi, S. M., Zhu, W., Ellsworth, W., Beroza, G. (2019). *Unsupervised Clustering of Seismic Signals Using Deep Convolutional Autoencoders*. IEEE GRSL. DOI: 10.1109/LGRS.2019.2909218
- Hasabnis, P., Nilot, E., Li, Y. E. (2024). *Introducing USED: Urban Seismic Event Detection*. Computers & Geosciences. DOI: 10.1016/j.cageo.2024.105561
- Saadia, A., Fotopoulos, G. (2023). *Unsupervised clustering of ambient seismic noise in an urban environment*. Computers & Geosciences. DOI: 10.1016/j.cageo.2023.105405
- Andajani, R. D. et al. (2026). *Seismometer-Based Pedestrian Monitoring Using Spectral Feature Extraction and Deep Learning*. Earth Systems and Environment. DOI: 10.1007/s41748-025-01003-4
- Xie, J., Girshick, R., Farhadi, A. (2016). *Unsupervised Deep Embedding for Clustering Analysis*. ICML.
- Cheng, X. et al. (2025). *Multistage deep clustering of urban ambient noise for seismic imaging*. Geophysical Journal International. DOI: 10.1093/gji/ggaf273

---

*Documento generado como parte del proyecto educativo de monitoreo sísmico de diapiros de lodo con Raspberry Shake.*
