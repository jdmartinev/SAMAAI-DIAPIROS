"""SeisLM frozen embeddings -> Isolation Forest -> segment calibration."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

SAMPLE_RATE = 100
WINDOW_SAMPLES = 3001
STRIDE_SAMPLES = 1500


def load_waveform(path, sample_rate=None, missing_components='error'):
    """Return (3, 30000) in ZNE order. Never silently invent components."""
    path = Path(path)
    if path.suffix.lower() == '.npy':
        if sample_rate != SAMPLE_RATE:
            raise ValueError('NPY requires --sample-rate 100; resample upstream if necessary.')
        x = np.load(path, allow_pickle=False)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[0] not in (1, 3):
            raise ValueError('NPY shape must be (N,), (1,N), or (3,N) in ZNE order.')
    else:
        from obspy import read
        st = read(str(path))
        if len({(t.stats.network, t.stats.station, t.stats.location, t.stats.channel[:-1]) for t in st}) != 1:
            raise ValueError('Use one station/location/instrument family per file.')
        if any(t.stats.sampling_rate != SAMPLE_RATE for t in st):
            raise ValueError('Model input requires 100 Hz; resample upstream.')
        if st.get_gaps():
            raise ValueError('Gaps/overlaps require explicit upstream repair.')
        st.merge(method=0)
        channels = {t.stats.channel[-1]: t for t in st}
        if len(channels) != len(st) or set(channels) not in ({'Z'}, {'Z', 'N', 'E'}):
            raise ValueError('Expected vertical Z only, or aligned ZNE channels.')
        if len({(str(t.stats.starttime), t.stats.npts) for t in st}) != 1:
            raise ValueError('Components must have identical start times and lengths.')
        if any(np.ma.isMaskedArray(t.data) and np.ma.getmaskarray(t.data).any() for t in st):
            raise ValueError('Masked samples are not accepted.')
        x = np.stack([channels[c].data for c in ('ZNE' if len(st)==3 else 'Z')])
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] != 30000 or not np.isfinite(x).all():
        raise ValueError('Expected exactly 30000 finite samples per channel (300 seconds).')
    if np.any(np.std(x, axis=1) == 0):
        raise ValueError('Flat components: inspect as a data-quality issue.')
    if x.shape[0] == 1:
        if missing_components != 'copy':
            raise ValueError('Vertical-only data requires explicit --missing-components copy.')
        x = np.repeat(x, 3, axis=0)
    return x


def make_windows(x):
    starts = list(range(0, x.shape[1]-WINDOW_SAMPLES+1, STRIDE_SAMPLES))
    last = x.shape[1]-WINDOW_SAMPLES
    if starts[-1] != last:
        starts.append(last)
    windows = np.stack([x[:, s:s+WINDOW_SAMPLES] for s in starts])
    windows = windows - windows.mean(axis=-1, keepdims=True)
    std = windows.std(axis=-1, keepdims=True)
    if np.any(std <= 1e-10):
        raise ValueError('Flat/near-flat subwindow: inspect as data quality.')
    return (windows/std).astype(np.float32), np.asarray(starts)/SAMPLE_RATE


class SeisLMEncoder:
    """Loader and feature extraction adapted from the official feature demo."""
    def __init__(self, checkpoint, device='cpu', batch_size=4):
        import torch
        from seisLM.model.foundation.pretrained_models import LitMultiDimWav2Vec2
        self.torch, self.device, self.batch_size = torch, device, batch_size
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        # Official Lightning checkpoint contains serialized ConfigDict metadata.
        # Load ONLY an official/trusted checkpoint: weights_only=False unpickles it.
        self.model = LitMultiDimWav2Vec2.load_from_checkpoint(
            str(checkpoint), map_location='cpu', weights_only=False).model
        if self.model.config.input_dim != 3:
            raise ValueError('This adapter expects the official 3-component checkpoint.')
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, windows):
        embeddings = []
        with self.torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                batch = self.torch.from_numpy(windows[start:start+self.batch_size]).to(self.device)
                out = self.model.wav2vec2(input_values=batch, output_hidden_states=True)
                embeddings.append(out.hidden_states[-1].mean(dim=1).cpu().numpy())
        return np.concatenate(embeddings).astype(np.float32)


def segment_score(scores, top_k=3):
    return float(np.sort(scores)[-min(top_k, len(scores)):].mean())


def score_embeddings(embeddings, starts, table, output, quantile=.99, top_k=3):
    """Fits exclusively on fit windows; threshold uses calibration segments."""
    if not 0 < quantile < 1 or top_k < 1:
        raise ValueError('Invalid quantile/top-k.')
    for split in ('fit', 'calibration', 'test'):
        if not (table.split == split).any():
            raise ValueError(f'Missing {split} split.')
    if not table.split.isin(['fit', 'calibration', 'test']).all():
        raise ValueError('Unknown split.')
    dims = set()
    for i, e in enumerate(embeddings):
        if e.ndim != 2 or not len(e) or not np.isfinite(e).all() or len(e) != len(starts[i]):
            raise ValueError('Invalid embeddings or offsets.')
        dims.add(e.shape[1])
    if len(dims) != 1:
        raise ValueError('Embedding dimensions differ.')
    fit = np.concatenate([embeddings[i] for i in range(len(table)) if table.iloc[i].split == 'fit'])
    scaler = StandardScaler().fit(fit)
    detector = IsolationForest(n_estimators=300, random_state=42).fit(scaler.transform(fit))
    window_scores = [-detector.score_samples(scaler.transform(e)) for e in embeddings]
    scores = [segment_score(s, top_k) for s in window_scores]
    threshold = float(np.quantile([s for s, split in zip(scores, table.split) if split == 'calibration'], quantile))
    result = table.copy()
    result['score'], result['threshold'] = scores, threshold
    result['anomaly'] = [bool(s > threshold) if split == 'test' else None for s, split in zip(scores, table.split)]
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    result.sort_values('score', ascending=False).to_csv(output/'segments.csv', index=False)
    pd.DataFrame([dict(path=table.iloc[i].path, split=table.iloc[i].split,
                       start_seconds=float(t), end_seconds=float(t+WINDOW_SAMPLES/SAMPLE_RATE), score=float(s))
                  for i in range(len(table)) for t, s in zip(starts[i], window_scores[i])]).to_csv(output/'windows.csv', index=False)
    joblib.dump(dict(scaler=scaler, detector=detector, threshold=threshold, top_k=top_k), output/'detector.joblib')
    return result, threshold


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(manifest, checkpoint, output, sample_rate=None, missing_components='error',
        device='cpu', batch_size=4, quantile=.99, top_k=3):
    manifest = Path(manifest).resolve()
    table = pd.read_csv(manifest)
    if not {'path', 'split'} <= set(table):
        raise ValueError('Manifest requires path,split.')
    paths = [(manifest.parent/str(p)).resolve() for p in table.path]
    hashes = [file_hash(p) for p in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate recording bytes, possibly across splits.')
    table['path'] = [str(p) for p in paths]
    encoder = SeisLMEncoder(checkpoint, device, batch_size)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    embeddings, offsets = [], []
    for i, path in enumerate(paths):
        windows, starts = make_windows(load_waveform(path, sample_rate, missing_components))
        e = encoder.encode(windows)
        embeddings.append(e); offsets.append(starts)
        np.savez_compressed(output/f'embeddings_{i:05d}.npz', embeddings=e, starts=starts,
                            path=str(path), recording_sha256=hashes[i])
    result, threshold = score_embeddings(embeddings, offsets, table, output, quantile, top_k)
    (output/'config.json').write_text(json.dumps(dict(
        encoder='SeisLM final hidden state temporal mean', checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=file_hash(checkpoint), sample_rate=100, window_samples=3001,
        stride_samples=1500, missing_components=missing_components, device=device,
        batch_size=batch_size, quantile=quantile, top_k=top_k, threshold=threshold,
        seed=42, recording_sha256=hashes), indent=2))
    print(result[result.split == 'test'].sort_values('score', ascending=False).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('results_seislm'))
    p.add_argument('--sample-rate', type=int)
    p.add_argument('--missing-components', choices=['error', 'copy'], default='error')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--quantile', type=float, default=.99)
    p.add_argument('--top-k', type=int, default=3)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch-size must be positive.')
    run(**vars(a))
