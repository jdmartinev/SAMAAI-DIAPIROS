"""Test preprocessing and detector/calibration; no pretrained inference claim."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from detect_segments import load_waveform, make_windows, score_embeddings


class PipelineTests(unittest.TestCase):
    def test_vertical_requires_explicit_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'trace.npy'
            np.save(path, np.sin(np.arange(30000)/10).astype(np.float32))
            with self.assertRaises(ValueError):
                load_waveform(path, 100)
            x = load_waveform(path, 100, 'copy')
            self.assertEqual(x.shape, (3, 30000))
            np.testing.assert_array_equal(x[0], x[2])
            windows, starts = make_windows(x)
            self.assertEqual(windows.shape, (19, 3, 3001))
            self.assertAlmostEqual(starts[-1]+30.01, 300)
            np.testing.assert_allclose(windows.mean(axis=-1), 0, atol=1e-6)
            np.testing.assert_allclose(windows.std(axis=-1), 1, atol=1e-6)

    def test_nonfinite_and_flat_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.npy'
            for values in [np.zeros(30000), np.full(30000, np.nan)]:
                np.save(path, values)
                with self.assertRaises(ValueError):
                    load_waveform(path, 100, 'copy')

    def test_calibration_and_fit_do_not_use_test(self):
        # Random vectors are test doubles, NOT SeisLM embeddings.
        rng = np.random.default_rng(42)
        split = ['fit']*12+['calibration']*12+['test']*2
        table = pd.DataFrame(dict(path=[f'fake_{i}' for i in range(26)], split=split))
        embeddings = [rng.normal(size=(19, 8)).astype(np.float32) for _ in split]
        starts = [np.arange(19)*15.0 for _ in split]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _, threshold1 = score_embeddings(embeddings, starts, table, out/'a')
            # Change only test vectors; reference fit and threshold must be identical.
            embeddings[-1] = np.full((19, 8), 100, dtype=np.float32)
            result, threshold2 = score_embeddings(embeddings, starts, table, out/'b')
            self.assertEqual(threshold1, threshold2)
            self.assertTrue(result.iloc[-1].anomaly)
            artifact = joblib.load(out/'b'/'detector.joblib')
            fit = np.concatenate(embeddings[:12])
            np.testing.assert_allclose(artifact['scaler'].mean_, fit.mean(axis=0), atol=1e-6)
            self.assertTrue(result[result.split != 'test'].anomaly.isna().all())
            self.assertTrue((out/'b'/'windows.csv').is_file())
            self.assertTrue((out/'b'/'segments.csv').is_file())


if __name__ == '__main__':
    unittest.main()
