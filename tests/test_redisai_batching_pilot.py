import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np
import torch


spec = importlib.util.spec_from_file_location(
    "redisai_batching_pilot", Path(__file__).resolve().parents[1] / "scripts/redisai_batching_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class BatchWitnessTest(unittest.TestCase):
    def test_scripted_dynamic_batch_and_split(self):
        model = torch.jit.script(pilot.BatchWitness())
        serialized = io.BytesIO()
        torch.jit.save(model, serialized)
        serialized.seek(0)
        model = torch.jit.load(serialized)
        inputs = [torch.tensor([[0.], [1.]]), torch.tensor([[10.], [11.], [12.]])]
        for x in inputs:
            pilot.check_output(x.numpy(), model(x).numpy(), len(x))
        merged = model(torch.cat(inputs)).numpy()
        for x, output in zip(inputs, np.split(merged, [2])):
            pilot.check_output(x.numpy(), output, 5)

    def test_rejects_wrong_batch_or_values(self):
        x = np.array([[1.], [2.]], dtype=np.float32)
        with self.assertRaises(AssertionError):
            pilot.check_output(x, pilot.expected(x, 2), 5)
        bad = pilot.expected(x, 5)
        bad[0, 0] = -1
        with self.assertRaises(AssertionError):
            pilot.check_output(x, bad, 5)


if __name__ == "__main__":
    unittest.main()
