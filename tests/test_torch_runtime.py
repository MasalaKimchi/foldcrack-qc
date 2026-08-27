from __future__ import annotations

import unittest
from types import SimpleNamespace

from foldcrack_qc._torch_runtime import (
    select_torch_device,
    synchronize_torch_device,
)


class _MPSBackend:
    def __init__(self, *, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _MPSRuntime:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _FailingMPSRuntime:
    @staticmethod
    def synchronize() -> None:
        raise RuntimeError("simulated backend failure")


def _fake_torch(*, mps_available: bool, with_synchronize: bool = True) -> object:
    mps = _MPSRuntime() if with_synchronize else SimpleNamespace()
    return SimpleNamespace(
        backends=SimpleNamespace(mps=_MPSBackend(available=mps_available)),
        mps=mps,
    )


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_preserves_mps_then_cpu_policy(self) -> None:
        self.assertEqual(
            select_torch_device("auto", torch_module=_fake_torch(mps_available=True)),
            "mps",
        )
        self.assertEqual(
            select_torch_device("auto", torch_module=_fake_torch(mps_available=False)),
            "cpu",
        )

    def test_explicit_cpu_and_case_normalization_are_preserved(self) -> None:
        torch = _fake_torch(mps_available=True)
        self.assertEqual(select_torch_device("CPU", torch_module=torch), "cpu")
        self.assertEqual(select_torch_device("AUTO", torch_module=torch), "mps")

    def test_explicit_unavailable_mps_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
            select_torch_device("mps", torch_module=_fake_torch(mps_available=False))

    def test_invalid_device_preserves_validation_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "'auto', 'mps', or 'cpu'"):
            select_torch_device("cuda", torch_module=_fake_torch(mps_available=False))


class SynchronizationTests(unittest.TestCase):
    def test_mps_synchronization_is_called(self) -> None:
        torch = _fake_torch(mps_available=True)
        synchronize_torch_device(torch, "mps")
        self.assertEqual(torch.mps.synchronize_calls, 1)

    def test_non_mps_is_always_a_noop(self) -> None:
        torch = _fake_torch(mps_available=False, with_synchronize=False)
        synchronize_torch_device(torch, "cpu", require_available=True)

    def test_missing_hook_supports_permissive_and_fail_closed_policies(self) -> None:
        torch = _fake_torch(mps_available=True, with_synchronize=False)
        synchronize_torch_device(torch, "mps")
        with self.assertRaisesRegex(TypeError, "synchronization is unavailable"):
            synchronize_torch_device(torch, "mps", require_available=True)

    def test_backend_synchronization_errors_propagate(self) -> None:
        torch = SimpleNamespace(mps=_FailingMPSRuntime())
        with self.assertRaisesRegex(RuntimeError, "simulated backend failure"):
            synchronize_torch_device(torch, "mps")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
