"""Hierarchical cache on the unified memory pool.

HiCache addresses the device buffers with the ids the controller holds, and
under `--enable-unified-memory` those are VIRTUAL while the L2 kernels index
per-layer views in kernel-facing space (and the state pool by physical slot).
On top of that the pool RELOCATES pages under compaction, and the conv/SSM
views are envelope-strided rather than a contiguous per-slot array.

So the guard has to be numerical, and it has to force a real host round trip:
a small device pool plus a dozen distinct long prefixes evicts the target off
the device, and re-requesting it can only be served by loading back through
L2. If any of the translate, the staging, or the move gate were wrong, the
reloaded KV would differ.

The reference is the SAME pool with HiCache off -- not the static pool. Unified
and static legitimately differ in reduction order here (measured up to 1.8 in
logprob on a fresh prompt for this model), so a static baseline would drown the
signal; against unified-without-HiCache the expectation is bit-equality.

    python -m pytest test/registered/hicache/test_hicache_unified_memory.py -v
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="extra-b", runner_config="2-gpu-large")

# Smallest in-tree GDN hybrid: MHA full attention + gated-delta-net state, i.e.
# both a per-layer-view sub-pool and an envelope-strided state sub-pool.
MODEL = "Qwen/Qwen3.5-0.8B"

_BASE_ARGS = [
    "--trust-remote-code",
    "--enable-unified-memory",
    "--linear-attn-backend",
    "triton",
    "--mamba-backend",
    "triton",
    "--mem-fraction-static",
    "0.6",
    # A small device pool is what makes the host tier reachable at all.
    "--max-total-tokens",
    "8192",
    "--max-mamba-cache-size",
    "64",
    "--enable-cache-report",
]
_HICACHE_ARGS = _BASE_ARGS + ["--enable-hierarchical-cache", "--hicache-ratio", "4"]

_PREFIX = (
    "The following is a detailed technical description of a distributed inference "
    "system with paged attention, radix prefix caching and hierarchical offload. "
) * 90
_TARGET = _PREFIX + " Question one:"


def _generate(base_url, text, max_new_tokens=32, logprobs=True):
    payload = {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
    }
    if logprobs:
        payload["return_logprob"] = True
        payload["logprob_start_len"] = 0
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=600)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    lp = (
        [t[0] for t in data["meta_info"]["output_token_logprobs"]] if logprobs else None
    )
    return data["text"], lp


class TestUnifiedMemoryHiCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.hicache_url = "http://127.0.0.1:8157"
        cls.reference_url = "http://127.0.0.1:8158"
        cls.process_hicache = popen_launch_server(
            cls.model,
            cls.hicache_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_HICACHE_ARGS,
        )
        cls.process_reference = popen_launch_server(
            cls.model,
            cls.reference_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_BASE_ARGS + ["--base-gpu-id", "1"],
        )

    @classmethod
    def tearDownClass(cls):
        for proc in (
            getattr(cls, "process_hicache", None),
            getattr(cls, "process_reference", None),
        ):
            if proc is not None:
                kill_process_tree(proc.pid)

    def _force_host_round_trip(self):
        """Evict the target off the device so the next hit must come from L2."""
        for i in range(12):
            _generate(
                self.hicache_url,
                _PREFIX + f" filler variant {i}. Question:",
                max_new_tokens=8,
                logprobs=False,
            )

    def test_load_back_matches_no_hicache(self):
        """The sharp one: KV that made a device->host->device round trip must
        produce the same logprobs as a run that never left the device."""
        cold_text, cold_lp = _generate(self.hicache_url, _TARGET)
        self._force_host_round_trip()
        warm_text, warm_lp = _generate(self.hicache_url, _TARGET)
        ref_text, ref_lp = _generate(self.reference_url, _TARGET)

        self.assertEqual(cold_text, ref_text)
        self.assertEqual(warm_text, ref_text)
        for label, lp in (("cold", cold_lp), ("after-L2-reload", warm_lp)):
            delta = max(abs(a - b) for a, b in zip(lp, ref_lp))
            self.assertAlmostEqual(
                delta,
                0.0,
                places=5,
                msg=f"{label} diverged from the no-HiCache reference by {delta}",
            )

    def test_server_survives_the_round_trip(self):
        """A wrong move gate or a missed free shows up as the idle memory-leak
        invariant aborting the scheduler rather than as bad output."""
        self._force_host_round_trip()
        for url in (self.hicache_url, self.reference_url):
            resp = requests.get(f"{url}/health", timeout=30)
            self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
