import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class HandwritingClient:
    def __init__(self, base_url: str, output_dir: Path, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.output_dir = output_dir
        self.timeout_s = timeout_s
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=20))
    def generate(
        self,
        text: str,
        style: int,
        bias: float,
        seed: int,
        request_id: str,
    ) -> Tuple[Path, Dict[str, Any]]:
        payload = {
            "text": text,
            "style": style,
            "bias": bias,
            "seed": seed % (2**32),
            "request_id": request_id,
        }
        response = self._client.post(f"{self.base_url}/generate", json=payload)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Handwriting service error {response.status_code}: {response.text}"
            )
        data = response.json()
        svg_filename = data.get("svg_filename")
        if not svg_filename:
            raise RuntimeError("Handwriting service did not return svg_filename")
        svg_path = self.output_dir / svg_filename
        _wait_for_file(svg_path, timeout_s=self.timeout_s)
        return svg_path, data.get("meta", {})


def _wait_for_file(path: Path, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {path}")
