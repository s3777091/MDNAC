from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


API_ROOT = Path(__file__).resolve().parents[1] / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


class FakeTransport:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def get_text(self, url, params=None, headers=None):
        key = (url, tuple(sorted((params or {}).items())))
        self.calls.append(key)
        if key in self._responses:
            return self._responses[key]

        actual_params = dict(key[1])
        matching_keys = [
            candidate_key
            for candidate_key in self._responses
            if candidate_key[0] == url
            and all(actual_params.get(name) == value for name, value in candidate_key[1])
        ]
        if matching_keys:
            return self._responses[max(matching_keys, key=lambda candidate_key: len(candidate_key[1]))]

        raise KeyError(key)


def sorted_params(**kwargs):
    return tuple(sorted(kwargs.items()))


class SpanCompletionAPITests(unittest.TestCase):
    def test_builds_span_prompt_from_ncbi_result_without_returning_output(self) -> None:
        from fastapi.testclient import TestClient
        from interfere.server import SPAN_COMPLETION_ROUTE, create_app

        raw_input = (
            "uncharacterized protein LOC111693495 Trichogramma pretiosum "
            "RefSeq host Alabama argillacea"
        )
        query = f"{raw_input} AND protein[Filter]"
        left = "MRPVASVNLFLKTR"
        missing_span = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHI"
        right = "ALHLYGSKEDFNRTLCLSFCALRRLQLYSIEDEIRKELSTFGSGNDTRLIDHVNKALKSCKNLL"
        full_sequence = f"{left}{missing_span}{right}"

        responses = {
            (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                sorted_params(
                    db="protein",
                    email="test@test.com",
                    idtype="acc",
                    retmax=1,
                    retmode="json",
                    retstart=0,
                    term=query,
                    tool="microbial-dna-compiler",
                ),
            ): json.dumps(
                {"esearchresult": {"count": "1", "idlist": ["XP_0111693495.1"]}}
            ),
            (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                sorted_params(
                    db="protein",
                    email="test@test.com",
                    id="XP_0111693495.1",
                    retmode="json",
                    tool="microbial-dna-compiler",
                ),
            ): json.dumps(
                {
                    "result": {
                        "uids": ["101"],
                        "101": {
                            "accessionversion": "XP_0111693495.1",
                            "caption": "XP_0111693495",
                            "gene": "LOC111693495",
                            "host": "Alabama argillacea",
                            "keywords": "RefSeq",
                            "organism": "Trichogramma pretiosum",
                            "product": "uncharacterized protein LOC111693495",
                            "slen": len(full_sequence),
                            "title": (
                                "uncharacterized protein LOC111693495 "
                                "[Trichogramma pretiosum]."
                            ),
                        },
                    }
                }
            ),
            (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                sorted_params(
                    db="protein",
                    email="test@test.com",
                    id="XP_0111693495.1",
                    retmode="text",
                    rettype="fasta",
                    tool="microbial-dna-compiler",
                ),
            ): (
                ">XP_0111693495.1 uncharacterized protein LOC111693495 "
                "[Trichogramma pretiosum].\n"
                f"{full_sequence}\n"
            ),
        }
        transport = FakeTransport(responses)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = _write_api_config(Path(temp_dir))
            with patch.dict(os.environ, {"MICROBIAL_DATA_NCBI_EMAIL": "test@test.com"}), patch(
                "interfere.server._new_http_transport",
                return_value=transport,
            ):
                client = TestClient(create_app(config_path=config_path, environment="local"))
                response = client.post(
                    SPAN_COMPLETION_ROUTE,
                    json={
                        "raw_input": raw_input,
                        "source": "ncbi",
                        "limit": 1,
                        "mask_policy": "random_span",
                        "mask_start": 14,
                        "mask_length": 48,
                        "left_flank_size": 64,
                        "right_flank_size": 64,
                    },
                )

        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual({"instruction", "input"}, set(payload))
        self.assertEqual(
            (
                "task protein span completion; labels protein sequence; "
                "description uncharacterized protein LOC111693495 [Trichogramma pretiosum].; "
                "organism Trichogramma pretiosum; keywords RefSeq; gene LOC111693495; "
                "product uncharacterized protein LOC111693495; host Alabama argillacea"
            ),
            payload["instruction"],
        )
        self.assertIn("mask_policy random_span", payload["input"])
        self.assertIn("mask_start 14", payload["input"])
        self.assertIn("mask_length 48", payload["input"])
        self.assertIn("<MASK_48>", payload["input"])
        self.assertIn("partial_sequence", payload["input"])
        self.assertNotIn("output", payload)
        self.assertNotIn(missing_span, json.dumps(payload))
        self.assertNotIn(full_sequence, json.dumps(payload))


def _write_api_config(root: Path) -> Path:
    config_path = root / "config.yaml"
    config_path.write_text(
        f"""
environment: local

environments:
  local:
    model:
      path: {root.as_posix()}/model.onnx
      device: cpu
    server:
      host: 127.0.0.1
      port: 8000
      reload: false
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


if __name__ == "__main__":
    unittest.main()
