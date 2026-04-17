from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from libs.data.utilities.http_index_download import (
    DirectoryEntry,
    default_output_dir,
    download_directory,
    download_entry,
    extract_directory_entries,
    filter_entries,
)


class _FakeHeaders(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeResponse:
    def __init__(self, payload: bytes, content_length: int | None = None, fail_after_reads: int | None = None) -> None:
        self._payload = payload
        self._offset = 0
        self._read_count = 0
        self._fail_after_reads = fail_after_reads
        headers = _FakeHeaders()
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, amount: int = -1) -> bytes:
        if self._fail_after_reads is not None and self._read_count >= self._fail_after_reads:
            raise OSError("network interrupted")
        self._read_count += 1
        if self._offset >= len(self._payload):
            return b""
        if amount < 0:
            amount = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk


class HttpIndexDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/http-index-download")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_extracts_only_direct_file_entries(self):
        html = """
        <html><body><pre>
        <a href="../">Parent Directory</a>
        <a href="subdir/">subdir/</a>
        <a href="alpha.txt">alpha.txt</a>
        <a href="beta.faa.gz">beta.faa.gz</a>
        <a href="?C=N;O=D">sort</a>
        </pre></body></html>
        """

        entries = extract_directory_entries(
            html,
            "https://ftp.ncbi.nlm.nih.gov/refseq/release/vertebrate_other/",
        )

        self.assertEqual(["alpha.txt", "beta.faa.gz"], [entry.name for entry in entries])

    def test_filters_entries_by_include_and_exclude_patterns(self):
        html = """
        <html><body><pre>
        <a href="viral.1.protein.faa.gz">viral.1.protein.faa.gz</a>
        <a href="viral.1.protein.gpff.gz">viral.1.protein.gpff.gz</a>
        <a href="viral.1.genomic.gbff.gz">viral.1.genomic.gbff.gz</a>
        </pre></body></html>
        """

        entries = extract_directory_entries(html, "https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/")
        filtered = filter_entries(
            entries,
            include_patterns=("*.protein.*",),
            exclude_patterns=("*.gpff.gz",),
        )

        self.assertEqual(["viral.1.protein.faa.gz"], [entry.name for entry in filtered])

    def test_builds_default_output_dir_from_url(self):
        output_dir = default_output_dir("https://ftp.ncbi.nlm.nih.gov/refseq/release/vertebrate_other/")

        self.assertEqual(
            Path("data/raw/refseq_bacteria_protein/vertebrate_other"),
            output_dir,
        )

    def test_builds_generic_default_output_dir_for_non_refseq_urls(self):
        output_dir = default_output_dir("https://example.com/files/protein/")

        self.assertEqual(
            Path("data/downloads/example.com/files/protein"),
            output_dir,
        )

    def test_skips_existing_file_when_size_matches_server(self):
        entry = DirectoryEntry(name="alpha.txt", url="https://example.com/alpha.txt")
        temp_dir = self.root / "skip-existing"
        temp_dir.mkdir(parents=True, exist_ok=True)
        destination = temp_dir / entry.name
        destination.write_bytes(b"abc")

        with patch(
            "libs.data.utilities.http_index_download.urlopen",
            return_value=_FakeResponse(b"abc", content_length=3),
        ):
            result = download_entry(entry, temp_dir)

        self.assertEqual("skipped", result.status)
        self.assertEqual(b"abc", destination.read_bytes())
        self.assertFalse((temp_dir / "alpha.txt.part").exists())

    def test_replaces_existing_file_when_size_differs(self):
        entry = DirectoryEntry(name="alpha.txt", url="https://example.com/alpha.txt")
        temp_dir = self.root / "replace-existing"
        temp_dir.mkdir(parents=True, exist_ok=True)
        destination = temp_dir / entry.name
        destination.write_bytes(b"a")

        with patch(
            "libs.data.utilities.http_index_download.urlopen",
            return_value=_FakeResponse(b"abc", content_length=3),
        ):
            result = download_entry(entry, temp_dir)

        self.assertEqual("replaced", result.status)
        self.assertEqual(b"abc", destination.read_bytes())
        self.assertFalse((temp_dir / "alpha.txt.part").exists())

    def test_preserves_existing_file_when_redownload_fails(self):
        entry = DirectoryEntry(name="alpha.txt", url="https://example.com/alpha.txt")
        temp_dir = self.root / "preserve-existing"
        temp_dir.mkdir(parents=True, exist_ok=True)
        destination = temp_dir / entry.name
        destination.write_bytes(b"stable")

        with patch(
            "libs.data.utilities.http_index_download.urlopen",
            return_value=_FakeResponse(b"abcdef", content_length=6, fail_after_reads=1),
        ):
            with self.assertRaises(OSError):
                download_entry(entry, temp_dir, force=True)

        self.assertEqual(b"stable", destination.read_bytes())
        self.assertFalse((temp_dir / "alpha.txt.part").exists())

    def test_download_directory_skips_existing_file_without_history_tracking(self):
        temp_dir = self.root / "directory-skip"
        temp_dir.mkdir(parents=True, exist_ok=True)
        destination = temp_dir / "alpha.txt"
        destination.write_bytes(b"abc")

        with patch(
            "libs.data.utilities.http_index_download.fetch_index_html",
            return_value='<a href="alpha.txt">alpha.txt</a>',
        ), patch(
            "libs.data.utilities.http_index_download.urlopen",
            return_value=_FakeResponse(b"abc", content_length=3),
        ):
            output_dir, results = download_directory(
                "https://example.com/files/",
                output_dir=temp_dir,
            )

        self.assertEqual(temp_dir, output_dir)
        self.assertEqual(["skipped"], [result.status for result in results])
        self.assertFalse((temp_dir / "history.json").exists())


if __name__ == "__main__":
    unittest.main()
