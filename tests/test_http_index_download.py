from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from libs.data.utilities.http_index_download import (
    DirectoryEntry,
    default_output_dir,
    download_directory,
    download_entry,
    extract_directory_entries,
    filter_entries,
)
from libs.data.utilities.refseq_history import load_refseq_history, resolve_refseq_history_path, save_refseq_history


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
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir, entry.name)
            destination.write_bytes(b"abc")

            with patch(
                "libs.data.utilities.http_index_download.urlopen",
                return_value=_FakeResponse(b"abc", content_length=3),
            ):
                result = download_entry(entry, temp_dir)

            self.assertEqual("skipped", result.status)
            self.assertEqual(b"abc", destination.read_bytes())
            self.assertFalse(Path(temp_dir, "alpha.txt.part").exists())

    def test_replaces_existing_file_when_size_differs(self):
        entry = DirectoryEntry(name="alpha.txt", url="https://example.com/alpha.txt")
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir, entry.name)
            destination.write_bytes(b"a")

            with patch(
                "libs.data.utilities.http_index_download.urlopen",
                return_value=_FakeResponse(b"abc", content_length=3),
            ):
                result = download_entry(entry, temp_dir)

            self.assertEqual("replaced", result.status)
            self.assertEqual(b"abc", destination.read_bytes())
            self.assertFalse(Path(temp_dir, "alpha.txt.part").exists())

    def test_preserves_existing_file_when_redownload_fails(self):
        entry = DirectoryEntry(name="alpha.txt", url="https://example.com/alpha.txt")
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir, entry.name)
            destination.write_bytes(b"stable")

            with patch(
                "libs.data.utilities.http_index_download.urlopen",
                return_value=_FakeResponse(b"abcdef", content_length=6, fail_after_reads=1),
            ):
                with self.assertRaises(OSError):
                    download_entry(entry, temp_dir, force=True)

            self.assertEqual(b"stable", destination.read_bytes())
            self.assertFalse(Path(temp_dir, "alpha.txt.part").exists())

    def test_download_directory_bootstraps_existing_file_into_history(self):
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir, "alpha.txt")
            destination.write_bytes(b"abc")

            with patch(
                "libs.data.utilities.http_index_download.fetch_index_html",
                return_value='<a href="alpha.txt">alpha.txt</a>',
            ), patch(
                "libs.data.utilities.http_index_download.urlopen",
                return_value=_FakeResponse(b"abc", content_length=3),
            ):
                output_dir, results, history_path = download_directory(
                    "https://example.com/files/",
                    output_dir=temp_dir,
                )

            self.assertEqual(Path(temp_dir), output_dir)
            self.assertEqual(["skipped"], [result.status for result in results])
            history = load_refseq_history(history_path, input_root=temp_dir)
            archive_entry = history["archives"]["alpha.txt"]
            self.assertTrue(archive_entry["present_on_disk"])
            self.assertEqual("skipped", archive_entry["last_download_status"])
            self.assertEqual("pending", archive_entry["build_status"])

    def test_download_directory_uses_history_to_skip_deleted_compiled_file(self):
        with TemporaryDirectory() as temp_dir:
            history_path = resolve_refseq_history_path(temp_dir)
            history = load_refseq_history(history_path, input_root=temp_dir)
            history["archives"]["alpha.txt"] = {
                "file_name": "alpha.txt",
                "relative_path": "alpha.txt",
                "group_name": Path(temp_dir).name,
                "kind": "unknown",
                "build_status": "compiled",
                "expected_size": 3,
                "compiled_local_size": 3,
                "compiled_modified_time_ns": 1,
                "present_on_disk": False,
            }
            save_refseq_history(history_path, history)

            with patch(
                "libs.data.utilities.http_index_download.fetch_index_html",
                return_value='<a href="alpha.txt">alpha.txt</a>',
            ), patch(
                "libs.data.utilities.http_index_download.urlopen",
                return_value=_FakeResponse(b"abc", content_length=3),
            ):
                output_dir, results, saved_history_path = download_directory(
                    "https://example.com/files/",
                    output_dir=temp_dir,
                )

            self.assertEqual(Path(temp_dir), output_dir)
            self.assertEqual(history_path, saved_history_path)
            self.assertEqual(["recorded"], [result.status for result in results])
            self.assertFalse(Path(temp_dir, "alpha.txt").exists())

            history = load_refseq_history(history_path, input_root=temp_dir)
            archive_entry = history["archives"]["alpha.txt"]
            self.assertFalse(archive_entry["present_on_disk"])
            self.assertEqual("recorded", archive_entry["last_download_status"])
            self.assertEqual("compiled", archive_entry["build_status"])


if __name__ == "__main__":
    unittest.main()
