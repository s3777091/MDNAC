"""Unit tests for ObjectStore abstraction and CatalogRepository."""

from __future__ import annotations

import unittest
from pathlib import Path
import shutil

from libs.data.backends.object_store import LocalObjectStore
from libs.data.backends.catalog import CatalogRepository
from libs.data.entities import DatasetArtifact


class LocalObjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/object-store-test")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = LocalObjectStore(root=self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_put_and_get_text(self) -> None:
        self.store.put_text("hello/world.txt", "content here")
        result = self.store.get_text("hello/world.txt")
        self.assertEqual("content here", result)

    def test_get_text_returns_none_for_missing_key(self) -> None:
        result = self.store.get_text("does/not/exist.txt")
        self.assertIsNone(result)

    def test_list_keys_with_prefix(self) -> None:
        self.store.put_text("data/a.txt", "a")
        self.store.put_text("data/b.txt", "b")
        self.store.put_text("other/c.txt", "c")
        keys = self.store.list_keys("data/")
        self.assertEqual(sorted(keys), ["data/a.txt", "data/b.txt"])

    def test_delete_prefix(self) -> None:
        self.store.put_text("prefix/a.txt", "a")
        self.store.put_text("prefix/b.txt", "b")
        self.store.delete_prefix("prefix/")
        keys = self.store.list_keys("prefix/")
        self.assertEqual(keys, [])


class CatalogRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/artifacts/catalog-repo-test")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = LocalObjectStore(root=self.root)
        self.catalog = CatalogRepository(object_store=self.store, catalog_key="datasets.csv")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_read_rows_empty_when_no_catalog(self) -> None:
        rows = self.catalog.read_rows()
        self.assertEqual(rows, [])

    def _make_artifact(self, source: str = "ncbi", name: str = "ds1", records: int = 10) -> DatasetArtifact:
        return DatasetArtifact(
            source_name=source,
            dataset_name=name,
            storage_mode="local",
            snapshot_id="snap-001",
            current_location="/data/ds1",
            file_locations={},
            record_count=records,
        )

    def test_upsert_and_read_back(self) -> None:
        self.catalog.upsert(self._make_artifact())
        rows = self.catalog.read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset_name"], "ds1")

    def test_upsert_updates_existing_row(self) -> None:
        self.catalog.upsert(self._make_artifact(records=10))
        self.catalog.upsert(self._make_artifact(records=20))
        rows = self.catalog.read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["record_count"], "20")

    def test_remove_row(self) -> None:
        self.catalog.upsert(self._make_artifact(source="ncbi", name="ds1"))
        self.catalog.upsert(self._make_artifact(source="ncbi", name="ds2"))
        self.catalog.remove("ncbi", "ds1")
        rows = self.catalog.read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset_name"], "ds2")

    def test_list_datasets(self) -> None:
        self.catalog.upsert(self._make_artifact(source="ena", name="alpha"))
        self.catalog.upsert(self._make_artifact(source="ncbi", name="beta"))
        datasets = self.catalog.list_datasets()
        names = sorted(d.dataset_name for d in datasets)
        self.assertEqual(names, ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()

