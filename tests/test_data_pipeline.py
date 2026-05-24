from __future__ import annotations

from http.client import IncompleteRead
import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from libs.data.backends.local import LocalDatasetRepository
from libs.data.backends.manager import DatasetManager
from libs.data.config import DataConfig
from libs.data.entities import FetchRequest, SequenceRecord
from libs.data.hub import MicrobialDataHub
from libs.data.sources.ddbj import DdbjSequenceSource
from libs.data.sources.ena import EnaSequenceSource
from libs.data.sources.ncbi import NcbiSequenceSource
from libs.data.training import SequenceNormalizationConfig, SequenceTokenizer, normalize_records
from libs.data.utilities.exceptions import SourceConfigurationError
from libs.data.utilities.http import UrllibHttpTransport


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
            if candidate_key[0] == url and all(actual_params.get(name) == value for name, value in candidate_key[1])
        ]
        if matching_keys:
            return self._responses[max(matching_keys, key=lambda candidate_key: len(candidate_key[1]))]

        raise KeyError(key)


def sorted_params(**kwargs):
    return tuple(sorted(kwargs.items()))


class EnaSequenceSourceTests(unittest.TestCase):
    def test_fetches_records_from_query(self):
        request = FetchRequest(dataset_name="ENA Coding Proteins", query="tax_tree(2)", limit=2)
        responses = {
            (
                "https://www.ebi.ac.uk/ena/portal/api/search",
                sorted_params(
                    fields="accession,description,scientific_name,base_count,protein_id,parent_accession,sequence_version",
                    format="tsv",
                    limit=2,
                    query="tax_tree(2)",
                    result="coding",
                ),
            ): (
                "accession\tdescription\tscientific_name\tbase_count\tprotein_id\tparent_accession\tsequence_version\n"
                "CAL001\tDNA gyrase subunit B\tEscherichia coli\t24\tCAL001.1\tOZ001.1\t1\n"
                "CAL002\tSugar phosphatase\tEscherichia coli\t18\tCAL002.1\tOZ001.1\t1\n"
            ),
            (
                "https://www.ebi.ac.uk/ena/browser/api/embl/CAL001,CAL002",
                sorted_params(),
            ): (
                "ID   CAL001; SV 1; linear; genomic DNA; STD; PRO; 24 BP.\n"
                "PA   OZ001.1\nDE   DNA gyrase subunit B\nOS   Escherichia coli\n"
                "FT   CDS             OZ001.1:1..24\nFT                   /protein_id=\"CAL001.1\"\n"
                "FT                   /translation=\"MPEP\nFT                   TIDE\"\n//\n"
                "ID   CAL002; SV 1; linear; genomic DNA; STD; PRO; 18 BP.\n"
                "PA   OZ001.1\nDE   Sugar phosphatase\nOS   Escherichia coli\n"
                "FT   CDS             OZ001.1:25..42\nFT                   /protein_id=\"CAL002.1\"\n"
                "FT                   /translation=\"GLY\nFT                   SERQ\"\n//\n"
            ),
        }

        records = EnaSequenceSource(transport=FakeTransport(responses)).fetch(request)

        self.assertEqual(2, len(records))
        self.assertEqual("CAL001.1", records[0].sequence_version)
        self.assertEqual("MPEPTIDE", records[0].sequence)
        self.assertEqual("GLYSERQ", records[1].sequence)

    def test_resolves_versioned_coding_accessions_from_query(self):
        request = FetchRequest(dataset_name="ENA Coding Proteins", query="tax_tree(2)", limit=2)
        responses = {
            (
                "https://www.ebi.ac.uk/ena/portal/api/search",
                sorted_params(fields="protein_id", format="tsv", limit=2, query="tax_tree(2)", result="coding"),
            ): "protein_id\nCAL001.1\nCAL002.4\n",
        }

        resolved = EnaSequenceSource(transport=FakeTransport(responses)).resolve_accessions(request)
        self.assertEqual(("CAL001.1", "CAL002.4"), resolved)

    def test_resolves_coding_accessions_from_query_without_offset_pagination(self):
        request = FetchRequest(dataset_name="ENA Coding Proteins", query="tax_tree(2)", limit=4)
        responses = {
            (
                "https://www.ebi.ac.uk/ena/portal/api/search",
                sorted_params(fields="protein_id", format="tsv", limit=4, query="tax_tree(2)", result="coding"),
            ): "protein_id\nCAL001.1\nCAL002.1\nCAL003.1\n",
        }

        with patch.object(EnaSequenceSource, "_DEFAULT_PORTAL_PAGE_SIZE", 2):
            transport = FakeTransport(responses)
            source = EnaSequenceSource(transport=transport)
            resolved = source.resolve_accessions(request)

        self.assertEqual(("CAL001.1", "CAL002.1", "CAL003.1"), resolved)
        self.assertEqual(
            [
                (
                    "https://www.ebi.ac.uk/ena/portal/api/search",
                    sorted_params(fields="protein_id", format="tsv", limit=4, query="tax_tree(2)", result="coding"),
                )
            ],
            transport.calls,
        )

    def test_requires_bounded_query_limit_for_unbounded_query_requests(self):
        request = FetchRequest(dataset_name="ENA Coding Proteins", query="tax_tree(2)", limit=0)

        with self.assertRaisesRegex(SourceConfigurationError, "positive limit"):
            EnaSequenceSource(transport=FakeTransport({})).resolve_accessions(request)

    def test_fetches_coding_records_in_embl_batches(self):
        request = FetchRequest(dataset_name="ENA Coding Proteins", query="tax_tree(2)", limit=4, batch_size=2)
        responses = {
            (
                "https://www.ebi.ac.uk/ena/portal/api/search",
                sorted_params(
                    fields="accession,description,scientific_name,base_count,protein_id,parent_accession,sequence_version",
                    format="tsv",
                    limit=4,
                    query="tax_tree(2)",
                    result="coding",
                ),
            ): (
                "accession\tdescription\tscientific_name\tbase_count\tprotein_id\tparent_accession\tsequence_version\n"
                "CAL001\tDNA gyrase subunit B\tEscherichia coli\t24\tCAL001.1\tOZ001.1\t1\n"
                "CAL002\tSugar phosphatase\tEscherichia coli\t18\tCAL002.1\tOZ001.1\t1\n"
                "CAL003\tStress protein\tEscherichia coli\t15\tCAL003.1\tOZ001.1\t1\n"
                "CAL004\tMembrane protein\tEscherichia coli\t12\tCAL004.1\tOZ001.1\t1\n"
            ),
            (
                "https://www.ebi.ac.uk/ena/browser/api/embl/CAL001,CAL002",
                sorted_params(),
            ): (
                "ID   CAL001; SV 1; linear; genomic DNA; STD; PRO; 24 BP.\n"
                "PA   OZ001.1\nDE   DNA gyrase subunit B\nOS   Escherichia coli\n"
                "FT   CDS             OZ001.1:1..24\nFT                   /protein_id=\"CAL001.1\"\n"
                "FT                   /translation=\"MPEP\nFT                   TIDE\"\n//\n"
                "ID   CAL002; SV 1; linear; genomic DNA; STD; PRO; 18 BP.\n"
                "PA   OZ001.1\nDE   Sugar phosphatase\nOS   Escherichia coli\n"
                "FT   CDS             OZ001.1:25..42\nFT                   /protein_id=\"CAL002.1\"\n"
                "FT                   /translation=\"GLY\nFT                   SERQ\"\n//\n"
            ),
            (
                "https://www.ebi.ac.uk/ena/browser/api/embl/CAL003,CAL004",
                sorted_params(),
            ): (
                "ID   CAL003; SV 1; linear; genomic DNA; STD; PRO; 15 BP.\n"
                "PA   OZ001.1\nDE   Stress protein\nOS   Escherichia coli\n"
                "FT   CDS             OZ001.1:43..57\nFT                   /protein_id=\"CAL003.1\"\n"
                "FT                   /translation=\"PEP\nFT                   TIDE\"\n//\n"
                "ID   CAL004; SV 1; linear; genomic DNA; STD; PRO; 12 BP.\n"
                "PA   OZ001.1\nDE   Membrane protein\nOS   Escherichia coli\n"
                "FT   CDS             OZ001.1:58..69\nFT                   /protein_id=\"CAL004.1\"\n"
                "FT                   /translation=\"MKL\nFT                   V\"\n//\n"
            ),
        }

        transport = FakeTransport(responses)
        records = EnaSequenceSource(transport=transport).fetch(request)

        self.assertEqual(4, len(records))
        self.assertIn(("https://www.ebi.ac.uk/ena/browser/api/embl/CAL001,CAL002", sorted_params()), transport.calls)
        self.assertIn(("https://www.ebi.ac.uk/ena/browser/api/embl/CAL003,CAL004", sorted_params()), transport.calls)


class DdbjSequenceSourceTests(unittest.TestCase):
    def test_fetches_records_from_accessions(self):
        request = FetchRequest(dataset_name="Petase Seeds", accessions=("AB000001", "AB000002"), limit=2, batch_size=2)
        responses = {
            (
                "https://getentry.ddbj.nig.ac.jp/getentry",
                sorted_params(accession_number="AB000001,AB000002", database="aa", filetype="text", format="flatfile", limit=2),
            ): (
                "LOCUS       AB000001                8 aa    linear   BCT 01-JAN-2020\nDEFINITION  PETase alpha candidate.\n"
                "ACCESSION   AB000001\nVERSION     AB000001.1\nSOURCE      Ideonella sakaiensis\n  ORGANISM  Ideonella sakaiensis\n//\n"
                "LOCUS       AB000002                7 aa    linear   BCT 01-JAN-2020\nDEFINITION  Nitrogen fixation candidate.\n"
                "ACCESSION   AB000002\nVERSION     AB000002.3\nSOURCE      Azotobacter vinelandii\n  ORGANISM  Azotobacter vinelandii\n//\n"
            ),
            (
                "https://getentry.ddbj.nig.ac.jp/getentry",
                sorted_params(accession_number="AB000001,AB000002", database="aa", filetype="text", format="fasta", limit=2),
            ): ">AB000001 PETase alpha candidate.\nMPEPTIDE\n>AB000002 Nitrogen fixation candidate.\nGLYSERQ\n",
        }

        records = DdbjSequenceSource(transport=FakeTransport(responses)).fetch(request)
        self.assertEqual(2, len(records))
        self.assertEqual("Ideonella sakaiensis", records[0].organism)
        self.assertEqual("AB000002.3", records[1].sequence_version)

    def test_requires_accessions(self):
        with self.assertRaises(SourceConfigurationError):
            DdbjSequenceSource(transport=FakeTransport({})).fetch(FetchRequest(dataset_name="Needs Accessions", query="petase", limit=1))


class NcbiSequenceSourceTests(unittest.TestCase):
    def test_fetches_records_from_query(self):
        request = FetchRequest(dataset_name="NCBI Proteins", query="txid2[Organism:exp]", limit=2)
        responses = {
            (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                sorted_params(db="protein", idtype="acc", retmax=2, retmode="json", retstart=0, term="txid2[Organism:exp]", tool="microbial-dna-compiler"),
            ): json.dumps({"esearchresult": {"count": "2", "idlist": ["NCBI001.1", "NCBI002.1"]}}),
            (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                sorted_params(db="protein", id="NCBI001.1,NCBI002.1", retmode="json", tool="microbial-dna-compiler"),
            ): json.dumps(
                {
                    "result": {
                        "uids": ["101", "102"],
                        "101": {"caption": "NCBI001", "title": "Nitrogen fixer candidate", "slen": 8, "biomol": "protein", "moltype": "aa", "sourcedb": "genbank", "organism": "Bacillus subtilis", "accessionversion": "NCBI001.1"},
                        "102": {"caption": "NCBI002", "title": "Plastic degrader candidate", "slen": 7, "biomol": "protein", "moltype": "aa", "sourcedb": "refseq", "organism": "Pseudomonas putida", "accessionversion": "NCBI002.1"},
                    }
                }
            ),
            (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                sorted_params(db="protein", id="NCBI001.1,NCBI002.1", retmode="text", rettype="fasta", tool="microbial-dna-compiler"),
            ): ">NCBI001.1 Nitrogen fixer candidate\nMPEPTIDE\n>NCBI002.1 Plastic degrader candidate\nGLYSERQ\n",
        }

        records = NcbiSequenceSource(transport=FakeTransport(responses), email="test@test.com").fetch(request)
        self.assertEqual(2, len(records))
        self.assertEqual("NCBI002.1", records[1].sequence_version)
        self.assertEqual("refseq", records[1].metadata["sourcedb"])


class UrllibHttpTransportTests(unittest.TestCase):
    def test_retries_incomplete_read(self):
        class _Headers:
            @staticmethod
            def get_content_charset():
                return "utf-8"

        class _Response:
            def __init__(self, payload: bytes | None = None, error: Exception | None = None) -> None:
                self.headers = _Headers()
                self._payload = payload
                self._error = error

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                if self._error is not None:
                    raise self._error
                return self._payload or b""

        transport = UrllibHttpTransport(max_retries=1, backoff_base=1.0)
        with patch(
            "libs.data.utilities.http.urlopen",
            side_effect=[
                _Response(error=IncompleteRead(b"partial", 7)),
                _Response(payload=b"ok"),
            ],
        ), patch("libs.data.utilities.http.time.sleep"):
            self.assertEqual("ok", transport.get_text("https://example.com"))


class DataConfigTests(unittest.TestCase):
    def setUp(self):
        self.config_path = Path("tests/artifacts/config.test.yaml")
        self.env_path = Path("tests/artifacts/config.test.env")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text("storage_mode: local\ndata_root: data-store\ndefault_batch_size: 7\nminio:\n  endpoint_url: http://minio.internal:9000\n  secure: false\n", encoding="utf-8")
        self.env_path.write_text("MICROBIAL_DATA_MINIO_ACCESS_KEY=test-access\nMICROBIAL_DATA_MINIO_SECRET_KEY=test-secret\nMINIO_ROOT_USER=test-access\nMINIO_ROOT_PASSWORD=test-secret\n", encoding="utf-8")

    def tearDown(self):
        if self.config_path.exists():
            self.config_path.unlink()
        if self.env_path.exists():
            self.env_path.unlink()

    def test_loads_from_yaml(self):
        with patch.dict(os.environ, {}, clear=True):
            config = DataConfig.load(self.config_path, env_path=self.env_path)

        self.assertEqual("local", config.storage_mode)
        self.assertEqual(self.config_path.parent / "data-store", config.data_root)
        self.assertEqual(7, config.default_batch_size)
        self.assertEqual("http://minio.internal:9000", config.minio.endpoint_url)
        self.assertFalse(config.minio.secure)


class SequenceNormalizationTests(unittest.TestCase):
    def test_normalizes_protein_sequences_and_deduplicates(self):
        records = [
            SequenceRecord(accession="ENA001", source_name="ena", description="candidate", organism="Bacillus subtilis", sequence="mpept!de", sequence_length=8),
            SequenceRecord(accession="ENA002", source_name="ena", description="duplicate", organism="Bacillus subtilis", sequence="MPEPTXDE", sequence_length=8),
        ]

        normalized_records, report = normalize_records(records, SequenceNormalizationConfig(sequence_type="protein", deduplicate_sequences=True))

        self.assertEqual(1, len(normalized_records))
        self.assertEqual("MPEPTXDE", normalized_records[0].sequence)
        self.assertEqual("protein", normalized_records[0].metadata["sequence_type"])
        self.assertEqual(1, report.dropped_reasons["duplicate_sequence"])

    def test_filters_ambiguous_proteins(self):
        records = [SequenceRecord(accession="ENA003", source_name="ena", description="ambiguous", organism="Synthetic consortium", sequence="M?**?", sequence_length=5)]
        normalized_records, report = normalize_records(records, SequenceNormalizationConfig(sequence_type="protein", max_ambiguous_ratio=0.40))

        self.assertEqual([], normalized_records)
        self.assertEqual(1, report.dropped_reasons["too_ambiguous"])


class SequenceTokenizerTests(unittest.TestCase):
    def test_encode_decode_round_trip_matches_training_text(self):
        tokenizer = SequenceTokenizer.from_sequence_type("protein")
        text = "<|protein|>MPEPTXDE<|endoftext|>\n<|protein|>GLYSERQ<|endoftext|>\n"

        self.assertEqual(text, tokenizer.decode(tokenizer.encode(text)))

    def test_bpe_tokenizer_has_no_unk_and_raises_on_unknown_characters(self):
        tokenizer = SequenceTokenizer.from_text(
            "<|protein|>MPEPTIDE<|endoftext|>\n<|protein|>MPEPTXDE<|endoftext|>\n",
            sequence_type="protein",
            vocab_size=32,
        )

        self.assertNotIn("<|unk|>", tokenizer.special_tokens)
        self.assertGreater(len(tokenizer.bpe_merges), 0)
        with self.assertRaises(ValueError):
            tokenizer.encode("<|protein|>MPEPTZDE<|endoftext|>")

    def test_text_file_tokenizer_resume_continues_from_checkpoint(self):
        root = Path("tests/artifacts/tokenizer-resume")
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        train_path = root / "train.txt"
        train_text = (
            "<|protein|>MPEPTIDEMPEPTIDE<|endoftext|>\n"
            "<|protein|>GLYSERQGLYSERQ<|endoftext|>\n"
            "<|protein|>MPEPTXDEMPEPTXDE<|endoftext|>\n"
        )
        train_path.write_text(train_text, encoding="utf-8")

        def stop_after_first_merge(event):
            if event.get("event") == "tokenizer_checkpoint_saved" and int(event.get("completed_merges", 0)) >= 1:
                raise RuntimeError("stop after checkpoint")

        interrupted = SequenceTokenizer.from_sequence_type("protein")
        with self.assertRaisesRegex(RuntimeError, "stop after checkpoint"):
            interrupted.train_from_text_file(
                train_path,
                vocab_size=36,
                cache_dir=root,
                resume=True,
                progress_callback=stop_after_first_merge,
            )

        self.assertTrue(list(root.glob("sequence-tokenizer-resume-*.state.json")))
        resumed_events: list[dict[str, object]] = []
        resumed = SequenceTokenizer.from_sequence_type("protein")
        stats = resumed.train_from_text_file(
            train_path,
            vocab_size=36,
            cache_dir=root,
            resume=True,
            progress_callback=resumed_events.append,
        )

        self.assertEqual(3, stats.record_count)
        self.assertEqual(train_text, resumed.decode(resumed.encode(train_text)))
        self.assertTrue(any(event.get("event") == "tokenizer_resume_loaded" for event in resumed_events))
        self.assertFalse(list(root.glob("sequence-tokenizer-resume-*")))
        shutil.rmtree(root, ignore_errors=True)

    def test_text_file_tokenizer_parallel_workers_match_serial(self):
        root = Path("tests/artifacts/tokenizer-parallel")
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        train_path = root / "train.txt"
        train_text = "".join(
            f"<|protein|>{'MPEPTIDEGLYSERQ' if index % 2 else 'GLYSERQMPEPTIDE'}<|endoftext|>\n"
            for index in range(30)
        )
        train_path.write_text(train_text, encoding="utf-8")

        serial = SequenceTokenizer.from_sequence_type("protein")
        serial_stats = serial.train_from_text_file(
            train_path,
            vocab_size=40,
            cache_dir=root,
            worker_count=1,
        )

        parallel_events: list[dict[str, object]] = []
        parallel = SequenceTokenizer.from_sequence_type("protein")
        parallel_stats = parallel.train_from_text_file(
            train_path,
            vocab_size=40,
            cache_dir=root,
            worker_count=2,
            progress_callback=parallel_events.append,
        )

        self.assertEqual(serial_stats, parallel_stats)
        self.assertEqual(json.loads(serial.to_json()), json.loads(parallel.to_json()))
        self.assertTrue(
            any(event.get("workers") == 2 for event in parallel_events)
            or any(event.get("event") == "tokenizer_parallel_disabled" for event in parallel_events)
        )
        shutil.rmtree(root, ignore_errors=True)


class TrainingHubLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.data_root = Path("tests/artifacts/data-root")
        shutil.rmtree(self.data_root, ignore_errors=True)
        self.config = DataConfig(storage_mode="local", data_root=self.data_root, default_batch_size=5)
        self.manager = DatasetManager(LocalDatasetRepository(config=self.config))
        self.transport = FakeTransport(
            {
                ("https://www.ebi.ac.uk/ena/portal/api/search", sorted_params(fields="protein_id", format="tsv", limit=2, query="tax_tree(2)", result="coding")): "protein_id\nCAL001.1\nCAL002.1\n",
                (
                    "https://www.ebi.ac.uk/ena/portal/api/search",
                    sorted_params(fields="accession,description,scientific_name,base_count,protein_id,parent_accession,sequence_version", format="tsv", limit=2, query="tax_tree(2)", result="coding"),
                ): (
                    "accession\tdescription\tscientific_name\tbase_count\tprotein_id\tparent_accession\tsequence_version\n"
                    "CAL001\tNitrogen fixer candidate\tBacillus subtilis\t24\tCAL001.1\tOZ001.1\t1\n"
                    "CAL002\tPlastic degrader candidate\tPseudomonas putida\t24\tCAL002.1\tOZ001.1\t1\n"
                ),
                ("https://www.ebi.ac.uk/ena/browser/api/embl/CAL001,CAL002", sorted_params()): (
                    "ID   CAL001; SV 1; linear; genomic DNA; STD; PRO; 24 BP.\nPA   OZ001.1\nDE   Nitrogen fixer candidate\nOS   Bacillus subtilis\n"
                    "FT   CDS             OZ001.1:1..24\nFT                   /protein_id=\"CAL001.1\"\nFT                   /translation=\"MPEP\nFT                   TIDE\"\n//\n"
                    "ID   CAL002; SV 1; linear; genomic DNA; STD; PRO; 24 BP.\nPA   OZ001.1\nDE   Plastic degrader candidate\nOS   Pseudomonas putida\n"
                    "FT   CDS             OZ001.1:25..48\nFT                   /protein_id=\"CAL002.1\"\nFT                   /translation=\"MPEX\nFT                   TIDE\"\n//\n"
                ),
                (
                    "https://www.ebi.ac.uk/ena/portal/api/search",
                    sorted_params(fields="accession,description,scientific_name,base_count,protein_id,parent_accession,sequence_version", format="tsv", limit=1, query='protein_id="CAL001.1" OR accession="CAL001"', result="coding"),
                ): "accession\tdescription\tscientific_name\tbase_count\tprotein_id\tparent_accession\tsequence_version\nCAL001\tNitrogen fixer candidate\tBacillus subtilis\t24\tCAL001.1\tOZ001.1\t1\n",
                ("https://www.ebi.ac.uk/ena/browser/api/embl/CAL001", sorted_params()): "ID   CAL001; SV 1; linear; genomic DNA; STD; PRO; 24 BP.\nPA   OZ001.1\nDE   Nitrogen fixer candidate\nOS   Bacillus subtilis\nFT   CDS             OZ001.1:1..24\nFT                   /protein_id=\"CAL001.1\"\nFT                   /translation=\"MPEP\nFT                   TIDE\"\n//\n",
                (
                    "https://www.ebi.ac.uk/ena/portal/api/search",
                    sorted_params(fields="accession,description,scientific_name,base_count,protein_id,parent_accession,sequence_version", format="tsv", limit=1, query='protein_id="CAL002.1" OR accession="CAL002"', result="coding"),
                ): "accession\tdescription\tscientific_name\tbase_count\tprotein_id\tparent_accession\tsequence_version\nCAL002\tPlastic degrader candidate\tPseudomonas putida\t24\tCAL002.1\tOZ001.1\t1\n",
                ("https://www.ebi.ac.uk/ena/browser/api/embl/CAL002", sorted_params()): "ID   CAL002; SV 1; linear; genomic DNA; STD; PRO; 24 BP.\nPA   OZ001.1\nDE   Plastic degrader candidate\nOS   Pseudomonas putida\nFT   CDS             OZ001.1:25..48\nFT                   /protein_id=\"CAL002.1\"\nFT                   /translation=\"MPEX\nFT                   TIDE\"\n//\n",
            }
        )
        self.hub = MicrobialDataHub(sources=[EnaSequenceSource(transport=self.transport)], dataset_manager=self.manager, config=self.config)

    def tearDown(self):
        shutil.rmtree(self.data_root, ignore_errors=True)

    def test_collects_and_prepares_protein_training_data(self):
        artifact = self.hub.collect("ena", FetchRequest(dataset_name="Nitrogen Fixers", query="tax_tree(2)", limit=2), sequence_type="protein")
        self.assertEqual(2, artifact.record_count)
        self.assertIn("<|protein|>MPEPTIDE<|endoftext|>", Path(artifact.train_txt_path).read_text(encoding="utf-8"))
        self.assertEqual("protein", SequenceTokenizer.load_map(artifact.tokenizer_map_path).sequence_type)

        request = FetchRequest(dataset_name="Nitrogen Fixers", query="tax_tree(2)", limit=2, batch_size=1)
        session_dir = self.config.sessions_root / "ena" / "nitrogen-fixers"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "accessions.txt").write_text("CAL001.1\nCAL002.1\n", encoding="utf-8")
        (session_dir / "train.txt").write_text("<|protein|>MPEPTIDE<|endoftext|>\n", encoding="utf-8")
        (session_dir / "manifest.json").write_text(
            "{\n  \"source_name\": \"ena\",\n  \"dataset_name\": \"nitrogen-fixers\",\n  \"storage_mode\": \"local\",\n  \"sequence_type\": \"protein\",\n  \"vocab_size\": 256,\n  \"processed_count\": 1,\n  \"total_count\": 2,\n  \"record_count\": 1,\n  \"dropped_record_count\": 0,\n  \"dropped_reasons\": {},\n  \"is_complete\": false,\n  \"request\": {\"query\": \"tax_tree(2)\", \"limit\": 2, \"batch_size\": 1, \"extra_fields\": [], \"include_suppressed\": false},\n  \"normalization\": {\"sequence_type\": \"protein\", \"min_length\": 0, \"max_length\": null, \"invalid_base_policy\": \"replace_with_x\", \"max_ambiguous_ratio\": 1.0, \"deduplicate_sequences\": true}\n}\n",
            encoding="utf-8",
        )
        session = self.hub.prepare_training_data(source_name="ena", request=request, sequence_type="protein", normalization=SequenceNormalizationConfig(sequence_type="protein"))
        self.assertTrue(session.is_complete)
        self.assertEqual(2, session.record_count)
        self.assertIn("<|protein|>MPEXTIDE<|endoftext|>", Path(session.train_txt_path).read_text(encoding="utf-8"))

    def test_delete_moves_dataset_to_trash(self):
        self.hub.collect("ena", FetchRequest(dataset_name="Nitrogen Fixers", query="tax_tree(2)", limit=2), sequence_type="protein")
        delete_result = self.hub.delete_dataset("ena", "nitrogen-fixers", permanent=False)
        self.assertTrue(delete_result.deleted)
        self.assertEqual([], self.hub.list_datasets("ena"))


class DdbjIncrementalPreparationTests(unittest.TestCase):
    def setUp(self):
        self.data_root = Path("tests/artifacts/ddbj-data-root")
        shutil.rmtree(self.data_root, ignore_errors=True)
        self.config = DataConfig(storage_mode="local", data_root=self.data_root, default_batch_size=5)
        self.manager = DatasetManager(LocalDatasetRepository(config=self.config))
        self.transport = FakeTransport(
            {
                ("https://getentry.ddbj.nig.ac.jp/getentry", sorted_params(accession_number="AB000001,AB000002", database="aa", filetype="text", format="flatfile", limit=2)): "LOCUS       AB000001                8 aa    linear   BCT 01-JAN-2020\nDEFINITION  PETase alpha candidate.\nACCESSION   AB000001\nVERSION     AB000001.1\nSOURCE      Ideonella sakaiensis\n  ORGANISM  Ideonella sakaiensis\n//\nLOCUS       AB000002                7 aa    linear   BCT 01-JAN-2020\nDEFINITION  Nitrogen fixation candidate.\nACCESSION   AB000002\nVERSION     AB000002.1\nSOURCE      Azotobacter vinelandii\n  ORGANISM  Azotobacter vinelandii\n//\n",
                ("https://getentry.ddbj.nig.ac.jp/getentry", sorted_params(accession_number="AB000001,AB000002", database="aa", filetype="text", format="fasta", limit=2)): ">AB000001 PETase alpha candidate.\nMPEPTIDE\n>AB000002 Nitrogen fixation candidate.\nGLYSERQ\n",
                ("https://getentry.ddbj.nig.ac.jp/getentry", sorted_params(accession_number="AB000003", database="aa", filetype="text", format="flatfile", limit=1)): "LOCUS       AB000003                8 aa    linear   BCT 01-JAN-2020\nDEFINITION  Stress response candidate.\nACCESSION   AB000003\nVERSION     AB000003.1\nSOURCE      Bacillus subtilis\n  ORGANISM  Bacillus subtilis\n//\n",
                ("https://getentry.ddbj.nig.ac.jp/getentry", sorted_params(accession_number="AB000003", database="aa", filetype="text", format="fasta", limit=1)): ">AB000003 Stress response candidate.\nPEPTIDER\n",
                ("https://getentry.ddbj.nig.ac.jp/getentry", sorted_params(accession_number="AB000001", database="aa", filetype="text", format="flatfile", limit=1)): "LOCUS       AB000001                8 aa    linear   BCT 01-JAN-2020\nDEFINITION  PETase alpha candidate.\nACCESSION   AB000001\nVERSION     AB000001.1\nSOURCE      Ideonella sakaiensis\n  ORGANISM  Ideonella sakaiensis\n//\n",
                ("https://getentry.ddbj.nig.ac.jp/getentry", sorted_params(accession_number="AB000002", database="aa", filetype="text", format="flatfile", limit=1)): "LOCUS       AB000002                8 aa    linear   BCT 01-JAN-2021\nDEFINITION  Nitrogen fixation candidate updated.\nACCESSION   AB000002\nVERSION     AB000002.2\nSOURCE      Azotobacter vinelandii\n  ORGANISM  Azotobacter vinelandii\n//\n",
                ("https://getentry.ddbj.nig.ac.jp/getentry", sorted_params(accession_number="AB000002", database="aa", filetype="text", format="fasta", limit=1)): ">AB000002 Nitrogen fixation candidate updated.\nMPEPTIDE\n",
            }
        )
        self.hub = MicrobialDataHub(sources=[DdbjSequenceSource(transport=self.transport)], dataset_manager=self.manager, config=self.config)

    def tearDown(self):
        shutil.rmtree(self.data_root, ignore_errors=True)

    def test_incremental_rebuilds_stay_protein_only(self):
        normalization = SequenceNormalizationConfig(sequence_type="protein")
        first_session = self.hub.prepare_training_data("ddbj", FetchRequest(dataset_name="DDBJ Seeds", accessions=("AB000001", "AB000002"), limit=2, batch_size=2), sequence_type="protein", normalization=normalization)
        self.assertEqual(2, first_session.record_count)

        first_call_count = len(self.transport.calls)
        expanded_session = self.hub.prepare_training_data("ddbj", FetchRequest(dataset_name="DDBJ Seeds", accessions=("AB000001", "AB000002", "AB000003"), limit=3, batch_size=2), sequence_type="protein", normalization=normalization)
        self.assertEqual(3, expanded_session.record_count)
        self.assertIn(("https://getentry.ddbj.nig.ac.jp/getentry", sorted_params(accession_number="AB000003", database="aa", filetype="text", format="fasta", limit=1)), self.transport.calls[first_call_count:])

        deduped_session = self.hub.prepare_training_data("ddbj", FetchRequest(dataset_name="DDBJ Seeds", accessions=("AB000001", "AB000002"), limit=2, batch_size=1), sequence_type="protein", normalization=normalization)
        self.assertEqual(1, deduped_session.record_count)
        self.assertIn("<|protein|>MPEPTIDE<|endoftext|>", Path(deduped_session.train_txt_path).read_text(encoding="utf-8"))


class NcbiTrainingHubLifecycleTests(unittest.TestCase):
    def test_prepares_training_data_from_ncbi_query(self):
        data_root = Path("tests/artifacts/ncbi-data-root")
        shutil.rmtree(data_root, ignore_errors=True)
        config = DataConfig(storage_mode="local", data_root=data_root, default_batch_size=5)
        manager = DatasetManager(LocalDatasetRepository(config=config))
        transport = FakeTransport(
            {
                ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", sorted_params(db="protein", idtype="acc", retmax=2, retmode="json", retstart=0, term="txid2[Organism:exp]", tool="microbial-dna-compiler")): json.dumps({"esearchresult": {"count": "2", "idlist": ["NCBI001.1", "NCBI002.1"]}}),
                ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", sorted_params(db="protein", id="NCBI001.1", retmode="json", tool="microbial-dna-compiler")): json.dumps({"result": {"uids": ["101"], "101": {"caption": "NCBI001", "title": "Nitrogen fixer candidate", "slen": 8, "biomol": "protein", "moltype": "aa", "sourcedb": "genbank", "organism": "Bacillus subtilis", "accessionversion": "NCBI001.1"}}}),
                ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", sorted_params(db="protein", id="NCBI001.1", retmode="text", rettype="fasta", tool="microbial-dna-compiler")): ">NCBI001.1 Nitrogen fixer candidate\nMPEPTIDE\n",
                ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", sorted_params(db="protein", id="NCBI002.1", retmode="json", tool="microbial-dna-compiler")): json.dumps({"result": {"uids": ["102"], "102": {"caption": "NCBI002", "title": "Plastic degrader candidate", "slen": 8, "biomol": "protein", "moltype": "aa", "sourcedb": "refseq", "organism": "Pseudomonas putida", "accessionversion": "NCBI002.1"}}}),
                ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", sorted_params(db="protein", id="NCBI002.1", retmode="text", rettype="fasta", tool="microbial-dna-compiler")): ">NCBI002.1 Plastic degrader candidate\nMPEXTIDE\n",
            }
        )
        hub = MicrobialDataHub(sources=[NcbiSequenceSource(transport=transport, email="test@test.com")], dataset_manager=manager, config=config)

        session = hub.prepare_training_data(
            source_name="ncbi",
            request=FetchRequest(dataset_name="NCBI Nitrogen Fixers", query="txid2[Organism:exp]", limit=2, batch_size=1),
            sequence_type="protein",
            normalization=SequenceNormalizationConfig(sequence_type="protein"),
        )

        self.assertTrue(session.is_complete)
        self.assertEqual(2, session.record_count)
        self.assertIn("<|protein|>MPEXTIDE<|endoftext|>", Path(session.train_txt_path).read_text(encoding="utf-8"))
        shutil.rmtree(data_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
