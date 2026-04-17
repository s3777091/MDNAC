from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from libs.data.config import DATA_CONFIG, DataConfig
from libs.data.training.raw_pipeline.defaults import DEFAULT_KEYWORD_RULES, PROFILE_OUTPUT_DIR
from libs.data.training.raw_pipeline.helpers import (
    extract_feature_sequence,
    feature_text,
    infer_organism_from_headers,
    match_rules,
    slugify,
)
from libs.data.training.raw_pipeline.models import (
    ProfileKeywordRule,
    ProfileSequencePair,
    RawTensorDatasetArtifact,
)
from libs.data.training.raw_pipeline.persistence import persist_pairs
from libs.data.utilities.parsers import ParsedAnnotationFeature, parse_fasta, parse_genbank_records, parse_gff_features


class RawDataPipeline:
    def __init__(
        self,
        config: DataConfig | None = None,
        keyword_rules: Sequence[ProfileKeywordRule] | None = None,
    ) -> None:
        self._config = config or DATA_CONFIG
        self._keyword_rules = tuple(keyword_rules or DEFAULT_KEYWORD_RULES)

    def build_pairs_from_fasta_and_gff(
        self,
        fasta_path: Path | str,
        annotation_path: Path | str,
        organism: str | None = None,
        sequence_type: str = "dna",
        keyword_rules: Sequence[ProfileKeywordRule] | None = None,
    ) -> list[ProfileSequencePair]:
        fasta_entries = parse_fasta(Path(fasta_path).read_text(encoding="utf-8"))
        sequences_by_id = {entry.accession: entry.sequence for entry in fasta_entries}
        headers_by_id = {entry.accession: entry.header for entry in fasta_entries}
        features = parse_gff_features(Path(annotation_path).read_text(encoding="utf-8"))

        inferred_organism = organism or infer_organism_from_headers(headers_by_id.values())
        return self._build_pairs_from_features(
            features=features,
            sequences_by_id=sequences_by_id,
            organism_by_id={feature.sequence_id: inferred_organism for feature in features if inferred_organism}
            if inferred_organism
            else {},
            sequence_type=sequence_type,
            keyword_rules=keyword_rules,
        )

    def build_pairs_from_genbank(
        self,
        genbank_path: Path | str,
        sequence_type: str = "dna",
        keyword_rules: Sequence[ProfileKeywordRule] | None = None,
    ) -> list[ProfileSequencePair]:
        records = parse_genbank_records(Path(genbank_path).read_text(encoding="utf-8"))
        features: list[ParsedAnnotationFeature] = []
        sequences_by_id: dict[str, str] = {}
        organism_by_id: dict[str, str] = {}
        for record in records:
            sequences_by_id[record.accession] = record.sequence
            if record.organism:
                organism_by_id[record.accession] = record.organism
            features.extend(record.features)

        return self._build_pairs_from_features(
            features=features,
            sequences_by_id=sequences_by_id,
            organism_by_id=organism_by_id,
            sequence_type=sequence_type,
            keyword_rules=keyword_rules,
        )

    def prepare_from_fasta_and_gff(
        self,
        dataset_name: str,
        fasta_path: Path | str,
        annotation_path: Path | str,
        organism: str | None = None,
        output_dir: Path | str | None = None,
        sequence_type: str = "dna",
        kmer_size: int = 3,
        profile_vocab_size: int = 256,
        max_profile_length: int | None = None,
        max_sequence_length: int | None = None,
        keyword_rules: Sequence[ProfileKeywordRule] | None = None,
    ) -> RawTensorDatasetArtifact:
        pairs = self.build_pairs_from_fasta_and_gff(
            fasta_path=fasta_path,
            annotation_path=annotation_path,
            organism=organism,
            sequence_type=sequence_type,
            keyword_rules=keyword_rules,
        )
        return self._persist_pairs(
            dataset_name=dataset_name,
            pairs=pairs,
            output_dir=output_dir,
            sequence_type=sequence_type,
            kmer_size=kmer_size,
            profile_vocab_size=profile_vocab_size,
            max_profile_length=max_profile_length,
            max_sequence_length=max_sequence_length,
        )

    def prepare_from_genbank(
        self,
        dataset_name: str,
        genbank_path: Path | str,
        output_dir: Path | str | None = None,
        sequence_type: str = "dna",
        kmer_size: int = 3,
        profile_vocab_size: int = 256,
        max_profile_length: int | None = None,
        max_sequence_length: int | None = None,
        keyword_rules: Sequence[ProfileKeywordRule] | None = None,
    ) -> RawTensorDatasetArtifact:
        pairs = self.build_pairs_from_genbank(
            genbank_path=genbank_path,
            sequence_type=sequence_type,
            keyword_rules=keyword_rules,
        )
        return self._persist_pairs(
            dataset_name=dataset_name,
            pairs=pairs,
            output_dir=output_dir,
            sequence_type=sequence_type,
            kmer_size=kmer_size,
            profile_vocab_size=profile_vocab_size,
            max_profile_length=max_profile_length,
            max_sequence_length=max_sequence_length,
        )

    def _build_pairs_from_features(
        self,
        features: Iterable[ParsedAnnotationFeature],
        sequences_by_id: dict[str, str],
        organism_by_id: dict[str, str],
        sequence_type: str,
        keyword_rules: Sequence[ProfileKeywordRule] | None = None,
    ) -> list[ProfileSequencePair]:
        active_rules = tuple(keyword_rules or self._keyword_rules)
        seen: set[tuple[str, int, int, str]] = set()
        pairs: list[ProfileSequencePair] = []

        for feature in features:
            sequence = sequences_by_id.get(feature.sequence_id)
            if not sequence:
                continue

            annotation_text = feature_text(feature)
            if not annotation_text:
                continue

            matched_rules = match_rules(annotation_text, active_rules)
            if not matched_rules:
                continue

            sequence_fragment = extract_feature_sequence(feature, sequence, sequence_type)
            if not sequence_fragment:
                continue

            organism = organism_by_id.get(feature.sequence_id, "")
            for rule, matched_keywords in matched_rules:
                dedupe_key = (feature.sequence_id, feature.start, feature.end, rule.label)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                pairs.append(
                    ProfileSequencePair(
                        profile=rule.render_profile(organism=organism, feature_type=feature.feature_type),
                        sequence=sequence_fragment,
                        accession=feature.sequence_id,
                        organism=organism,
                        feature_type=feature.feature_type,
                        start=feature.start,
                        end=feature.end,
                        strand=feature.strand,
                        matched_label=rule.label,
                        matched_keywords=matched_keywords,
                        metadata={"annotation_text": annotation_text},
                    )
                )

        return pairs

    def _persist_pairs(
        self,
        dataset_name: str,
        pairs: Sequence[ProfileSequencePair],
        output_dir: Path | str | None,
        sequence_type: str,
        kmer_size: int,
        profile_vocab_size: int,
        max_profile_length: int | None,
        max_sequence_length: int | None,
    ) -> RawTensorDatasetArtifact:
        resolved_output_dir = (
            Path(output_dir)
            if output_dir is not None
            else self._config.data_root / PROFILE_OUTPUT_DIR / slugify(dataset_name)
        )
        return persist_pairs(
            dataset_name=dataset_name,
            pairs=pairs,
            output_dir=resolved_output_dir,
            sequence_type=sequence_type,
            kmer_size=kmer_size,
            profile_vocab_size=profile_vocab_size,
            max_profile_length=max_profile_length,
            max_sequence_length=max_sequence_length,
        )
