from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class ProfileKeywordRule:
    label: str
    keywords: tuple[str, ...]
    template: str = "{label} {entity} in {organism}"

    def render_profile(self, organism: str, feature_type: str) -> str:
        label = self.label[:1].upper() + self.label[1:]
        entity = _profile_entity(feature_type)
        if organism:
            return self.template.format(label=label, entity=entity, organism=organism)
        return f"{label} {entity}"


@dataclass(slots=True, frozen=True)
class ProfileSequencePair:
    profile: str
    sequence: str
    accession: str
    organism: str
    feature_type: str
    start: int
    end: int
    strand: str
    matched_label: str
    matched_keywords: tuple[str, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "sequence": self.sequence,
            "accession": self.accession,
            "organism": self.organism,
            "feature_type": self.feature_type,
            "start": self.start,
            "end": self.end,
            "strand": self.strand,
            "matched_label": self.matched_label,
            "matched_keywords": list(self.matched_keywords),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class RawTensorDatasetArtifact:
    dataset_name: str
    output_dir: str
    pair_count: int
    pairs_path: str
    tensor_dataset_path: str
    profile_tokenizer_path: str
    sequence_tokenizer_path: str
    manifest_path: str
    kmer_size: int
    profile_vocab_size: int
    sequence_vocab_size: int


def _profile_entity(feature_type: str) -> str:
    normalized = feature_type.strip().lower()
    if normalized in {"gene", "cds", "mrna", "transcript"}:
        return "gene"
    return normalized or "feature"
