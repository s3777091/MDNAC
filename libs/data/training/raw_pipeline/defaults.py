from __future__ import annotations

from pathlib import Path

from libs.data.training.raw_pipeline.models import ProfileKeywordRule


PROFILE_OUTPUT_DIR = Path("model") / "datasets"


def _build_rules(definitions: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[ProfileKeywordRule, ...]:
    return tuple(ProfileKeywordRule(label=label, keywords=keywords) for label, keywords in definitions)


ABIOTIC_STRESS_RULES = _build_rules(
    (
        (
            "drought tolerant",
            (
                "drought tolerance",
                "drought tolerant",
                "response to water deprivation",
                "water deficit",
                "dehydration",
            ),
        ),
        (
            "salt tolerant",
            ("salt tolerance", "salt tolerant", "salinity", "salt stress", "response to salt stress", "ionic stress"),
        ),
        (
            "heat tolerant",
            ("heat stress", "heat shock", "thermotolerance", "thermal stress", "response to heat", "high temperature"),
        ),
        (
            "cold tolerant",
            (
                "cold stress",
                "cold tolerance",
                "cold acclimation",
                "freezing tolerance",
                "response to cold",
                "low temperature",
            ),
        ),
        (
            "oxidative stress response",
            (
                "oxidative stress",
                "response to oxidative stress",
                "cellular response to oxidative stress",
                "reactive oxygen species",
                "superoxide dismutase",
                "catalase",
                "peroxidase",
            ),
        ),
        (
            "osmotic stress response",
            (
                "osmotic stress",
                "response to osmotic stress",
                "cellular response to osmotic stress",
                "osmoregulation",
                "osmoprotectant",
                "compatible solute",
                "osmolarity",
            ),
        ),
    )
)

BIOTIC_AND_RESISTANCE_RULES = _build_rules(
    (
        (
            "disease resistance",
            (
                "disease resistance",
                "defense response",
                "immune response",
                "pathogen response",
                "systemic acquired resistance",
            ),
        ),
        (
            "antibiotic resistance",
            (
                "antibiotic resistance",
                "antimicrobial resistance",
                "beta-lactamase",
                "multidrug resistance",
                "drug resistance",
                "drug efflux",
                "aminoglycoside",
            ),
        ),
        (
            "metal resistance",
            ("metal resistance", "metal ion resistance", "heavy metal", "copper resistance", "zinc resistance", "cadmium resistance"),
        ),
        (
            "biofilm formation",
            (
                "biofilm",
                "biofilm formation",
                "adhesion protein",
                "quorum sensing",
                "surface attachment",
                "exopolysaccharide",
            ),
        ),
        (
            "CRISPR defense",
            ("crispr", "crispr-associated", "cas protein", "cas nuclease", "spacer acquisition", "interference complex"),
        ),
    )
)

METABOLISM_AND_PHYSIOLOGY_RULES = _build_rules(
    (
        (
            "photosynthesis",
            (
                "photosynthesis",
                "photosystem",
                "photosystem i",
                "photosystem ii",
                "photosynthetic electron transport",
                "light harvesting",
                "light-harvesting complex",
                "chlorophyll binding",
                "thylakoid",
                "rubisco",
            ),
        ),
        (
            "nitrogen fixation",
            ("nitrogen fixation", "nitrogenase", "nitrogenase reductase", "nitrogen fixation protein", "nifh", "nifd", "nifk"),
        ),
        (
            "carbon metabolism",
            (
                "carbon fixation",
                "carbon metabolic process",
                "calvin cycle",
                "glycolysis",
                "tca cycle",
                "tricarboxylic acid",
                "pentose phosphate",
            ),
        ),
        (
            "transport",
            (
                "transporter",
                "transmembrane transport",
                "abc transporter",
                "major facilitator superfamily",
                "ion channel",
                "efflux pump",
                "permease",
            ),
        ),
        (
            "signal transduction",
            (
                "signal transduction",
                "two-component system",
                "histidine kinase",
                "histidine-protein kinase",
                "response regulator",
                "sensor kinase",
                "phosphorelay",
            ),
        ),
    )
)

DNA_AND_RNA_RULES = _build_rules(
    (
        (
            "DNA replication",
            ("dna replication", "replication initiator", "replication fork", "dna polymerase", "primase", "replicative helicase"),
        ),
        (
            "DNA repair",
            (
                "dna repair",
                "dna damage response",
                "base excision repair",
                "nucleotide excision repair",
                "double-strand break repair",
                "mismatch repair",
                "recombinational repair",
                "photolyase",
            ),
        ),
        (
            "recombination",
            ("recombination", "homologous recombination", "reca", "recombinase", "site-specific recombination"),
        ),
        (
            "mobile element",
            ("transposase", "integrase", "site-specific recombinase", "transposable element", "insertion sequence"),
        ),
        (
            "transcription regulation",
            (
                "transcription factor",
                "transcription regulator",
                "transcriptional regulator",
                "dna-binding transcription factor",
                "sigma factor",
                "repressor",
                "activator",
            ),
        ),
        (
            "RNA processing",
            ("rna processing", "rrna processing", "trna processing", "rna helicase", "ribonuclease", "splicing factor", "mrna maturation"),
        ),
        (
            "RNA silencing",
            (
                "sirna",
                "mirna",
                "small rna",
                "gene silencing by rna",
                "post-transcriptional gene silencing",
                "rna interference",
                "rna-directed dna methylation",
                "argonaute",
                "dicer",
            ),
        ),
        (
            "translation",
            (
                "ribosomal protein",
                "translation initiation",
                "translation elongation",
                "translation termination",
                "translation factor",
                "aminoacyl-trna synthetase",
                "ribosome biogenesis",
            ),
        ),
    )
)


DEFAULT_KEYWORD_RULES = (
    *ABIOTIC_STRESS_RULES,
    *BIOTIC_AND_RESISTANCE_RULES,
    *METABOLISM_AND_PHYSIOLOGY_RULES,
    *DNA_AND_RNA_RULES,
)
