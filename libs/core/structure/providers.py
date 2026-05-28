from __future__ import annotations

from .types import ExternalStructureProviderSpec


def recommended_structure_providers() -> tuple[ExternalStructureProviderSpec, ...]:
    return (
        ExternalStructureProviderSpec(
            name="AlphaFold 3",
            provider_type="closed_or_restricted_server/model",
            recommended_role="highest-accuracy final validation for structure and complexes",
            strengths=(
                "strong structure prediction across proteins and molecular complexes",
                "best used as final verifier when license and access allow it",
            ),
            limitations=(
                "not ideal as a default local dependency",
                "licensing and access constraints may limit training-pipeline integration",
            ),
            install_hint="Use the official DeepMind distribution or server path allowed by your license.",
            license_note="Check current AlphaFold 3 terms before commercial or automated use.",
        ),
        ExternalStructureProviderSpec(
            name="Boltz-2",
            provider_type="open_local_model",
            recommended_role="local structure and affinity scoring for generated candidates",
            strengths=(
                "practical local candidate ranking",
                "supports biomolecular complex and affinity-oriented workflows",
            ),
            limitations=(
                "GPU and model-weight management should stay outside the core package",
                "treat scores as filters, not proof of biological activity",
            ),
            install_hint="Install Boltz in a separate environment and expose it through a provider adapter.",
        ),
        ExternalStructureProviderSpec(
            name="Protenix",
            provider_type="open_local_model",
            recommended_role="AlphaFold-3-style local structure validation",
            strengths=(
                "useful when a local AF3-style pipeline is required",
                "good fit for batch validation of generated proteins",
            ),
            limitations=(
                "large model/runtime footprint",
                "quality depends on templates/MSA/features and chosen checkpoint",
            ),
            install_hint="Install Protenix separately and write a thin adapter returning StructurePrediction.",
        ),
        ExternalStructureProviderSpec(
            name="Chai-1",
            provider_type="open_local_model",
            recommended_role="local multimodal structure prediction fallback or ensemble member",
            strengths=(
                "useful independent model for ensemble disagreement checks",
                "practical for protein and complex prediction workflows",
            ),
            limitations=(
                "large dependency surface",
                "should not be imported in core training code",
            ),
            install_hint="Run Chai-1 as an external command or service provider.",
        ),
        ExternalStructureProviderSpec(
            name="ESMFold",
            provider_type="sequence_only_local_model",
            recommended_role="fast sequence-only foldability screen",
            strengths=(
                "does not require MSA",
                "good early filter when throughput matters",
            ),
            limitations=(
                "usually weaker than modern AF3-style tools for final validation",
                "less direct for ligand/complex/affinity questions",
            ),
            install_hint="Use as a fast first-pass structure confidence provider.",
        ),
        ExternalStructureProviderSpec(
            name="ProstT5",
            provider_type="sequence_structure_language_model",
            recommended_role="AA-to-3Di structural-token channel and lightweight structure plausibility scorer",
            strengths=(
                "bridges amino-acid tokens and Foldseek 3Di structure tokens",
                "fits auxiliary structure-token training and candidate scoring",
            ),
            limitations=(
                "3Di plausibility is not a full 3D fold validation",
                "should complement, not replace, final structure prediction",
            ),
            install_hint="Use transformers with Rostlab/ProstT5 in an optional adapter.",
        ),
    )
