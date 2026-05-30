"""
mdnac - Microbial DNA Compiler

Clean-architecture package for microbial protein data pipelines,
model training, and inference.

Package structure:
    mdnac.domain         - Domain entities, value objects, exceptions
    mdnac.application    - Use cases and application services
    mdnac.infrastructure - Storage, HTTP, config adapters
    mdnac.ml             - ML models, training, tokenizer
    mdnac.interfaces     - CLI and notebook adapters
"""

__version__ = "0.2.0"
