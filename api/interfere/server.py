import argparse
import os
from pathlib import Path
from typing import Any, Literal

from .config import (
    APISettings,
    CONFIG_PATH_ENV_VAR,
    ENVIRONMENT_ENV_VAR,
    generation_kwargs,
    load_config,
)
from .inference import InferenceAPI
from libs.protein_completion.masking import (
    is_standard_amino_acid_text,
    make_span_completion_example,
)


SPAN_COMPLETION_ROUTE = "/protein-span-completion/prompt"


def create_app(
    *,
    config_path: str | Path | None = None,
    environment: str | None = None,
):
    from fastapi import Body, FastAPI, HTTPException
    from pydantic import BaseModel, Field

    settings = load_config(config_path=config_path, environment=environment)
    api_cache: dict[str, InferenceAPI] = {}

    class GenerateRequest(BaseModel):
        prompt: str = ""
        max_new_tokens: int | None = Field(default=None, ge=0)
        temperature: float | None = Field(default=None, ge=0.0)
        top_k: int | None = Field(default=None, gt=0)
        seed: int | None = None
        stop_at_endoftext: bool | None = None
        ensure_protein_prompt: bool | None = None

    class SpanCompletionRequest(BaseModel):
        raw_input: str = Field(..., min_length=1)
        source: Literal["ncbi", "ena"] = "ncbi"
        limit: int = Field(default=1, gt=0)
        mask_policy: str = "random_span"
        mask_start: int = Field(default=0, ge=0)
        mask_length: int = Field(default=48, gt=0)
        left_flank_size: int = Field(default=64, ge=0)
        right_flank_size: int = Field(default=64, ge=0)

    app = FastAPI(
        title="MDNAC Protein API",
        version="0.1.0",
    )
    app.state.settings = settings

    def get_api() -> InferenceAPI:
        cache_key = f"{settings.model.path}|{settings.model.device}"
        if cache_key not in api_cache:
            api_cache[cache_key] = InferenceAPI.load(
                model_path=settings.model.path,
                device_name=settings.model.device,
            )
        return api_cache[cache_key]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "environment": settings.environment,
            "model_path": str(settings.model.path),
            "device": settings.model.device,
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        try:
            api = get_api()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "ready",
            "environment": settings.environment,
            "model_path": str(api.session.onnx_path),
            "providers": list(api.session.providers),
        }

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        kwargs = _merge_generation_request(settings, payload)
        try:
            result = get_api().generate_protein(request.prompt, **kwargs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    @app.post(SPAN_COMPLETION_ROUTE)
    def span_completion(request: SpanCompletionRequest = Body(...)) -> dict[str, str]:
        (
            FetchRequest,
            DataNotFoundError,
            SourceConfigurationError,
            NcbiSequenceSource,
            EnaSequenceSource,
        ) = _span_completion_data_dependencies()

        try:
            query = _build_source_query(request.raw_input, source_name=request.source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        transport = _new_http_transport()
        sequence_source = (
            NcbiSequenceSource(transport=transport)
            if request.source == "ncbi"
            else EnaSequenceSource(transport=transport)
        )
        fetch_request = FetchRequest(
            dataset_name="protein-span-completion",
            query=query,
            limit=request.limit,
            extra_fields=("gene", "product", "host", "keywords"),
        )
        try:
            records = sequence_source.fetch(fetch_request)
        except DataNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SourceConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"{request.source.upper()} source fetch failed: {exc}",
            ) from exc

        if not records:
            raise HTTPException(status_code=404, detail="No sequence records found.")

        mask_end = request.mask_start + request.mask_length
        try:
            record = _select_span_completion_record(records, mask_end=mask_end)
            source_row = _build_span_source_row(record, raw_input=request.raw_input)
            span_row = make_span_completion_example(
                source_row,
                source_index=0,
                mask_start=request.mask_start,
                mask_end=mask_end,
                mask_policy=request.mask_policy,
                left_flank_size=request.left_flank_size,
                right_flank_size=request.right_flank_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "instruction": span_row["instruction"],
            "input": span_row["input"],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the local MDNAC protein HTTP API.",
    )
    parser.add_argument("--config", default=None, help="Path to api/config.yaml.")
    parser.add_argument("--env", default=None, help="Environment name from config.yaml.")
    parser.add_argument("--host", default=None, help="Override configured host.")
    parser.add_argument("--port", type=int, default=None, help="Override configured port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    args = parser.parse_args()

    import uvicorn

    if args.config:
        os.environ[CONFIG_PATH_ENV_VAR] = str(Path(args.config).expanduser().resolve())
    if args.env:
        os.environ[ENVIRONMENT_ENV_VAR] = str(args.env)

    settings = load_config(config_path=args.config, environment=args.env)
    uvicorn.run(
        "interfere.server:create_app",
        factory=True,
        host=args.host or settings.server.host,
        port=args.port or settings.server.port,
        reload=bool(args.reload or settings.server.reload),
    )


def _merge_generation_request(
    settings: APISettings,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    kwargs = generation_kwargs(settings.generation)
    for key in tuple(kwargs):
        if request_payload.get(key) is not None:
            kwargs[key] = request_payload[key]
    return kwargs


def _build_source_query(raw_input: str, *, source_name: str) -> str:
    query = " ".join(raw_input.split())
    if not query:
        raise ValueError("raw_input must not be empty.")
    if source_name == "ncbi":
        return query
    return query


def _select_span_completion_record(
    records: list[Any],
    *,
    mask_end: int,
) -> Any:
    sequenced_records = [
        record for record in records if _compact_protein_sequence(record.sequence)
    ]
    if not sequenced_records:
        raise ValueError("No fetched sequence record contains a protein sequence.")

    long_enough_records = [
        record
        for record in sequenced_records
        if len(_compact_protein_sequence(record.sequence)) > mask_end
    ]
    if not long_enough_records:
        raise ValueError("No fetched sequence is long enough for the requested mask span.")

    standard_records = [
        record
        for record in long_enough_records
        if is_standard_amino_acid_text(_compact_protein_sequence(record.sequence))
    ]
    return (standard_records or long_enough_records)[0]


def _build_span_source_row(record: Any, *, raw_input: str) -> dict[str, Any]:
    return {
        "instruction": build_instruction_from_record(record, raw_input),
        "input": "",
        "output": record.sequence,
        "accession": record.accession,
        "metadata": record.metadata,
        "output_format": "single protein sequence",
    }


def build_instruction_from_record(record: Any, raw_input: str) -> str:
    del raw_input
    fields = [
        ("description", record.description),
        ("organism", record.organism),
        ("keywords", _metadata_value(record.metadata, "keywords")),
        ("gene", _metadata_value(record.metadata, "gene")),
        ("product", _metadata_value(record.metadata, "product")),
        ("host", _metadata_value(record.metadata, "host")),
    ]
    parts = ["task protein span completion", "labels protein sequence"]
    parts.extend(f"{name} {value}" for name, value in fields if value)
    return "; ".join(parts)


def _metadata_value(metadata: dict[str, str], field_name: str) -> str:
    value = str(metadata.get(field_name) or "").strip()
    return " ".join(value.split())


def _compact_protein_sequence(sequence: str) -> str:
    return "".join(str(sequence or "").split()).upper()


def _new_http_transport() -> Any:
    from libs.data.utilities.http import UrllibHttpTransport

    return UrllibHttpTransport()


def _span_completion_data_dependencies() -> tuple[Any, ...]:
    from libs.data.entities import FetchRequest
    from libs.data.sources.ena import EnaSequenceSource
    from libs.data.sources.ncbi import NcbiSequenceSource
    from libs.data.utilities.exceptions import DataNotFoundError, SourceConfigurationError

    return (
        FetchRequest,
        DataNotFoundError,
        SourceConfigurationError,
        NcbiSequenceSource,
        EnaSequenceSource,
    )


__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()
