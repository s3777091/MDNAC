# ── Cell: List all parts & select which ones to process ───────────────────────
# Paste this BEFORE the annotate_s3_instruction_jsonl_3di() call in the notebook.

from libs.data.training.streaming import list_minio_text_parts

# List all available parts
parts = list_minio_text_parts(
    prefix_uri=SOURCE_PREFIX_URI,
    s3_client=s3_client,
    suffixes=(".jsonl",),
)

print(f"Found {len(parts)} parts:\n")
for i, part in enumerate(parts):
    size_mb = (part.size or 0) / (1024 * 1024)
    print(f"  [{i}] {part.uri}  ({size_mb:.1f} MB)")

# ─────────────────────────────────────────────────────────────────────────────
# Pick parts to process.  Examples:
#   SELECTED = [2, 3, 5]       → specific indices
#   SELECTED = range(3, 10)    → parts 3–9
#   SELECTED = range(len(parts))  → all parts
# ─────────────────────────────────────────────────────────────────────────────
SELECTED = [0]  # ← change this to the indices you want

selected_part_uris = [parts[i].uri for i in SELECTED]
print(f"\n✅ Selected {len(selected_part_uris)} part(s):")
for uri in selected_part_uris:
    print(f"  • {uri}")


# ── Cell: Run annotation on selected parts only ──────────────────────────────
# Replace the old prefix_uri call with this:

summary = annotate_s3_instruction_jsonl_3di(
    provider=provider,
    part_uris=selected_part_uris,        # ← use part_uris instead of prefix_uri
    output_prefix_uri=OUTPUT_PREFIX_URI,
    s3_client=s3_client,
    config=config,
    cache_dir=CACHE_DIR,
    cache_path=CACHE_DB,
    batch_size=BATCH_SIZE,
    overwrite=OVERWRITE,
    skip_existing=True,
    upload_manifest=True,
    progress_callback=print,
)

summary.to_dict()
