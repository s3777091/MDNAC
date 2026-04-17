from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import URLError

from libs.data.utilities.http_index_download import (
    default_output_dir,
    download_directory,
    ensure_directory_url,
    extract_directory_entries,
    fetch_index_html,
    filter_entries,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download all files listed in a simple HTTP directory index."
    )
    parser.add_argument("url", help="Directory URL, for example https://ftp.ncbi.nlm.nih.gov/refseq/release/vertebrate_other/")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory. Default: RefSeq release URLs go to data/raw/refseq_bacteria_protein/<group>/, otherwise data/downloads/<host>/<path>/",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help='Glob pattern to keep. Repeatable. Example: --include "*.protein.faa.gz"',
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help='Glob pattern to skip. Repeatable. Example: --exclude "*.gpff.gz"',
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List matching files without downloading them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    resolved_url = ensure_directory_url(args.url)
    target_dir = args.output_dir or default_output_dir(resolved_url)

    if args.list_only:
        try:
            html = fetch_index_html(resolved_url)
        except URLError as exc:
            print(f"[error] failed to reach {resolved_url}: {exc.reason}")
            return 2
        entries = extract_directory_entries(html, resolved_url)
        filtered_entries = filter_entries(entries, include_patterns=args.include, exclude_patterns=args.exclude)
        print(f"[index] {resolved_url}")
        print(f"[files] {len(filtered_entries)}")
        for entry in filtered_entries:
            print(entry.name)
        print(f"[target] {target_dir}")
        return 0

    try:
        output_dir, download_results = download_directory(
            resolved_url,
            output_dir=target_dir,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
            force=args.force,
        )
    except URLError as exc:
        print(f"[error] failed to reach {resolved_url}: {exc.reason}")
        return 2

    status_counts = {"downloaded": 0, "replaced": 0, "skipped": 0}
    for result in download_results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    print(f"[source] {resolved_url}")
    print(f"[target] {output_dir}")
    print(f"[matched] {len(download_results)} files")
    print(f"[downloaded] {status_counts['downloaded']}")
    print(f"[replaced] {status_counts['replaced']}")
    print(f"[skipped] {status_counts['skipped']}")
    for result in download_results:
        print(f"{result.status}\t{result.entry.name}")
    if download_results and status_counts["downloaded"] == 0 and status_counts["replaced"] == 0:
        print("[warning] all matching files are already present on disk; nothing new was downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
