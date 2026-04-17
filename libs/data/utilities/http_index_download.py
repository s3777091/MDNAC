from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


_DEFAULT_USER_AGENT = "MicrobialDNACompiler/0.2"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    name: str
    url: str


@dataclass(frozen=True, slots=True)
class DownloadResult:
    entry: DirectoryEntry
    path: Path
    status: str
    expected_size: int | None = None


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)
                return


def ensure_directory_url(url: str) -> str:
    stripped = url.strip()
    if not stripped:
        raise ValueError("Directory URL must not be empty.")
    if stripped.endswith("/"):
        return stripped
    return stripped + "/"


def fetch_index_html(url: str, user_agent: str = _DEFAULT_USER_AGENT) -> str:
    request = Request(url, headers={"User-Agent": user_agent}, method="GET")
    with urlopen(request, timeout=60) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset)


def extract_directory_entries(index_html: str, directory_url: str) -> list[DirectoryEntry]:
    base_url = ensure_directory_url(directory_url)
    base_path = urlparse(base_url).path
    parser = _HrefParser()
    parser.feed(index_html)

    entries: list[DirectoryEntry] = []
    seen_names: set[str] = set()
    for href in parser.hrefs:
        if href in {"../", "./"} or href.startswith(("#", "?")):
            continue

        resolved_url = urljoin(base_url, href)
        parsed = urlparse(resolved_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if not parsed.path.startswith(base_path):
            continue

        relative_path = parsed.path[len(base_path) :]
        if not relative_path or "/" in relative_path.strip("/"):
            continue
        if relative_path.endswith("/"):
            continue

        file_name = Path(relative_path).name
        if not file_name or file_name in seen_names:
            continue

        seen_names.add(file_name)
        entries.append(DirectoryEntry(name=file_name, url=resolved_url))

    return sorted(entries, key=lambda entry: entry.name)


def filter_entries(
    entries: Iterable[DirectoryEntry],
    include_patterns: Iterable[str] = (),
    exclude_patterns: Iterable[str] = (),
) -> list[DirectoryEntry]:
    includes = tuple(pattern.strip() for pattern in include_patterns if pattern.strip())
    excludes = tuple(pattern.strip() for pattern in exclude_patterns if pattern.strip())

    filtered: list[DirectoryEntry] = []
    for entry in entries:
        if includes and not any(fnmatch(entry.name, pattern) for pattern in includes):
            continue
        if excludes and any(fnmatch(entry.name, pattern) for pattern in excludes):
            continue
        filtered.append(entry)
    return filtered


def default_output_dir(directory_url: str, root_dir: Path | str = Path("data/downloads")) -> Path:
    parsed = urlparse(ensure_directory_url(directory_url))
    refseq_release_output_dir = _refseq_release_output_dir(parsed)
    if refseq_release_output_dir is not None:
        return refseq_release_output_dir

    relative_parts = [parsed.netloc, *[part for part in parsed.path.split("/") if part]]
    return Path(root_dir, *relative_parts)


def _refseq_release_output_dir(parsed_url) -> Path | None:
    if parsed_url.netloc != "ftp.ncbi.nlm.nih.gov":
        return None

    path_parts = [part for part in parsed_url.path.split("/") if part]
    if len(path_parts) < 3:
        return None
    if path_parts[:2] != ["refseq", "release"]:
        return None

    release_group = path_parts[2]
    metadata_groups = {
        "announcements",
        "release-catalog",
        "release-error-notice",
        "release-notes",
        "release-statistics",
    }
    if release_group in metadata_groups:
        return None

    # Keep RefSeq protein downloads under the project's raw-data area.
    return Path("data/raw/refseq_bacteria_protein", release_group)


def _content_length(headers) -> int | None:
    raw_value = headers.get("Content-Length")
    if raw_value is None:
        return None
    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError):
        return None
    if parsed_value < 0:
        return None
    return parsed_value


def download_entry(
    entry: DirectoryEntry,
    output_dir: Path | str,
    force: bool = False,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> DownloadResult:
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / entry.name
    temp_path = destination_path.with_name(f"{destination_path.name}.part")
    destination_existed = destination_path.exists()

    request = Request(entry.url, headers={"User-Agent": user_agent}, method="GET")
    with urlopen(request, timeout=60) as response:
        expected_size = _content_length(response.headers)
        if destination_existed and not force:
            existing_size = destination_path.stat().st_size
            if expected_size is None or existing_size == expected_size:
                temp_path.unlink(missing_ok=True)
                return DownloadResult(
                    entry=entry,
                    path=destination_path,
                    status="skipped",
                    expected_size=expected_size,
                )

        bytes_written = 0
        temp_path.unlink(missing_ok=True)
        try:
            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_written += len(chunk)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    if expected_size is not None and bytes_written != expected_size:
        temp_path.unlink(missing_ok=True)
        raise OSError(
            f"Incomplete download for {entry.name}: expected {expected_size} bytes, got {bytes_written}."
        )

    temp_path.replace(destination_path)
    status = "replaced" if destination_existed else "downloaded"
    return DownloadResult(
        entry=entry,
        path=destination_path,
        status=status,
        expected_size=expected_size,
    )


def download_directory(
    directory_url: str,
    output_dir: Path | str | None = None,
    include_patterns: Iterable[str] = (),
    exclude_patterns: Iterable[str] = (),
    force: bool = False,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> tuple[Path, list[DownloadResult]]:
    resolved_url = ensure_directory_url(directory_url)
    html = fetch_index_html(resolved_url, user_agent=user_agent)
    entries = extract_directory_entries(html, resolved_url)
    filtered_entries = filter_entries(entries, include_patterns=include_patterns, exclude_patterns=exclude_patterns)

    target_dir = Path(output_dir) if output_dir is not None else default_output_dir(resolved_url)
    results: list[DownloadResult] = []
    for entry in filtered_entries:
        results.append(
            download_entry(
                entry,
                target_dir,
                force=force,
                user_agent=user_agent,
            )
        )

    return target_dir, results
