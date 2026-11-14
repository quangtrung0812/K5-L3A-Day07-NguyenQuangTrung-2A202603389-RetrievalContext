"""Download and verify the public 124-page WEF Global Risks Report 2024 PDF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import pymupdf

SOURCE_URL = "https://www3.weforum.org/docs/WEF_The_Global_Risks_Report_2024.pdf"
DEFAULT_PDF_PATH = Path("data/advanced/raw/WEF_Global_Risks_2024.pdf")
DEFAULT_META_PATH = Path("data/advanced/raw/source_metadata.json")


def download_wef_report(
    output_path: Path = DEFAULT_PDF_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    url: str = SOURCE_URL,
) -> dict[str, object]:
    """Download the WEF Global Risks Report 2024 PDF if missing and verify page count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            output_path.write_bytes(resp.read())

    data = output_path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    doc = pymupdf.open(output_path)
    page_count = len(doc)
    doc.close()
    if page_count < 100:
        raise RuntimeError(f"Expected >= 100 pages, got {page_count} in {output_path}")

    meta = {
        "title": "The Global Risks Report 2024 (19th Edition)",
        "publisher": "World Economic Forum",
        "published": "January 2024",
        "license_and_access": "Public WEF report (CC BY-NC-ND 4.0 / public download URL)",
        "source_url": url,
        "local_path": str(output_path).replace("\\", "/"),
        "pages": page_count,
        "size_bytes": len(data),
        "sha256": sha256,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WEF Global Risks Report 2024.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META_PATH)
    args = parser.parse_args()
    meta = download_wef_report(args.output, args.meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
