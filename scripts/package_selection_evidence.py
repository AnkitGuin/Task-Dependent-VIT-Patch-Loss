"""Build a private, checksum-verifiable research evidence bundle; never uploads it."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(output):
    if output.exists():
        raise FileExistsError("Preserve existing bundles; choose a new output filename")
    roots = ["src", "tests", "scripts", "configs", "docs", "reports/selection", "paper"]
    roots += [
        f"runs/{name}"
        for name in [
            "selection_primary",
            "selection_controls",
            "selection_sensitivity",
            "selection_logme_profile",
            "selection_permutation",
            "selection_regularization",
            "selection_probe_grid",
        ]
    ]
    files = []
    excluded_suffixes = {".pyc", ".aux", ".log", ".bbl", ".blg", ".out", ".toc"}
    for root in roots:
        for item in Path(root).rglob("*"):
            if item.is_file() and "__pycache__" not in item.parts and item.suffix not in excluded_suffixes:
                files.append(item)
    for name in [
        ".gitignore",
        ".github/workflows/ci.yml",
        "README.md",
        "pyproject.toml",
        "requirements-lock.txt",
        "LICENSE",
        "data/selection_local/assets.json",
    ]:
        if Path(name).is_file():
            files.append(Path(name))
    files = sorted(set(files))
    manifest = {
        p.as_posix(): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in files
    }
    description = """PRIVATE RESEARCH EVIDENCE - NOT A PUBLICATION OR PUBLIC RELEASE

Read docs/refinement_results.md and docs/selection_reproducibility.md first.
The original primary protocol and post-primary follow-ups are distinct cohorts.
Source, protocols, selected heads/operators, validation curves, pretest choices,
predictions, generated reports and the unpublished manuscript are included.
Original CIFAR images, full pretrained models, caches, virtualenv and Git history
are excluded. Reacquire assets using the checksum-verifying preparation script.
This bundle has not been uploaded. Review licensing before redistribution.

MANIFEST.json contains the byte count and SHA256 of each original file. Verify
these after extraction before rerunning anything. The sibling .sha256 file
fingerprints the complete ZIP. ZIP CRC integrity is checked during packaging.
Software checks and this evidence do not establish novelty or journal acceptance.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("EVIDENCE_README.txt", description)
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        for path in files:
            archive.write(path, path.as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        for name, metadata in manifest.items():
            content = archive.read(name)
            if len(content) != metadata["bytes"] or hashlib.sha256(content).hexdigest() != metadata["sha256"]:
                raise ValueError(f"Bundle content verification failed: {name}")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{checksum}  {output.name}\n")
    result = {
        "archive": str(output),
        "files": len(files),
        "uncompressed_bytes": sum(v["bytes"] for v in manifest.values()),
        "archive_bytes": output.stat().st_size,
        "sha256": checksum,
        "crc_and_all_file_sha256_verified": True,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/selection_evidence_20261114.zip"))
    package(parser.parse_args().output)
