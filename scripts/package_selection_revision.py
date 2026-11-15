"""Create versioned, fully verified local source and research-evidence releases."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from prepare_revision_repository import README as PUBLIC_README

VERSION = "2026.11.15-v2"
EXCLUDED = {".pyc", ".aux", ".log", ".bbl", ".blg", ".out", ".toc", ".synctex.gz"}


def files_under(roots):
    files = []
    for root in roots:
        path = Path(root)
        for f in path.rglob("*"):
            if (f.is_file() and "__pycache__" not in f.parts and f.suffix not in EXCLUDED
                    and not any("warmup" in x or x in {"pdf_validation", "archive_prerevision_20261115", "archive_preselection_20261114"} for x in f.parts)
                    and (f.suffix != ".npy" or "runs" in f.parts)):
                files.append(f)
    return files


def create_archive(output, files, description, overrides=None):
    if output.exists():
        raise FileExistsError(f"Preserve versioned releases: {output}")
    overrides = overrides or {}
    manifest = {}
    for f in sorted(set(files)):
        content = overrides.get(f.as_posix(), f.read_bytes())
        manifest[f.as_posix()] = dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        archive.writestr("RELEASE_README.txt",description)
        archive.writestr("RELEASE_VERSION.txt",VERSION+"\n")
        archive.writestr("MANIFEST.json",json.dumps(manifest,indent=2)+"\n")
        for name in manifest:
            if name in overrides:
                archive.writestr(name,overrides[name])
            else:
                archive.write(name,name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, record in manifest.items():
            content = archive.read(name)
            assert len(content)==record["bytes"]
            assert hashlib.sha256(content).hexdigest()==record["sha256"]
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(digest+"  "+output.name+"\n")
    record = dict(version=VERSION,archive=output.as_posix(),files=len(manifest),bytes=output.stat().st_size,
                  sha256=digest,all_archived_files_sha256_verified=True,zip_crc_verified=True,uploaded=False)
    output.with_suffix(".zip.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("artifacts/revision_release"))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    common = files_under(["src","tests","scripts","configs","docs","paper","reports/selection","reports/selection_revision",".github","licenses"])
    common += [Path(p) for p in ["README.md","pyproject.toml","requirements-lock.txt","requirements-revision-lock.txt","LICENSE","THIRD_PARTY_NOTICES.md","CITATION.cff",".gitignore",".gitattributes","Makefile",
                                 "data/selection_local/assets.json","data/selection_local/svhn_assets.json"] if Path(p).is_file()]
    description = """Frozen Linear Patch Selection: revision release, author Ankit Guin

Start with README.md and docs/selection_revision_reproducibility.md.
This version preserves final, preliminary and exploratory evidence boundaries.
The paper's main results use only runs/selection_revision_final.
The source archive contains code, configurations, tests, documentation, all
generated manuscript tables/figures and machine-readable summary reports.
The companion evidence archive adds all final and supporting historical records,
operators, selected heads, validation paths, frozen choices and predictions.
Both archives retain the same relative paths and can be extracted together.

No raw dataset images, full downloaded pretrained weights, credentials, Git
history, virtual environment or aborted/warm-up revision cohort is included.
Use explicit preparation scripts to obtain checksum-verified upstream assets.
MANIFEST.json fingerprints every original file; .zip.sha256 fingerprints the ZIP.
Both ZIP CRC and every extracted-content SHA256 were checked during packaging.

Versioned public location:
https://github.com/AnkitGuin/Task-Dependent-VIT-Patch-Loss/releases/tag/v2026.11.15-v2
No archival DOI is claimed. Author/journal metadata must be finalized for submission.
Measured results do not guarantee journal acceptance or generalize to all domains.
"""
    source = create_archive(args.output/f"patch-selection-source-{VERSION}.zip",common,description,
                            overrides={"README.md": PUBLIC_README.encode("utf-8")})
    cohorts = ["selection_revision_final","selection_primary","selection_controls","selection_sensitivity",
               "selection_logme_profile","selection_permutation","selection_regularization","selection_probe_grid"]
    evidence_files = files_under([f"runs/{c}" for c in cohorts]+["licenses"])+[Path("THIRD_PARTY_NOTICES.md")]
    evidence = create_archive(args.output/f"patch-selection-evidence-{VERSION}.zip",evidence_files,description)
    print(json.dumps(dict(source=source,evidence=evidence),indent=2))


if __name__ == "__main__":
    main()
