"""Publish an authorized revision release using credentials scoped to this GitHub repo.

Credentials are obtained from Git's configured helper, remain in process memory,
and are never printed or written to disk. Run --check before preparing a release.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = "AnkitGuin/Task-Dependent-VIT-Patch-Loss"
TAG = "v2026.11.15-v2"


def credentials(checkout):
    result = subprocess.run(["git", "credential", "fill"], cwd=checkout,
                            input="protocol=https\nhost=github.com\npath="+REPOSITORY+".git\n\n",
                            capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError("Git credential helper could not authenticate this repository")
    fields = dict(line.split("=",1) for line in result.stdout.splitlines() if "=" in line)
    if not fields.get("password"):
        raise RuntimeError("No repository credential was returned by Git")
    return fields["password"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout",type=Path,default=Path("../github-release"))
    parser.add_argument("--artifacts",type=Path,default=Path("artifacts/revision_release"))
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--prepare",action="store_true",help="Create a draft without uploading/publishing")
    parser.add_argument("--commit",help="Exact already-pushed commit for this release")
    args = parser.parse_args()
    token = credentials(args.checkout.resolve())
    def request(url,method="GET",payload=None,binary=None,authenticated=True):
        headers = {"Accept":"application/vnd.github+json","User-Agent":"patch-selection-revision-release",
                   "X-GitHub-Api-Version":"2022-11-28"}
        if authenticated:
            headers["Authorization"] = "Bearer "+token
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        if binary is not None:
            data = binary
            headers["Content-Type"] = "application/octet-stream"
        try:
            with urllib.request.urlopen(urllib.request.Request(url,data=data,headers=headers,method=method),timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as ex:
            if ex.code == 404:
                return None
            raise RuntimeError(f"GitHub request failed with HTTP {ex.code}") from None
    base = "https://api.github.com/repos/"+REPOSITORY
    repo = request(base)
    if not repo:
        raise RuntimeError("Repository is inaccessible")
    if repo.get("private"):
        raise RuntimeError("Repository is private; public availability cannot be asserted")
    if not repo.get("permissions",{}).get("push"):
        raise RuntimeError("This GitHub identity lacks repository write permission")
    if args.check:
        print(json.dumps(dict(repository=repo["full_name"],public=True,push_permission=True,default_branch=repo["default_branch"])))
        return
    if not args.commit:
        raise ValueError("Specify the exact pushed source commit with --commit")
    release = request(base+"/releases/tags/"+TAG)
    if release is None:
        all_releases = request(base+"/releases?per_page=100")
        release = next((r for r in all_releases if r["tag_name"]==TAG),None)
    if release is None:
        release = request(base+"/releases","POST",dict(tag_name=TAG,target_commitish=args.commit,
                          name="Frozen linear patch selection: journal revision v2",draft=True,prerelease=False,
                          body="Full-test, content-clean revision by Ankit Guin: CIFAR-10, CIFAR-100 and SVHN; 420 candidate heads; competitive LogME/H-score/probe baselines; fixed-policy, paired-uncertainty and resource audit. Source and evidence archives include per-file SHA256 manifests; upstream raw images/full weights are excluded. See the reproduction guide and limitations in the repository. This release is research evidence, not a claim of journal acceptance."))
    if args.prepare:
        print(json.dumps(dict(draft_release_created=release["draft"],release_id=release["id"],tag=TAG)))
        return
    files = sorted(args.artifacts.glob("*.zip"))+sorted(args.artifacts.glob("*.zip.sha256"))
    if len(files)!=4:
        raise ValueError("Expected the source/evidence ZIPs and their two external SHA256 files")
    uploaded = {r["name"]:r for r in request(base+f"/releases/{release['id']}/assets")}
    verified = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        asset = uploaded.get(path.name)
        if asset is None:
            url = release["upload_url"].split("{")[0]+"?"+urllib.parse.urlencode(dict(name=path.name))
            asset = request(url,"POST",binary=path.read_bytes())
        if asset["size"]!=path.stat().st_size:
            raise ValueError("Uploaded release size mismatch: "+path.name)
        if asset.get("digest") and asset["digest"]!="sha256:"+digest:
            raise ValueError("Uploaded release digest mismatch: "+path.name)
        verified.append(dict(name=path.name,bytes=asset["size"],sha256=digest,server_digest=asset.get("digest"),
                             url=asset["browser_download_url"]))
        print(f"VERIFIED UPLOAD {path.name}",flush=True)
    if release["draft"]:
        release = request(base+f"/releases/{release['id']}","PATCH",dict(draft=False))
    public = request(base+"/releases/tags/"+TAG,authenticated=False)
    if not public or public["draft"]:
        raise RuntimeError("Published release is not anonymously accessible")
    record = dict(repository=repo["html_url"],tag=TAG,source_commit=args.commit,release_url=public["html_url"],
                  anonymously_accessible=True,assets=verified)
    target = Path("reports/selection_revision/public_release.json")
    target.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()
