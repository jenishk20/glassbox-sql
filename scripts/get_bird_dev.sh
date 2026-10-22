#!/usr/bin/env bash
# Download BIRD dev (the original 2024-06-27 release) and check it is the exact file we used.
#
#   bash scripts/get_bird_dev.sh            # into data/bird/
#   bash scripts/get_bird_dev.sh /some/dir  # anywhere else (then set BIRD_DIR=/some/dir/dev_20240627)
#
# Needs: curl, unzip, python3. About 350 MB download, 1.4 GB unpacked.
set -euo pipefail

TARGET="${1:-data/bird}"
URL="https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
DEV_JSON_SHA256="630272f2b1c44d8cef2c3b246f623355cf0bbc1e832c81061df895530dfc2f06"

mkdir -p "$TARGET"
cd "$TARGET"

if [ ! -f dev.zip ]; then
  echo "downloading $URL (346 MB)..."
  curl -L --fail -o dev.zip.part "$URL"
  mv dev.zip.part dev.zip
fi

echo "unpacking..."
unzip -q -o dev.zip && rm -rf __MACOSX
cd dev_20240627
unzip -q -o dev_databases.zip && rm -rf __MACOSX dev_databases.zip

if [ ! -f dev_databases/california_schools/california_schools.sqlite ]; then
  echo "ERROR: unexpected layout, dev_databases/<db>/<db>.sqlite not found in $(pwd)" >&2
  exit 1
fi

actual=$(python3 -c "import hashlib; print(hashlib.sha256(open('dev.json','rb').read()).hexdigest())")
if [ "$actual" != "$DEV_JSON_SHA256" ]; then
  echo "ERROR: dev.json checksum $actual does not match the release we used ($DEV_JSON_SHA256)" >&2
  exit 1
fi

echo "BIRD dev ready in $(pwd): $(ls dev_databases | wc -l | tr -d ' ') databases, dev.json checksum OK"
