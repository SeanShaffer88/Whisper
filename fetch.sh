#!/bin/sh
# Download the pinned upstream sources into vendor/ and verify SHA-256.
# Sources are NEVER committed; vendor/ is gitignored.
set -eu
cd "$(dirname "$0")"
# shellcheck disable=SC1091
. ./pins.txt
mkdir -p vendor

sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

fetch() {
    url="$1"; file="$2"; want="$3"
    if [ ! -f "vendor/$file" ]; then
        echo "downloading $url"
        curl -fL "$url" -o "vendor/$file"
    fi
    got=$(sha_of "vendor/$file")
    if [ "$got" != "$want" ]; then
        echo "SHA-256 MISMATCH for $file" >&2
        echo "  pinned: $want" >&2
        echo "  actual: $got" >&2
        exit 1
    fi
    echo "verified $file ($got)"
}

fetch "$WHISPER_URL" "whisper-$WHISPER_VERSION.tar.gz" "$WHISPER_SHA256"
fetch "$TEST_MODEL_URL" "$TEST_MODEL_NAME" "$TEST_MODEL_SHA256"

rm -rf vendor/src-whisper
mkdir -p vendor/tmp-whisper
tar -xzf "vendor/whisper-$WHISPER_VERSION.tar.gz" -C vendor/tmp-whisper
mv "vendor/tmp-whisper/$WHISPER_DIR" vendor/src-whisper
rm -rf vendor/tmp-whisper
# Tar extraction preserves the release-date mtime, which cargo's
# build-script fingerprint can mistake for "unchanged" if a build ran
# while the tree was absent. Stamp the header so the probe re-runs.
touch vendor/src-whisper/include/whisper.h
echo "sources ready in vendor/src-whisper; test model at vendor/$TEST_MODEL_NAME"
