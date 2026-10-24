
import os
import time
import urllib.request
import hashlib

URL = "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz"
OUT = os.path.join("data_cifar100", "cifar-100-python.tar.gz")
TOTAL = 169001437
CHUNK = 1024 * 1024
EXPECTED_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Restart the corrupted archive and reserve exactly the expected size.
with open(OUT, "w+b") as f:
    f.truncate(TOTAL)

    for start in range(0, TOTAL, CHUNK):
        end = min(start + CHUNK, TOTAL) - 1
        expected = end - start + 1

        for attempt in range(1, 9):
            try:
                req = urllib.request.Request(
                    URL,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "CIFAR100-research-downloader/1.0",
                    },
                )

                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"Expected HTTP 206, got {response.status}"
                        )

                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(
                        f"bytes {start}-{end}/"
                    ):
                        raise RuntimeError(
                            f"Unexpected Content-Range: {content_range}"
                        )

                    data = response.read()

                if len(data) != expected:
                    raise RuntimeError(
                        f"Expected {expected} bytes, received {len(data)}"
                    )

                f.seek(start)
                f.write(data)
                f.flush()
                break

            except Exception as exc:
                print(
                    f"Chunk {start:,}-{end:,}, "
                    f"attempt {attempt}/8 failed: {exc}"
                )
                if attempt == 8:
                    raise
                time.sleep(min(attempt * 2, 15))

        completed = end + 1
        print(
            f"Progress: {completed:,}/{TOTAL:,} bytes "
            f"({completed * 100 / TOTAL:.1f}%)"
        )

print("Checking archive size and MD5...")
actual_size = os.path.getsize(OUT)

md5 = hashlib.md5()
with open(OUT, "rb") as f:
    for block in iter(lambda: f.read(1024 * 1024), b""):
        md5.update(block)

actual_md5 = md5.hexdigest()
print("File size:   ", actual_size)
print("Expected size:", TOTAL)
print("Actual MD5:  ", actual_md5)
print("Expected MD5:", EXPECTED_MD5)

if actual_size != TOTAL or actual_md5 != EXPECTED_MD5:
    raise SystemExit("Verification failed. Do not extract the archive.")

print("SUCCESS: CIFAR-100 archive verified.")
