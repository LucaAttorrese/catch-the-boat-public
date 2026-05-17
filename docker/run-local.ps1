# Windows PowerShell wrapper — same behaviour as docker/run-local.sh.
#
# Examples:
#   .\docker\run-local.ps1 python evaluation/evaluate.py --scenario hard --headless
#   .\docker\run-local.ps1 python docker/benchmark.py
#   .\docker\run-local.ps1 pytest tests/
#
# Override the image tag with $env:CHALLENGE_IMAGE.

$ErrorActionPreference = "Stop"

$Image  = if ($env:CHALLENGE_IMAGE)  { $env:CHALLENGE_IMAGE }  else { "challenge-eval:base" }
$Cpus   = if ($env:CHALLENGE_CPUS)   { $env:CHALLENGE_CPUS }   else { "6" }
$Memory = if ($env:CHALLENGE_MEMORY) { $env:CHALLENGE_MEMORY } else { "8g" }

# Default tag is `challenge-eval:base` (built locally from
# docker/Dockerfile.eval). To run against the organizer-published
# image with the reference simulator baked in, pull and re-tag:
#   docker pull ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon
#   docker tag  ghcr.io/skyeusoftware/catch-the-boat:2026-hackathon challenge-eval:full
# Then either set `$env:CHALLENGE_IMAGE = "challenge-eval:full"` or
# invoke this script with CHALLENGE_IMAGE=challenge-eval:full prefixed.
# The published image is public — no login required.

# Build if the image isn't already there.
docker image inspect $Image 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Image '$Image' not found locally. Building from docker/Dockerfile.eval ..."
    docker build -f docker/Dockerfile.eval -t $Image .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# On Windows we cannot pass --user with a Linux UID, so files inside the
# container land as the container's default user. That's fine for dev.
docker run --rm -it `
    --cpus=$Cpus `
    --memory=$Memory `
    -e OPENBLAS_NUM_THREADS=$Cpus `
    -e OMP_NUM_THREADS=$Cpus `
    -e MKL_NUM_THREADS=$Cpus `
    -v "${PWD}:/workspace" `
    -w /workspace `
    $Image @args
