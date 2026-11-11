param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here
try {
    & $Python main.py train
    if ($LASTEXITCODE -ne 0) { throw "training failed" }

    & $Python main.py test
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }

    & $Python main.py evaluate --seconds 20
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed" }

    & $Python main.py demo --mass 4 --seconds 8
    if ($LASTEXITCODE -ne 0) { throw "demo failed" }
}
finally {
    Pop-Location
}
