param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here
try {
    & $Python main.py --test
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }

    & $Python main.py --demo --output-dir results
    if ($LASTEXITCODE -ne 0) { throw "demo failed" }
}
finally {
    Pop-Location
}
