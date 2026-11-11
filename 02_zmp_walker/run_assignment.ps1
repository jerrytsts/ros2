param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here
try {
    & $Python main.py test
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }

    & $Python main.py run
    if ($LASTEXITCODE -ne 0) { throw "report failed" }
}
finally {
    Pop-Location
}
