$llamaQuantize = "llama-quantize.exe" 

$inputFile   = "model-f16.gguf"
$outputFile  = "model-TaIQ2_M.gguf"
$imatrixFile = "model.imatrix.gguf"
$baseType    = "iq2_m"
$threads     = "10"

$quantArgs = @()
$quantArgs += "--imatrix"
$quantArgs += $imatrixFile

Get-Content "tensor_types_per_layer.txt" | ForEach-Object {
    $line = $_.Trim()
    if ($line -and $line.Contains("=")) {
        $quantArgs += "--tensor-type"
        $quantArgs += $line
    }
}

$quantArgs += $inputFile
$quantArgs += $outputFile
$quantArgs += $baseType
$quantArgs += $threads


Write-Host "RUN TaQuants" -ForegroundColor Green
& $llamaQuantize @quantArgs
