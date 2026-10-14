param(
    [string]$OutputDirectory = ""
)

$platformRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $platformRoot
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $platformRoot "data\manifests"
}
$dynamicRoot = Join-Path $workspaceRoot "journal_work\repo_sources\Novel_Cyber_Infrastructure_src\Novel-Cyber-Infrastructure-Enabling-AI-Driven-Analytics-and-Digital-Twins-for-Real-Time-3D-Printing-main\database\dynamic"
$visionManifestPath = Join-Path $OutputDirectory "vision_operational_manifest.csv"
$sourceManifestPath = Join-Path $OutputDirectory "source_file_manifest.csv"

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

function Get-TextSha256 {
    param([string]$Text)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        return ([System.BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

$imageFiles = Get-ChildItem -LiteralPath $dynamicRoot -Recurse -File -ErrorAction Stop |
    Where-Object { $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg", ".png") } |
    Sort-Object FullName

$visionRows = [System.Collections.Generic.List[object]]::new()
$sourceRows = [System.Collections.Generic.List[object]]::new()

foreach ($file in $imageFiles) {
    $relativeDynamic = [System.IO.Path]::GetRelativePath($dynamicRoot, $file.FullName)
    $parts = $relativeDynamic -split '[\\/]'
    if ($parts.Count -lt 3) { continue }
    $sessionId = $parts[0]
    $stream = $parts[1]
    $captureId = $file.BaseName -replace '_(raw|processed)$', ''
    $timestampToken = ""
    if ($captureId -match '^(\d{8}_\d{6}_\d{6})') { $timestampToken = $Matches[1] }
    $relativeWorkspace = [System.IO.Path]::GetRelativePath($workspaceRoot, $file.FullName).Replace("\", "/")
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $recordHash = Get-TextSha256 -Text $relativeWorkspace

    $visionRows.Add([pscustomobject][ordered]@{
        record_id = $recordHash.Substring(0, 20)
        dataset_id = "local_operational_camera"
        source_type = "recorded_operational"
        source_group_id = $sessionId
        capture_id = $captureId
        capture_timestamp_token = $timestampToken
        stream = $stream
        relative_path = $relativeWorkspace
        bytes = $file.Length
        sha256 = $hash
        label_status = "unlabeled"
        partition = "replay_only"
        confirmatory_eligible = 0
        exclusion_reason = "no_detection_ground_truth_and_only_two_source_sessions"
    })

    $sourceRows.Add([pscustomobject][ordered]@{
        source_id = "novel_cyber_infrastructure"
        category = "image"
        relative_path = $relativeWorkspace
        bytes = $file.Length
        sha256 = $hash
        modified_utc = $file.LastWriteTimeUtc.ToString("o")
    })
}

$visionRows | Export-Csv -LiteralPath $visionManifestPath -NoTypeInformation -Encoding utf8

$existingSourceRows = @()
if (Test-Path -LiteralPath $sourceManifestPath) {
    $existingSourceRows = Import-Csv -LiteralPath $sourceManifestPath | Where-Object {
        -not ($_.category -eq "image" -and $_.relative_path -like "*/database/dynamic/*")
    }
}
@($existingSourceRows) + @($sourceRows) |
    Sort-Object relative_path -Unique |
    Export-Csv -LiteralPath $sourceManifestPath -NoTypeInformation -Encoding utf8

$groupCounts = $visionRows | Group-Object source_group_id | ForEach-Object {
    [pscustomobject]@{ group = $_.Name; count = $_.Count }
}
$streamCounts = $visionRows | Group-Object stream | ForEach-Object {
    [pscustomobject]@{ stream = $_.Name; count = $_.Count }
}

[pscustomobject]@{
    record_count = $visionRows.Count
    groups = $groupCounts
    streams = $streamCounts
    output = $visionManifestPath
} | ConvertTo-Json -Depth 4
