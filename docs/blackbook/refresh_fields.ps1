# Refresh the TOC, List of Figures and List of Tables of the built report and export a PDF.
# Uses Microsoft Word if installed, otherwise WPS Writer (both expose the same COM object model).
param(
    [string]$Docx = (Join-Path $PSScriptRoot "TruthLens_Final_Project_Report.docx")
)
$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::ChangeExtension($Docx, ".pdf")
$app = $null
foreach ($progId in @("Word.Application", "KWPS.Application")) {
    try { $app = New-Object -ComObject $progId; break } catch { }
}
if ($null -eq $app) { throw "Neither Word nor WPS Writer is available." }
$app.Visible = $false
try {
    $doc = $app.Documents.Open($Docx)
    # TOC fields: Contents, List of Figures (Figure Caption style), List of Tables (Table Caption style)
    for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) { $doc.TablesOfContents.Item($i).Update() }
    $doc.Fields.Update() | Out-Null
    for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) { $doc.TablesOfContents.Item($i).UpdatePageNumbers() }
    $doc.Save()
    $doc.SaveAs2($Pdf, 17)   # 17 = PDF
    $doc.Close($false)
    Write-Output "refreshed $Docx"
    Write-Output "exported  $Pdf"
} finally {
    $app.Quit()
}
