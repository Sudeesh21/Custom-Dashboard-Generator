# =========================================================
# AUTOMATED WINDOWS EXPORTER INSTALLER
# =========================================================
# This script checks for the windows_exporter service.
# If not found, it downloads the MSI and installs it silently on port 9182.

$exporterVersion = "0.31.3"
$url = "https://github.com/prometheus-community/windows_exporter/releases/download/v$exporterVersion/windows_exporter-$exporterVersion-amd64.msi"
$outputFile = "$env:TEMP\windows_exporter.msi"
$listenPort = "9182"

Write-Host "`n--- Checking Windows Exporter Status ---" -ForegroundColor Cyan

# Check if the service already exists
if (Get-Service "windows_exporter" -ErrorAction SilentlyContinue) {
    Write-Host "Success: 'windows_exporter' service is already installed." -ForegroundColor Green
    Write-Host "Ensuring it is running..."
    Start-Service "windows_exporter" -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Green
}
else {
    # It's not installed, so we download and install it.
    Write-Host "Service not found. Starting installation process..." -ForegroundColor Yellow
    Write-Host "(Internet connection required to download installer)"
    
    Try {
        Write-Host "Downloading Windows Exporter v$exporterVersion MSI..."
        # Download the file, requiring TLS 1.2 for GitHub security
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $outputFile -ErrorAction Stop

        Write-Host "Installing silently on port $listenPort..."
        # Run the MSI installer silently (/qn) and set the listen port
        # RECTIFIED: Added specific collectors to ensure consistency with dashboard
        $installArgs = "/i `"$outputFile`" ENABLED_COLLECTORS=cpu,cs,logical_disk,net,os,service,system LISTEN_PORT=$listenPort /qn"
        Start-Process msiexec.exe -ArgumentList $installArgs -Wait -Verb RunAs -ErrorAction Stop

        Write-Host "Installation complete." -ForegroundColor Green
        
        # Clean up the downloaded file to save space
        Remove-Item $outputFile -ErrorAction SilentlyContinue
    }
    Catch {
        Write-Host "`n[ERROR] INSTALLATION FAILED:" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host "Please ensure you have internet access and ran this script as Administrator."
        exit 1
    }
}

Write-Host "`nWindows Exporter ready." -ForegroundColor Cyan
Write-Host "------------------------------------------`n"