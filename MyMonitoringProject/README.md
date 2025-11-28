# Prerequisites
Before running this project, ensure you have the following:

**1. OS:** Windows 10, Windows 11, or Windows Server.

**2. Docker Desktop:** Installed and currently running in the background.

**3. Permissions:** You must have Administrator privileges on the machine to install the exporter service.

**4. Internet Connection:** Required during the first run to download Docker images and the Windows Exporter installer.

# Installation & Usage Step-by-Step
We have streamlined the entire setup process into a single automated script.

### Step 1: Download the Project

Clone this repository or download the ZIP file and extract it to a folder on your computer.

*Example:* `C:\Users\YourUser\Documents\MyMonitoringProject`
### Step 2: Start Docker Desktop 
Ensure Docker Desktop is running. You should see the Docker whale icon in your system tray.

### Step 3: Run the Automated Setup Script
 This script will automatically download and install the necessary Windows service and then build and start all Docker containers.

1. Navigate to the project folder in File Explorer.

2. Locate the `START_PROJECT.bat` file.

3. Right-click the file and select `Run as Administrator`.

4. A command window will open showing the installation progress. Wait for it to display "SETUP COMPLETE - SYSTEMS ONLINE".

# Accessing the Dashboards
Once the setup script completes, your monitoring stack is active running in the background.

### Grafana (Dashboards)	`http://localhost:3001`	
*User:* `admin`
*Pass:* `admin`

### Prometheus (Metrics UI)	 `http://localhost:9090/targets`
