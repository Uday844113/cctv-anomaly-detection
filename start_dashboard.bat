@echo off
cd /d "F:\EMS(CNV)\cctv-anomaly-detection"
"F:\EMS(CNV)\venv\Scripts\python.exe" -m streamlit run dashboard/app.py --server.port 8502 --server.address 127.0.0.1