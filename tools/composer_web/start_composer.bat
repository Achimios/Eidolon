@echo off
rem Composer (HTML): starts the local server and opens the page in the browser.
cd /d "%~dp0..\.."
".venv\Scripts\python.exe" "tools\composer_web\serve.py" %*
