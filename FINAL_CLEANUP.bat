@echo off
REM Final Cleanup - Remove documentation files, keep only essentials

echo Removing temporary documentation files...

del /F /Q DEEP_CLEANUP.bat 2>nul
del /F /Q PERFORMANCE_IMPROVEMENT_GUIDE.md 2>nul
del /F /Q REPO_STATUS.md 2>nul
del /F /Q airgap\4-pipeline\train_improved.sh 2>nul

echo.
echo ✅ Cleanup complete!
echo.
echo Final structure:
echo   README.md - Main documentation
echo   START_HERE.md - Quick start guide
echo   VISUAL_GUIDE.md - Visual reference
echo   QUICK_REF.txt - Quick commands
echo   gcp/ - GCP deployment
echo   airgap/ - Air-gap deployment
echo   + Core framework files
echo.
pause
