# Train Side-View Processor (Complete)

This project splits a side-view train video into per-coach clips, extracts representative frames, performs door detection (heuristic and optional YOLO), and generates HTML + PDF reports.

## Quick steps (Windows)

1. Open this folder in VS Code.
2. Create and activate a venv:
   - PowerShell (temporary bypass if needed):
     ```powershell
     python -m venv .venv
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     .venv\Scripts\Activate.ps1
     ```
   - Or use Command Prompt (no policy issues):
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Place your input video at `data/Raw_video/DHN-wagon/` (e.g., `DHN_side_view1.mp4`).
5. Edit `config/default.yaml` if needed (train_number, input_video, frames_per_coach).
6. Run (from project root):
   ```powershell
   python -m src.pipeline all --config config/default.yaml
   ```
7. Outputs:
   - Clips: `outputs/Processed_Video/<train_number>/`
   - Report: `reports/side_view/<train_number>/report.html` and `.pdf`

If you prefer running via VS Code tasks, use Terminal → Run Task → Run: All.

