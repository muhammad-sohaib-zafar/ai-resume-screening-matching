# AI Resume Screening & Matching Dashboard

## Run in VS Code

1. Open this folder in VS Code.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate it:

   **Windows PowerShell**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

The dashboard supports:
- Job description text input
- Multiple PDF resume uploads
- PDF text extraction using PyPDF2
- TF-IDF and cosine similarity scoring
- Ranked candidate leaderboard
- Skill match percentage
- Missing keyword detection
- Plotly radar comparison for Python, SQL, ML, and Analytics
- CSV export
