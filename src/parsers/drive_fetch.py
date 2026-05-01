"""Drive fetch helper — download running log files from Google Drive.

Handles OAuth user flow, token caching, and the native-Sheet-vs-xlsx distinction
(native Sheets are exported to xlsx on the fly; xlsx files download as-is).

ONE-TIME SETUP (~10 min):
  1. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
  2. Go to https://console.cloud.google.com, create a new project (or reuse one).
  3. APIs & Services → Enable APIs → search for and enable "Google Drive API".
  4. APIs & Services → Credentials → Create Credentials → OAuth client ID.
     - Application type: Desktop app
     - Name: anything (e.g. "running-log-local")
     - Download the JSON.
  5. Save it as ~/.config/running-log/credentials.json
  6. First run opens a browser for consent; token is cached afterward at
     ~/.config/running-log/token.json and refreshes automatically.

(If step 2-4 feels excessive and you'd rather install rclone and run `rclone config`,
see the alternative at the bottom of this docstring.)

USAGE:
  from drive_fetch import get_drive_service, download_file, find_log_by_year

  svc = get_drive_service()
  # Known file IDs
  download_file(svc, ADJUSTMENTS_ID, '/tmp/adjustments.xlsx')
  # Or look up by year
  current_id = find_log_by_year(svc, 2026, FOLDER_ID)
  download_file(svc, current_id, '/tmp/Running_Log_2026.xlsx')

ALTERNATIVE — rclone:
  Install rclone, run `rclone config` (interactive; opens browser). Then from Python:
    subprocess.run(['rclone', 'copy', 'gdrive:Running Log 2026', '/tmp/'], check=True)
  No GCP console required, but adds rclone as a binary dependency.
"""
import os
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.paths import DATA_DIR, OUTPUT_DIR


# Known Drive IDs for Max's running log data
FOLDER_ID = '0AEcLRNUY5jL_Uk9PVA'
ADJUSTMENTS_ID = '1EnfRO7iFG7KAO6QxrnI-wToRm1OOrCFQADC2W3zHN6w'

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CONFIG_DIR = os.path.expanduser('~/.config/running-log')
CREDENTIALS_PATH = os.path.join(CONFIG_DIR, 'credentials.json')
TOKEN_PATH = os.path.join(CONFIG_DIR, 'token.json')

XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
NATIVE_SHEET_MIME = 'application/vnd.google-apps.spreadsheet'


def _abort_with_reauth_instructions(reason: str) -> 'None':
    """Print a clean message + exit non-zero. Used when the cached token is
    unrecoverable (expired refresh-token, revoked grant, corrupted file).
    The message tells the user exactly how to recover; no stack trace."""
    sys.stderr.write(
        f"\nERROR: Drive auth failed — {reason}\n\n"
        f"Refresh the token by deleting the cache and re-running:\n"
        f"    rm {TOKEN_PATH}\n"
        f"    ./scripts/run_pipeline.sh\n\n"
        f"The first re-run will open a browser for Google OAuth consent\n"
        f"and write a fresh ~/.config/running-log/token.json.\n"
    )
    sys.exit(1)


def get_drive_service():
    """Authenticate and return a Drive API service object. Caches token between runs.

    Aborts cleanly (no traceback) on token failures: expired refresh-token,
    revoked grant, or unreadable cache file. The error message tells the
    user how to re-authenticate.
    """
    # Imports deferred so the rest of the pipeline doesn't require google-auth when unused
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
    from google_auth_oauthlib.flow import InstalledAppFlow

    os.makedirs(CONFIG_DIR, exist_ok=True)
    creds = None
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except (ValueError, KeyError) as e:
            _abort_with_reauth_instructions(f"cached token unreadable ({e})")
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as e:
                _abort_with_reauth_instructions(
                    f"refresh token expired or revoked ({e})")
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"No OAuth credentials at {CREDENTIALS_PATH}. "
                    "See this module's docstring for one-time setup steps."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as f:
            f.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)


def download_file(service, file_id, dest_path):
    """Download a Drive file to dest_path.

    Native Google Sheets are exported to xlsx. Uploaded xlsx files download as-is.
    """
    from googleapiclient.http import MediaIoBaseDownload

    meta = service.files().get(fileId=file_id, fields='mimeType,name').execute()
    if meta['mimeType'] == NATIVE_SHEET_MIME:
        request = service.files().export_media(fileId=file_id, mimeType=XLSX_MIME)
    else:
        request = service.files().get_media(fileId=file_id)

    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
    with io.FileIO(dest_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return dest_path


def find_log_by_year(service, year, folder_id=FOLDER_ID):
    """Find the log file for a given year by title match within the folder.

    Tries 'Running Log YYYY' (the canonical name) first, then fuzzy match.
    Returns the file ID or raises FileNotFoundError.
    """
    exact_q = (f"name = 'Running Log {year}' and '{folder_id}' in parents "
               "and trashed = false")
    results = service.files().list(q=exact_q, fields='files(id, name, mimeType)',
                                   pageSize=10).execute()
    files = results.get('files', [])
    if not files:
        fuzzy_q = (f"name contains 'Running Log {year}' and '{folder_id}' in parents "
                   "and trashed = false")
        results = service.files().list(q=fuzzy_q, fields='files(id, name, mimeType)',
                                       pageSize=10).execute()
        files = results.get('files', [])
    if not files:
        raise FileNotFoundError(f"No log file found for year {year} in folder {folder_id}")
    # Prefer native Sheet over xlsx duplicates
    files.sort(key=lambda f: (f['mimeType'] != NATIVE_SHEET_MIME, f['name']))
    return files[0]['id']


def fetch_all_historical_logs(service, years, dest_dir, folder_id=FOLDER_ID):
    """Download every 'Running Log YYYY' log for the given years into dest_dir.

    Returns a dict {year: local_path}. Files are named Running_Log_YYYY.xlsx
    so freeze_historical.py's path resolver picks them up.
    """
    os.makedirs(dest_dir, exist_ok=True)
    paths = {}
    for year in years:
        file_id = find_log_by_year(service, year, folder_id)
        dest = os.path.join(dest_dir, f"Running_Log_{year}.xlsx")
        download_file(service, file_id, dest)
        paths[year] = dest
    return paths


def build_snapshot(service, current_year, out_path, folder_id=FOLDER_ID,
                   keep_xlsx=False, work_dir=None):
    """Fetch the current-year log + adjustments from Drive and write a
    snapshot CSV at `out_path`. This is the one-shot refresh for the
    hot-path input build_dataset consumes.

    If `keep_xlsx` is True, the downloaded xlsx files are kept in `work_dir`
    (default: tempfile.mkdtemp). Otherwise they're discarded after the
    snapshot is written.
    """
    import tempfile
    import shutil
    import snapshot

    tmp = work_dir or tempfile.mkdtemp(prefix="running-snapshot-")
    try:
        log_id = find_log_by_year(service, current_year, folder_id)
        log_xlsx = os.path.join(tmp, f"Running_Log_{current_year}.xlsx")
        download_file(service, log_id, log_xlsx)

        adj_xlsx = os.path.join(tmp, "adjustments.xlsx")
        download_file(service, ADJUSTMENTS_ID, adj_xlsx)

        snapshot.snapshot_from_xlsx(log_xlsx, current_year, adj_xlsx, out_path)
    finally:
        if not keep_xlsx and (work_dir is None):
            shutil.rmtree(tmp, ignore_errors=True)
    return out_path


# ---------- CLI ----------

def _cmd_snapshot(args):
    service = get_drive_service()
    build_snapshot(service, args.year, args.out,
                   keep_xlsx=args.keep_xlsx, work_dir=args.work_dir)


def _cmd_fetch_historical(args):
    service = get_drive_service()
    years = range(args.start_year, args.end_year + 1)
    paths = fetch_all_historical_logs(service, years, args.out_dir)
    for yr, p in sorted(paths.items()):
        print(f"  {yr}: {p}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Drive fetch helpers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot",
                       help="Fetch current-year log + adjustments, emit snapshot CSV")
    s.add_argument("--year", type=int, required=True,
                   help="Current year to fetch (e.g. 2026)")
    s.add_argument("--out", default=str(DATA_DIR / "drive_snapshot.csv"),
                   help=f"Output snapshot CSV path (default: {DATA_DIR / 'drive_snapshot.csv'})")
    s.add_argument("--keep-xlsx", action="store_true",
                   help="Keep the downloaded xlsx files for inspection")
    s.add_argument("--work-dir",
                   help="Directory to download xlsx into (default: tempdir)")
    s.set_defaults(func=_cmd_snapshot)

    h = sub.add_parser("fetch-historical",
                       help="Download per-year Running Log xlsx files for freeze")
    h.add_argument("--start-year", type=int, default=2016)
    h.add_argument("--end-year", type=int, required=True,
                   help="Last year to fetch (inclusive)")
    h.add_argument("--out-dir", default=str(OUTPUT_DIR / "logs"),
                   help=f"Directory to download xlsx into (default: {OUTPUT_DIR / 'logs'})")
    h.set_defaults(func=_cmd_fetch_historical)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
