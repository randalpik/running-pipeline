/**
 * Apps Script bound to the current-year Running Log workbook.
 * Watches edits to the Workout column and triggers the build-and-deploy
 * GitHub Actions workflow via repository_dispatch.
 *
 * Behavior:
 *   - onSheetEdit fires on every cell change. If the edit is a single-cell
 *     edit on the Workout column of any data sheet, queue it under the row
 *     in ScriptProperties with a current timestamp. Subsequent edits to the
 *     same row reset the timestamp (debounce).
 *   - processPending runs every minute. For each queued entry whose last
 *     edit is more than SETTLE_MS ago, re-read the cell value. If it's
 *     non-empty, dispatch a regular pipeline run. If the value contains
 *     "race@" (case-insensitive), additionally dispatch a fit run.
 *
 * Setup (do once after copying the workbook to a new year):
 *   1. Extensions → Apps Script (the script comes with the sheet copy).
 *   2. Project Settings → Script Properties → set:
 *        GITHUB_PAT       fine-grained PAT, "Actions: write" on the repo.
 *                         Reuse the same PAT used by the admin "Run
 *                         pipeline" button (lives in Netlify env as
 *                         GITHUB_DISPATCH_TOKEN).
 *        GITHUB_REPO      e.g. "randalpik/running-pipeline"
 *        WORKOUT_HEADER   optional, defaults to "Workout"
 *   3. In the Apps Script editor, choose the function "setupTriggers"
 *      from the dropdown and click Run. Grant OAuth consent when prompted.
 *      That installs the onEdit + 1-min cron triggers.
 *
 * Reusing the PAT across Netlify and Apps Script is fine. PAT is read-only
 * in the sense that the only thing it can do on the repo is dispatch
 * workflows.
 */

const WORKOUT_HEADER_DEFAULT = 'Workout';
const HEADER_ROW = 1;
const SETTLE_MS = 60_000;
const QUEUE_PREFIX = 'pending_';

/**
 * Run this once from the editor after copying the workbook to a new year.
 * Idempotent: re-running won't create duplicate triggers.
 */
function setupTriggers() {
  const existing = ScriptApp.getProjectTriggers();
  const haveEdit = existing.some(
    (t) =>
      t.getHandlerFunction() === 'onSheetEdit' &&
      t.getEventType() === ScriptApp.EventType.ON_EDIT
  );
  const haveCron = existing.some(
    (t) =>
      t.getHandlerFunction() === 'processPending' &&
      t.getEventType() === ScriptApp.EventType.CLOCK
  );

  if (!haveEdit) {
    ScriptApp.newTrigger('onSheetEdit')
      .forSpreadsheet(SpreadsheetApp.getActiveSpreadsheet())
      .onEdit()
      .create();
    Logger.log('Installed onEdit trigger');
  } else {
    Logger.log('onEdit trigger already installed');
  }

  if (!haveCron) {
    ScriptApp.newTrigger('processPending')
      .timeBased()
      .everyMinutes(1)
      .create();
    Logger.log('Installed 1-minute cron trigger');
  } else {
    Logger.log('1-minute cron trigger already installed');
  }
}

/**
 * Installable onEdit handler. Filters to single-cell edits on the Workout
 * column (located by header match in row 1) and queues for settle.
 */
function onSheetEdit(e) {
  if (!e || !e.range) return;
  const range = e.range;
  if (range.getNumRows() !== 1 || range.getNumColumns() !== 1) return;

  const row = range.getRow();
  if (row <= HEADER_ROW) return;

  const sheet = range.getSheet();
  const props = PropertiesService.getScriptProperties();
  const headerName = props.getProperty('WORKOUT_HEADER') || WORKOUT_HEADER_DEFAULT;
  const workoutCol = findHeaderColumn_(sheet, headerName);
  if (workoutCol === null) return;
  if (range.getColumn() !== workoutCol) return;

  const key = QUEUE_PREFIX + sheet.getSheetId() + '_' + row;
  props.setProperty(
    key,
    JSON.stringify({
      sheetId: sheet.getSheetId(),
      sheetName: sheet.getName(),
      row: row,
      col: workoutCol,
      lastEdit: Date.now(),
    })
  );
}

/**
 * Time-based handler (runs every minute). Drains queued edits whose
 * settle timer has elapsed. Idempotent; safe to overlap.
 */
function processPending() {
  const props = PropertiesService.getScriptProperties();
  const all = props.getProperties();
  const now = Date.now();
  const ss = SpreadsheetApp.getActiveSpreadsheet();

  for (const key in all) {
    if (!key.startsWith(QUEUE_PREFIX)) continue;

    let entry;
    try {
      entry = JSON.parse(all[key]);
    } catch (err) {
      props.deleteProperty(key);
      continue;
    }

    if (now - entry.lastEdit < SETTLE_MS) continue;

    const sheet = findSheet_(ss, entry.sheetId, entry.sheetName);
    if (!sheet) {
      props.deleteProperty(key);
      continue;
    }

    const value = sheet.getRange(entry.row, entry.col).getValue();
    const text = value == null ? '' : String(value).trim();

    if (!text) {
      props.deleteProperty(key);
      continue;
    }

    dispatchWorkflow_('pipeline-run', { fit: false, historical: false });
    if (/race@/i.test(text)) {
      dispatchWorkflow_('pipeline-run-fit', { fit: true, historical: false });
    }

    props.deleteProperty(key);
  }
}

function findHeaderColumn_(sheet, headerName) {
  const lastCol = sheet.getLastColumn();
  if (lastCol < 1) return null;
  const headers = sheet.getRange(HEADER_ROW, 1, 1, lastCol).getValues()[0];
  const target = String(headerName).toLowerCase().trim();
  for (let i = 0; i < headers.length; i++) {
    if (String(headers[i]).toLowerCase().trim() === target) return i + 1;
  }
  return null;
}

function findSheet_(ss, sheetId, sheetName) {
  const sheets = ss.getSheets();
  for (const s of sheets) {
    if (s.getSheetId() === sheetId) return s;
  }
  return ss.getSheetByName(sheetName) || null;
}

function dispatchWorkflow_(eventType, clientPayload) {
  const props = PropertiesService.getScriptProperties();
  const token = props.getProperty('GITHUB_PAT');
  const repo = props.getProperty('GITHUB_REPO');
  if (!token || !repo) {
    console.error('Missing GITHUB_PAT or GITHUB_REPO in Script Properties');
    return;
  }

  const url = 'https://api.github.com/repos/' + repo + '/dispatches';
  const resp = UrlFetchApp.fetch(url, {
    method: 'post',
    contentType: 'application/json',
    headers: {
      Authorization: 'Bearer ' + token,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    },
    payload: JSON.stringify({
      event_type: eventType,
      client_payload: clientPayload,
    }),
    muteHttpExceptions: true,
  });
  const code = resp.getResponseCode();
  if (code < 200 || code >= 300) {
    console.error(
      'Dispatch ' + eventType + ' failed: ' + code + ' ' + resp.getContentText()
    );
  } else {
    console.log('Dispatched ' + eventType);
  }
}
