/**
 * Apps Script bound to the current-year Running Log workbook.
 *
 * This file contains TWO independent systems that share the workbook:
 *
 *   (A) Pipeline dispatcher — watches user edits and triggers the
 *       build-and-deploy GitHub Actions workflow via workflow_dispatch.
 *       Uses an installable onEdit trigger (onSheetEdit). Setup requires
 *       running setupTriggers() once.
 *
 *   (B) Legacy spreadsheet automation — Max's pre-existing onEdit
 *       routine that expands shorthand codes, totals miles/paces by
 *       run type, and rebuilds the sorted summary lists (weather,
 *       partners, conditions, wind, time of day, shoes, location).
 *       Runs on the simple onEdit() trigger, which Apps Script wires
 *       up automatically by name; no setup needed.
 *
 * The two systems are orthogonal:
 *   - Different function names (onEdit vs onSheetEdit).
 *   - Simple triggers and installable triggers don't fire on each
 *     other's script-driven cell writes, so the legacy routine's
 *     setValue() calls do not re-fire the dispatcher.
 *   - No shared helper names (asSecs/asTime are legacy-only;
 *     findHeaderColumn_/findSheet_/dispatchWorkflow_ are dispatcher-only).
 *
 * Pipeline dispatcher behavior:
 *   The Workout column is the "this day's data is finalized" signal —
 *   it's edited once per day, last (modulo weight, which is fine to lag).
 *   The dispatcher fires a `workflow_dispatch` event against
 *   build-and-deploy.yml (same endpoint and PAT as the admin "Run
 *   pipeline" button) when:
 *     - The user edits the Workout column to a non-empty value. Body
 *       sent with `inputs: {fit: false, historical: false}`. If the
 *       value matches /race@/i, a second dispatch with `fit: true`
 *       is also sent so the Bayesian refit job runs.
 *     - The user edits the Weather column (col 8 abbreviation slot)
 *       to a value beginning with `.r` or `.l`. Those codes drive the
 *       legacy onEdit to autofill the Workout cell to "rec." or "long."
 *       via setValue, which does not re-fire onEdit triggers — so the
 *       dispatcher watches the source column instead. `..` (skip) and
 *       plain weather words do not dispatch; the manual workout-column
 *       edit later in the day handles those.
 *
 * Setup (do once after copying the workbook to a new year):
 *   1. Extensions → Apps Script (the script comes with the sheet copy).
 *   2. Project Settings → Script Properties → set:
 *        GITHUB_PAT             fine-grained PAT, "Actions: write" on
 *                               the repo. Reuse the PAT used by the
 *                               admin "Run pipeline" button (Netlify
 *                               env GITHUB_DISPATCH_TOKEN).
 *        GITHUB_REPO            e.g. "randalpik/running-pipeline"
 *        WORKOUT_HEADER         optional, defaults to "Workout"
 *        ABBREV_HEADER          optional, defaults to "Weather"
 *        GITHUB_REF             optional, defaults to "main"
 *        GITHUB_WORKFLOW_FILE   optional, defaults to "build-and-deploy.yml"
 *   3. In the Apps Script editor, choose the function "setupTriggers"
 *      from the dropdown and click Run. Grant OAuth consent when prompted.
 *      That installs the onSheetEdit installable trigger. The legacy
 *      onEdit() trigger does not need installation; Apps Script picks
 *      it up by name. setupTriggers also performs a one-shot migration:
 *      it removes any prior `processPending` time-based trigger from
 *      the old debounce design and drains leftover `pending_*`
 *      ScriptProperties keys.
 *
 * Reusing the PAT across Netlify and Apps Script is fine. PAT is read-only
 * in the sense that the only thing it can do on the repo is dispatch
 * workflows.
 */

const WORKOUT_HEADER_DEFAULT = 'Workout';
const ABBREV_HEADER_DEFAULT = 'Weather';
// Headers are in row 2 of this workbook (row 1 holds totals/labels). The
// legacy onEdit iterates daily data starting at i=2 (0-indexed) = row 3
// (1-indexed), confirming the header is row 2.
const HEADER_ROW = 2;

/**
 * Run this once from the editor after copying the workbook to a new year.
 * Idempotent: re-running won't create duplicate triggers.
 *
 * Also performs a one-shot migration from the previous debounce design:
 * removes any leftover `processPending` time-based trigger and drains
 * `pending_*` ScriptProperties keys.
 */
function setupTriggers() {
  const existing = ScriptApp.getProjectTriggers();
  const haveEdit = existing.some(
    (t) =>
      t.getHandlerFunction() === 'onSheetEdit' &&
      t.getEventType() === ScriptApp.EventType.ON_EDIT
  );

  if (!haveEdit) {
    ScriptApp.newTrigger('onSheetEdit')
      .forSpreadsheet(SpreadsheetApp.getActiveSpreadsheet())
      .onEdit()
      .create();
    Logger.log('Installed onSheetEdit trigger');
  } else {
    Logger.log('onSheetEdit trigger already installed');
  }

  // Migration: remove the old 1-minute processPending cron if present.
  let removedCrons = 0;
  for (const t of existing) {
    if (t.getHandlerFunction() === 'processPending') {
      ScriptApp.deleteTrigger(t);
      removedCrons++;
    }
  }
  Logger.log(
    removedCrons > 0
      ? 'Removed ' + removedCrons + ' legacy processPending cron trigger(s)'
      : 'No legacy processPending cron triggers to remove'
  );

  // Migration: drain any leftover pending_* queue properties from the
  // old debounce design.
  const props = PropertiesService.getScriptProperties();
  const all = props.getProperties();
  let drained = 0;
  for (const key in all) {
    if (key.indexOf('pending_') === 0) {
      props.deleteProperty(key);
      drained++;
    }
  }
  Logger.log(
    drained > 0
      ? 'Drained ' + drained + ' legacy pending_* queue propertie(s)'
      : 'No legacy pending_* queue properties to drain'
  );
}

/**
 * Installable onEdit handler. Dispatches a `workflow_dispatch` event
 * instantly on user edits to either the Workout column or the Weather
 * (abbreviation) column. See the file header for the trigger semantics.
 *
 * Logs every entry and decision branch via console.log so the Executions
 * page in the Apps Script editor shows what's happening. If you don't see
 * an "[onSheetEdit] fired" line in Executions when you edit a cell, the
 * installable trigger isn't firing — re-run setupTriggers and grant OAuth.
 */
function onSheetEdit(e) {
  console.log(
    '[onSheetEdit] fired: ' +
      JSON.stringify({
        hasEvent: !!e,
        hasRange: !!(e && e.range),
        value: e && e.value,
        oldValue: e && e.oldValue,
      })
  );

  if (!e || !e.range) {
    console.log('[onSheetEdit] bail: no event/range');
    return;
  }
  const range = e.range;
  if (range.getNumRows() !== 1 || range.getNumColumns() !== 1) {
    console.log(
      '[onSheetEdit] bail: multi-cell edit (' +
        range.getNumRows() +
        'x' +
        range.getNumColumns() +
        ')'
    );
    return;
  }
  if (range.getRow() <= HEADER_ROW) {
    console.log('[onSheetEdit] bail: header row');
    return;
  }

  const sheet = range.getSheet();
  const props = PropertiesService.getScriptProperties();
  const workoutHeader = props.getProperty('WORKOUT_HEADER') || WORKOUT_HEADER_DEFAULT;
  const abbrevHeader = props.getProperty('ABBREV_HEADER') || ABBREV_HEADER_DEFAULT;
  const workoutCol = findHeaderColumn_(sheet, workoutHeader);
  const abbrevCol = findHeaderColumn_(sheet, abbrevHeader);

  const col = range.getColumn();
  // e.value reflects what the user typed. Capture it before the simple
  // onEdit (which may overwrite the abbreviation cell with weather text).
  const value = String(e.value == null ? '' : e.value);

  console.log(
    '[onSheetEdit] context: ' +
      JSON.stringify({
        sheet: sheet.getName(),
        row: range.getRow(),
        col: col,
        workoutHeader: workoutHeader,
        workoutCol: workoutCol,
        abbrevHeader: abbrevHeader,
        abbrevCol: abbrevCol,
        value: value,
      })
  );

  if (workoutCol === null) {
    console.log(
      '[onSheetEdit] WARN: header "' +
        workoutHeader +
        '" not found in row 1 — Workout column edits will not dispatch'
    );
  }
  if (abbrevCol === null) {
    console.log(
      '[onSheetEdit] WARN: header "' +
        abbrevHeader +
        '" not found in row 1 — abbreviation column edits will not dispatch'
    );
  }

  if (workoutCol !== null && col === workoutCol) {
    const text = value.trim();
    if (!text) {
      console.log('[onSheetEdit] workout col cleared; no dispatch');
      return;
    }
    console.log('[onSheetEdit] workout col edit → dispatch (fit=false)');
    dispatchWorkflow_({ fit: false, historical: false });
    if (/race@/i.test(text)) {
      console.log('[onSheetEdit] race@ detected → also dispatch (fit=true)');
      dispatchWorkflow_({ fit: true, historical: false });
    }
    return;
  }

  if (abbrevCol !== null && col === abbrevCol) {
    // Dispatch only on `.r` / `.l`, the codes that drive the legacy onEdit
    // to autofill the workout cell. `..` (skip) and plain weather words
    // are not finalization signals — wait for the user's manual workout
    // edit.
    const lower = value.toLowerCase();
    if (
      lower.charAt(0) === '.' &&
      (lower.charAt(1) === 'r' || lower.charAt(1) === 'l')
    ) {
      console.log(
        '[onSheetEdit] abbrev col code "' + lower.slice(0, 2) + '" → dispatch (fit=false)'
      );
      dispatchWorkflow_({ fit: false, historical: false });
    } else {
      console.log(
        '[onSheetEdit] abbrev col edit "' + value + '" not a .r/.l code; no dispatch'
      );
    }
    return;
  }

  console.log(
    '[onSheetEdit] edit on col ' +
      col +
      ' is neither workout nor abbreviation; no dispatch'
  );
}

/**
 * Diagnostic helper. Run from the editor (Run → debugConfig) to confirm
 * Script Properties are set, headers resolve, and the onSheetEdit trigger
 * is installed. Output appears in the Executions page.
 */
function debugConfig() {
  const props = PropertiesService.getScriptProperties();
  const token = props.getProperty('GITHUB_PAT');
  const repo = props.getProperty('GITHUB_REPO');
  const workoutHeader = props.getProperty('WORKOUT_HEADER') || WORKOUT_HEADER_DEFAULT;
  const abbrevHeader = props.getProperty('ABBREV_HEADER') || ABBREV_HEADER_DEFAULT;

  console.log(
    '[debugConfig] script properties: ' +
      JSON.stringify({
        GITHUB_PAT: token ? '(set, length=' + token.length + ')' : '(MISSING)',
        GITHUB_REPO: repo || '(MISSING)',
        WORKOUT_HEADER: workoutHeader,
        ABBREV_HEADER: abbrevHeader,
      })
  );

  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheets = ss.getSheets();
  for (const sheet of sheets) {
    const workoutCol = findHeaderColumn_(sheet, workoutHeader);
    const abbrevCol = findHeaderColumn_(sheet, abbrevHeader);
    console.log(
      '[debugConfig] sheet "' +
        sheet.getName() +
        '" → workoutCol=' +
        workoutCol +
        ', abbrevCol=' +
        abbrevCol +
        (workoutCol === null || abbrevCol === null ? '  ← header lookup FAILED' : '')
    );
  }

  const triggers = ScriptApp.getProjectTriggers();
  const summary = triggers.map((t) => ({
    handler: t.getHandlerFunction(),
    eventType: String(t.getEventType()),
    triggerSource: String(t.getTriggerSource()),
  }));
  console.log('[debugConfig] installed triggers: ' + JSON.stringify(summary));

  const haveSheetEdit = triggers.some(
    (t) =>
      t.getHandlerFunction() === 'onSheetEdit' &&
      t.getEventType() === ScriptApp.EventType.ON_EDIT
  );
  console.log(
    '[debugConfig] onSheetEdit installable trigger: ' +
      (haveSheetEdit ? 'OK' : 'MISSING — re-run setupTriggers')
  );
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

/**
 * POSTs a workflow_dispatch event for the build-and-deploy workflow.
 * Mirrors site/netlify/functions/admin-run-pipeline.ts so the same PAT
 * (Actions: write) works for both. Inputs are coerced to "true"/"false"
 * strings — that's what GitHub Actions expects for boolean inputs.
 */
function dispatchWorkflow_(inputs) {
  const props = PropertiesService.getScriptProperties();
  const token = props.getProperty('GITHUB_PAT');
  const repo = props.getProperty('GITHUB_REPO');
  if (!token || !repo) {
    console.error('Missing GITHUB_PAT or GITHUB_REPO in Script Properties');
    return;
  }

  const ref = props.getProperty('GITHUB_REF') || 'main';
  const workflowFile =
    props.getProperty('GITHUB_WORKFLOW_FILE') || 'build-and-deploy.yml';
  const url =
    'https://api.github.com/repos/' +
    repo +
    '/actions/workflows/' +
    workflowFile +
    '/dispatches';
  const body = {
    ref: ref,
    inputs: {
      fit: inputs && inputs.fit ? 'true' : 'false',
      historical: inputs && inputs.historical ? 'true' : 'false',
    },
  };

  const resp = UrlFetchApp.fetch(url, {
    method: 'post',
    contentType: 'application/json',
    headers: {
      Authorization: 'Bearer ' + token,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    },
    payload: JSON.stringify(body),
    muteHttpExceptions: true,
  });
  const code = resp.getResponseCode();
  if (code < 200 || code >= 300) {
    console.error(
      'Dispatch failed: ' +
        code +
        ' url=' +
        url +
        ' inputs=' +
        JSON.stringify(body.inputs) +
        ' body=' +
        resp.getContentText()
    );
  } else {
    console.log(
      'Dispatched workflow_dispatch inputs=' + JSON.stringify(body.inputs)
    );
  }
}

// ============================================================================
// Legacy spreadsheet automation
// ----------------------------------------------------------------------------
// The functions below are Max's pre-existing onEdit routine, preserved
// verbatim. They run on the simple onEdit trigger (auto-wired by name) and
// are independent of the pipeline dispatcher above. Do not edit casually —
// the spreadsheet's day-to-day data entry depends on this exact behavior.
// ============================================================================

function onEdit() {
  var sheet = SpreadsheetApp.getActiveSheet();
  var data = sheet.getDataRange().getValues();
  var totalMiles = [0, 0, 0, 0, 0, 0];
  var totalPace = [0, 0, 0, 0, 0, 0];
  var totalNum = [0, 0, 0, 0, 0, 0];
  var filledCells = 0;
  var itemLists = [[], [], [], [], [], [], []]; //List order: weathers, partners, conditions, wind, time of day, shoes, location
  var countLists = [[], [], [], [], [], [], []]; //Pulled from spreadsheet columns 7, 9, 10, 11, 12, 13, 14
  var locDistances = [];
  var locTimes = [];
  var locPaces = [];
  var shoeDistances = [];
  var mergedLists = [[], [], [], [], [], [], []];
  var colors = ['#d9ead3', '#d9d2e9', '#fff2cc', '#ffe599', '#f9cb9c', '#fce5cd'];
  var color, i, j, k, n, s, col;
  var presets = [[], [], [], [], [], []]; //weather, conditions, wind, time of day, shoes, location

  const daysInYear = 365;
  const ts = 17; //total start column in case more parameters get added

  presets[0] = {"r": "rec.", "l": "long."};
  presets[1] = {".": "clear", "c": "cloudy", "o": "overcast", "l": "light rain", "h": "heavy rain", "i": "inside", "f": "foggy", "s": "snow"};
  presets[2] = {".": "dry", "w": "wet", "i": "inside", "c": "icy", "m": "muddy", "s": "snow"};
  presets[3] = {".": "low", "m": "moderate", "h": "high"};
  presets[4] = {".": "morning", "e": "early", "a": "afternoon", "l": "late"};
  presets[5] = {".": data[10][ts + 10], "t": data[13][ts + 10], "x": data[12][ts + 10], "r": data[11][ts + 10], "h": data[14][ts + 10]};
  presets[6] = {
    ".": data[9][ts + 10],
    "12": "12 south",
    "bc": "boulder creek",
    "bm": "belle meade",
    "bt": "boulder turnpike",
    "eb": "east boulder",
    "eh": "english hill",
    "ev": "everest",
    "gw": "greenway",
    "hm": "hartman",
    "ir": "indoor rec",
    "lc": "love circle",
    "ls": "lake samm",
    "mc": "mccabe",
    "mg": "magnolia",
    "ng": "north greenway",
    "nn": "nike>nature",
    "np": "nike>powerline",
    "or": "outdoor rec",
    "p1": "powerline 1",
    "p2": "powerline 2",
    "p3": "powerline 3",
    "pl": "pipeline",
    "rc": "rollercoaster",
    "rh": "RHS track",
    "rp": "rose park",
    "rt": "river trail",
    "sb": "suburbia",
    "sl": "shoreline",
    "tm": "table mesa",
    "ws": "watershed",
  };

  for (i=2;i<daysInYear+2;i++) { //iterates through each day
    if(data[i][3] !== "") {
      filledCells++;
    }

    if(data[i][7].charAt(0) == ".") { //uses code presets to populate info cells if a code was given
      var runTypePreset = presets[0][data[i][7].charAt(1)];
      if (runTypePreset !== undefined) { //skip workout autofill when char[1] is '.'; user fills it manually
        sheet.getRange(i+1,9).setValue(runTypePreset);
      }
      var weatherPreset = presets[1][data[i][7].charAt(2)];
      sheet.getRange(i+1,8).setValue(weatherPreset);
      sheet.getRange(i+1,10).setValue("solo");
      sheet.getRange(i+1,11).setValue(presets[2][data[i][7].charAt(3)]);
      sheet.getRange(i+1,12).setValue(presets[3][data[i][7].charAt(4)]);
      sheet.getRange(i+1,13).setValue(presets[4][data[i][7].charAt(5)]);
      sheet.getRange(i+1,14).setValue(presets[5][data[i][7].charAt(6)]);
      sheet.getRange(i+1,15).setValue(presets[6][data[i][7].slice(7)]);
      data[i][7] = weatherPreset; //force update
      if (runTypePreset !== undefined) {
        data[i][8] = runTypePreset; //force update
      }
    }

    //totals distances and paces for workouts where min/mile pace is stated
    color = sheet.getRange(i+1,9).getBackground();
    if(colors.indexOf(color) != -1) {
      n = colors.indexOf(color);
      if (n<2) { //if recovery or long run, gets distance from total workout distance, and adds to total
        totalMiles[n] += data[i][4];
        var pace = asTime(data[i][5]*60/data[i][4]);
        if(data[i][8].includes(".")) { //autofills pace from time and distance if needed, adding hill sprints
          var newText = data[i][8].replace(".", data[i][8].endsWith(".") ? `@${pace}` : `@${pace}/`);
          sheet.getRange(i+1,9).setValue(newText);
          data[i][8] = newText; //to update for later in the script
        }
      } else { //if quality workout, gets distance from workout notes, converts to miles, and adds distance to total
        totalMiles[n] += data[i][8].slice(data[i][8].indexOf(",") + 2, data[i][8].indexOf("@") - 1)/1609;
      }
      totalPace[n] += asSecs(data[i][8], data[i][8].indexOf("@") + 1);
      totalNum[n]++; //increments the number of workouts of that type
    }

    //handles listing of weather, partners, conditions, time of day, and location
    for (j=0;j<7;j++) { //for clarity, j is the identifier of the type of list being made
      col = (j == 0) ? 7 : (j + 8); //function to map this identifier to the actual column as it appears in the spreadsheet
      if (data[i][col] != "") {
        if(j == 1) { //handles separating lists of partner names if necessary
          s = data[i][col].split(", ");
          for(k=0;k<s.length;k++) {
            if(itemLists[j].indexOf(s[k]) == -1) {
              itemLists[j].push(s[k]);
              countLists[j].push(0);
            }
            countLists[j][itemLists[j].indexOf(s[k])]++;
          }
        } else { //handles all other lists with only one item per cell
          if(itemLists[j].indexOf(data[i][col]) == -1) {
            itemLists[j].push(data[i][col]);
            countLists[j].push(0);
            if(j == 5) {
              shoeDistances.push(0);
            }
            if(j == 6) {
              locDistances.push(0);
              locTimes.push(0);
            }
          }
          countLists[j][itemLists[j].indexOf(data[i][col])]++;
          if(j == 5) {
            shoeDistances[itemLists[j].indexOf(data[i][col])] += data[i][4];
          }
          if(j == 6) {
            locDistances[itemLists[j].indexOf(data[i][col])] += data[i][4];
            locTimes[itemLists[j].indexOf(data[i][col])] += data[i][5];
          }
        }
      }
    }
  }

  data = sheet.getDataRange().getValues();

  for (i=0;i<locDistances.length;i++) { //finds pace for each location based on time and distance
    locPaces[i] = asTime(60 * locTimes[i] / locDistances[i]);
  }

  for (i=0;i<7;i++) { //sorts and prints each list with its count
    for(j=0;j<itemLists[i].length;j++) {
      if(i<5) { //concatenates items and counts into single objects to be sorted, and pace if applicable
        mergedLists[i].push({count: countLists[i][j], item: itemLists[i][j]});
      } else if (i<6) {
        mergedLists[i].push({count: countLists[i][j], item: itemLists[i][j], distance: shoeDistances[j]});
      } else {
        mergedLists[i].push({count: countLists[i][j], item: itemLists[i][j], pace: locPaces[j], distance: locDistances[j]});
      }
    }
    mergedLists[i] = mergedLists[i].sort((a, b) => a.distance ? b.distance - a.distance : b.count - a.count);
    var offset = i == 6 ? 1 : 0;
    for(j=0;j<mergedLists[i].length;j++) {
      sheet.getRange(18+j,ts+1+(2*i)+offset).setValue(mergedLists[i][j].item);
      sheet.getRange(18+j,ts+2+(2*i)+offset).setValue(mergedLists[i][j].count);
      if(i>4) {
        sheet.getRange(18+j,ts+3+(2*i)+offset).setValue(mergedLists[i][j].distance);
      }
      if(i>5) {
        sheet.getRange(18+j,ts+4+(2*i)+offset).setValue(mergedLists[i][j].pace);
      }
    }
    for(j=0;j<5;j++) { //erases any list entry that could have been left behind
      sheet.getRange(18+mergedLists[i].length+j,ts+1+(2*i)+offset).setValue("");
      sheet.getRange(18+mergedLists[i].length+j,ts+2+(2*i)+offset).setValue("");
      if(i>4) {
        sheet.getRange(18+mergedLists[i].length+j,ts+3+(2*i)+offset).setValue("");
      }
      if(i>5) {
        sheet.getRange(18+mergedLists[i].length+j,ts+4+(2*i)+offset).setValue("");
      }
    }
  }

  for (i=0;i<6;i++) {
    sheet.getRange(i+10,ts+2).setValue(Math.round(totalMiles[i] * 10)/10); //prints total distance rounded to tenths of a mile
    sheet.getRange(i+10,ts+3).setValue(totalNum[i] == 0 ? "" : asTime(totalPace[i]/totalNum[i])); //prints average pace, or zero if no workouts were completed of this type
  }
  var totalDistance = data[3][ts+1];
  var averageDistance = totalDistance / filledCells;
  sheet.getRange(4, ts+4).setValue((daysInYear * averageDistance).toFixed(1));
  var totalTime = asSecs(data[4][ts+1], 0);
  var averageTime = totalTime / filledCells;
  sheet.getRange(5, ts+4).setValue(asTime(parseInt(daysInYear * averageTime)));
}

function asSecs(s, index) { //converts a time in x:xx format to seconds
  var separator = s.slice(index).indexOf(":") + index;
  var secs = s.slice(separator + 1, separator + 3);
  secs = (secs.charAt(0) == "0") ? parseInt(secs.slice(1, 2)) : parseInt(secs); //removes leading zero on seconds to avoid being interpreted in octal
  return parseInt(s.slice(index, separator) * 60) + secs;
}

function asTime(secs) { //converts a time in seconds to x:xx format
  return Math.floor(secs/60) + ":" + ((secs % 60) < 10 ? "0" : "") + Math.floor(secs % 60);
}
