import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session, send_file
import torch
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import requests
import time
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import glob

# Import your prediction utilities
from src.models.ecg_cnn import ECGCNN, ECGLSTM
from src.utils.model_utils import ModelSaver

app = Flask(__name__)
app.secret_key = 'your-very-secure-secret'

UPLOAD_FOLDER = './uploads'
MODEL_FOLDER = './saved_models'
ALLOWED_EXTENSIONS = {'csv', 'npy'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility: list .pth models
def list_models():
    files = []
    for fname in os.listdir(MODEL_FOLDER):
        if fname.endswith('.pth'):
            files.append(fname)
    return sorted(files)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_path, device):
    saver = ModelSaver(MODEL_FOLDER)
    model_data = torch.load(model_path, map_location=device)
    model_type = model_data.get('model_type', 'cnn')
    num_classes = len(model_data.get('class_names', []))
    input_length = model_data.get('input_length', 1000)
    if model_type == 'cnn':
        model = ECGCNN(input_length=input_length, num_classes=num_classes)
    else:
        model = ECGLSTM(input_size=12, num_classes=num_classes)
    model.load_state_dict(model_data['model_state_dict'])
    model = model.to(device).eval()
    class_names = model_data.get('class_names', [])
    return model, class_names, input_length

def ecg_file_to_array(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        # Try to infer header automatically and take only numeric columns
        df = pd.read_csv(filepath, dtype=str)
        # Drop non-numeric columns (like id, timestamp etc)
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(axis=1, how='all') # remove empty cols
        if numeric_df.shape[1] == 0:
            raise ValueError("No numeric data columns found in the uploaded CSV. Please check your file format.")
        arr = numeric_df.values.astype(np.float32)
        return arr
    elif ext == '.npy':
        arr = np.load(filepath)
        return arr.astype(np.float32)
    else:
        raise ValueError('Unsupported file type')

def preprocess_signal(arr, input_length):
    """
    Reshape ECG data for model input. Accepts vectors (any dimension),
    replicates to 12 leads or pads/crops/truncates as necessary.
    Output: (12, input_length) float32
    """
    arr = np.asarray(arr)
    # If 1D: treat as a single lead, tile
    if arr.ndim == 1:
        arr = np.tile(arr[None, :], (12, 1))
    elif arr.ndim == 2:
        if arr.shape[0] == 12:
            pass  # already (12, L)
        elif arr.shape[1] == 12:
            arr = arr.T  # (L, 12) -> (12, L)
        elif arr.shape[0] == 1:
            arr = np.tile(arr, (12, 1))  # (1, L) -> (12, L)
        elif arr.shape[1] == 1:
            arr = np.tile(arr.T, (12, 1))  # (L, 1) -> (12, L)
        elif arr.shape[0] < 12:
            arr = np.pad(arr, ((0, 12 - arr.shape[0]), (0, 0)), 'constant')
        elif arr.shape[1] < 12:
            arr = np.pad(arr, ((0, 0), (0, 12 - arr.shape[1])), 'constant')
        elif arr.shape[0] > 12:
            arr = arr[:12, :]
        elif arr.shape[1] > 12:
            arr = arr[:, :12]
    else:
        raise ValueError("Could not interpret ECG file shape. Expected a 1D or 2D array.")
    # Crop or pad time axis
    if arr.shape[1] >= input_length:
        arr = arr[:, :input_length]
    else:
        arr = np.pad(arr, ((0, 0), (0, input_length - arr.shape[1])), 'constant')
    return arr.astype(np.float32)

def predict_ecg(model, arr, device):
    x = torch.tensor(arr[None, :, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x)
        out = out.cpu().numpy()[0]
    return out

# Mapping for short code to full form
diagnosis_map = {
    "MI": "Myocardial Infarction",
    "NORM": "Normal ECG",
    "STTC": "ST/T Changes",
    "HYP": "Hypertrophy",
    "CD": "Conduction Disturbance"
}

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key="
)

def _local_summary(top3, preds):
    if not top3:
        return "No AI summary available."
    # top diagnosis
    primary_short, primary_full, primary_score = top3[0]
    risk = f"{primary_score*100:.1f}%"
    advice = ""
    if primary_short == 'MI' and primary_score >= 0.5:
        advice = (
            "Features suggest possible myocardial infarction. Urgent evaluation is recommended: "
            "call emergency services, obtain ECG confirmation, cardiac enzymes, and consider aspirin "
            "if no contraindications."
        )
    elif primary_short in ('STTC', 'CD', 'HYP') and primary_score >= 0.5:
        advice = (
            "Abnormal ECG features detected. Arrange prompt cardiology review, repeat ECG, and risk factor management."
        )
    else:
        advice = "No high-risk pattern detected. If symptomatic, seek medical advice; otherwise continue routine care."
    return (
        f"Most likely: {primary_full} ({primary_short}) with estimated probability {risk}. "
        f"Top differentials: "
        + ", ".join([f"{s} ({f})" for s, f, _ in top3[1:]])
        + ". " + advice
    )

def summarize_with_gemini(top3, preds):
    diag_lines = [f"{short} ({longname}): {score*100:.1f}%" for short, longname, score in top3]
    all_diags = "\n".join(diag_lines)
    prompt = (
        "Here are the diagnostic probabilities for a patient's ECG scan (as given by an AI model):\n"
        f"{all_diags}\n"
        "Summarize these as a crisp clinical report: (1) Say which diagnoses are most likely and their risk, (2) Give clear recommendations for immediate action if a serious issue (like MI) is present (hospitalization, medicines if relevant),\n"
        "and (3) If nothing serious, advise on healthy next steps. Be concise, as for a doctor's letter, and do not repeat the values directly. Use clear language."
    )
    data = {"contents":[{"parts":[{"text":prompt}]}]}
    # If no API key, fall back immediately
    if not GEMINI_API_KEY:
        return _local_summary(top3, preds)
    url = GEMINI_URL + GEMINI_API_KEY
    # Try up to 2 attempts with simple backoff for rate limits
    for attempt in range(2):
        try:
            resp = requests.post(url, json=data, timeout=20)
            if resp.status_code == 429:
                if attempt == 0:
                    time.sleep(2.0)
                    continue
                return _local_summary(top3, preds)
            resp.raise_for_status()
            res = resp.json()
            ai_report = res['candidates'][0]['content']['parts'][0]['text']
            return ai_report
        except Exception:
            if attempt == 0:
                time.sleep(1.0)
                continue
            return _local_summary(top3, preds)

def find_record_pairs(directory, suffix):
    "Return list of base names (without extension) for valid pairs. Suffix is '_lr' or '_hr'."
    base_paths = glob.glob(os.path.join(directory, f'*{suffix}.hea'))
    pairs = []
    for hea_path in base_paths:
        dat_path = os.path.splitext(hea_path)[0] + '.dat'
        if os.path.exists(dat_path):
            # Just the file's base name
            pairs.append(os.path.basename(os.path.splitext(hea_path)[0]))
    return sorted(pairs)

@app.route('/', methods=['GET'])
def index():
    models = list_models()
    # Scan for 100 Hz and 500 Hz record pairs
    records100 = find_record_pairs('DATA/records100', '_lr')
    records500 = find_record_pairs('DATA/records500', '_hr')
    return render_template('index.html', models=models, records100=records100, records500=records500)

@app.route('/predict', methods=['POST'])
def predict():
    notes = request.form.get('notes', '').strip()
    model_name = request.form['model_name']
    model_path = os.path.join(MODEL_FOLDER, model_name)

    # Debug: log incoming form and uploaded files presence
    try:
        print(f"PREDICT: received form keys={list(request.form.keys())}", flush=True)
        print(f"PREDICT: files keys={list(request.files.keys())}", flush=True)
    except Exception:
        print('PREDICT: cannot read request metadata', flush=True)

    # Only handle manual uploads of .hea + .dat for either 100 Hz or 500 Hz
    hea_100 = request.files.get('hea_100')
    dat_100 = request.files.get('dat_100')
    hea_500 = request.files.get('hea_500')
    dat_500 = request.files.get('dat_500')
    record_file = None
    freq_name = None

    # Debug print file info without consuming streams
    def _file_info(f):
        if not f:
            return None
        name = getattr(f, 'filename', None)
        size = getattr(f, 'content_length', None)
        try:
            # try to peek one byte non-destructively
            stream = getattr(f, 'stream', None)
            pos = None
            if stream:
                try:
                    pos = stream.tell()
                except Exception:
                    pos = None
                try:
                    chunk = stream.read(1)
                    has = bool(chunk)
                except Exception:
                    has = None
                if pos is not None:
                    try:
                        stream.seek(pos)
                    except Exception:
                        pass
            else:
                has = None
        except Exception:
            has = None
        return {'filename': name, 'content_length': size, 'has_bytes': has}

    print('PREDICT: hea_100=', _file_info(hea_100), ' dat_100=', _file_info(dat_100), flush=True)
    print('PREDICT: hea_500=', _file_info(hea_500), ' dat_500=', _file_info(dat_500), flush=True)

    # Helper: check present, named & non-empty (keeps previous behavior)
    def file_valid(f):
        try:
            return f and f.filename and f.filename.strip() and f.stream and f.filename != 'None' and len(f.read()) > 0
        finally:
            try:
                if f and getattr(f, 'stream', None):
                    f.stream.seek(0)
            except Exception:
                pass
    # After read(), ensure seek(0) before use
    if hea_100 and getattr(hea_100, 'stream', None):
        try: hea_100.stream.seek(0)
        except Exception: pass
    if dat_100 and getattr(dat_100, 'stream', None):
        try: dat_100.stream.seek(0)
        except Exception: pass
    if hea_500 and getattr(hea_500, 'stream', None):
        try: hea_500.stream.seek(0)
        except Exception: pass
    if dat_500 and getattr(dat_500, 'stream', None):
        try: dat_500.stream.seek(0)
        except Exception: pass

    # Use the first valid upload present (100Hz then 500Hz)
    match100 = file_valid(hea_100) and file_valid(dat_100)
    if match100:
        base_hea = os.path.splitext(os.path.basename(hea_100.filename))[0]
        base_dat = os.path.splitext(os.path.basename(dat_100.filename))[0]
        if base_hea != base_dat:
            flash(f"100 Hz file base names do not match: '{hea_100.filename}' and '{dat_100.filename}'")
            return redirect(request.url)
        freq_name = '100 Hz'
        record_file = (hea_100, dat_100)
    else:
        match500 = file_valid(hea_500) and file_valid(dat_500)
        if match500:
            base_hea = os.path.splitext(os.path.basename(hea_500.filename))[0]
            base_dat = os.path.splitext(os.path.basename(dat_500.filename))[0]
            if base_hea != base_dat:
                flash(f"500 Hz file base names do not match: '{hea_500.filename}' and '{dat_500.filename}'")
                return redirect(request.url)
            freq_name = '500 Hz'
            record_file = (hea_500, dat_500)
        else:
            msg = []
            if not (file_valid(hea_100) or file_valid(dat_100)) and not (file_valid(hea_500) or file_valid(dat_500)):
                msg.append('You must upload a .hea and .dat file for either 100 Hz or 500 Hz!')
            if hea_100 and not file_valid(hea_100):
                msg.append(f"'100 Hz .hea' file is missing or empty: '{hea_100.filename}'")
            if dat_100 and not file_valid(dat_100):
                msg.append(f"'100 Hz .dat' file is missing or empty: '{dat_100.filename}'")
            if hea_500 and not file_valid(hea_500):
                msg.append(f"'500 Hz .hea' file is missing or empty: '{hea_500.filename}'")
            if dat_500 and not file_valid(dat_500):
                msg.append(f"'500 Hz .dat' file is missing or empty: '{dat_500.filename}'")
            flash("; ".join(msg) or "Missing required files.")
            return redirect(request.url)

    # Reset file pointer after check
    record_file[0].seek(0)
    record_file[1].seek(0)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Preserve original base name so .hea references match the .dat file
        hea_stem = os.path.splitext(os.path.basename(record_file[0].filename))[0]
        base = os.path.join(tmpdir, hea_stem)
        hea_path = base + '.hea'
        dat_path = base + '.dat'
        record_file[0].save(hea_path)
        record_file[1].save(dat_path)
        import wfdb
        try:
            signal, meta = wfdb.rdsamp(base)
            age = str(meta.get('age', '')) if 'age' in meta else ''
            sex = str(meta.get('sex', '')) if 'sex' in meta else ''
            patient_id = meta.get('record_name', '')
        except Exception as e:
            flash(f"Error reading WFDB files: {e}")
            return redirect(url_for('index'))
        arr = signal.T
        device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        model, class_names, input_length = load_model(model_path, device)
        try:
            arr = preprocess_signal(arr, input_length)
            preds = predict_ecg(model, arr, device)
            decoded = [(cls, diagnosis_map.get(cls, cls), float(score)) for cls, score in zip(class_names, preds)]
            top3 = sorted(decoded, key=lambda x: -x[2])[:3]
            ai_report = summarize_with_gemini(top3, decoded)
            user_filename = (record_file[0].filename or '') + ' / ' + (record_file[1].filename or '')
            # persist in session for PDF export
            session['last_result'] = {
                'top3': top3,
                'preds': decoded,
                'filename': user_filename,
                'freq_name': freq_name,
                'name': '',
                'age': age,
                'sex': sex,
                'patient_id': patient_id,
                'notes': notes,
                'ai_report': ai_report,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return render_template('result.html',
                top3=top3, preds=decoded, filename=user_filename,
                freq_name=freq_name, name='', age=age, sex=sex, patient_id=patient_id, notes=notes,
                ai_report=ai_report
            )
        except Exception as e:
            flash(f"Prediction error: {str(e)}")
            return redirect(url_for('index'))

# --- PDF Export Route ---
@app.route('/export_pdf', methods=['GET'])
def export_pdf():
    data = session.get('last_result')
    if not data:
        flash('No report found to export. Please run a prediction first.')
        return redirect(url_for('index'))
    # Build PDF via ReportLab (no external system deps)
    pdf_io = BytesIO()
    doc = SimpleDocTemplate(pdf_io, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story = []
    title = Paragraph('ECG AI Clinical Report', styles['Title'])
    story.append(title)
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated at: {data.get('generated_at', '')}", styles['Normal']))
    story.append(Paragraph("This report is generated by AI and may contain inaccuracies. Please consult a clinician for a professional diagnosis.", styles['Normal']))
    story.append(Spacer(1, 8))

    story.append(Paragraph('<b>Study Details</b>', styles['Heading3']))
    story.append(Paragraph(f"File: {data.get('filename','-')}", styles['Normal']))
    if data.get('freq_name'):
        story.append(Paragraph(f"Scanned Frequency: {data.get('freq_name')}", styles['Normal']))
    story.append(Spacer(1, 6))

    story.append(Paragraph('<b>Patient Information</b>', styles['Heading3']))
    story.append(Paragraph(f"Name: {data.get('name') or '-'}", styles['Normal']))
    story.append(Paragraph(f"Age: {data.get('age') or '-'}", styles['Normal']))
    story.append(Paragraph(f"Sex: {data.get('sex') or '-'}", styles['Normal']))
    story.append(Paragraph(f"Patient ID: {data.get('patient_id') or '-'}", styles['Normal']))
    story.append(Paragraph(f"Notes: {data.get('notes') or '-'}", styles['Normal']))
    story.append(Spacer(1, 6))

    story.append(Paragraph('<b>AI Summary</b>', styles['Heading3']))
    story.append(Paragraph(data.get('ai_report') or 'Not available.', styles['Normal']))
    story.append(Spacer(1, 6))

    story.append(Paragraph('<b>Top Predictions</b>', styles['Heading3']))
    top3 = data.get('top3') or []
    for short, longname, conf in top3:
        story.append(Paragraph(f"{longname} ({short}) — {conf*100:.1f}%", styles['Normal']))
    story.append(Spacer(1, 6))

    story.append(Paragraph('<b>All Class Probabilities</b>', styles['Heading3']))
    preds = data.get('preds') or []
    table_data = [['Short Code', 'Full Diagnosis', 'Confidence']]
    for short, longname, conf in preds:
        table_data.append([short, longname, f"{conf*100:.1f}%"])
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (2,1), (2,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
    ]))
    story.append(tbl)

    doc.build(story)
    pdf_io.seek(0)
    filename = (data.get('patient_id') or 'ecg_report') + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf'
    return send_file(pdf_io, as_attachment=True, download_name=filename, mimetype='application/pdf')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
