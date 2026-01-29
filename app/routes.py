import base64
import io
import json
import os
import smtplib
import sqlite3
import tempfile
from datetime import datetime
from email.message import EmailMessage

import cv2
import numpy as np
import pytesseract
import requests
from dotenv import load_dotenv
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, Response
from openai import OpenAI
from google import genai
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

# PDF and DOCX extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# DeepFace for accurate face recognition
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

from .models import DB_PATH, init_db

main = Blueprint("main", __name__)
load_dotenv()
init_db()


def _get_api_key():
    """Return OpenAI API key from common env names."""
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")


def _get_gemini_api_key():
    """Return Gemini API key from env."""
    return os.getenv("GEMINI_API_KEY")


def _invoke_gemini_text(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
    max_output_tokens: int = 500,
):
    """Call Gemini via Apps Script proxy using GEMINI_API_KEY and GEMINI_APPS_SCRIPT_URL."""

    api_key = _get_gemini_api_key()
    appscript_url = os.getenv("GEMINI_APPS_SCRIPT_URL")
    if not api_key or not appscript_url:
        _debug("gemini.config_missing", {"has_key": bool(api_key), "has_url": bool(appscript_url)})
        return None

    payload = {
        "apiKey": api_key,
        "systemPrompt": system_prompt,
        "userPrompt": user_prompt,
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }

    try:
        resp = requests.post(appscript_url, json=payload, timeout=25)
        if not resp.ok:
            _debug("gemini.appscript_http_error", {"status": resp.status_code, "text": resp.text[:500]})
            return None
        data = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
        text = data.get("text") or data.get("reply") or data.get("content")
        if text:
            return str(text).strip()
        # Fallback: if Apps Script returns raw string body
        if isinstance(resp.text, str) and resp.text.strip():
            return resp.text.strip()
        _debug("gemini.appscript_empty", "No text content returned")
        return None
    except Exception as exc:  # noqa: BLE001
        _debug("gemini.appscript_exception", str(exc))
        return None


def _invoke_openai_text(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
    max_output_tokens: int = 500,
):
    """Call OpenAI chat directly as a fallback when Gemini is unavailable."""

    api_key = _get_api_key()
    if not api_key:
        _debug("openai.config_missing", "No OPENAI_API_KEY found")
        return None

    client = OpenAI(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        text = completion.choices[0].message.content if completion.choices else ""
        return text.strip() if text else None
    except Exception as exc:  # noqa: BLE001
        _debug("openai.chat_error", str(exc))
        return None


def _invoke_chat_response(system_prompt: str, user_prompt: str, *, temperature: float = 0.7, max_output_tokens: int = 500):
    """Try Gemini first, then fall back to OpenAI so chat always works when one provider is down."""

    reply = _invoke_gemini_text(
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    if reply:
        return reply

    _debug("chat.fallback", "Gemini unavailable; using OpenAI")
    return _invoke_openai_text(
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def _synthesize_speech(text: str, *, voice: str = "alloy", audio_format: str = "mp3"):
    """Return (base64_audio, mime) for a short reply using OpenAI TTS; gracefully fallback on failure."""

    api_key = _get_api_key()
    if not api_key or not text:
        _debug("tts.skipped", {"has_key": bool(api_key), "has_text": bool(text)})
        return None, None

    # Keep payload small to respect API limits and response size for the browser.
    trimmed = text.strip()
    max_chars = 1800
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars]

    client = OpenAI(api_key=api_key)
    try:
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=trimmed,
            response_format=audio_format,
        )
        audio_bytes = resp.read() if hasattr(resp, "read") else bytes(resp)
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        mime = "audio/mpeg" if audio_format == "mp3" else f"audio/{audio_format}"
        _debug("tts.generated", {"bytes": len(audio_bytes)})
        return encoded, mime
    except Exception as exc:  # noqa: BLE001
        _debug("tts.error", str(exc))
        return None, None


def _transcribe_audio(file_path: str):
    """Transcribe an audio file using OpenAI Whisper; return (text, error)."""

    api_key = _get_api_key()
    if not api_key:
        return None, "Missing OPENAI_API_KEY"

    client = OpenAI(api_key=api_key)
    try:
        with open(file_path, "rb") as fh:
            resp = client.audio.transcriptions.create(model="whisper-1", file=fh)
        text = (resp.text or "").strip()
        return text, None if text else "Empty transcription"
    except Exception as exc:  # noqa: BLE001
        _debug("audio.transcribe_error", str(exc))
        return None, str(exc)


def _analyze_audio_insights(transcript: str, character: str, fir_context: str):
    """Use LLM to summarize interrogation audio; return dict with insights."""

    api_key = _get_api_key()
    if not api_key or not transcript:
        return None

    client = OpenAI(api_key=api_key)
    system_msg = (
        "You are an investigative analyst. Given a transcript and the FIR context, produce STRICT JSON with keys: "
        "summary (3-5 bullet sentences), risks (2-4 contradiction/risk notes grounded in FIR), actions (3-5 next steps), "
        "verdict (one of truthful/uncertain/lying), verdict_reason (1-2 sentences citing key contradiction or support), "
        "contradictions (array of short objects: {claim, fir_statement, assessment}), signal (0-1 float; 1 = truthful, 0 = lying). "
        "Verdict rule: if the statement clearly conflicts with the FIR (e.g., FIR says Ravi is Akhil's classmate but Ravi denies knowing Akhil), mark verdict=lying and push signal near 0-0.2. "
        "Use 'uncertain' only when evidence is thin or mixed. The risks list must reflect contradictions with FIR when present."
    )
    user_msg = (
        "Character: " + (character or "Unknown") + "\n"
        "FIR context:\n" + (fir_context or "[none]") + "\n"
        "Transcript:\n" + transcript
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.35,
            max_tokens=450,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content if completion.choices else ""
        return json.loads(raw) if raw else None
    except Exception as exc:  # noqa: BLE001
        _debug("audio.analysis_error", str(exc))
        return None


def _sample_video_frames(video_path: str, max_frames: int = 4):
    """Grab a handful of face-focused frames to keep LLM payload tiny."""

    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(total_frames // max_frames, 1)

        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None

        for idx in range(max_frames):
            target = idx * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            region = frame
            if face_cascade and not face_cascade.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                    region = frame[y : y + h, x : x + w]

            rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not ok:
                continue
            frames.append(base64.b64encode(buf).decode("ascii"))
    finally:
        cap.release()

    return frames


def _analyze_facial_truth_signals(frames_b64: list[str], transcript: str, fir_context: str, character: str):
    """Ask the LLM to read facial cues from a handful of video frames."""

    api_key = _get_api_key()
    if not api_key or not frames_b64:
        return None

    client = OpenAI(api_key=api_key)
    system_msg = (
        "You review interrogation video stills. Assess micro-expressions and body cues for honesty. "
        "Respond STRICT JSON with keys: expression_summary (2-4 bulletish sentences), cues (array of short cues), "
        "verdict (truthful/uncertain/lying), confidence (0-1 float), recommended_checks (2-3 quick suggestions). "
        "Be non-accusatory; when unsure lean to 'uncertain'."
    )

    transcript_snippet = (transcript or "")[:700]
    fir_snippet = (fir_context or "")[:700]

    content = [
        {
            "type": "text",
            "text": (
                "Character: "
                + (character or "Unknown")
                + "\nFIR cues (truncated):\n"
                + fir_snippet
                + "\nTranscript snippet:\n"
                + transcript_snippet
            ),
        }
    ]

    for frame_b64 in frames_b64[:4]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": content},
            ],
            temperature=0.25,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content if completion.choices else ""
        return json.loads(raw) if raw else None
    except Exception as exc:  # noqa: BLE001
        _debug("video.expression_error", str(exc))
        return None


def _debug(label: str, payload):
    """Lightweight debug printer to aid troubleshooting without leaking secrets."""
    try:
        print(f"[DEBUG] {label}: {payload}")
    except Exception:
        print(f"[DEBUG] {label}: <unprintable>")


def _insert_criminal(name: str, alias: str, notes: str, image_bytes: bytes, mime_type: str = "image/jpeg"):
    """Persist a criminal record with image BLOB into SQLite."""

    if not name or not image_bytes:
        raise ValueError("Name and image are required")

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO criminals (name, alias, image, mime_type, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name.strip(), alias.strip() if alias else None, image_bytes, mime_type or "image/jpeg", notes.strip() if notes else None),
        )
        conn.commit()


def _fetch_criminals(limit: int = 20):
    """Return recent criminal rows without loading BLOBs."""

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, alias, notes, created_at, mime_type, LENGTH(image) as size_bytes
            FROM criminals
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    records = []
    for row in rows:
        rec = {
            "id": row[0],
            "name": row[1],
            "alias": row[2],
            "notes": row[3],
            "created_at": row[4],
            "mime_type": row[5],
            "size_bytes": row[6],
        }
        records.append(rec)
    return records


def _decode_image(image_bytes: bytes):
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _save_temp_image(image_bytes: bytes, suffix=".jpg"):
    """Save image bytes to a temporary file and return the path."""
    if not image_bytes:
        return None
    try:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(path, "wb") as f:
            f.write(image_bytes)
        return path
    except Exception:
        return None


def _verify_faces_deepface(img1_bytes: bytes, img2_bytes: bytes):
    """Use DeepFace to verify if two images contain the same person.
    
    Returns:
        dict with 'verified' (bool), 'distance' (float), 'similarity' (float 0-1)
        or None if verification fails
    """
    if not DEEPFACE_AVAILABLE:
        return None
    
    path1 = _save_temp_image(img1_bytes)
    path2 = _save_temp_image(img2_bytes)
    
    if not path1 or not path2:
        return None
    
    try:
        # Use VGG-Face model - good balance of accuracy and speed
        # enforce_detection=False allows processing even if face not clearly detected
        result = DeepFace.verify(
            img1_path=path1,
            img2_path=path2,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
        )
        
        # DeepFace returns distance (lower = more similar)
        # Convert to similarity score (0-1, higher = more similar)
        distance = result.get("distance", 1.0)
        threshold = result.get("threshold", 0.4)
        verified = result.get("verified", False)
        
        # Normalize distance to similarity (0-1)
        # VGG-Face typically uses cosine distance with threshold ~0.4
        # Distance 0 = identical, Distance > threshold = different person
        if distance <= 0:
            similarity = 1.0
        elif distance >= 1.0:
            similarity = 0.0
        else:
            # Map distance to similarity: 0 distance = 1.0 sim, threshold distance = 0.5 sim
            similarity = max(0.0, 1.0 - (distance / (threshold * 2)))
        
        return {
            "verified": verified,
            "distance": distance,
            "threshold": threshold,
            "similarity": round(similarity, 3),
        }
    except Exception as exc:
        _debug("deepface.verify_error", str(exc))
        return None
    finally:
        # Clean up temp files
        try:
            if path1 and os.path.exists(path1):
                os.remove(path1)
            if path2 and os.path.exists(path2):
                os.remove(path2)
        except Exception:
            pass


def _get_face_embedding_deepface(image_bytes: bytes):
    """Extract face embedding using DeepFace.
    
    Returns numpy array embedding or None if extraction fails.
    """
    if not DEEPFACE_AVAILABLE:
        return None
    
    path = _save_temp_image(image_bytes)
    if not path:
        return None
    
    try:
        # Get embedding using VGG-Face
        embeddings = DeepFace.represent(
            img_path=path,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
        )
        
        if embeddings and len(embeddings) > 0:
            # Return the embedding vector (typically 2622-d for VGG-Face)
            return np.array(embeddings[0]["embedding"], dtype=np.float32)
        return None
    except Exception as exc:
        _debug("deepface.embedding_error", str(exc))
        return None
    finally:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _detect_face(gray_img):
    """Detect and return the largest face region, or None if no face found."""
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        return None
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return None
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    # Return largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return gray_img[y : y + h, x : x + w]


def _compute_lbp_histogram(gray_face, grid_x=8, grid_y=8):
    """Compute Local Binary Pattern histogram for face recognition.
    
    LBP is more discriminative for faces than ORB averaging.
    Returns a normalized histogram feature vector.
    """
    if gray_face is None:
        return None
    
    # Resize to consistent size
    face = cv2.resize(gray_face, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Compute LBP manually (8 neighbors, radius 1)
    rows, cols = face.shape
    lbp = np.zeros((rows - 2, cols - 2), dtype=np.uint8)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = face[i, j]
            code = 0
            code |= (face[i-1, j-1] >= center) << 7
            code |= (face[i-1, j  ] >= center) << 6
            code |= (face[i-1, j+1] >= center) << 5
            code |= (face[i  , j+1] >= center) << 4
            code |= (face[i+1, j+1] >= center) << 3
            code |= (face[i+1, j  ] >= center) << 2
            code |= (face[i+1, j-1] >= center) << 1
            code |= (face[i  , j-1] >= center) << 0
            lbp[i-1, j-1] = code
    
    # Compute spatial histogram (divide into grid cells)
    hist = []
    cell_h = lbp.shape[0] // grid_y
    cell_w = lbp.shape[1] // grid_x
    
    for gy in range(grid_y):
        for gx in range(grid_x):
            cell = lbp[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            h, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256))
            hist.extend(h)
    
    hist = np.array(hist, dtype=np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist


def _compute_face_features(image_bytes: bytes):
    """Extract multiple face features for robust matching.
    
    Returns dict with:
    - lbp_hist: LBP histogram (most discriminative for faces)
    - intensity_hist: grayscale intensity histogram
    - face_detected: whether a face was found
    """
    if not image_bytes:
        return None
    
    img = _decode_image(image_bytes)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try to detect face
    face_region = _detect_face(gray)
    face_detected = face_region is not None
    
    if not face_detected:
        # Use center crop as fallback
        h, w = gray.shape
        min_dim = min(h, w)
        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2
        face_region = gray[start_y : start_y + min_dim, start_x : start_x + min_dim]
    
    # Normalize size
    face_region = cv2.resize(face_region, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Apply histogram equalization for lighting normalization
    face_region = cv2.equalizeHist(face_region)
    
    # Compute LBP histogram
    lbp_hist = _compute_lbp_histogram(face_region)
    
    # Compute intensity histogram
    intensity_hist, _ = np.histogram(face_region.ravel(), bins=64, range=(0, 256))
    intensity_hist = intensity_hist.astype(np.float32)
    norm = np.linalg.norm(intensity_hist)
    if norm > 0:
        intensity_hist = intensity_hist / norm
    
    return {
        "lbp_hist": lbp_hist,
        "intensity_hist": intensity_hist,
        "face_detected": face_detected,
    }


def _compare_faces(features1, features2):
    """Compare two face feature sets and return similarity score 0-1.
    
    Uses weighted combination of LBP histogram similarity and intensity histogram.
    Returns lower scores when faces don't match well.
    """
    if features1 is None or features2 is None:
        return 0.0
    
    lbp1, lbp2 = features1.get("lbp_hist"), features2.get("lbp_hist")
    int1, int2 = features1.get("intensity_hist"), features2.get("intensity_hist")
    face1, face2 = features1.get("face_detected", False), features2.get("face_detected", False)
    
    scores = []
    
    # LBP histogram comparison (most important for face identity)
    if lbp1 is not None and lbp2 is not None:
        # Use histogram intersection for LBP
        lbp_sim = np.minimum(lbp1, lbp2).sum()
        scores.append(("lbp", lbp_sim, 0.7))  # 70% weight
    
    # Intensity histogram comparison
    if int1 is not None and int2 is not None:
        int_sim = np.minimum(int1, int2).sum()
        scores.append(("intensity", int_sim, 0.3))  # 30% weight
    
    if not scores:
        return 0.0
    
    # Weighted average
    total_weight = sum(s[2] for s in scores)
    weighted_sum = sum(s[1] * s[2] for s in scores)
    similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Apply penalty if face detection status differs
    if face1 != face2:
        similarity *= 0.7  # 30% penalty
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, similarity))


def _face_embedding_cv(image_bytes: bytes):
    """Return a compact face embedding using OpenCV ORB (no external deps).
    
    DEPRECATED: Use _compute_face_features() instead for better accuracy.
    Kept for backward compatibility.
    """

    if not image_bytes:
        return None

    img = _decode_image(image_bytes)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try to focus on the largest detected face; otherwise use the full frame.
    face_region = gray
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                face_region = gray[y : y + h, x : x + w]

    face_region = cv2.resize(face_region, (256, 256), interpolation=cv2.INTER_AREA)

    try:
        orb = cv2.ORB_create(500)
        keypoints, descriptors = orb.detectAndCompute(face_region, None)
        if descriptors is None or len(descriptors) == 0:
            return None
        # ORB descriptors are (N, 32); average to a stable 32-d vector.
        vec = descriptors.astype(np.float32).mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm
    except Exception as exc:  # noqa: BLE001
        _debug("face.orb_error", str(exc))
        return None


def _image_phash(image_bytes: bytes):
    """Compute a perceptual hash (DCT-based) for similarity matching."""

    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        block = dct[:8, :8]
        mean = block[1:, 1:].mean()
        bits = (block > mean).astype(np.uint8)
        return bits.flatten()
    except Exception as exc:  # noqa: BLE001
        _debug("phash.error", str(exc))
        return None


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    if a is None or b is None or a.shape != b.shape:
        return 64
    return int(np.count_nonzero(a != b))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def _find_best_image_match(query_bytes: bytes):
    """Return best matching criminal record using DeepFace for accurate face recognition.
    
    Uses deep learning face embeddings (VGG-Face) for reliable identity matching.
    Falls back to embedding cosine similarity if direct verification fails.
    """

    if not DEEPFACE_AVAILABLE:
        _debug("match.deepface_unavailable", "DeepFace not installed")
        return None

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, alias, notes, created_at, mime_type, image
            FROM criminals
            ORDER BY datetime(created_at) DESC
            LIMIT 100
            """
        )
        rows = cur.fetchall()

    if not rows:
        return None

    # Get embedding for query image once
    query_embedding = _get_face_embedding_deepface(query_bytes)
    
    best = None
    best_score = -1.0
    best_verified = False

    for row in rows:
        cid, name, alias, notes, created_at, mime_type, blob = row

        # Method 1: Try direct face verification (most accurate)
        verify_result = _verify_faces_deepface(query_bytes, blob)
        
        if verify_result:
            score = verify_result["similarity"]
            verified = verify_result["verified"]
            method = "DeepFace VGG-Face"
            
            if verified:
                method += " ✓ Verified"
            
            # Prioritize verified matches
            is_better = False
            if verified and not best_verified:
                is_better = True
            elif verified == best_verified and score > best_score:
                is_better = True
            elif not best_verified and score > best_score:
                is_better = True
            
            if is_better:
                best_score = score
                best_verified = verified
                best = {
                    "id": cid,
                    "name": name,
                    "alias": alias,
                    "notes": notes,
                    "created_at": created_at,
                    "mime_type": mime_type or "image/jpeg",
                    "image_b64": base64.b64encode(blob).decode("ascii"),
                    "score": round(score, 3),
                    "method": method,
                    "verified": verified,
                }
        else:
            # Method 2: Fallback to embedding cosine similarity
            if query_embedding is not None:
                db_embedding = _get_face_embedding_deepface(blob)
                if db_embedding is not None:
                    # Cosine similarity: 1 = identical, 0 = orthogonal, -1 = opposite
                    cos_sim = _cosine_similarity(query_embedding, db_embedding)
                    # Map from [-1,1] to [0,1]
                    score = (cos_sim + 1.0) / 2.0
                    
                    if score > best_score and not best_verified:
                        best_score = score
                        best = {
                            "id": cid,
                            "name": name,
                            "alias": alias,
                            "notes": notes,
                            "created_at": created_at,
                            "mime_type": mime_type or "image/jpeg",
                            "image_b64": base64.b64encode(blob).decode("ascii"),
                            "score": round(score, 3),
                            "method": "DeepFace embedding similarity",
                            "verified": False,
                        }

    # Add confidence warnings
    if best:
        if best_score < 0.3:
            best["low_confidence"] = True
            best["message"] = "Very low confidence - likely not a match"
        elif best_score < 0.5 and not best_verified:
            best["low_confidence"] = True
            best["message"] = "Low confidence - verify manually"
    
    return best


def _get_email_credentials():
    """Return email user/password from env (do not hardcode secrets)."""
    user = os.getenv("REPORT_EMAIL_USER") or os.getenv("EMAIL_USER")
    password = os.getenv("REPORT_EMAIL_PASSWORD") or os.getenv("EMAIL_PASSWORD")
    return user, password


def _is_valid_email_format(address: str) -> bool:
    """Basic RFC5322-ish email format check (does not verify mailbox exists).

    This prevents obviously invalid inputs but cannot guarantee that the
    mailbox is real or reachable – that depends on the remote mail server.
    """
    if not address or "@" not in address:
        return False
    local, _, domain = address.rpartition("@");
    if not local or not domain or "." not in domain:
        return False
    if " " in address:
        return False
    return True


def _wants_json() -> bool:
    """Heuristic to decide if the client expects a JSON response (AJAX).

    Used so /report/email can stay on the same page via fetch() while
    still supporting classic form submissions with redirects.
    """
    accept = (request.headers.get("Accept") or "").lower()
    requested_with = (request.headers.get("X-Requested-With") or "").lower()
    return "application/json" in accept or requested_with == "xmlhttprequest"


def _generate_pdf_report(
    cleaned_text: str,
    characters: list,
    character_profiles: list,
    general_questions: list,
    roadmap_steps: list,
    locations: list,
) -> bytes:
    """Generate a compact PDF summary for the FIR analysis and return bytes."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin_x = 20 * mm
    margin_top = height - 25 * mm
    margin_bottom = 20 * mm

    text_obj = pdf.beginText()
    text_obj.setTextOrigin(margin_x, margin_top)
    text_obj.setFont("Helvetica-Bold", 16)
    text_obj.textLine("JUSTICE AI - FIR Investigation Report")
    text_obj.setFont("Helvetica", 9)
    text_obj.textLine("")
    text_obj.textLine(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    text_obj.textLine("")

    def add_section(title: str, body_lines: list[str]):
        nonlocal text_obj
        if text_obj.getY() < margin_bottom + 40:
            pdf.drawText(text_obj)
            pdf.showPage()
            text_obj = pdf.beginText()
            text_obj.setTextOrigin(margin_x, margin_top)
        text_obj.setFont("Helvetica-Bold", 12)
        text_obj.textLine(title)
        text_obj.setFont("Helvetica", 9)
        text_obj.textLine("")
        max_width = width - 2 * margin_x
        for raw in body_lines:
            if not raw:
                text_obj.textLine("")
                continue
            wrapped = simpleSplit(raw, "Helvetica", 9, max_width)
            for line in wrapped:
                if text_obj.getY() < margin_bottom:
                    pdf.drawText(text_obj)
                    pdf.showPage()
                    text_obj = pdf.beginText()
                    text_obj.setTextOrigin(margin_x, margin_top)
                    text_obj.setFont("Helvetica", 9)
                text_obj.textLine(line)
        text_obj.textLine("")

    # Section: FIR text (trim to keep file small)
    snippet = (cleaned_text or "No text provided.").strip()
    if len(snippet) > 4000:
        snippet = snippet[:4000] + "\n[... truncated for brevity in PDF ...]"
    add_section("FIR Text (cleaned)", snippet.splitlines() or [snippet])

    # Section: Characters and questions
    if characters:
        lines = []
        for person in characters:
            name = str(person.get("name", "Unknown")).strip()
            qs = person.get("questions") or []
            lines.append(f"- {name}:")
            for q in qs:
                lines.append(f"   • {q}")
            lines.append("")
    else:
        lines = ["No specific characters detected; using general questions only."]
    add_section("Character-based Questions", lines)

    # Section: General questions
    if general_questions:
        g_lines = [f"• {q}" for q in general_questions]
    else:
        g_lines = ["No general questions available."]
    add_section("General Investigative Questions", g_lines)

    # Section: Character profiles
    if character_profiles:
        p_lines = []
        for profile in character_profiles:
            p_lines.append(f"- {profile.get('name', 'Unknown')}")
            p_lines.append(f"   Phone: {profile.get('phone', 'Not captured')}")
            p_lines.append(f"   Address: {profile.get('address', 'Not captured')}")
            notes = profile.get("notes") or "No additional notes found"
            p_lines.append(f"   Notes: {notes}")
            p_lines.append("")
    else:
        p_lines = ["No character profiles could be extracted from the FIR text."]
    add_section("Character Profiles", p_lines)

    # Section: Roadmap
    if roadmap_steps:
        r_lines = []
        for idx, step in enumerate(roadmap_steps, start=1):
            r_lines.append(f"{idx}. {step.get('title', 'Untitled step')}")
            detail = step.get("detail") or ""
            if detail:
                r_lines.append(f"   {detail}")
            r_lines.append("")
    else:
        r_lines = ["No roadmap steps available."]
    add_section("Investigation Roadmap", r_lines)

    # Section: Locations
    if locations:
        loc_lines = []
        for loc in locations:
            loc_lines.append(f"- {loc.get('title', 'Location')}: {loc.get('address', '')}")
    else:
        loc_lines = ["No locations were detected in the FIR text."]
    add_section("Locations / Addresses", loc_lines)

    pdf.drawText(text_obj)
    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.read()

@main.route("/")
def index():
    return render_template("landing.html")


@main.route("/upload", methods=["POST"])
def upload():
    # Accept both "image" (used by dashboard fetch) and "file" (generic form name)
    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename.strip() == "":
        flash("Please choose a file to upload.", "error")
        return redirect(url_for("main.index"))

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    
    # Supported file types for research paper reviewer
    supported_extensions = {".pdf", ".doc", ".docx"}
    
    if ext not in supported_extensions:
        flash("Unsupported file type. Please upload a PDF or Word document.", "error")
        return redirect(url_for("main.index"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    text = ""
    try:
        if ext == ".pdf":
            text = extract_text_from_pdf(temp_path)
        elif ext in {".doc", ".docx"}:
            text = extract_text_from_docx(temp_path)
        else:
            text = "Unsupported file format."
    except Exception as exc:  # noqa: BLE001
        flash(f"Text extraction failed: {exc}", "error")
        return redirect(url_for("main.index"))
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    extracted_text = text.strip() if text else "No text detected in the document."

    _debug("upload.extracted_text_len", len(extracted_text))

    return render_template(
        "results.html",
        filename=filename,
        extracted_text=extracted_text,
        word_count=len(extracted_text.split()),
        char_count=len(extracted_text),
    )


@main.route("/rescan", methods=["POST"])
def rescan():
    edited_text = (request.form.get("edited_text") or "").strip()
    if not edited_text:
        flash("Text is empty. Please edit the extracted content or upload again.", "error")
        return redirect(url_for("main.index"))

    # Use the user-edited text as the new source of truth; keep a cleaned copy for analysis.
    user_text = edited_text
    cleaned_text = format_text(user_text)
    analysis_text = cleaned_text.strip() if cleaned_text else user_text

    _debug("rescan.user_text_len", len(user_text))
    _debug("rescan.cleaned_text_len", len(cleaned_text))

    characters, general_questions = build_questions(analysis_text)
    character_profiles = extract_character_profiles(analysis_text)
    roadmap_steps = build_roadmap(analysis_text, characters, general_questions)
    locations = extract_locations(analysis_text)
    legal_sections, legal_audit = build_judicial_sections(analysis_text)

    _debug("rescan.characters_count", len(characters))
    _debug("rescan.general_questions_count", len(general_questions))
    _debug("rescan.profiles_count", len(character_profiles))
    _debug("rescan.character_profiles_data", character_profiles)
    _debug("rescan.roadmap_steps_count", len(roadmap_steps))
    _debug("rescan.locations_count", len(locations))
    _debug("rescan.legal_sections_count", len(legal_sections))
    _debug("rescan.legal_audit", legal_audit)

    return render_template(
        "dashboard.html",
        # Show the exact text the user edited to avoid unexpected rewrites.
        extracted_text=user_text,
        character_questions=characters,
        character_profiles=character_profiles,
        general_questions=general_questions,
        roadmap_steps=roadmap_steps,
        locations=locations,
        legal_sections=legal_sections,
        legal_audit=legal_audit,
    )


@main.route("/admin", methods=["GET", "POST"])
def admin_panel():
    """Simple admin panel to add criminal images into SQLite."""

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        alias = (request.form.get("alias") or "").strip()
        notes = (request.form.get("notes") or "").strip()
        image_file = request.files.get("image")

        if not name:
            flash("Name is required.", "error")
            return redirect(url_for("main.admin_panel"))

        if not image_file or image_file.filename.strip() == "":
            flash("Please choose an image file to upload.", "error")
            return redirect(url_for("main.admin_panel"))

        mime = image_file.mimetype or "image/jpeg"
        if not mime.startswith("image/"):
            flash("Only image uploads are allowed.", "error")
            return redirect(url_for("main.admin_panel"))

        image_bytes = image_file.read()
        max_bytes = 6 * 1024 * 1024  # 6 MB guardrail
        if len(image_bytes) > max_bytes:
            flash("Image is too large. Please keep it under 6 MB.", "error")
            return redirect(url_for("main.admin_panel"))

        try:
            _insert_criminal(name=name, alias=alias, notes=notes, image_bytes=image_bytes, mime_type=mime)
            flash("Criminal record saved.", "success")
        except Exception as exc:  # noqa: BLE001
            _debug("admin.insert_error", str(exc))
            flash("Failed to save record. Please try again.", "error")

        return redirect(url_for("main.admin_panel"))

    records = _fetch_criminals(limit=30)
    return render_template("admin.html", records=records, db_path=str(DB_PATH))


@main.route("/match-image", methods=["POST"])
def match_image():
    """Compare an uploaded image against stored criminal images using ORB embeddings with pHash fallback."""

    # Accept both "image" (used by dashboard fetch) and "file" (generic form name)
    file = request.files.get("image") or request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No image provided"}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
        return jsonify({"error": "Unsupported image type"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty image"}), 400
    if len(image_bytes) > 6 * 1024 * 1024:
        return jsonify({"error": "Image too large (6MB max)"}), 400

    best = _find_best_image_match(image_bytes)
    if not best:
        return jsonify({"match": None, "message": "No matches available in the database"}), 200

    return jsonify({"match": best}), 200


@main.route("/chat", methods=["POST"])
def chat():
    """Handle chatbot messages and return AI-generated responses based on FIR context."""
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    fir_context = (data.get("context") or "").strip()

    if not user_message:
        return jsonify({"reply": "Please enter a message."}), 400

    system_prompt = """You are an AI Investigation Advisor assistant for law enforcement. You have analyzed an FIR (First Information Report) document and can answer questions about the case.

Your role:
- Provide helpful investigative insights and suggestions
- Answer questions about the case details, characters, timeline, and locations
- Suggest investigative approaches and next steps
- Help identify potential leads or inconsistencies
- Be professional, precise, and objective

Important guidelines:
- Only provide advisory information, not legal advice
- Base your responses on the FIR content provided
- If information is not available in the FIR, say so clearly
- Be concise but thorough
- Maintain confidentiality and professionalism

FIR Document Content:
"""

    try:
        prompt = f"{system_prompt}{fir_context}\n\nUser message:\n{user_message}"
        reply = _invoke_chat_response(
            system_prompt,
            prompt,
            temperature=0.7,
            max_output_tokens=500,
        )
        if not reply:
            return jsonify({"reply": "AI service is not configured. Please contact the administrator."}), 500
        reply = reply.strip()
        _debug("chat.reply_length", len(reply))
        audio_b64, mime = _synthesize_speech(reply)
        payload = {"reply": reply, "audio": audio_b64, "mime": mime}
        return jsonify(payload)
    except Exception as exc:
        _debug("chat.error", str(exc))
        return jsonify({"reply": "I apologize, but I encountered an error processing your request. Please try again."}), 500


@main.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    """Transcribe an interrogation audio and extract insights for a selected character."""

    audio_file = request.files.get("audio")
    character = (request.form.get("character") or "General").strip() or "General"
    fir_context = (request.form.get("context") or "").strip()

    if not audio_file or audio_file.filename.strip() == "":
        return jsonify({"error": "Please attach an audio file."}), 400

    ext = os.path.splitext(audio_file.filename)[1].lower()
    audio_exts = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    is_video = ext in video_exts
    if ext not in audio_exts | video_exts:
        return jsonify({"error": "Unsupported clip type. Use audio (wav/mp3/m4a/aac/ogg/flac/webm) or video (mp4/mov/mkv/avi/webm/m4v)."}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        audio_file.save(tmp.name)
        temp_path = tmp.name

    transcript = None
    try:
        transcript, err = _transcribe_audio(temp_path)
        if err or not transcript:
            return jsonify({"error": f"Transcription failed: {err or 'no text'}"}), 500

        # Debug visibility: show transcript snippet for troubleshooting verdict issues
        _debug("audio.transcript_text", (transcript or "")[:800])

        insights = _analyze_audio_insights(transcript, character, fir_context) or {}
        _debug("audio.insights_raw", insights)

        video_frames = []
        video_expression = None
        if is_video:
            video_frames = _sample_video_frames(temp_path, max_frames=4)
            _debug("video.frames_captured", len(video_frames))
            video_expression = _analyze_facial_truth_signals(video_frames, transcript, fir_context, character)
            _debug("video.expression", video_expression)
            if video_expression:
                try:
                    expr_conf = float(video_expression.get("confidence", 0.5))
                except (TypeError, ValueError):
                    expr_conf = 0.5
                try:
                    audio_signal = float(insights.get("signal")) if insights and "signal" in insights else 0.5
                except (TypeError, ValueError):
                    audio_signal = 0.5
                combined_signal = max(0.0, min(1.0, (audio_signal + expr_conf) / 2))
                insights["signal"] = combined_signal
                insights["facial_verdict"] = video_expression.get("verdict")
                insights["facial_confidence"] = expr_conf
                insights["facial_cues"] = video_expression.get("cues")
                insights["facial_summary"] = video_expression.get("expression_summary")

        payload = {
            "transcript": transcript,
            "insights": insights,
            "video": {
                "is_video": is_video,
                "frames": len(video_frames),
                "expression": video_expression,
            },
        }
        return jsonify(payload)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@main.route("/report/pdf", methods=["POST"])
def download_report_pdf():
    """Generate a PDF report for the current FIR context and return it as a download."""
    raw_text = (request.form.get("report_text") or "").strip()
    if not raw_text:
        flash("FIR text is empty. Please ensure there is content before generating a report.", "error")
        # Always redirect to a GET-safe route to avoid 405 errors from POST-only pages.
        return redirect(url_for("main.index"))

    cleaned = format_text(raw_text)
    analysis_text = cleaned.strip() if cleaned else raw_text

    characters, general_questions = build_questions(analysis_text)
    character_profiles = extract_character_profiles(analysis_text)
    roadmap_steps = build_roadmap(analysis_text, characters, general_questions)
    locations = extract_locations(analysis_text)

    try:
        pdf_bytes = _generate_pdf_report(
            analysis_text,
            characters,
            character_profiles,
            general_questions,
            roadmap_steps,
            locations,
        )
        _debug("report.pdf_size_bytes", len(pdf_bytes))
        headers = {
            "Content-Type": "application/pdf",
            "Content-Disposition": "attachment; filename=firm_report.pdf",
        }
        return Response(pdf_bytes, headers=headers)
    except Exception as exc:  # noqa: BLE001
        _debug("report.pdf_error", str(exc))
        flash("Failed to generate PDF report. Please try again.", "error")
        return redirect(url_for("main.index"))


def _send_report_email(recipient: str, pdf_bytes: bytes):
    """Send the generated PDF report to the provided email address via SMTP."""
    user, password = _get_email_credentials()
    _debug("report.email_user_present", bool(user))
    if not user or not password:
        raise RuntimeError("Email credentials are not configured on the server.")

    msg = EmailMessage()
    msg["Subject"] = "JUSTICE AI - FIR Investigation Report"
    msg["From"] = user
    msg["To"] = recipient
    msg.set_content(
        """Attached is the PDF report generated from the FIR analysis in JUSTICE AI.\n\nThis is an automated message."""
    )
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename="fir_report.pdf")

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)


@main.route("/report/email", methods=["POST"])
def email_report_pdf():
    """Generate a PDF report and email it to the provided address."""
    raw_text = (request.form.get("report_text") or "").strip()
    recipient = (request.form.get("email") or "").strip()

    if not raw_text:
        msg = "FIR text is empty. Please ensure there is content before generating a report."
        if _wants_json():
            return jsonify({"status": "error", "message": msg}), 400
        flash(msg, "error")
        return redirect(url_for("main.index"))

    if not _is_valid_email_format(recipient):
        msg = "Please provide a valid, correctly formatted email address."
        if _wants_json():
            return jsonify({"status": "error", "message": msg}), 400
        flash(msg, "error")
        return redirect(url_for("main.index"))

    cleaned = format_text(raw_text)
    analysis_text = cleaned.strip() if cleaned else raw_text

    characters, general_questions = build_questions(analysis_text)
    character_profiles = extract_character_profiles(analysis_text)
    roadmap_steps = build_roadmap(analysis_text, characters, general_questions)
    locations = extract_locations(analysis_text)

    try:
        pdf_bytes = _generate_pdf_report(
            analysis_text,
            characters,
            character_profiles,
            general_questions,
            roadmap_steps,
            locations,
        )
        _send_report_email(recipient, pdf_bytes)
        # NOTE: SMTP cannot guarantee that the mailbox actually exists –
        # it only confirms that our server accepted the message for delivery.
        msg = "Report Sent Successfully!"
        if _wants_json():
            return jsonify({"status": "ok", "message": msg})
        flash(msg, "success")
    except Exception as exc:  # noqa: BLE001
        _debug("report.email_error", str(exc))
        msg = "Failed to send email report. Please verify the email configuration and try again."
        if _wants_json():
            return jsonify({"status": "error", "message": msg}), 500
        flash(msg, "error")

    return redirect(url_for("main.index"))


def build_questions(extracted_text: str):
    """Use OpenAI to draft questions by character (names in the text); fall back to generic prompts."""
    default_general = [
        "Summarize the incident timeline in under 5 bullet points.",
        "What evidence is already collected and what is pending?",
        "Identify any contradictions in the statements within the FIR.",
        "List immediate investigative next steps for the case team.",
    ]

    api_key = _get_api_key()
    if not api_key or not extracted_text:
        return [], default_general

    client = OpenAI(api_key=api_key)
    system_msg = (
        "You are an assistant that extracts character/person names (people mentioned) from an FIR text "
        "and drafts precise investigative questions per character. Respond ONLY in JSON "
        "with keys characters (array of {name, questions}) and general_questions (array)."
    )
    user_msg = (
        "FIR text:\n" + extracted_text + "\n"
        "Return 2-4 questions per character. If no clear characters, leave the characters array empty "
        "and put 4 high-value general investigative questions in general_questions."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content if completion.choices else ""
        data = json.loads(raw_content) if raw_content else {}
        characters = data.get("characters") or []
        general = data.get("general_questions") or default_general

        # Normalize shape
        normalized_chars = []
        for item in characters:
            name = (item or {}).get("name")
            qs = (item or {}).get("questions") or []
            if name and qs:
                normalized_chars.append({"name": str(name).strip(), "questions": [str(q).strip() for q in qs if q]})

        return normalized_chars, general
    except Exception:
        return [], default_general


def build_judicial_sections(extracted_text: str):
    """Ask OpenAI for matching IPC/CrPC sections and fall back to curated suggestions."""

    fallback = [
        {
            "code": "IPC 154",
            "statute": "Information in cognizable cases",
            "summary": "Ensures FIR registration when cognizable offences are alleged, safeguarding complainant rights.",
            "reason": "Always cite to validate that the FIR is procedurally compliant before downstream action.",
            "bailable": "Not applicable",
            "punishment": "Sets duty for police officers; no direct punishment but non-compliance invites departmental action.",
            "confidence": 0.38,
            "origin": "fallback",
        },
        {
            "code": "IPC 420",
            "statute": "Cheating and dishonestly inducing delivery of property",
            "summary": "Covers deceitful acts where victims are induced to hand over money, valuables, or signatures.",
            "reason": "Trigger when FIR narrates misrepresentation, forged promises, or siphoning of funds.",
            "bailable": "Non-bailable",
            "punishment": "Up to 7 years imprisonment and fine.",
            "confidence": 0.41,
            "origin": "fallback",
        },
        {
            "code": "CrPC 41",
            "statute": "When police may arrest without warrant",
            "summary": "Guides lawful arrest only when necessity criteria are met, protecting Article 21 rights.",
            "reason": "Remind IOs to document reasons for custodial steps cited inside the FIR narrative.",
            "bailable": "Context dependent",
            "punishment": "Procedural safeguard; non-compliance invites judicial scrutiny.",
            "confidence": 0.33,
            "origin": "fallback",
        },
    ]

    audit = {
        "status": "fallback",
        "model": None,
        "notes": "Showing curated starter sections.",
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    if not extracted_text:
        audit["notes"] = "No FIR text provided; showing curated starter sections."
        return fallback, audit

    api_key = _get_api_key()
    if not api_key:
        audit["notes"] = "OPENAI_API_KEY missing; using curated fallback list."
        return fallback, audit

    client = OpenAI(api_key=api_key)
    system_msg = (
        "You are a senior Indian criminal-law researcher. Given an FIR narrative, identify the top Indian statutes "
        "(IPC, CrPC, IT Act, Evidence Act, etc.) that plausibly apply. Respond ONLY in JSON with key 'sections' as an array. "
        "Each item must include: code (e.g., 'IPC 420'), statute (short title), summary (1-2 lines), reason (why it matches), "
        "bailable (Yes/No/Depends), punishment (max penalty), confidence (0-1 float), and optional keywords (array). "
        "Limit to 3-5 items ordered by relevance."
    )
    user_msg = (
        "FIR text:\n" + extracted_text + "\n"
        "Focus on actual legal applicability; prefer IPC/CrPC sections unless specialised laws are explicitly indicated."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.25,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content if completion.choices else ""
        data = json.loads(raw_content) if raw_content else {}
        sections_raw = data.get("sections") or []

        sections = []
        for item in sections_raw:
            code = (item or {}).get("code")
            statute = (item or {}).get("statute") or (item or {}).get("title")
            summary = (item or {}).get("summary")
            reason = (item or {}).get("reason")
            bailable = (item or {}).get("bailable") or "Depends"
            punishment = (item or {}).get("punishment") or "As per statute"
            confidence = (item or {}).get("confidence")
            keywords = (item or {}).get("keywords") or []
            if code and statute and summary:
                try:
                    conf_value = float(confidence) if confidence is not None else None
                except (TypeError, ValueError):
                    conf_value = None
                sections.append(
                    {
                        "code": str(code).strip(),
                        "statute": str(statute).strip(),
                        "summary": str(summary).strip(),
                        "reason": str(reason).strip() if reason else "LLM rationale unavailable.",
                        "bailable": str(bailable).strip(),
                        "punishment": str(punishment).strip(),
                        "confidence": conf_value,
                        "keywords": [str(k).strip() for k in keywords if k],
                        "origin": "openai",
                    }
                )

        if sections:
            audit = {
                "status": "llm",
                "model": "gpt-4o-mini",
                "notes": data.get("disclaimer") or "Generated via OpenAI using FIR narrative.",
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }
            return sections, audit
    except Exception as exc:  # noqa: BLE001
        _debug("legal_sections.error", str(exc))

    audit["notes"] = "OpenAI response unavailable; using curated fallback list."
    return fallback, audit


def extract_locations(extracted_text: str):
    """Extract probable addresses/places from FIR text with titles."""
    api_key = _get_api_key()
    if not api_key or not extracted_text:
        return []

    client = OpenAI(api_key=api_key)
    system_msg = (
        "You extract location mentions (addresses, landmarks, stations, villages, streets) from FIR text. "
        "Respond ONLY in JSON with key locations as an array of objects {title, address}. "
        "Keep 1-5 items, concise human-readable titles, and full address text when available."
    )
    user_msg = "FIR text:\n" + extracted_text

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content if completion.choices else ""
        data = json.loads(raw_content) if raw_content else {}
        items = data.get("locations") or []

        cleaned = []
        for item in items:
            title = (item or {}).get("title")
            address = (item or {}).get("address")
            if title and address:
                cleaned.append({"title": str(title).strip(), "address": str(address).strip()})
        return cleaned
    except Exception:
        return []


def extract_character_profiles(extracted_text: str):
    """Extract character-level info (phone, address, notes) from FIR text."""
    api_key = _get_api_key()
    _debug("extract_character_profiles.api_key_present", bool(api_key))
    if not api_key or not extracted_text:
        _debug("extract_character_profiles.early_return", "no api_key or text")
        return []

    client = OpenAI(api_key=api_key)
    system_msg = (
        "You are an expert at extracting person-level details from FIR (First Information Report) text. "
        "For EVERY person mentioned, extract their name, phone number (mobile/landline), residential/office address, "
        "and any additional notes (role, relation, occupation, age, etc.). "
        "If a detail is explicitly mentioned in the text, you MUST include it. "
        "Respond ONLY in JSON with key 'characters' containing an array of objects: "
        "{name: string, phone: string or null, address: string or null, notes: string or null}. "
        "Extract ALL persons mentioned, typically 1-5 people."
    )
    user_msg = "FIR text:\n" + extracted_text + "\n\nExtract ALL person details including any phone numbers, addresses, and notes mentioned."

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content if completion.choices else ""
        _debug("extract_character_profiles.raw_response", raw_content[:500] if raw_content else "<empty>")
        data = json.loads(raw_content) if raw_content else {}
        items = data.get("characters") or []
        _debug("extract_character_profiles.items_count", len(items))

        cleaned = []
        for item in items:
            _debug("extract_character_profiles.item", item)
            name = (item or {}).get("name")
            phone = (item or {}).get("phone")
            address = (item or {}).get("address")
            notes = (item or {}).get("notes")
            if name:
                profile = {
                    "name": str(name).strip(),
                    "phone": str(phone).strip() if phone else "Not captured in FIR",
                    "address": str(address).strip() if address else "Not captured in FIR",
                    "notes": str(notes).strip() if notes else "No additional notes found",
                }
                _debug("extract_character_profiles.profile", profile)
                cleaned.append(profile)
        _debug("extract_character_profiles.final_count", len(cleaned))
        return cleaned
    except Exception as e:
        _debug("extract_character_profiles.exception", str(e))
        return []


def build_roadmap(extracted_text: str, characters: list, general_questions: list):
    """Generate an investigation roadmap based on FIR text and drafted questions."""
    default_steps = [
        {"title": "Stabilize scene", "detail": "Secure location, preserve evidence, and ensure officer/bodycam logging."},
        {"title": "Collect statements", "detail": "Interview complainant and nearest witnesses; capture contradictions early."},
        {"title": "Evidence sweep", "detail": "Gather CCTV, digital traces, call records, and physical items in a strict chain-of-custody."},
        {"title": "Corroborate timeline", "detail": "Align statements with evidence to confirm sequence of events and identify gaps."},
        {"title": "Targeted follow-ups", "detail": "Schedule re-interviews for inconsistencies; run background checks on key persons."},
        {"title": "Action items", "detail": "Issue summons, warrants (if applicable), and set deadlines for forensic reports."},
    ]

    api_key = _get_api_key()
    if not api_key or not extracted_text:
        return default_steps

    client = OpenAI(api_key=api_key)
    system_msg = (
        "You design a concise investigation roadmap from FIR text and drafted questions. "
        "Return STRICT JSON with key steps (array of {title, detail}). Keep 5-8 steps, each actionable and chronological."
    )

    questions_context = {
        "characters": characters,
        "general_questions": general_questions,
    }

    user_msg = (
        "FIR text:\n" + extracted_text + "\n"
        "Drafted questions (context):\n" + json.dumps(questions_context, ensure_ascii=False) + "\n"
        "Produce steps that state where to start and the next efficient investigative moves."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.35,
            max_tokens=700,
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content if completion.choices else ""
        data = json.loads(raw_content) if raw_content else {}
        steps = data.get("steps") or []

        normalized = []
        for item in steps:
            title = (item or {}).get("title")
            detail = (item or {}).get("detail")
            if title and detail:
                normalized.append({"title": str(title).strip(), "detail": str(detail).strip()})

        return normalized or default_steps
    except Exception:
        return default_steps


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    if not PDFPLUMBER_AVAILABLE:
        raise ImportError("pdfplumber is not installed. Please run: pip install pdfplumber")
    
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:
        raise ValueError(f"Failed to extract text from PDF: {exc}")
    
    return "\n\n".join(text_parts)


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx is not installed. Please run: pip install python-docx")
    
    text_parts = []
    try:
        doc = DocxDocument(docx_path)
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
    except Exception as exc:
        raise ValueError(f"Failed to extract text from DOCX: {exc}")
    
    return "\n\n".join(text_parts)


def format_text(raw_text: str) -> str:
    """Lightly format OCR output via OpenAI for readability."""
    api_key = _get_api_key()
    if not api_key or not raw_text:
        return raw_text

    client = OpenAI(api_key=api_key)
    system_msg = (
        "You clean OCR output from FIR images. Fix casing/spaces, keep original meaning, "
        "and return plain text only (no JSON, no Markdown). Preserve names, dates, and numbers."
    )
    user_msg = "OCR text:\n" + raw_text

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        cleaned = completion.choices[0].message.content if completion.choices else ""
        return cleaned.strip() if cleaned else raw_text
    except Exception:
        return raw_text


def extract_text(image_path: str) -> str:
    """High-confidence OCR pipeline with upscale, denoise, and threshold."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to read the uploaded image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale if small to help OCR
    h, w = gray.shape
    scale = 2.0 if max(h, w) < 1200 else 1.0
    if scale != 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Mild denoise
    denoised = cv2.fastNlMeansDenoising(equalized, h=10, templateWindowSize=7, searchWindowSize=21)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )

    # Invert if background is dark
    white_ratio = np.mean(binary) / 255.0
    if white_ratio < 0.5:
        binary = cv2.bitwise_not(binary)

    config = "--oem 3 --psm 6"  # LSTM OCR, assume a block of text

    # Run OCR on both processed and grayscale; pick longer confident text
    text_bin = pytesseract.image_to_string(binary, config=config)
    text_gray = pytesseract.image_to_string(denoised, config=config)

    candidate = text_bin if len(text_bin.strip()) >= len(text_gray.strip()) else text_gray
    cleaned = candidate.strip()
    return cleaned or "No text detected."