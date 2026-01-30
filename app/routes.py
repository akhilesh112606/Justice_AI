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
    pdf_base64 = None
    pdf_positions = None
    is_pdf = ext == ".pdf"
    
    try:
        if ext == ".pdf":
            text = extract_text_from_pdf(temp_path)
            # Extract text positions for accurate highlight mapping
            pdf_positions = extract_pdf_text_with_positions(temp_path)
            # Read PDF file as base64 for frontend rendering
            with open(temp_path, "rb") as pdf_file:
                import base64
                pdf_base64 = base64.b64encode(pdf_file.read()).decode("utf-8")
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

    # Analyze formatting suggestions with position data
    formatting_data = analyze_formatting_suggestions(extracted_text, filename, pdf_positions)
    _debug("upload.formatting_score", formatting_data.get("score", 0))

    # Analyze plagiarism
    plagiarism_data = analyze_plagiarism(extracted_text, filename)
    # Optionally enhance with LLM if score is concerning
    if plagiarism_data.get("overall_score", 100) < 80:
        plagiarism_data = enhance_plagiarism_with_llm(extracted_text, plagiarism_data)
    _debug("upload.plagiarism_score", plagiarism_data.get("overall_score", 0))

    # Analyze citations
    citation_data = analyze_citations(extracted_text, filename)
    _debug("upload.citation_score", citation_data.get("citation_score", 0))

    # Analyze rewrite opportunities
    rewrite_data = analyze_rewrite_opportunities(
        extracted_text,
        formatting_data=formatting_data,
        plagiarism_data=plagiarism_data,
        citation_data=citation_data
    )
    _debug("upload.rewrite_opportunities", rewrite_data.get("stats", {}).get("total_opportunities", 0))

    return render_template(
        "results.html",
        filename=filename,
        extracted_text=extracted_text,
        word_count=len(extracted_text.split()),
        char_count=len(extracted_text),
        formatting_data=formatting_data,
        plagiarism_data=plagiarism_data,
        citation_data=citation_data,
        rewrite_data=rewrite_data,
        pdf_base64=pdf_base64,
        is_pdf=is_pdf,
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
    """Handle chatbot messages and return AI-generated responses based on research paper context."""
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    paper_context = (data.get("context") or "").strip()

    if not user_message:
        return jsonify({"reply": "Please enter a message."}), 400

    system_prompt = """You are an AI Research Paper Review Assistant named "ReviewerAI". You have access to the full analysis of a research paper including its text content, plagiarism analysis, citation analysis, formatting suggestions, and rewrite opportunities.

Your role:
- Provide specific, actionable feedback on academic writing and paper structure
- Answer questions about the paper's content, methodology, findings, and analysis results
- Explain plagiarism findings in detail and suggest how to address them
- Help improve citations, reference formatting, and academic integrity
- Suggest concrete ways to strengthen arguments and improve clarity
- Guide users through improving specific sections of their paper
- Provide writing tips and academic best practices

Important guidelines:
- Always reference the specific analysis data provided when answering
- Be specific - mention actual scores, issues, and suggestions from the analysis
- Focus on constructive, actionable feedback with examples when possible
- If the user asks about something not in the analysis, say so clearly
- Be encouraging but honest about areas needing improvement
- Prioritize high-impact improvements first
- Use markdown formatting for clear, structured responses

You have access to:
- The paper's extracted text content
- Plagiarism analysis with originality scores and issues
- Citation analysis with reference quality metrics
- Formatting suggestions for each page
- Rewrite opportunities with priority levels

"""

    try:
        prompt = f"{system_prompt}\n{paper_context}\n\n---\nUser Question: {user_message}\n\nProvide a helpful, specific response based on the paper analysis above:"
        reply = _invoke_chat_response(
            system_prompt,
            prompt,
            temperature=0.7,
            max_output_tokens=800,
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


@main.route("/generate-rewrite", methods=["POST"])
def generate_rewrite():
    """Generate a rewrite for a specific issue or section.
    
    Expects JSON with:
    - type: 'targeted' | 'section' | 'contributions'
    - For targeted: original_text, issue_type, issue_description, suggested_fix
    - For section: section_name, section_text, feedback, full_context
    - For contributions: text, current_contributions
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        rewrite_type = data.get("type", "targeted")
        
        if rewrite_type == "targeted":
            result = generate_targeted_rewrite(
                original_text=data.get("original_text", ""),
                issue_type=data.get("issue_type", "general"),
                issue_description=data.get("issue_description", ""),
                suggested_fix=data.get("suggested_fix", "")
            )
        elif rewrite_type == "section":
            result = generate_section_rewrite(
                section_name=data.get("section_name", "general"),
                section_text=data.get("section_text", ""),
                feedback=data.get("feedback", ""),
                full_context=data.get("full_context", "")
            )
        elif rewrite_type == "contributions":
            result = generate_contribution_bullets(
                text=data.get("text", ""),
                current_contributions=data.get("current_contributions", [])
            )
        else:
            return jsonify({"error": f"Unknown rewrite type: {rewrite_type}"}), 400
        
        return jsonify(result)
        
    except Exception as exc:
        _debug("generate_rewrite.error", str(exc))
        return jsonify({"error": str(exc)}), 500


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


def extract_pdf_text_with_positions(pdf_path: str) -> dict:
    """Extract text from PDF with position information for each page.
    
    Returns dict with:
    - pages: list of page data with words, lines, and bounding boxes
    - page_dimensions: list of (width, height) for each page
    """
    if not PDFPLUMBER_AVAILABLE:
        return {"pages": [], "page_dimensions": []}
    
    result = {"pages": [], "page_dimensions": []}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_width = float(page.width)
                page_height = float(page.height)
                result["page_dimensions"].append((page_width, page_height))
                
                page_data = {
                    "page_number": page_num,
                    "width": page_width,
                    "height": page_height,
                    "words": [],
                    "lines": [],
                    "text": ""
                }
                
                # Extract words with bounding boxes
                words = page.extract_words(keep_blank_chars=True, use_text_flow=True)
                for word in words:
                    page_data["words"].append({
                        "text": word.get("text", ""),
                        "x0": word.get("x0", 0) / page_width,  # Normalize to 0-1
                        "y0": word.get("top", 0) / page_height,
                        "x1": word.get("x1", 0) / page_width,
                        "y1": word.get("bottom", 0) / page_height,
                    })
                
                # Group words into lines (same y position)
                if words:
                    current_line = []
                    current_y = None
                    tolerance = 5  # pixels tolerance for same line
                    
                    for word in sorted(words, key=lambda w: (w.get("top", 0), w.get("x0", 0))):
                        word_y = word.get("top", 0)
                        if current_y is None or abs(word_y - current_y) <= tolerance:
                            current_line.append(word)
                            current_y = word_y if current_y is None else current_y
                        else:
                            if current_line:
                                line_text = " ".join(w.get("text", "") for w in current_line)
                                page_data["lines"].append({
                                    "text": line_text,
                                    "x0": min(w.get("x0", 0) for w in current_line) / page_width,
                                    "y0": min(w.get("top", 0) for w in current_line) / page_height,
                                    "x1": max(w.get("x1", 0) for w in current_line) / page_width,
                                    "y1": max(w.get("bottom", 0) for w in current_line) / page_height,
                                    "word_count": len(current_line)
                                })
                            current_line = [word]
                            current_y = word_y
                    
                    # Don't forget the last line
                    if current_line:
                        line_text = " ".join(w.get("text", "") for w in current_line)
                        page_data["lines"].append({
                            "text": line_text,
                            "x0": min(w.get("x0", 0) for w in current_line) / page_width,
                            "y0": min(w.get("top", 0) for w in current_line) / page_height,
                            "x1": max(w.get("x1", 0) for w in current_line) / page_width,
                            "y1": max(w.get("bottom", 0) for w in current_line) / page_height,
                            "word_count": len(current_line)
                        })
                
                page_data["text"] = page.extract_text() or ""
                result["pages"].append(page_data)
                
    except Exception as exc:
        _debug("extract_pdf_positions.error", str(exc))
    
    return result


def find_text_position_in_page(search_text: str, page_data: dict) -> dict:
    """Find the position of text within a page's extracted data.
    
    Returns normalized coordinates (0-1 range) or None if not found.
    """
    if not search_text or not page_data.get("lines"):
        return None
    
    search_lower = search_text.lower().strip()[:50]  # Limit search length
    
    # First try to find in lines
    for line in page_data["lines"]:
        line_text = line.get("text", "").lower()
        if search_lower in line_text:
            return {
                "x": line["x0"],
                "y": line["y0"],
                "width": line["x1"] - line["x0"],
                "height": line["y1"] - line["y0"]
            }
    
    # Try word-level search
    for word in page_data.get("words", []):
        word_text = word.get("text", "").lower()
        if search_lower.split()[0] in word_text if search_lower.split() else False:
            return {
                "x": word["x0"],
                "y": word["y0"],
                "width": word["x1"] - word["x0"],
                "height": word["y1"] - word["y0"]
            }
    
    return None


def analyze_formatting_suggestions(text: str, filename: str = "", pdf_positions: dict = None) -> dict:
    """Analyze document text and generate formatting/alignment suggestions using AI.
    
    Args:
        text: Extracted text from the document
        filename: Original filename
        pdf_positions: Optional dict from extract_pdf_text_with_positions for accurate mapping
    
    Returns a dict with:
    - pages: list of page-level suggestions with accurate position data
    - overall: overall document suggestions
    - score: formatting quality score 0-100
    """
    
    if not text or len(text.strip()) < 50:
        return {
            "pages": [],
            "overall": {"suggestions": ["Document is too short for analysis."]},
            "score": 0,
            "error": None
        }
    
    import re  # For regex matching in position calculation
    
    # Split text into approximate pages (roughly 3000 chars per page)
    page_size = 3000
    pages_text = []
    for i in range(0, len(text), page_size):
        page_content = text[i:i + page_size]
        if page_content.strip():
            pages_text.append(page_content)
    
    if not pages_text:
        pages_text = [text]
    
    # Use actual PDF pages if available
    if pdf_positions and pdf_positions.get("pages"):
        pages_text = [p.get("text", "") for p in pdf_positions["pages"] if p.get("text")]
    
    # Try AI-based analysis first
    api_key = _get_api_key()
    
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            
            # Analyze each page
            pages_suggestions = []
            
            for page_num, page_text in enumerate(pages_text[:10], 1):  # Limit to first 10 pages
                system_msg = """You are an expert document formatting and academic writing reviewer. 
Analyze the given text excerpt from a research paper and provide specific, actionable formatting and alignment suggestions.

Focus on:
1. Paragraph structure and spacing
2. Heading hierarchy and formatting
3. Text alignment issues
4. Citation formatting
5. List and bullet formatting
6. Figure/table reference formatting
7. Section organization
8. Academic writing style
9. Consistency issues
10. Typography concerns (if detectable)

Respond ONLY in JSON with this exact structure:
{
    "suggestions": [
        {
            "type": "heading|paragraph|alignment|citation|list|figure|section|style|consistency|typography",
            "severity": "critical|major|minor|info",
            "title": "Short title (max 8 words)",
            "description": "Detailed description of the issue",
            "location": "Approximate location description (e.g., 'paragraph 2', 'beginning of page')",
            "fix": "Suggested fix or improvement",
            "affected_text": "Short snippet of affected text if applicable (max 50 chars)"
        }
    ],
    "page_score": 0-100,
    "summary": "One sentence summary of page formatting quality"
}

Provide 3-7 suggestions per page, prioritizing the most impactful issues."""

                user_msg = f"Page {page_num} of document '{filename}':\n\n{page_text[:2500]}"
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.3,
                    max_tokens=1200,
                )
                
                raw = response.choices[0].message.content.strip()
                
                # Parse JSON
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()
                
                import json
                page_data = json.loads(raw)
                page_data["page_number"] = page_num
                page_data["char_start"] = (page_num - 1) * page_size
                page_data["char_end"] = min(page_num * page_size, len(text))
                
                # Calculate positions for each suggestion
                if pdf_positions and page_num <= len(pdf_positions.get("pages", [])):
                    pdf_page_data = pdf_positions["pages"][page_num - 1]
                    lines = pdf_page_data.get("lines", [])
                    total_lines = len(lines)
                    
                    for idx, suggestion in enumerate(page_data.get("suggestions", [])):
                        affected_text = suggestion.get("affected_text", "")
                        location_desc = suggestion.get("location", "").lower()
                        
                        # Try to find position by affected text first
                        if affected_text:
                            pos = find_text_position_in_page(affected_text, pdf_page_data)
                            if pos:
                                suggestion["location_x"] = pos["x"]
                                suggestion["location_y"] = pos["y"]
                                suggestion["location_width"] = min(pos["width"] + 0.1, 0.9 - pos["x"])
                                suggestion["location_height"] = min(pos["height"] + 0.02, 0.1)
                                continue
                        
                        # Parse location description for line/paragraph numbers
                        line_match = re.search(r'line\s*(\d+)', location_desc)
                        para_match = re.search(r'paragraph\s*(\d+)', location_desc)
                        
                        if line_match and total_lines > 0:
                            line_num = min(int(line_match.group(1)), total_lines) - 1
                            if 0 <= line_num < total_lines:
                                line_data = lines[line_num]
                                suggestion["location_x"] = line_data["x0"]
                                suggestion["location_y"] = line_data["y0"]
                                suggestion["location_width"] = line_data["x1"] - line_data["x0"]
                                suggestion["location_height"] = line_data["y1"] - line_data["y0"] + 0.01
                                continue
                        
                        if para_match and total_lines > 0:
                            # Estimate paragraph position (assume ~5 lines per paragraph)
                            para_num = int(para_match.group(1))
                            est_line = min((para_num - 1) * 5, total_lines - 1)
                            if 0 <= est_line < total_lines:
                                line_data = lines[est_line]
                                suggestion["location_x"] = line_data["x0"]
                                suggestion["location_y"] = line_data["y0"]
                                suggestion["location_width"] = 0.85
                                # Cover multiple lines for paragraph
                                end_line = min(est_line + 4, total_lines - 1)
                                suggestion["location_height"] = lines[end_line]["y1"] - line_data["y0"]
                                continue
                        
                        # Position based on keywords in location
                        if "beginning" in location_desc or "start" in location_desc or "top" in location_desc:
                            y_pos = 0.1
                        elif "end" in location_desc or "bottom" in location_desc:
                            y_pos = 0.75
                        elif "middle" in location_desc or "center" in location_desc:
                            y_pos = 0.4
                        else:
                            # Distribute suggestions evenly down the page
                            y_pos = 0.1 + (idx * 0.12)
                        
                        suggestion["location_x"] = 0.08
                        suggestion["location_y"] = min(y_pos, 0.85)
                        suggestion["location_width"] = 0.84
                        suggestion["location_height"] = 0.04
                
                pages_suggestions.append(page_data)
            
            # Generate overall document suggestions
            overall_system = """You are an expert document formatting reviewer. Based on the document text, provide overall formatting and structure suggestions for the entire document.

Respond ONLY in JSON:
{
    "suggestions": [
        {
            "type": "structure|consistency|style|formatting",
            "title": "Short title",
            "description": "Detailed description",
            "priority": "high|medium|low"
        }
    ],
    "overall_score": 0-100,
    "grade": "A|B|C|D|F",
    "strengths": ["strength 1", "strength 2"],
    "improvements_needed": ["improvement 1", "improvement 2"]
}"""
            
            overall_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": overall_system},
                    {"role": "user", "content": f"Full document text (truncated):\n\n{text[:6000]}"}
                ],
                temperature=0.3,
                max_tokens=800,
            )
            
            overall_raw = overall_response.choices[0].message.content.strip()
            if "```json" in overall_raw:
                overall_raw = overall_raw.split("```json")[1].split("```")[0].strip()
            elif "```" in overall_raw:
                overall_raw = overall_raw.split("```")[1].split("```")[0].strip()
            
            overall_data = json.loads(overall_raw)
            
            # Calculate average score
            page_scores = [p.get("page_score", 70) for p in pages_suggestions]
            avg_score = sum(page_scores) / len(page_scores) if page_scores else 70
            
            return {
                "pages": pages_suggestions,
                "overall": overall_data,
                "score": int((avg_score + overall_data.get("overall_score", 70)) / 2),
                "total_pages": len(pages_text),
                "error": None
            }
            
        except Exception as exc:
            _debug("formatting_analysis.ai_error", str(exc))
    
    # Fallback: Rule-based analysis
    return _analyze_formatting_rules(text, pages_text, filename, pdf_positions)


def _analyze_formatting_rules(text: str, pages_text: list, filename: str, pdf_positions: dict = None) -> dict:
    """Rule-based formatting analysis fallback when AI is unavailable."""
    
    import re
    
    pages_suggestions = []
    
    for page_num, page_text in enumerate(pages_text[:10], 1):
        suggestions = []
        
        # Get PDF page data for positioning if available
        pdf_page_data = None
        if pdf_positions and page_num <= len(pdf_positions.get("pages", [])):
            pdf_page_data = pdf_positions["pages"][page_num - 1]
        
        # Check for very long paragraphs (>500 chars without line break)
        paragraphs = page_text.split('\n\n')
        for i, para in enumerate(paragraphs):
            if len(para) > 500:
                suggestion = {
                    "type": "paragraph",
                    "severity": "minor",
                    "title": "Long paragraph detected",
                    "description": f"Paragraph {i+1} is quite long ({len(para)} characters). Consider breaking it into smaller paragraphs for better readability.",
                    "location": f"Paragraph {i+1}",
                    "fix": "Split into 2-3 shorter paragraphs with clear topic sentences.",
                    "affected_text": para[:50] + "..." if len(para) > 50 else para
                }
                # Calculate position
                if pdf_page_data:
                    pos = find_text_position_in_page(para[:50], pdf_page_data)
                    if pos:
                        suggestion["location_x"] = pos["x"]
                        suggestion["location_y"] = pos["y"]
                        suggestion["location_width"] = 0.84
                        suggestion["location_height"] = 0.08
                    else:
                        # Estimate based on paragraph index
                        suggestion["location_x"] = 0.08
                        suggestion["location_y"] = 0.1 + (i * 0.15)
                        suggestion["location_width"] = 0.84
                        suggestion["location_height"] = 0.08
                else:
                    suggestion["location_x"] = 0.08
                    suggestion["location_y"] = 0.1 + (i * 0.15)
                    suggestion["location_width"] = 0.84
                    suggestion["location_height"] = 0.08
                suggestions.append(suggestion)
        
        # Check for inconsistent spacing
        if '  ' in page_text:
            double_space_count = page_text.count('  ')
            suggestions.append({
                "type": "alignment",
                "severity": "minor",
                "title": "Inconsistent spacing",
                "description": f"Found {double_space_count} instances of double spaces.",
                "location": "Throughout the page",
                "fix": "Replace double spaces with single spaces for consistency.",
                "affected_text": ""
            })
        
        # Check for potential heading issues (lines that look like headings but aren't formatted)
        lines = page_text.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and len(stripped) < 60 and stripped.isupper() and len(stripped) > 5:
                if not stripped.endswith(':'):
                    suggestions.append({
                        "type": "heading",
                        "severity": "info",
                        "title": "Potential heading detected",
                        "description": "This line appears to be a heading. Ensure proper heading formatting is applied.",
                        "location": f"Line {i+1}",
                        "fix": "Apply consistent heading style (bold, larger font, proper spacing).",
                        "affected_text": stripped[:50]
                    })
        
        # Check for citation patterns
        citation_patterns = [
            r'\([A-Z][a-z]+,?\s*\d{4}\)',  # (Author, 2020)
            r'\[[0-9]+\]',  # [1]
            r'\([0-9]+\)',  # (1)
        ]
        found_citations = set()
        for pattern in citation_patterns:
            if re.search(pattern, page_text):
                found_citations.add(pattern)
        
        if len(found_citations) > 1:
            suggestions.append({
                "type": "citation",
                "severity": "major",
                "title": "Mixed citation styles",
                "description": "Multiple citation formats detected. Academic papers should use consistent citation style.",
                "location": "Throughout the page",
                "fix": "Choose one citation style (APA, IEEE, etc.) and apply consistently.",
                "affected_text": ""
            })
        
        # Check for bullet/list formatting
        bullet_chars = ['•', '-', '*', '○', '►']
        bullets_found = [c for c in bullet_chars if c in page_text]
        if len(bullets_found) > 1:
            suggestions.append({
                "type": "list",
                "severity": "minor",
                "title": "Inconsistent list markers",
                "description": f"Multiple bullet styles found: {', '.join(bullets_found)}",
                "location": "List sections",
                "fix": "Use consistent bullet markers throughout the document.",
                "affected_text": ""
            })
        
        # Check for figure/table references
        fig_refs = re.findall(r'[Ff]igure\s*\.?\s*\d+|[Ff]ig\s*\.?\s*\d+', page_text)
        table_refs = re.findall(r'[Tt]able\s*\.?\s*\d+', page_text)
        
        if fig_refs or table_refs:
            fig_patterns = set([ref.split()[0].lower() for ref in fig_refs])
            if len(fig_patterns) > 1:
                suggestions.append({
                    "type": "figure",
                    "severity": "minor",
                    "title": "Inconsistent figure references",
                    "description": "Mix of 'Figure' and 'Fig.' abbreviations detected.",
                    "location": "Figure references",
                    "fix": "Use either 'Figure' or 'Fig.' consistently throughout.",
                    "affected_text": ""
                })
        
        # Calculate page score
        severity_scores = {"critical": 20, "major": 10, "minor": 5, "info": 2}
        deductions = sum(severity_scores.get(s.get("severity", "info"), 0) for s in suggestions)
        page_score = max(40, 100 - deductions)
        
        # Add position data to all suggestions that don't have it
        for idx, suggestion in enumerate(suggestions):
            if "location_x" not in suggestion:
                affected_text = suggestion.get("affected_text", "")
                location_desc = suggestion.get("location", "").lower()
                
                # Try to find by affected text
                if pdf_page_data and affected_text:
                    pos = find_text_position_in_page(affected_text, pdf_page_data)
                    if pos:
                        suggestion["location_x"] = pos["x"]
                        suggestion["location_y"] = pos["y"]
                        suggestion["location_width"] = min(0.84, pos["width"] + 0.2)
                        suggestion["location_height"] = min(0.06, pos["height"] + 0.02)
                        continue
                
                # Try to find by line number
                if pdf_page_data and pdf_page_data.get("lines"):
                    lines = pdf_page_data["lines"]
                    total_lines = len(lines)
                    
                    line_match = re.search(r'line\s*(\d+)', location_desc)
                    if line_match and total_lines > 0:
                        line_num = min(int(line_match.group(1)), total_lines) - 1
                        if 0 <= line_num < total_lines:
                            line_data = lines[line_num]
                            suggestion["location_x"] = line_data["x0"]
                            suggestion["location_y"] = line_data["y0"]
                            suggestion["location_width"] = line_data["x1"] - line_data["x0"]
                            suggestion["location_height"] = line_data["y1"] - line_data["y0"] + 0.01
                            continue
                
                # Default: distribute evenly with some variety
                y_offset = 0.08 + (idx * 0.11)
                if "throughout" in location_desc or "multiple" in location_desc:
                    y_offset = 0.35 + (idx * 0.05)
                elif "beginning" in location_desc or "start" in location_desc:
                    y_offset = 0.08 + (idx * 0.03)
                elif "end" in location_desc:
                    y_offset = 0.7 + (idx * 0.03)
                
                suggestion["location_x"] = 0.08
                suggestion["location_y"] = min(y_offset, 0.85)
                suggestion["location_width"] = 0.84
                suggestion["location_height"] = 0.045
        
        pages_suggestions.append({
            "page_number": page_num,
            "suggestions": suggestions[:7],
            "page_score": page_score,
            "summary": f"Page {page_num}: {len(suggestions)} formatting suggestions identified.",
            "char_start": (page_num - 1) * 3000,
            "char_end": min(page_num * 3000, len(text))
        })
    
    # Overall suggestions
    overall_suggestions = []
    
    # Document length check
    word_count = len(text.split())
    if word_count < 500:
        overall_suggestions.append({
            "type": "structure",
            "title": "Short document",
            "description": f"Document contains only {word_count} words. Research papers typically have more content.",
            "priority": "medium"
        })
    
    # Check abstract
    if "abstract" not in text.lower()[:2000]:
        overall_suggestions.append({
            "type": "structure",
            "title": "Abstract may be missing",
            "description": "No 'Abstract' heading detected in the beginning of the document.",
            "priority": "high"
        })
    
    # Check references section
    if "references" not in text.lower()[-3000:] and "bibliography" not in text.lower()[-3000:]:
        overall_suggestions.append({
            "type": "structure",
            "title": "References section may be missing",
            "description": "No 'References' or 'Bibliography' section detected at the end.",
            "priority": "high"
        })
    
    # Calculate overall score
    page_scores = [p.get("page_score", 70) for p in pages_suggestions]
    avg_score = sum(page_scores) / len(page_scores) if page_scores else 70
    
    if avg_score >= 90:
        grade = "A"
    elif avg_score >= 80:
        grade = "B"
    elif avg_score >= 70:
        grade = "C"
    elif avg_score >= 60:
        grade = "D"
    else:
        grade = "F"
    
    return {
        "pages": pages_suggestions,
        "overall": {
            "suggestions": overall_suggestions,
            "overall_score": int(avg_score),
            "grade": grade,
            "strengths": ["Document was successfully parsed", "Text extraction completed"],
            "improvements_needed": [s["title"] for s in overall_suggestions[:3]]
        },
        "score": int(avg_score),
        "total_pages": len(pages_text),
        "error": None
    }


# ============================================================================
# PLAGIARISM DETECTION SYSTEM - ENHANCED VERSION
# ============================================================================

def _web_search_check(query: str, num_results: int = 5) -> list:
    """
    Search the web for matching content using DuckDuckGo (no API key required).
    Returns list of potential source matches.
    """
    import urllib.request
    import urllib.parse
    
    results = []
    
    # Clean and truncate query for search
    query_clean = " ".join(query.split()[:15])  # First 15 words
    
    try:
        # Use DuckDuckGo HTML search (no API needed)
        encoded_query = urllib.parse.quote_plus(query_clean)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
            # Parse results from HTML
            import re
            
            # Extract result snippets and URLs
            result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]*)</a>'
            
            urls = re.findall(result_pattern, html)
            snippets = re.findall(snippet_pattern, html)
            
            for i, (url_match, title) in enumerate(urls[:num_results]):
                snippet = snippets[i] if i < len(snippets) else ""
                
                # Clean up the URL (DuckDuckGo redirects)
                actual_url = url_match
                if "uddg=" in url_match:
                    url_parts = urllib.parse.parse_qs(urllib.parse.urlparse(url_match).query)
                    actual_url = url_parts.get('uddg', [url_match])[0]
                
                results.append({
                    "url": actual_url,
                    "title": title.strip(),
                    "snippet": snippet.strip(),
                    "source": "web"
                })
                
    except Exception as exc:
        _debug("web_search.error", str(exc))
    
    return results


def _google_scholar_check(query: str) -> list:
    """
    Check content against Google Scholar for academic source matching.
    Returns list of potential academic source matches.
    """
    import urllib.request
    import urllib.parse
    
    results = []
    query_clean = " ".join(query.split()[:12])
    
    try:
        encoded_query = urllib.parse.quote_plus(query_clean)
        url = f"https://scholar.google.com/scholar?q={encoded_query}&hl=en"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
            import re
            
            # Extract titles and links from Scholar results
            title_pattern = r'<h3[^>]*class="gs_rt"[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>([^<]*(?:<[^/][^>]*>[^<]*</[^>]*>)*[^<]*)</a>'
            matches = re.findall(title_pattern, html, re.DOTALL)
            
            for url_match, title in matches[:3]:
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                results.append({
                    "url": url_match,
                    "title": clean_title,
                    "source": "google_scholar"
                })
                
    except Exception as exc:
        _debug("scholar_search.error", str(exc))
    
    return results


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using multiple methods.
    Returns similarity score 0-1.
    """
    if not text1 or not text2:
        return 0.0
    
    # Method 1: Word overlap (Jaccard)
    words1 = set(_tokenize_text(text1.lower()))
    words2 = set(_tokenize_text(text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    jaccard = len(words1 & words2) / len(words1 | words2)
    
    # Method 2: N-gram overlap
    shingles1 = _create_shingles(list(words1), 3)
    shingles2 = _create_shingles(list(words2), 3)
    
    shingle_sim = _jaccard_similarity(shingles1, shingles2)
    
    # Method 3: Sequence matching
    from difflib import SequenceMatcher
    seq_sim = SequenceMatcher(None, text1.lower()[:500], text2.lower()[:500]).ratio()
    
    # Weighted average
    return (jaccard * 0.3 + shingle_sim * 0.4 + seq_sim * 0.3)


def _extract_key_sentences(text: str, num_sentences: int = 10) -> list:
    """
    Extract the most distinctive sentences for plagiarism checking.
    Prioritizes sentences with specific claims, numbers, or unique phrasing.
    """
    import re
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 40]
    
    scored_sentences = []
    
    for sent in sentences:
        score = 0
        
        # Prefer sentences with numbers/statistics
        if re.search(r'\d+(?:\.\d+)?%|\d{4}|\d+\s*(?:million|billion|thousand)', sent):
            score += 3
        
        # Prefer sentences with specific claims
        claim_words = ['discovered', 'found', 'proved', 'demonstrated', 'showed', 
                       'revealed', 'concluded', 'established', 'confirmed']
        if any(word in sent.lower() for word in claim_words):
            score += 2
        
        # Prefer sentences with proper nouns (likely specific references)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sent)
        score += min(len(proper_nouns), 3)
        
        # Prefer longer, more specific sentences
        word_count = len(sent.split())
        if 15 <= word_count <= 40:
            score += 2
        elif word_count > 40:
            score += 1
        
        # Penalize very common sentence starters
        common_starters = ['this is', 'it is', 'there are', 'we can', 'the']
        if any(sent.lower().startswith(starter) for starter in common_starters):
            score -= 1
        
        scored_sentences.append((sent, score))
    
    # Sort by score and return top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored_sentences[:num_sentences]]


def _check_sentence_against_web(sentence: str) -> dict:
    """
    Check a single sentence against web sources.
    Returns match information if found.
    """
    web_results = _web_search_check(sentence, num_results=3)
    scholar_results = _google_scholar_check(sentence)
    
    all_results = web_results + scholar_results
    
    matches = []
    for result in all_results:
        snippet = result.get("snippet", "").lower()
        
        # Calculate how similar the snippet is to our sentence
        if snippet:
            similarity = _calculate_text_similarity(sentence, snippet)
            
            if similarity > 0.4:  # Threshold for potential match
                matches.append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "source": result.get("source", "web"),
                    "similarity": round(similarity * 100, 1),
                    "snippet": snippet[:200]
                })
    
    return {
        "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
        "matches": sorted(matches, key=lambda x: x["similarity"], reverse=True)[:3],
        "has_match": len(matches) > 0,
        "highest_similarity": max([m["similarity"] for m in matches]) if matches else 0
    }


def _detect_paraphrasing(text: str) -> dict:
    """
    Detect potential paraphrasing by analyzing sentence structure patterns.
    Paraphrased content often has unusual word choices or awkward phrasing.
    """
    import re
    from collections import Counter
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    indicators = []
    suspicious_count = 0
    
    for i, sent in enumerate(sentences):
        words = sent.lower().split()
        
        # Check for overly complex synonyms (thesaurus abuse)
        complex_words = [w for w in words if len(w) > 10]
        complex_ratio = len(complex_words) / len(words) if words else 0
        
        # Check for passive voice overuse (common in paraphrasing)
        passive_patterns = r'\b(is|are|was|were|been|being)\s+\w+ed\b'
        passive_count = len(re.findall(passive_patterns, sent.lower()))
        
        # Check for unusual word order
        # Sentences starting with adverbs or prepositional phrases
        unusual_starts = r'^(However|Moreover|Furthermore|Additionally|Consequently|Nevertheless|Interestingly|Surprisingly|Remarkably)'
        has_unusual_start = bool(re.match(unusual_starts, sent))
        
        # Check for synonym chains (multiple synonyms of same concept)
        word_freq = Counter(words)
        
        suspicion_score = 0
        reasons = []
        
        if complex_ratio > 0.2:
            suspicion_score += 2
            reasons.append("High density of complex words")
        
        if passive_count > 2:
            suspicion_score += 1
            reasons.append("Heavy passive voice usage")
        
        if has_unusual_start and i > 0:
            suspicion_score += 0.5
            reasons.append("Unusual sentence structure")
        
        if suspicion_score >= 2:
            suspicious_count += 1
            indicators.append({
                "sentence_num": i + 1,
                "text": sent[:80] + "..." if len(sent) > 80 else sent,
                "suspicion_score": round(suspicion_score, 1),
                "reasons": reasons
            })
    
    paraphrase_percentage = (suspicious_count / len(sentences) * 100) if sentences else 0
    
    return {
        "indicators": indicators[:10],
        "suspicious_sentence_count": suspicious_count,
        "total_sentences": len(sentences),
        "paraphrase_percentage": round(paraphrase_percentage, 1),
        "risk_level": "high" if paraphrase_percentage > 30 else ("medium" if paraphrase_percentage > 15 else "low")
    }


def _analyze_writing_patterns(text: str) -> dict:
    """
    Deep analysis of writing patterns to detect copied/patchwork content.
    """
    import re
    from collections import Counter
    
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    
    if len(paragraphs) < 2:
        return {
            "pattern_consistency": 100,
            "anomalies": [],
            "assessment": "insufficient_text"
        }
    
    paragraph_profiles = []
    
    for para in paragraphs:
        sentences = re.split(r'[.!?]+', para)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = para.lower().split()
        
        # Calculate various metrics
        profile = {
            "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "punctuation_density": len(re.findall(r'[,;:\-\(\)]', para)) / len(words) if words else 0,
            "question_ratio": len([s for s in sentences if '?' in s]) / len(sentences) if sentences else 0,
            "first_person_usage": len(re.findall(r'\b(I|we|my|our|us)\b', para, re.I)) / len(words) if words else 0,
            "passive_voice_ratio": len(re.findall(r'\b(is|are|was|were|been|being)\s+\w+ed\b', para.lower())) / len(sentences) if sentences else 0,
            "contraction_usage": len(re.findall(r"\b\w+'[a-z]+\b", para.lower())) / len(words) if words else 0,
        }
        paragraph_profiles.append(profile)
    
    # Detect anomalies (paragraphs that differ significantly from neighbors)
    anomalies = []
    
    for i, profile in enumerate(paragraph_profiles):
        if i == 0:
            continue
        
        prev_profile = paragraph_profiles[i - 1]
        
        # Compare with previous paragraph
        differences = []
        
        if abs(profile["avg_sentence_length"] - prev_profile["avg_sentence_length"]) > 10:
            differences.append("sentence length")
        if abs(profile["avg_word_length"] - prev_profile["avg_word_length"]) > 1.5:
            differences.append("word complexity")
        if abs(profile["punctuation_density"] - prev_profile["punctuation_density"]) > 0.05:
            differences.append("punctuation style")
        if abs(profile["first_person_usage"] - prev_profile["first_person_usage"]) > 0.03:
            differences.append("narrative voice")
        if abs(profile["passive_voice_ratio"] - prev_profile["passive_voice_ratio"]) > 0.3:
            differences.append("voice (active/passive)")
        
        if len(differences) >= 2:
            anomalies.append({
                "paragraph": i + 1,
                "differences": differences,
                "severity": "high" if len(differences) >= 3 else "medium",
                "text_preview": paragraphs[i][:100] + "..."
            })
    
    # Calculate overall consistency
    import statistics
    
    all_metrics = []
    for metric_name in ["avg_sentence_length", "avg_word_length", "punctuation_density"]:
        values = [p[metric_name] for p in paragraph_profiles]
        if len(values) > 1:
            cv = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) > 0 else 0
            all_metrics.append(cv)
    
    avg_variation = sum(all_metrics) / len(all_metrics) if all_metrics else 0
    pattern_consistency = max(0, 100 - avg_variation * 100)
    
    return {
        "pattern_consistency": round(pattern_consistency, 1),
        "anomalies": anomalies[:5],
        "paragraph_count": len(paragraphs),
        "assessment": "consistent" if pattern_consistency > 80 else ("moderate" if pattern_consistency > 60 else "inconsistent")
    }


def _ai_deep_analysis(text: str, preliminary_results: dict) -> dict:
    """
    Use AI for deep contextual plagiarism analysis.
    Identifies suspicious patterns that algorithmic methods might miss.
    """
    api_key = _get_api_key()
    gemini_key = _get_gemini_api_key()
    
    if not api_key and not gemini_key:
        return {"ai_available": False}
    
    # Only run deep analysis if there are concerns
    if preliminary_results.get("overall_score", 100) > 85:
        return {
            "ai_available": True,
            "analysis_skipped": True,
            "reason": "Preliminary score indicates low plagiarism risk"
        }
    
    try:
        system_prompt = """You are an expert academic integrity analyst. Analyze the text for signs of plagiarism, including:

1. **Patchwork Plagiarism**: Content stitched from multiple sources
2. **Paraphrasing Issues**: Poorly disguised copied content
3. **Style Inconsistencies**: Sudden changes in writing quality/style
4. **Citation Gaming**: Improper use of sources to mask copying
5. **AI-Generated Content**: Signs of AI-written text

Provide analysis in JSON format:
{
    "plagiarism_likelihood": 0-100 (0=original, 100=definitely plagiarized),
    "confidence": 0-1,
    "detected_issues": [
        {
            "type": "patchwork|paraphrase|style_shift|citation_gaming|ai_generated",
            "severity": "low|medium|high|critical",
            "description": "specific issue description",
            "evidence": "quoted text showing the issue",
            "location": "approximate location in text"
        }
    ],
    "ai_content_probability": 0-100,
    "overall_assessment": "brief 2-3 sentence summary",
    "recommendations": ["action item 1", "action item 2", "action item 3"]
}"""

        user_prompt = f"""Analyze this academic text for plagiarism indicators:

--- TEXT START ---
{text[:4000]}
--- TEXT END ---

Preliminary Analysis Results:
- Overall Score: {preliminary_results.get('overall_score', 'N/A')}/100
- Risk Level: {preliminary_results.get('risk_level', 'unknown')}
- Vocabulary Richness: {preliminary_results.get('component_scores', {}).get('vocabulary_richness', 'N/A')}%
- Style Consistency: {preliminary_results.get('component_scores', {}).get('style_consistency', 'N/A')}%
- Internal Duplicates: {preliminary_results.get('analyses', {}).get('internal_duplicates', {}).get('duplicate_percentage', 0)}%

Perform deep analysis and identify specific plagiarism concerns."""

        # Try Gemini first, then OpenAI
        result = _invoke_gemini_text(system_prompt, user_prompt, temperature=0.2, max_output_tokens=1000)
        
        if not result:
            result = _invoke_openai_text(system_prompt, user_prompt, temperature=0.2, max_output_tokens=1000)
        
        if not result:
            return {"ai_available": True, "error": "AI analysis failed"}
        
        # Parse JSON response
        import json
        
        raw = result.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        
        ai_result = json.loads(raw)
        ai_result["ai_available"] = True
        ai_result["analysis_completed"] = True
        
        return ai_result
        
    except Exception as exc:
        _debug("ai_deep_analysis.error", str(exc))
        return {"ai_available": True, "error": str(exc)}


def _tokenize_text(text: str) -> list:
    """Tokenize text into words, removing punctuation and converting to lowercase."""
    import re
    # Remove special characters but keep apostrophes in contractions
    text = re.sub(r"[^\w\s']", " ", text.lower())
    words = text.split()
    # Filter out very short words and numbers
    return [w for w in words if len(w) > 2 and not w.isdigit()]


def _create_shingles(words: list, n: int = 5) -> set:
    """Create n-gram shingles from a list of words."""
    if len(words) < n:
        return set([" ".join(words)])
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))


def _hash_shingle(shingle: str) -> int:
    """Create a hash value for a shingle."""
    import hashlib
    return int(hashlib.md5(shingle.encode()).hexdigest()[:8], 16)


def _compute_minhash_signature(shingles: set, num_hashes: int = 100) -> list:
    """Compute MinHash signature for a set of shingles."""
    import random
    random.seed(42)  # Fixed seed for reproducibility
    
    if not shingles:
        return [float('inf')] * num_hashes
    
    # Generate hash functions parameters
    max_val = 2**32 - 1
    hash_params = [(random.randint(1, max_val), random.randint(0, max_val)) for _ in range(num_hashes)]
    
    signature = []
    for a, b in hash_params:
        min_hash = float('inf')
        for shingle in shingles:
            h = _hash_shingle(shingle)
            hash_val = (a * h + b) % max_val
            min_hash = min(min_hash, hash_val)
        signature.append(min_hash)
    
    return signature


def _jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _detect_internal_duplicates(text: str) -> dict:
    """Detect repeated phrases and self-plagiarism within the document."""
    import re
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    duplicates = []
    duplicate_chars = 0
    total_chars = len(text)
    
    # Check for duplicate sentences
    seen_sentences = {}
    for i, sent in enumerate(sentences):
        # Normalize sentence for comparison
        normalized = " ".join(sent.lower().split())
        if len(normalized) < 40:
            continue
            
        if normalized in seen_sentences:
            duplicates.append({
                "text": sent[:100] + "..." if len(sent) > 100 else sent,
                "type": "exact_duplicate",
                "first_occurrence": seen_sentences[normalized],
                "second_occurrence": i + 1,
                "severity": "high"
            })
            duplicate_chars += len(sent)
        else:
            seen_sentences[normalized] = i + 1
    
    # Check for similar sentences (near-duplicates)
    sentence_shingles = []
    for sent in sentences:
        words = _tokenize_text(sent)
        shingles = _create_shingles(words, 3)  # Use 3-grams for sentences
        sentence_shingles.append((sent, shingles))
    
    for i in range(len(sentence_shingles)):
        for j in range(i + 2, len(sentence_shingles)):  # Skip adjacent sentences
            sent1, shingles1 = sentence_shingles[i]
            sent2, shingles2 = sentence_shingles[j]
            
            similarity = _jaccard_similarity(shingles1, shingles2)
            if similarity > 0.6 and similarity < 1.0:  # Similar but not exact
                # Avoid duplicate entries
                already_found = any(
                    d["first_occurrence"] == i + 1 and d["second_occurrence"] == j + 1 
                    for d in duplicates
                )
                if not already_found:
                    duplicates.append({
                        "text": sent1[:80] + "..." if len(sent1) > 80 else sent1,
                        "similar_to": sent2[:80] + "..." if len(sent2) > 80 else sent2,
                        "type": "near_duplicate",
                        "similarity": round(similarity * 100, 1),
                        "first_occurrence": i + 1,
                        "second_occurrence": j + 1,
                        "severity": "medium" if similarity < 0.8 else "high"
                    })
                    duplicate_chars += len(sent1) * (similarity - 0.5)
    
    duplicate_percentage = (duplicate_chars / total_chars * 100) if total_chars > 0 else 0
    
    return {
        "duplicates": duplicates[:10],  # Limit to top 10
        "duplicate_count": len(duplicates),
        "duplicate_percentage": round(min(duplicate_percentage, 100), 1)
    }


def _analyze_vocabulary_richness(text: str) -> dict:
    """Analyze vocabulary diversity and richness metrics."""
    words = _tokenize_text(text)
    
    if not words:
        return {
            "total_words": 0,
            "unique_words": 0,
            "vocabulary_richness": 0,
            "hapax_legomena": 0,
            "avg_word_length": 0,
            "assessment": "insufficient_text"
        }
    
    total_words = len(words)
    unique_words = len(set(words))
    
    # Type-Token Ratio (TTR)
    ttr = unique_words / total_words if total_words > 0 else 0
    
    # Hapax legomena (words appearing only once)
    from collections import Counter
    word_freq = Counter(words)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = hapax / total_words if total_words > 0 else 0
    
    # Average word length
    avg_word_length = sum(len(w) for w in words) / total_words if total_words > 0 else 0
    
    # Yule's K measure (vocabulary richness that's more stable across text lengths)
    if total_words > 0:
        freq_of_freqs = Counter(word_freq.values())
        m1 = total_words
        m2 = sum(f * (c ** 2) for c, f in freq_of_freqs.items())
        yules_k = 10000 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0
    else:
        yules_k = 0
    
    # Assessment based on metrics
    if ttr > 0.7:
        assessment = "excellent"
    elif ttr > 0.5:
        assessment = "good"
    elif ttr > 0.35:
        assessment = "moderate"
    else:
        assessment = "low"
    
    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "vocabulary_richness": round(ttr * 100, 1),
        "hapax_legomena": hapax,
        "hapax_ratio": round(hapax_ratio * 100, 1),
        "avg_word_length": round(avg_word_length, 1),
        "yules_k": round(yules_k, 2),
        "assessment": assessment
    }


def _detect_common_phrases(text: str) -> dict:
    """Detect overly common academic phrases that might indicate template use or plagiarism."""
    
    # Common academic boilerplate phrases (often copied)
    common_phrases = [
        "it is important to note that",
        "in this paper we",
        "the results show that",
        "as shown in figure",
        "as shown in table",
        "in conclusion",
        "in summary",
        "the purpose of this study",
        "the aim of this research",
        "literature review",
        "the findings suggest",
        "further research is needed",
        "according to the results",
        "based on the findings",
        "it can be concluded that",
        "the data indicates",
        "significant difference",
        "statistically significant",
        "the study reveals",
        "previous studies have shown",
        "research has demonstrated",
        "as mentioned earlier",
        "as discussed above",
        "it is evident that",
        "it is clear that",
        "in other words",
        "on the other hand",
        "in addition to",
        "furthermore",
        "moreover",
        "nevertheless",
        "however it should be noted",
        "it is worth mentioning",
        "this suggests that",
        "this indicates that",
    ]
    
    text_lower = text.lower()
    found_phrases = []
    
    for phrase in common_phrases:
        count = text_lower.count(phrase)
        if count > 0:
            found_phrases.append({
                "phrase": phrase,
                "count": count,
                "risk": "low" if count == 1 else ("medium" if count <= 3 else "high")
            })
    
    # Sort by count descending
    found_phrases.sort(key=lambda x: x["count"], reverse=True)
    
    # Calculate boilerplate score
    total_phrase_occurrences = sum(p["count"] for p in found_phrases)
    word_count = len(text.split())
    boilerplate_density = (total_phrase_occurrences * 5 / word_count * 100) if word_count > 0 else 0
    
    return {
        "common_phrases": found_phrases[:15],
        "total_occurrences": total_phrase_occurrences,
        "boilerplate_density": round(min(boilerplate_density, 100), 1),
        "risk_level": "low" if boilerplate_density < 5 else ("medium" if boilerplate_density < 15 else "high")
    }


def _analyze_style_consistency(text: str) -> dict:
    """Analyze writing style consistency across the document."""
    import re
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    
    if len(paragraphs) < 3:
        return {
            "consistency_score": 100,
            "variations": [],
            "assessment": "insufficient_paragraphs"
        }
    
    paragraph_metrics = []
    
    for para in paragraphs:
        sentences = re.split(r'[.!?]+', para)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            continue
        
        # Calculate metrics for each paragraph
        words = _tokenize_text(para)
        
        # Average sentence length
        avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Vocabulary richness for this paragraph
        unique_ratio = len(set(words)) / len(words) if words else 0
        
        # Complexity: ratio of long words (>6 chars)
        long_words = sum(1 for w in words if len(w) > 6)
        complexity = long_words / len(words) if words else 0
        
        paragraph_metrics.append({
            "avg_sentence_length": avg_sent_len,
            "vocabulary_richness": unique_ratio,
            "complexity": complexity,
            "word_count": len(words)
        })
    
    if len(paragraph_metrics) < 2:
        return {
            "consistency_score": 100,
            "variations": [],
            "assessment": "insufficient_data"
        }
    
    # Calculate standard deviations to detect inconsistencies
    import statistics
    
    avg_sent_lengths = [p["avg_sentence_length"] for p in paragraph_metrics]
    vocab_richnesses = [p["vocabulary_richness"] for p in paragraph_metrics]
    complexities = [p["complexity"] for p in paragraph_metrics]
    
    variations = []
    
    # Check for significant variations in sentence length
    if len(avg_sent_lengths) > 1:
        sent_std = statistics.stdev(avg_sent_lengths)
        sent_mean = statistics.mean(avg_sent_lengths)
        sent_cv = (sent_std / sent_mean * 100) if sent_mean > 0 else 0
        
        if sent_cv > 50:
            variations.append({
                "metric": "sentence_length",
                "description": f"Sentence length varies significantly (CV: {sent_cv:.1f}%)",
                "severity": "high" if sent_cv > 75 else "medium",
                "coefficient_of_variation": round(sent_cv, 1)
            })
    
    # Check for vocabulary richness variations
    if len(vocab_richnesses) > 1:
        vocab_std = statistics.stdev(vocab_richnesses)
        vocab_mean = statistics.mean(vocab_richnesses)
        vocab_cv = (vocab_std / vocab_mean * 100) if vocab_mean > 0 else 0
        
        if vocab_cv > 40:
            variations.append({
                "metric": "vocabulary_richness",
                "description": f"Vocabulary diversity varies between sections (CV: {vocab_cv:.1f}%)",
                "severity": "high" if vocab_cv > 60 else "medium",
                "coefficient_of_variation": round(vocab_cv, 1)
            })
    
    # Check for complexity variations
    if len(complexities) > 1:
        comp_std = statistics.stdev(complexities)
        comp_mean = statistics.mean(complexities)
        comp_cv = (comp_std / comp_mean * 100) if comp_mean > 0 else 0
        
        if comp_cv > 45:
            variations.append({
                "metric": "writing_complexity",
                "description": f"Writing complexity varies across sections (CV: {comp_cv:.1f}%)",
                "severity": "high" if comp_cv > 70 else "medium",
                "coefficient_of_variation": round(comp_cv, 1)
            })
    
    # Calculate consistency score
    total_cv = sum(v.get("coefficient_of_variation", 0) for v in variations)
    consistency_score = max(0, 100 - total_cv * 0.5)
    
    return {
        "consistency_score": round(consistency_score, 1),
        "variations": variations,
        "paragraph_count": len(paragraph_metrics),
        "assessment": "consistent" if consistency_score > 80 else ("moderate" if consistency_score > 60 else "inconsistent")
    }


def _analyze_citation_patterns(text: str) -> dict:
    """Analyze citation patterns and detect potential uncited content."""
    import re
    
    # Common citation patterns
    patterns = {
        "apa": r'\([A-Z][a-z]+(?:\s+(?:&|and)\s+[A-Z][a-z]+)*,?\s*\d{4}[a-z]?\)',  # (Smith, 2020) or (Smith & Jones, 2020)
        "ieee": r'\[\d+\]',  # [1]
        "numbered": r'\(\d+\)',  # (1)
        "harvard": r'[A-Z][a-z]+\s+\(\d{4}\)',  # Smith (2020)
        "footnote": r'\[\w+\d+\]',  # [ref1]
    }
    
    citation_counts = {}
    total_citations = 0
    
    for style, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            citation_counts[style] = len(matches)
            total_citations += len(matches)
    
    # Detect potential uncited quotes
    quote_patterns = [
        r'"[^"]{30,}"',  # Double quotes
        r"'[^']{30,}'",  # Single quotes
        r'„[^"]{30,}"',  # German-style quotes
    ]
    
    uncited_quotes = []
    for pattern in quote_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Check if there's a citation nearby
            match_pos = text.find(match)
            context = text[max(0, match_pos-50):min(len(text), match_pos+len(match)+50)]
            
            has_citation = any(re.search(p, context) for p in patterns.values())
            if not has_citation:
                uncited_quotes.append({
                    "quote": match[:80] + "..." if len(match) > 80 else match,
                    "issue": "Quoted text without visible citation"
                })
    
    # Detect statistics without citations
    stat_pattern = r'\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*(?:million|billion|thousand)'
    stats_found = re.findall(stat_pattern, text.lower())
    
    # Calculate citation density
    word_count = len(text.split())
    citation_density = (total_citations / (word_count / 100)) if word_count > 0 else 0
    
    # Determine primary citation style
    primary_style = max(citation_counts.items(), key=lambda x: x[1])[0] if citation_counts else "none"
    
    # Check for mixed citation styles
    styles_used = [s for s, c in citation_counts.items() if c > 0]
    mixed_styles = len(styles_used) > 1
    
    return {
        "total_citations": total_citations,
        "citation_styles": citation_counts,
        "primary_style": primary_style,
        "mixed_styles": mixed_styles,
        "citation_density": round(citation_density, 2),
        "uncited_quotes": uncited_quotes[:5],
        "statistics_count": len(stats_found),
        "assessment": "good" if citation_density > 2 else ("moderate" if citation_density > 0.5 else "low")
    }


def _calculate_text_fingerprint(text: str) -> dict:
    """Create a fingerprint of the text for comparison purposes."""
    words = _tokenize_text(text)
    
    # Create shingles
    shingles_5 = _create_shingles(words, 5)
    shingles_3 = _create_shingles(words, 3)
    
    # Compute MinHash signatures
    signature = _compute_minhash_signature(shingles_5, 50)
    
    # Create a simple hash of the entire text
    import hashlib
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    
    return {
        "shingle_count_5": len(shingles_5),
        "shingle_count_3": len(shingles_3),
        "minhash_signature": signature[:10],  # First 10 for display
        "document_hash": text_hash[:16],
        "word_count": len(words)
    }


def analyze_plagiarism(text: str, filename: str = "") -> dict:
    """
    Comprehensive plagiarism analysis using algorithmic methods.
    
    This function performs multiple analyses:
    1. Internal duplicate detection (self-plagiarism)
    2. Vocabulary richness analysis
    3. Common phrase detection
    4. Style consistency analysis
    5. Citation pattern analysis
    6. Text fingerprinting
    
    Returns a comprehensive plagiarism report with scores and details.
    """
    
    if not text or len(text.strip()) < 100:
        return {
            "overall_score": 0,
            "risk_level": "unknown",
            "error": "Text too short for analysis",
            "analyses": {}
        }
    
    # Run all analyses
    internal_duplicates = _detect_internal_duplicates(text)
    vocabulary = _analyze_vocabulary_richness(text)
    common_phrases = _detect_common_phrases(text)
    style = _analyze_style_consistency(text)
    fingerprint = _calculate_text_fingerprint(text)
    
    # Calculate component scores (higher = more original, lower = more suspicious)
    
    # Duplicate score: penalize based on duplicate percentage
    duplicate_score = max(0, 100 - internal_duplicates["duplicate_percentage"] * 2)
    
    # Vocabulary score: based on richness
    vocab_score = vocabulary["vocabulary_richness"]
    
    # Common phrase score: penalize heavy boilerplate usage
    phrase_score = max(0, 100 - common_phrases["boilerplate_density"] * 2)
    
    # Style score: consistency is good
    style_score = style["consistency_score"]
    
    # Calculate weighted overall score (without citations)
    weights = {
        "duplicates": 0.35,
        "vocabulary": 0.25,
        "phrases": 0.15,
        "style": 0.25
    }
    
    overall_score = (
        duplicate_score * weights["duplicates"] +
        vocab_score * weights["vocabulary"] +
        phrase_score * weights["phrases"] +
        style_score * weights["style"]
    )
    
    # Determine risk level
    if overall_score >= 85:
        risk_level = "low"
        risk_description = "Document appears to be largely original"
    elif overall_score >= 70:
        risk_level = "moderate"
        risk_description = "Some areas may need review"
    elif overall_score >= 50:
        risk_level = "elevated"
        risk_description = "Multiple indicators suggest potential issues"
    else:
        risk_level = "high"
        risk_description = "Significant concerns detected"
    
    # Generate summary issues
    issues = []
    
    if internal_duplicates["duplicate_percentage"] > 5:
        issues.append({
            "type": "internal_duplicates",
            "severity": "high" if internal_duplicates["duplicate_percentage"] > 15 else "medium",
            "message": f"{internal_duplicates['duplicate_percentage']}% of content appears to be repeated internally"
        })
    
    if vocabulary["assessment"] == "low":
        issues.append({
            "type": "vocabulary",
            "severity": "medium",
            "message": "Low vocabulary diversity may indicate copied content"
        })
    
    if common_phrases["risk_level"] == "high":
        issues.append({
            "type": "boilerplate",
            "severity": "medium",
            "message": "High usage of common academic phrases detected"
        })
    
    if style["assessment"] == "inconsistent":
        issues.append({
            "type": "style_variation",
            "severity": "high",
            "message": "Writing style varies significantly across sections"
        })
    
    # ============ ENHANCED ANALYSIS (NEW) ============
    
    # Paraphrasing detection
    paraphrasing = _detect_paraphrasing(text)
    
    # Writing pattern analysis
    writing_patterns = _analyze_writing_patterns(text)
    
    # Web search check (sample key sentences)
    web_matches = []
    key_sentences = _extract_key_sentences(text, num_sentences=5)
    
    for sent in key_sentences[:3]:  # Check top 3 most distinctive sentences
        match_result = _check_sentence_against_web(sent)
        if match_result["has_match"]:
            web_matches.append(match_result)
    
    # Calculate additional scores
    paraphrase_score = max(0, 100 - paraphrasing["paraphrase_percentage"] * 2)
    pattern_score = writing_patterns["pattern_consistency"]
    web_match_score = 100 - (len(web_matches) * 15)  # -15 for each web match
    web_match_score = max(0, min(100, web_match_score))
    
    # Update weights for enhanced scoring (without citations)
    enhanced_weights = {
        "duplicates": 0.22,
        "vocabulary": 0.18,
        "phrases": 0.12,
        "style": 0.18,
        "paraphrasing": 0.12,
        "patterns": 0.10,
        "web_matches": 0.08
    }
    
    enhanced_score = (
        duplicate_score * enhanced_weights["duplicates"] +
        vocab_score * enhanced_weights["vocabulary"] +
        phrase_score * enhanced_weights["phrases"] +
        style_score * enhanced_weights["style"] +
        paraphrase_score * enhanced_weights["paraphrasing"] +
        pattern_score * enhanced_weights["patterns"] +
        web_match_score * enhanced_weights["web_matches"]
    )
    
    # Use enhanced score as the primary score
    overall_score = enhanced_score
    
    # Update risk level with enhanced analysis
    if overall_score >= 85:
        risk_level = "low"
        risk_description = "Document appears to be largely original with high confidence"
    elif overall_score >= 70:
        risk_level = "moderate"
        risk_description = "Some areas may need review; minor concerns detected"
    elif overall_score >= 50:
        risk_level = "elevated"
        risk_description = "Multiple indicators suggest potential plagiarism issues"
    else:
        risk_level = "high"
        risk_description = "Significant plagiarism concerns detected - immediate review required"
    
    # Add enhanced issues
    if paraphrasing["risk_level"] in ["medium", "high"]:
        issues.append({
            "type": "paraphrasing",
            "severity": paraphrasing["risk_level"],
            "message": f"Potential paraphrasing detected in {paraphrasing['paraphrase_percentage']}% of sentences"
        })
    
    if writing_patterns["assessment"] == "inconsistent":
        issues.append({
            "type": "writing_patterns",
            "severity": "high",
            "message": f"Writing patterns vary significantly across {len(writing_patterns['anomalies'])} sections"
        })
    
    if web_matches:
        issues.append({
            "type": "web_matches",
            "severity": "critical",
            "message": f"{len(web_matches)} sentences found with potential web source matches"
        })
    
    # Build preliminary results for AI analysis
    preliminary_results = {
        "overall_score": overall_score,
        "risk_level": risk_level,
        "component_scores": {
            "vocabulary_richness": round(vocab_score, 1),
            "style_consistency": round(style_score, 1),
        },
        "analyses": {
            "internal_duplicates": internal_duplicates
        }
    }
    
    # Run AI deep analysis for suspicious documents
    ai_analysis = {}
    if overall_score < 80:
        ai_analysis = _ai_deep_analysis(text, preliminary_results)
        
        # Incorporate AI findings
        if ai_analysis.get("analysis_completed"):
            ai_likelihood = ai_analysis.get("plagiarism_likelihood", 0)
            
            # Blend AI assessment with algorithmic score
            if ai_likelihood > 50:
                # Reduce score if AI detects high likelihood
                overall_score = (overall_score * 0.6) + ((100 - ai_likelihood) * 0.4)
                
            # Add AI-detected issues
            for ai_issue in ai_analysis.get("detected_issues", []):
                if ai_issue.get("severity") in ["high", "critical"]:
                    issues.append({
                        "type": f"ai_{ai_issue.get('type', 'unknown')}",
                        "severity": ai_issue.get("severity"),
                        "message": ai_issue.get("description", "AI detected potential issue"),
                        "evidence": ai_issue.get("evidence", "")[:100]
                    })
    
    # Final risk level recalculation
    if overall_score >= 85:
        risk_level = "low"
        risk_description = "Document appears to be original with high confidence"
    elif overall_score >= 70:
        risk_level = "moderate"
        risk_description = "Some areas may need review"
    elif overall_score >= 50:
        risk_level = "elevated"
        risk_description = "Multiple indicators suggest potential issues"
    else:
        risk_level = "high"
        risk_description = "Significant concerns detected - requires thorough review"

    return {
        "overall_score": round(overall_score, 1),
        "originality_score": round(overall_score, 1),
        "risk_level": risk_level,
        "risk_description": risk_description,
        "issues": issues,
        "component_scores": {
            "duplicate_check": round(duplicate_score, 1),
            "vocabulary_richness": round(vocab_score, 1),
            "phrase_originality": round(phrase_score, 1),
            "style_consistency": round(style_score, 1),
            "paraphrasing_check": round(paraphrase_score, 1),
            "pattern_consistency": round(pattern_score, 1),
            "web_source_check": round(web_match_score, 1)
        },
        "analyses": {
            "internal_duplicates": internal_duplicates,
            "vocabulary": vocabulary,
            "common_phrases": common_phrases,
            "style_consistency": style,
            "fingerprint": fingerprint,
            "paraphrasing": paraphrasing,
            "writing_patterns": writing_patterns,
            "web_matches": web_matches,
            "ai_analysis": ai_analysis if ai_analysis else None
        },
        "web_sources_found": len(web_matches),
        "ai_enhanced": bool(ai_analysis.get("analysis_completed")),
        "ai_content_probability": ai_analysis.get("ai_content_probability", 0) if ai_analysis else 0,
        "word_count": len(text.split()),
        "filename": filename,
        "analysis_version": "2.0-enhanced"
    }


def enhance_plagiarism_with_llm(text: str, basic_analysis: dict) -> dict:
    """
    Use LLM to enhance plagiarism analysis with contextual understanding.
    This is optional and supplements the algorithmic analysis.
    """
    api_key = _get_api_key()
    
    if not api_key:
        return basic_analysis
    
    # Only use LLM if there are concerning signals
    if basic_analysis.get("overall_score", 100) > 80:
        basic_analysis["llm_enhanced"] = False
        return basic_analysis
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare a summary of concerns for LLM
        concerns = [issue["message"] for issue in basic_analysis.get("issues", [])]
        
        system_msg = """You are an academic integrity expert. Analyze the provided text excerpt and algorithmic analysis results. 
Provide a brief assessment (2-3 sentences) of the originality concerns and specific recommendations.
Respond in JSON with keys: assessment (string), recommendations (array of 3 strings), confidence (float 0-1)."""
        
        user_msg = f"""Document excerpt (first 2000 chars):
{text[:2000]}

Algorithmic Analysis Results:
- Overall Score: {basic_analysis.get('overall_score', 'N/A')}/100
- Risk Level: {basic_analysis.get('risk_level', 'unknown')}
- Detected Issues: {', '.join(concerns) if concerns else 'None significant'}
- Vocabulary Richness: {basic_analysis.get('component_scores', {}).get('vocabulary_richness', 'N/A')}%
- Style Consistency: {basic_analysis.get('component_scores', {}).get('style_consistency', 'N/A')}%

Provide your assessment and recommendations."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=400,
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        
        llm_result = json.loads(raw)
        
        basic_analysis["llm_enhanced"] = True
        basic_analysis["llm_assessment"] = llm_result.get("assessment", "")
        basic_analysis["llm_recommendations"] = llm_result.get("recommendations", [])
        basic_analysis["llm_confidence"] = llm_result.get("confidence", 0.5)
        
    except Exception as exc:
        _debug("plagiarism_llm.error", str(exc))
        basic_analysis["llm_enhanced"] = False
    
    return basic_analysis


def analyze_citations(text: str, filename: str = "") -> dict:
    """Comprehensive citation analysis for research papers.
    
    Analyzes:
    - Citation detection and extraction (various formats)
    - Citation density and distribution
    - Format consistency
    - Reference verification (simulated web check)
    - Citation-claim alignment
    - Missing citation detection
    - Reference quality metrics
    
    Returns dict with detailed citation analysis.
    """
    import re
    from collections import Counter
    from urllib.parse import quote_plus
    
    if not text or len(text.strip()) < 100:
        return {
            "total_citations": 0,
            "citation_score": 0,
            "error": "Document too short for citation analysis",
            "citations": [],
            "references": [],
            "issues": [],
            "verified_count": 0,
            "unverified_count": 0
        }
    
    # ============ CITATION DETECTION ============
    
    # Define citation patterns for different styles
    citation_patterns = {
        "numeric_bracket": r'\[(\d+(?:\s*[-–,]\s*\d+)*)\]',  # [1], [1-3], [1, 2, 3]
        "numeric_paren": r'\((\d+(?:\s*[-–,]\s*\d+)*)\)',  # (1), (1-3)
        "author_year": r'\(([A-Z][a-zA-Z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-zA-Z]+)?,?\s*\d{4}[a-z]?)\)',  # (Smith, 2020), (Smith et al., 2020)
        "author_year_no_paren": r'([A-Z][a-zA-Z]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][a-zA-Z]+)?)\s*\((\d{4}[a-z]?)\)',  # Smith (2020)
        "superscript_style": r'(?<=[a-zA-Z.,])(\d{1,3})(?=\s|[.,;:\)]|$)',  # Superscript-like numbers after text
    }
    
    detected_citations = []
    citation_styles_found = set()
    
    # Extract citations by pattern
    for style_name, pattern in citation_patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            citation_text = match.group(0)
            citation_content = match.group(1) if match.lastindex else match.group(0)
            position = match.start()
            
            # Get surrounding context (50 chars before and after)
            context_start = max(0, position - 60)
            context_end = min(len(text), position + len(citation_text) + 60)
            context = text[context_start:context_end]
            
            # Calculate approximate page and paragraph
            text_before = text[:position]
            approx_page = (len(text_before) // 3000) + 1
            paragraph_count = text_before.count('\n\n') + 1
            
            detected_citations.append({
                "text": citation_text,
                "content": citation_content,
                "style": style_name,
                "position": position,
                "context": context.strip(),
                "page": approx_page,
                "paragraph": paragraph_count
            })
            citation_styles_found.add(style_name)
    
    # Remove duplicates based on position (keep unique positions)
    seen_positions = set()
    unique_citations = []
    for cit in sorted(detected_citations, key=lambda x: x["position"]):
        # Allow positions within 5 chars to be considered same
        is_duplicate = any(abs(cit["position"] - pos) < 5 for pos in seen_positions)
        if not is_duplicate:
            unique_citations.append(cit)
            seen_positions.add(cit["position"])
    
    detected_citations = unique_citations
    
    # ============ REFERENCE EXTRACTION ============
    
    # Find references section
    references = []
    ref_section_match = re.search(
        r'(?:references|bibliography|works\s+cited|literature\s+cited)\s*\n',
        text,
        re.IGNORECASE
    )
    
    if ref_section_match:
        ref_section = text[ref_section_match.end():]
        
        # Split into individual references
        # Try numbered format first
        numbered_refs = re.split(r'\n\s*\[?\d{1,3}\]?\.?\s+', ref_section)
        
        if len(numbered_refs) > 2:
            for i, ref in enumerate(numbered_refs[1:], 1):  # Skip first empty
                ref_clean = ref.strip()
                if len(ref_clean) > 20:  # Valid reference
                    ref_info = parse_reference(ref_clean, i)
                    references.append(ref_info)
        else:
            # Try paragraph-based splitting
            paragraphs = ref_section.split('\n\n')
            for i, para in enumerate(paragraphs, 1):
                para_clean = para.strip()
                if len(para_clean) > 30 and not para_clean.lower().startswith(('appendix', 'figure', 'table')):
                    ref_info = parse_reference(para_clean, i)
                    references.append(ref_info)
                if len(references) >= 100:  # Limit
                    break
    
    # ============ REFERENCE VERIFICATION ============
    
    verified_refs = []
    unverified_refs = []
    
    for ref in references[:20]:  # Limit verification to first 20
        verification = verify_reference(ref)
        ref["verification"] = verification
        if verification["verified"]:
            verified_refs.append(ref)
        else:
            unverified_refs.append(ref)
    
    # ============ CITATION DENSITY ANALYSIS ============
    
    word_count = len(text.split())
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    citation_density = {
        "citations_per_1000_words": round((len(detected_citations) / max(word_count, 1)) * 1000, 2),
        "citations_per_sentence": round(len(detected_citations) / max(sentence_count, 1), 3),
        "total_citations": len(detected_citations),
        "total_references": len(references)
    }
    
    # Analyze citation distribution by page
    page_distribution = Counter(cit["page"] for cit in detected_citations)
    total_pages = max(page_distribution.keys()) if page_distribution else 1
    
    # ============ CITATION ISSUES DETECTION ============
    
    issues = []
    
    # Check for style inconsistency
    if len(citation_styles_found) > 1:
        # Filter out false positives
        major_styles = [s for s in citation_styles_found if s not in ["superscript_style"]]
        if len(major_styles) > 1:
            issues.append({
                "type": "inconsistency",
                "severity": "major",
                "title": "Mixed Citation Styles",
                "description": f"Multiple citation formats detected: {', '.join(major_styles)}. Academic papers should use consistent citation style.",
                "recommendation": "Choose one citation style (APA, IEEE, Chicago, etc.) and apply consistently throughout."
            })
    
    # Check citation density
    if citation_density["citations_per_1000_words"] < 5 and word_count > 1000:
        issues.append({
            "type": "density",
            "severity": "major",
            "title": "Low Citation Density",
            "description": f"Only {citation_density['citations_per_1000_words']} citations per 1000 words. Research papers typically have 15-30 citations per 1000 words.",
            "recommendation": "Add more citations to support claims, especially in introduction and related work sections."
        })
    elif citation_density["citations_per_1000_words"] > 50:
        issues.append({
            "type": "density",
            "severity": "minor",
            "title": "Very High Citation Density",
            "description": f"Unusually high citation density ({citation_density['citations_per_1000_words']} per 1000 words). This may indicate over-citation.",
            "recommendation": "Review if all citations are necessary and contribute meaningfully to the argument."
        })
    
    # Check for citation-reference mismatch
    cited_numbers = set()
    for cit in detected_citations:
        if cit["style"] in ["numeric_bracket", "numeric_paren"]:
            # Extract all numbers from citation
            nums = re.findall(r'\d+', cit["content"])
            for n in nums:
                cited_numbers.add(int(n))
    
    if cited_numbers and references:
        max_cited = max(cited_numbers) if cited_numbers else 0
        if max_cited > len(references):
            issues.append({
                "type": "mismatch",
                "severity": "critical",
                "title": "Citation-Reference Mismatch",
                "description": f"Citations reference up to [{max_cited}], but only {len(references)} references found.",
                "recommendation": "Ensure all cited numbers correspond to entries in the references section."
            })
    
    # Check for uncited references
    if references and cited_numbers:
        ref_numbers = set(range(1, len(references) + 1))
        uncited = ref_numbers - cited_numbers
        if uncited and len(uncited) > 2:
            issues.append({
                "type": "uncited",
                "severity": "minor",
                "title": "Uncited References",
                "description": f"References {list(uncited)[:5]}{'...' if len(uncited) > 5 else ''} are not cited in the text.",
                "recommendation": "Remove unused references or ensure all listed references are cited."
            })
    
    # Check for unverified/suspicious references
    if len(unverified_refs) > len(verified_refs) and len(references) > 5:
        issues.append({
            "type": "verification",
            "severity": "major",
            "title": "Many Unverifiable References",
            "description": f"{len(unverified_refs)} out of {len(references[:20])} checked references could not be verified.",
            "recommendation": "Double-check reference accuracy. Ensure DOIs, publication years, and author names are correct."
        })
    
    # Check for self-citation pattern
    self_citation_count = 0
    for ref in references:
        if ref.get("potential_self_cite"):
            self_citation_count += 1
    
    if self_citation_count > len(references) * 0.3 and len(references) > 5:
        issues.append({
            "type": "self_citation",
            "severity": "minor",
            "title": "High Self-Citation Ratio",
            "description": f"Approximately {self_citation_count} references may be self-citations ({round(self_citation_count/len(references)*100)}%).",
            "recommendation": "While self-citation is acceptable, excessive self-citation may raise concerns. Aim for diverse sources."
        })
    
    # ============ MISSING CITATION DETECTION ============
    
    # Statements that typically need citations
    claim_patterns = [
        (r'studies\s+(?:have\s+)?show(?:n|ed|s)?', "Research claim"),
        (r'research\s+(?:has\s+)?(?:demonstrate|indicate|suggest|reveal)', "Research claim"),
        (r'according\s+to', "Attribution"),
        (r'it\s+(?:has\s+been|is)\s+(?:proven|shown|demonstrated|established)', "Established fact claim"),
        (r'previous(?:ly)?\s+(?:work|research|studies)', "Prior work reference"),
        (r'\d+%\s+of', "Statistical claim"),
        (r'(?:first|originally)\s+(?:proposed|introduced|developed)\s+(?:by|in)', "Origin claim"),
        (r'widely\s+(?:accepted|known|used|adopted)', "Consensus claim"),
    ]
    
    unsupported_claims = []
    for pattern, claim_type in claim_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            pos = match.start()
            # Check if there's a citation within 100 characters after
            nearby_text = text[pos:min(pos + 150, len(text))]
            has_nearby_citation = any(
                re.search(p, nearby_text) for p in citation_patterns.values()
            )
            
            if not has_nearby_citation:
                context_start = max(0, pos - 30)
                context_end = min(len(text), pos + 100)
                unsupported_claims.append({
                    "type": claim_type,
                    "text": match.group(0),
                    "context": text[context_start:context_end].strip(),
                    "position": pos,
                    "page": (len(text[:pos]) // 3000) + 1
                })
    
    # Limit to most important unsupported claims
    if len(unsupported_claims) > 10:
        # Prioritize by claim type importance
        priority_order = ["Statistical claim", "Research claim", "Established fact claim", "Origin claim", "Prior work reference", "Attribution", "Consensus claim"]
        unsupported_claims.sort(key=lambda x: priority_order.index(x["type"]) if x["type"] in priority_order else 99)
        unsupported_claims = unsupported_claims[:10]
    
    if unsupported_claims:
        issues.append({
            "type": "missing_citation",
            "severity": "major",
            "title": f"{len(unsupported_claims)} Potentially Unsupported Claims",
            "description": "Several statements that typically require citations appear to lack proper references.",
            "recommendation": "Review flagged statements and add appropriate citations to support claims.",
            "claims": unsupported_claims[:5]  # Include top 5 examples
        })
    
    # ============ REFERENCE QUALITY ANALYSIS ============
    
    years = []
    for ref in references:
        if ref.get("year"):
            try:
                year = int(ref["year"])
                if 1900 < year <= 2026:
                    years.append(year)
            except ValueError:
                pass
    
    reference_quality = {
        "total_references": len(references),
        "verified_count": len(verified_refs),
        "unverified_count": len(unverified_refs),
        "avg_year": round(sum(years) / len(years)) if years else None,
        "newest_year": max(years) if years else None,
        "oldest_year": min(years) if years else None,
        "recent_refs_count": len([y for y in years if y >= 2020]),
        "outdated_refs_count": len([y for y in years if y < 2010])
    }
    
    # Check reference recency
    if years and reference_quality["avg_year"] and reference_quality["avg_year"] < 2015:
        issues.append({
            "type": "recency",
            "severity": "minor",
            "title": "Outdated References",
            "description": f"Average reference year is {reference_quality['avg_year']}. Consider citing more recent work.",
            "recommendation": "Include recent publications (2020+) to show awareness of current developments in the field."
        })
    
    # ============ CALCULATE CITATION SCORE ============
    
    score = 100
    
    # Deduct for issues
    severity_deductions = {"critical": 25, "major": 15, "minor": 5}
    for issue in issues:
        score -= severity_deductions.get(issue["severity"], 5)
    
    # Bonus for good practices
    if len(citation_styles_found) == 1:
        score += 5  # Consistent style
    if reference_quality["recent_refs_count"] > 5:
        score += 5  # Good recent references
    if len(verified_refs) > len(unverified_refs):
        score += 5  # Most refs verified
    
    score = max(0, min(100, score))
    
    # Determine risk level
    if score >= 85:
        risk_level = "excellent"
    elif score >= 70:
        risk_level = "good"
    elif score >= 50:
        risk_level = "needs_improvement"
    else:
        risk_level = "poor"
    
    # ============ LLM ENHANCEMENT ============
    
    llm_analysis = None
    api_key = _get_api_key()
    
    if api_key and len(detected_citations) > 0:
        try:
            client = OpenAI(api_key=api_key)
            
            # Prepare citation summary for LLM
            citation_summary = f"""
Document has {len(detected_citations)} citations and {len(references)} references.
Citation styles found: {', '.join(citation_styles_found)}
Citation density: {citation_density['citations_per_1000_words']} per 1000 words
Verified references: {len(verified_refs)}/{len(references[:20])} checked
Issues detected: {len(issues)}

Sample unsupported claims:
{json.dumps(unsupported_claims[:3], indent=2) if unsupported_claims else 'None detected'}

Sample references:
{json.dumps([{'title': r.get('title', 'Unknown')[:60], 'year': r.get('year'), 'verified': r.get('verification', {}).get('verified', False)} for r in references[:5]], indent=2)}
"""
            
            system_msg = """You are an academic citation expert. Analyze the citation practices in this research paper summary and provide:
1. An overall assessment of citation quality (2-3 sentences)
2. Specific recommendations for improvement (3-5 items)
3. A credibility assessment (how well does the paper support its claims with evidence)

Respond ONLY in JSON:
{
    "assessment": "Overall assessment text",
    "recommendations": ["rec1", "rec2", "rec3"],
    "credibility": "high|medium|low",
    "credibility_reason": "Brief explanation",
    "notable_issues": ["issue1", "issue2"]
}"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": citation_summary}
                ],
                temperature=0.3,
                max_tokens=500,
            )
            
            raw = response.choices[0].message.content.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            
            llm_analysis = json.loads(raw)
            
        except Exception as exc:
            _debug("citation_llm.error", str(exc))
    
    # ============ ADVANCED ANALYTICS ============
    
    # Citation Timeline (year distribution)
    year_distribution = Counter()
    for ref in references:
        if ref.get("year"):
            try:
                year = int(ref["year"])
                if 1950 <= year <= 2026:
                    year_distribution[year] = year_distribution.get(year, 0) + 1
            except ValueError:
                pass
    
    # Create timeline data sorted by year
    timeline_data = []
    if year_distribution:
        min_year = min(year_distribution.keys())
        max_year = max(year_distribution.keys())
        for year in range(min_year, max_year + 1):
            timeline_data.append({
                "year": year,
                "count": year_distribution.get(year, 0),
                "is_recent": year >= 2020
            })
    
    # Source Type Classification
    source_types = {"journal": 0, "conference": 0, "book": 0, "thesis": 0, "web": 0, "other": 0}
    for ref in references:
        raw_text = ref.get("raw_text", "").lower()
        if any(kw in raw_text for kw in ["journal", "transactions", "letters", "review"]):
            source_types["journal"] += 1
        elif any(kw in raw_text for kw in ["conference", "proceedings", "symposium", "workshop", "ieee", "acm"]):
            source_types["conference"] += 1
        elif any(kw in raw_text for kw in ["book", "chapter", "edition", "publisher", "press"]):
            source_types["book"] += 1
        elif any(kw in raw_text for kw in ["thesis", "dissertation", "phd", "master"]):
            source_types["thesis"] += 1
        elif ref.get("url") and not ref.get("doi"):
            source_types["web"] += 1
        else:
            source_types["other"] += 1
    
    # Author Diversity Analysis
    all_authors = []
    unique_first_authors = set()
    for ref in references:
        authors = ref.get("authors", "")
        if authors:
            # Extract first author's last name
            first_author_match = re.match(r'^([A-Z][a-zA-Z\-]+)', authors)
            if first_author_match:
                unique_first_authors.add(first_author_match.group(1).lower())
            # Count comma-separated authors
            author_count = len(re.findall(r'[A-Z][a-zA-Z]+', authors[:100]))
            all_authors.append(min(author_count, 10))
    
    author_diversity = {
        "unique_first_authors": len(unique_first_authors),
        "total_references": len(references),
        "diversity_ratio": round(len(unique_first_authors) / max(len(references), 1) * 100, 1),
        "avg_authors_per_ref": round(sum(all_authors) / max(len(all_authors), 1), 1) if all_authors else 0
    }
    
    # Section-wise Citation Distribution (based on position in document)
    section_distribution = {"introduction": 0, "methods": 0, "results": 0, "discussion": 0}
    doc_length = len(text)
    for cit in detected_citations:
        position_ratio = cit["position"] / max(doc_length, 1)
        if position_ratio < 0.2:
            section_distribution["introduction"] += 1
        elif position_ratio < 0.4:
            section_distribution["methods"] += 1
        elif position_ratio < 0.7:
            section_distribution["results"] += 1
        else:
            section_distribution["discussion"] += 1
    
    # Citation Clustering - find citation "bursts" (multiple citations close together)
    citation_clusters = []
    if len(detected_citations) > 1:
        cluster_start = 0
        current_cluster = [detected_citations[0]]
        
        for i in range(1, len(detected_citations)):
            curr_pos = detected_citations[i]["position"]
            prev_pos = detected_citations[i-1]["position"]
            
            if curr_pos - prev_pos < 200:  # Within 200 chars = same cluster
                current_cluster.append(detected_citations[i])
            else:
                if len(current_cluster) >= 2:
                    citation_clusters.append({
                        "size": len(current_cluster),
                        "position": current_cluster[0]["position"],
                        "page": current_cluster[0]["page"],
                        "citations": [c["text"] for c in current_cluster[:5]]
                    })
                current_cluster = [detected_citations[i]]
        
        # Don't forget last cluster
        if len(current_cluster) >= 2:
            citation_clusters.append({
                "size": len(current_cluster),
                "position": current_cluster[0]["position"],
                "page": current_cluster[0]["page"],
                "citations": [c["text"] for c in current_cluster[:5]]
            })
    
    # Reference Quality Score Breakdown
    quality_metrics = {
        "recency_score": min(100, (reference_quality.get("recent_refs_count", 0) / max(len(references), 1)) * 200),
        "verification_score": (len(verified_refs) / max(len(references[:20]), 1)) * 100 if references else 0,
        "completeness_score": sum(1 for r in references if r.get("title") and r.get("year") and r.get("authors")) / max(len(references), 1) * 100,
        "diversity_score": author_diversity["diversity_ratio"],
        "density_score": min(100, (citation_density.get("citations_per_1000_words", 0) / 20) * 100)
    }
    
    # Citation Health Indicators
    health_indicators = []
    
    if quality_metrics["recency_score"] >= 70:
        health_indicators.append({"label": "Recent Sources", "status": "good", "value": f"{reference_quality.get('recent_refs_count', 0)} refs from 2020+"})
    elif quality_metrics["recency_score"] >= 40:
        health_indicators.append({"label": "Recent Sources", "status": "warning", "value": "Some recent, needs more"})
    else:
        health_indicators.append({"label": "Recent Sources", "status": "poor", "value": "Mostly outdated sources"})
    
    if quality_metrics["verification_score"] >= 70:
        health_indicators.append({"label": "Reference Quality", "status": "good", "value": f"{len(verified_refs)} verified"})
    elif quality_metrics["verification_score"] >= 40:
        health_indicators.append({"label": "Reference Quality", "status": "warning", "value": "Partially verifiable"})
    else:
        health_indicators.append({"label": "Reference Quality", "status": "poor", "value": "Many unverifiable"})
    
    if len(citation_styles_found) == 1:
        health_indicators.append({"label": "Style Consistency", "status": "good", "value": list(citation_styles_found)[0].replace("_", " ").title()})
    else:
        health_indicators.append({"label": "Style Consistency", "status": "warning", "value": f"{len(citation_styles_found)} styles mixed"})
    
    if author_diversity["diversity_ratio"] >= 80:
        health_indicators.append({"label": "Author Diversity", "status": "good", "value": f"{author_diversity['unique_first_authors']} unique authors"})
    elif author_diversity["diversity_ratio"] >= 50:
        health_indicators.append({"label": "Author Diversity", "status": "warning", "value": "Moderate diversity"})
    else:
        health_indicators.append({"label": "Author Diversity", "status": "poor", "value": "Low diversity - expand sources"})
    
    return {
        "citation_score": score,
        "risk_level": risk_level,
        "total_citations": len(detected_citations),
        "total_references": len(references),
        "citations": detected_citations[:50],
        "references": references[:30],
        "verified_count": len(verified_refs),
        "unverified_count": len(unverified_refs),
        "citation_density": citation_density,
        "page_distribution": dict(page_distribution),
        "styles_found": list(citation_styles_found),
        "issues": issues,
        "unsupported_claims": unsupported_claims[:10],
        "reference_quality": reference_quality,
        "llm_analysis": llm_analysis,
        # Advanced Analytics
        "timeline_data": timeline_data,
        "source_types": source_types,
        "author_diversity": author_diversity,
        "section_distribution": section_distribution,
        "citation_clusters": citation_clusters[:10],
        "quality_metrics": quality_metrics,
        "health_indicators": health_indicators
    }


def parse_reference(ref_text: str, ref_number: int) -> dict:
    """Parse a reference string to extract structured information."""
    import re
    
    ref_info = {
        "number": ref_number,
        "raw_text": ref_text[:300],  # Limit length
        "title": None,
        "authors": None,
        "year": None,
        "journal": None,
        "doi": None,
        "url": None,
        "potential_self_cite": False
    }
    
    # Extract year
    year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
    if year_match:
        ref_info["year"] = year_match.group(0)
    
    # Extract DOI
    doi_match = re.search(r'10\.\d{4,}/[^\s]+', ref_text)
    if doi_match:
        ref_info["doi"] = doi_match.group(0).rstrip('.,;')
    
    # Extract URL
    url_match = re.search(r'https?://[^\s<>"]+', ref_text)
    if url_match:
        ref_info["url"] = url_match.group(0).rstrip('.,;)')
    
    # Extract title (usually in quotes or after year)
    title_match = re.search(r'"([^"]+)"', ref_text)
    if title_match:
        ref_info["title"] = title_match.group(1)
    else:
        # Try to extract title after authors (before journal/year pattern)
        parts = ref_text.split('.')
        if len(parts) >= 2:
            potential_title = parts[1].strip() if len(parts[1]) > 10 else None
            if potential_title and not re.match(r'^\d', potential_title):
                ref_info["title"] = potential_title[:150]
    
    # Extract authors (usually at beginning)
    author_match = re.match(r'^([A-Z][^.]+\.)', ref_text)
    if author_match:
        authors_text = author_match.group(1)
        # Check for et al. or multiple author pattern
        ref_info["authors"] = authors_text[:100]
    
    # Detect potential journal name
    journal_patterns = [
        r'(?:journal|proceedings|transactions|conference|symposium|workshop)\s+(?:of|on)\s+[^,.\d]+',
        r'in\s+([A-Z][^,]+(?:Conference|Workshop|Symposium|Proceedings))',
    ]
    for pattern in journal_patterns:
        match = re.search(pattern, ref_text, re.IGNORECASE)
        if match:
            ref_info["journal"] = match.group(0)[:100]
            break
    
    return ref_info


def verify_reference(ref: dict) -> dict:
    """Attempt to verify a reference exists.
    
    Uses heuristics and patterns to assess reference validity.
    In a production system, this would query CrossRef, Google Scholar, etc.
    """
    import re
    
    verification = {
        "verified": False,
        "confidence": 0.0,
        "source": "heuristic",
        "issues": []
    }
    
    # If DOI exists, consider it more likely valid
    if ref.get("doi"):
        verification["verified"] = True
        verification["confidence"] = 0.9
        verification["source"] = "doi_present"
        return verification
    
    # If URL exists and looks like a valid academic source
    if ref.get("url"):
        academic_domains = ['doi.org', 'arxiv.org', 'ieee.org', 'acm.org', 'springer.com', 
                          'sciencedirect.com', 'wiley.com', 'nature.com', 'plos.org',
                          'ncbi.nlm.nih.gov', 'pubmed', 'researchgate.net', 'academia.edu']
        url = ref["url"].lower()
        if any(domain in url for domain in academic_domains):
            verification["verified"] = True
            verification["confidence"] = 0.85
            verification["source"] = "academic_url"
            return verification
    
    # Check for well-formed reference
    has_authors = ref.get("authors") and len(ref["authors"]) > 5
    has_year = ref.get("year") and 1900 < int(ref["year"]) <= 2026
    has_title = ref.get("title") and len(ref["title"]) > 10
    
    confidence = 0.0
    if has_authors:
        confidence += 0.3
    if has_year:
        confidence += 0.25
    if has_title:
        confidence += 0.3
    if ref.get("journal"):
        confidence += 0.15
    
    verification["confidence"] = round(confidence, 2)
    verification["verified"] = confidence >= 0.6
    
    # Flag potential issues
    if not has_year:
        verification["issues"].append("Missing or invalid publication year")
    if not has_title:
        verification["issues"].append("Title not clearly identifiable")
    if not has_authors:
        verification["issues"].append("Authors not clearly specified")
    
    return verification


# ============================================================================
# REWRITE-TO-ACCEPT SYSTEM
# ============================================================================

def identify_paper_sections(text: str) -> dict:
    """Identify and extract major sections from a research paper.
    
    Returns dict with section names mapped to their text content.
    """
    import re
    
    # Common academic paper section patterns (without inline flags - we use re.IGNORECASE)
    section_patterns = [
        (r'\babstract\b', 'abstract'),
        (r'\bintroduction\b', 'introduction'),
        (r'\bliterature\s*review\b', 'literature_review'),
        (r'\brelated\s*work\b', 'related_work'),
        (r'\bbackground\b', 'background'),
        (r'\bmethodology\b|\bmethods?\b', 'methodology'),
        (r'\bexperiment(?:s|al)?\b', 'experiments'),
        (r'\bresults?\b', 'results'),
        (r'\bdiscussion\b', 'discussion'),
        (r'\bconclusions?\b', 'conclusion'),
        (r'\bfuture\s*work\b', 'future_work'),
        (r'\bcontributions?\b', 'contributions'),
        (r'\blimitations?\b', 'limitations'),
        (r'\breferences?\b|\bbibliography\b', 'references'),
        (r'\backnowledg(?:e)?ments?\b', 'acknowledgments'),
    ]
    
    sections = {}
    lines = text.split('\n')
    current_section = 'preamble'
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check if this line is a section header
        is_header = False
        for pattern, section_name in section_patterns:
            if re.match(f'^\\s*\\d*\\.?\\s*{pattern}\\s*$', stripped, re.IGNORECASE) or \
               (len(stripped) < 50 and re.search(pattern, stripped) and stripped.isupper()):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = section_name
                current_content = []
                is_header = True
                break
        
        if not is_header:
            current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    # Special handling for abstract (often at the beginning)
    if 'abstract' not in sections and 'preamble' in sections:
        preamble = sections['preamble']
        abstract_match = re.search(r'(?i)abstract[:\s]*(.+?)(?=\n\n|\bintroduction\b|$)', preamble, re.DOTALL)
        if abstract_match:
            sections['abstract'] = abstract_match.group(1).strip()
    
    return sections


def generate_section_rewrite(section_name: str, section_text: str, feedback: str, full_context: str = "") -> dict:
    """Generate a rewritten version of a section based on feedback.
    
    Args:
        section_name: Name of the section (abstract, introduction, etc.)
        section_text: Original section text
        feedback: Specific feedback/suggestion to address
        full_context: Full paper context for coherence
    
    Returns dict with original, rewrite, explanation, and improvements.
    """
    api_key = _get_api_key()
    
    if not api_key:
        return {
            "success": False,
            "error": "API key not configured",
            "original": section_text,
            "rewrite": None
        }
    
    client = OpenAI(api_key=api_key)
    
    # Section-specific prompts
    section_prompts = {
        'abstract': """You are an expert academic editor specializing in research paper abstracts.
An abstract should:
- State the problem/motivation (1-2 sentences)
- Describe the approach/methodology (1-2 sentences)
- Highlight key results/findings (1-2 sentences)
- State the significance/implications (1 sentence)
Keep it concise (150-300 words typically).""",

        'introduction': """You are an expert academic editor specializing in introductions.
A strong introduction should:
- Hook the reader with problem significance
- Establish context and background
- Clearly state the research gap
- Present the thesis/research questions
- Outline the paper structure
- Preview key contributions""",

        'methodology': """You are an expert academic editor specializing in methodology sections.
A good methodology should:
- Clearly describe the research design
- Explain data collection methods
- Detail analysis procedures
- Justify methodological choices
- Address validity and reliability
- Be reproducible by other researchers""",

        'results': """You are an expert academic editor specializing in results sections.
Results should:
- Present findings objectively
- Use clear data organization
- Reference figures/tables appropriately
- Highlight significant findings
- Avoid interpretation (save for discussion)""",

        'discussion': """You are an expert academic editor specializing in discussion sections.
A strong discussion should:
- Interpret key findings
- Compare with existing literature
- Address implications
- Acknowledge limitations
- Suggest future research directions""",

        'conclusion': """You are an expert academic editor specializing in conclusions.
A strong conclusion should:
- Summarize main findings
- Restate significance
- Present practical implications
- NOT introduce new information
- End with impact statement""",

        'contributions': """You are an expert academic editor specializing in contribution statements.
Strong contributions should:
- Be specific and measurable
- Distinguish from prior work
- Highlight novelty clearly
- Be ordered by importance
- Use action verbs"""
    }
    
    base_prompt = section_prompts.get(section_name, """You are an expert academic editor.
Improve the text for clarity, conciseness, and academic rigor.""")
    
    system_msg = f"""{base_prompt}

Your task is to rewrite the given section to address the specific feedback while:
1. Maintaining the author's core meaning and voice
2. Improving clarity and academic tone
3. Making it more compelling for reviewers
4. Ensuring proper academic language

Respond in JSON format:
{{
    "rewritten_text": "The improved section text",
    "changes_made": ["list of specific improvements"],
    "word_count_change": "increased/decreased/same",
    "confidence": 0.0-1.0
}}"""

    user_msg = f"""Section: {section_name.upper()}

FEEDBACK TO ADDRESS:
{feedback}

ORIGINAL TEXT:
{section_text[:3000]}

{f"PAPER CONTEXT (for coherence):{chr(10)}{full_context[:1500]}" if full_context else ""}

Please provide a rewritten version that addresses the feedback while maintaining academic quality."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.4,
            max_tokens=2000,
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Parse JSON
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        
        import json
        result = json.loads(raw)
        
        # Normalize response keys to match frontend expectations while keeping legacy keys
        resp = {
            "success": True,
            "original": section_text,
            "rewrite": result.get("rewritten_text", ""),
            "changes_made": result.get("changes_made", []),
            "word_count_change": result.get("word_count_change", "same"),
            "confidence": result.get("confidence", 0.7),
            "section": section_name
        }
        # Frontend expects these keys: original_text, rewritten_text, changes
        resp["original_text"] = resp.get("original")
        resp["rewritten_text"] = resp.get("rewrite")
        resp["changes"] = resp.get("changes_made")
        return resp
        
    except Exception as exc:
        _debug("section_rewrite.error", str(exc))
        # Return normalized failure payload
        return {
            "success": False,
            "error": str(exc),
            "original": section_text,
            "original_text": section_text,
            "rewrite": None,
            "rewritten_text": None
        }


def generate_targeted_rewrite(original_text: str, issue_type: str, issue_description: str, suggested_fix: str = "") -> dict:
    """Generate a targeted rewrite for a specific issue.
    
    Args:
        original_text: The text that needs improvement
        issue_type: Type of issue (grammar, clarity, structure, etc.)
        issue_description: Description of the problem
        suggested_fix: Optional suggested fix
    
    Returns dict with original, rewrite, and explanation.
    """
    api_key = _get_api_key()
    
    if not api_key:
        # Provide rule-based fixes for common issues
        return _rule_based_rewrite(original_text, issue_type, issue_description)
    
    client = OpenAI(api_key=api_key)
    
    issue_prompts = {
        'clarity': "Focus on making the text clearer and easier to understand. Remove ambiguity.",
        'grammar': "Fix grammatical errors while preserving meaning.",
        'conciseness': "Make the text more concise. Remove redundancy and wordiness.",
        'academic_tone': "Improve the academic tone. Make it more formal and scholarly.",
        'structure': "Improve sentence and paragraph structure for better flow.",
        'passive_voice': "Convert passive voice to active voice where appropriate.",
        'hedging': "Adjust hedging language (too much or too little certainty).",
        'citation': "Improve citation integration and referencing.",
        'transition': "Add or improve transitional phrases for better coherence.",
        'specificity': "Make claims more specific and concrete.",
    }
    
    specific_guidance = issue_prompts.get(issue_type, "Improve the text based on the described issue.")
    
    system_msg = f"""You are an expert academic editor providing targeted rewrites.

{specific_guidance}

Rules:
1. Keep the rewrite close to the original length (±20%)
2. Preserve the author's intended meaning
3. Maintain academic tone
4. Make minimal but impactful changes

Respond in JSON:
{{
    "rewritten_text": "improved text",
    "explanation": "brief explanation of changes",
    "improvement_score": 1-10
}}"""

    user_msg = f"""ISSUE TYPE: {issue_type}
PROBLEM: {issue_description}
{f"SUGGESTED FIX: {suggested_fix}" if suggested_fix else ""}

ORIGINAL TEXT:
{original_text[:1500]}

Provide a rewritten version that addresses this specific issue."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        raw = response.choices[0].message.content.strip()
        
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        
        import json
        result = json.loads(raw)
        
        return {
            "success": True,
            "original": original_text,
            "rewrite": result.get("rewritten_text", ""),
            "explanation": result.get("explanation", ""),
            "improvement_score": result.get("improvement_score", 5),
            "issue_type": issue_type
        }
        
    except Exception as exc:
        _debug("targeted_rewrite.error", str(exc))
        return _rule_based_rewrite(original_text, issue_type, issue_description)


def _rule_based_rewrite(original_text: str, issue_type: str, issue_description: str) -> dict:
    """Provide rule-based rewrites when LLM is unavailable."""
    import re
    
    rewrite = original_text
    changes = []
    
    if issue_type == 'passive_voice':
        # Simple passive to active conversion hints
        passive_patterns = [
            (r'\bwas\s+(\w+ed)\s+by\b', 'Consider: "[subject] \\1"'),
            (r'\bwere\s+(\w+ed)\s+by\b', 'Consider: "[subjects] \\1"'),
            (r'\bis\s+(\w+ed)\s+by\b', 'Consider: "[subject] \\1s"'),
        ]
        for pattern, suggestion in passive_patterns:
            if re.search(pattern, original_text, re.IGNORECASE):
                changes.append(f"Passive voice detected. {suggestion}")
    
    elif issue_type == 'conciseness':
        # Remove common filler phrases
        filler_phrases = [
            (r'\bin order to\b', 'to'),
            (r'\bdue to the fact that\b', 'because'),
            (r'\bat this point in time\b', 'now'),
            (r'\bin the event that\b', 'if'),
            (r'\bit is important to note that\b', 'notably,'),
            (r'\bthe fact that\b', 'that'),
            (r'\ba large number of\b', 'many'),
            (r'\bin spite of the fact that\b', 'although'),
        ]
        for pattern, replacement in filler_phrases:
            if re.search(pattern, rewrite, re.IGNORECASE):
                rewrite = re.sub(pattern, replacement, rewrite, flags=re.IGNORECASE)
                changes.append(f"Replaced wordy phrase with '{replacement}'")
    
    elif issue_type == 'academic_tone':
        # Improve academic tone
        informal_phrases = [
            (r'\bget\b', 'obtain'),
            (r'\bgot\b', 'obtained'),
            (r'\bshow\b', 'demonstrate'),
            (r'\bfind out\b', 'determine'),
            (r'\blook at\b', 'examine'),
            (r'\ba lot of\b', 'numerous'),
            (r'\bbig\b', 'significant'),
            (r'\bbad\b', 'adverse'),
            (r'\bgood\b', 'beneficial'),
        ]
        for pattern, replacement in informal_phrases:
            if re.search(pattern, rewrite, re.IGNORECASE):
                rewrite = re.sub(pattern, replacement, rewrite, flags=re.IGNORECASE)
                changes.append(f"Improved formality: '{pattern[2:-2]}' → '{replacement}'")
    
    elif issue_type == 'hedging':
        # Adjust hedging language
        over_hedging = [
            (r'\bseems to perhaps\b', 'may'),
            (r'\bmight possibly\b', 'may'),
            (r'\bcould potentially\b', 'could'),
        ]
        for pattern, replacement in over_hedging:
            if re.search(pattern, rewrite, re.IGNORECASE):
                rewrite = re.sub(pattern, replacement, rewrite, flags=re.IGNORECASE)
                changes.append(f"Reduced over-hedging")
    
    return {
        "success": True,
        "original": original_text,
        "rewrite": rewrite if changes else original_text,
        "explanation": "; ".join(changes) if changes else "No automatic fixes available. Manual review recommended.",
        "improvement_score": 5 if changes else 3,
        "issue_type": issue_type,
        "rule_based": True
    }


def generate_contribution_bullets(text: str, current_contributions: list = None) -> dict:
    """Generate improved contribution bullets for a paper.
    
    Args:
        text: Full paper text for context
        current_contributions: Existing contribution statements if any
    
    Returns dict with suggested contribution bullets.
    """
    api_key = _get_api_key()
    
    if not api_key:
        return {
            "success": False,
            "error": "API key not configured",
            "contributions": []
        }
    
    client = OpenAI(api_key=api_key)
    
    system_msg = """You are an expert academic editor specializing in framing research contributions.

Strong contribution statements should:
1. Start with action verbs (We propose, We introduce, We demonstrate, We develop)
2. Be specific and measurable
3. Clearly differentiate from prior work
4. Highlight novelty and impact
5. Be ordered by importance (most significant first)

Respond in JSON:
{
    "contributions": [
        {
            "bullet": "The contribution statement",
            "type": "methodological|theoretical|empirical|practical",
            "novelty_level": "high|medium|low",
            "explanation": "Why this is a contribution"
        }
    ],
    "summary": "One sentence summary of the work's impact"
}"""

    current_text = ""
    if current_contributions:
        current_text = f"\n\nCURRENT CONTRIBUTIONS (to improve):\n" + "\n".join(f"- {c}" for c in current_contributions)

    user_msg = f"""Based on the following paper content, generate 3-5 strong contribution statements that would impress reviewers.

PAPER CONTENT:
{text[:4000]}
{current_text}

Generate compelling, specific contribution bullets."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.5,
            max_tokens=1200,
        )
        
        raw = response.choices[0].message.content.strip()
        
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        
        import json
        result = json.loads(raw)
        
        return {
            "success": True,
            "contributions": result.get("contributions", []),
            "summary": result.get("summary", ""),
            "original_contributions": current_contributions
        }
        
    except Exception as exc:
        _debug("contribution_bullets.error", str(exc))
        return {
            "success": False,
            "error": str(exc),
            "contributions": []
        }


def analyze_rewrite_opportunities(text: str, formatting_data: dict = None, plagiarism_data: dict = None, citation_data: dict = None) -> dict:
    """Analyze the document and identify all rewrite opportunities.
    
    Consolidates feedback from all analysis modules into actionable rewrites.
    
    Returns comprehensive rewrite plan.
    """
    import re
    
    sections = identify_paper_sections(text)
    rewrite_opportunities = []
    
    # Priority levels: critical, high, medium, low
    
    # 1. Extract issues from formatting data
    if formatting_data and formatting_data.get("pages"):
        for page in formatting_data["pages"]:
            for suggestion in page.get("suggestions", []):
                opp = {
                    "id": f"format_{len(rewrite_opportunities)}",
                    "source": "formatting",
                    "type": suggestion.get("type", "general"),
                    "severity": suggestion.get("severity", "minor"),
                    "priority": "high" if suggestion.get("severity") in ["critical", "major"] else "medium",
                    "title": suggestion.get("title", "Formatting Issue"),
                    "description": suggestion.get("description", ""),
                    "affected_text": suggestion.get("affected_text", ""),
                    "location": suggestion.get("location", f"Page {page.get('page_number', '?')}"),
                    "suggested_fix": suggestion.get("fix", ""),
                    "page": page.get("page_number", 1),
                    "rewrite_ready": bool(suggestion.get("affected_text"))
                }
                rewrite_opportunities.append(opp)
    
    # 2. Extract issues from plagiarism data
    if plagiarism_data:
        # Style inconsistencies
        if plagiarism_data.get("style_analysis", {}).get("inconsistencies"):
            for inc in plagiarism_data["style_analysis"]["inconsistencies"][:5]:
                opp = {
                    "id": f"style_{len(rewrite_opportunities)}",
                    "source": "plagiarism",
                    "type": "style_consistency",
                    "severity": "minor",
                    "priority": "medium",
                    "title": "Style Inconsistency",
                    "description": inc,
                    "affected_text": "",
                    "location": "Throughout document",
                    "suggested_fix": "Maintain consistent writing style",
                    "rewrite_ready": False
                }
                rewrite_opportunities.append(opp)
        
        # Common phrases (potential plagiarism)
        if plagiarism_data.get("common_phrases", {}).get("flagged_phrases"):
            for phrase_info in plagiarism_data["common_phrases"]["flagged_phrases"][:5]:
                phrase = phrase_info.get("phrase", "") if isinstance(phrase_info, dict) else str(phrase_info)
                opp = {
                    "id": f"phrase_{len(rewrite_opportunities)}",
                    "source": "plagiarism",
                    "type": "common_phrase",
                    "severity": "major",
                    "priority": "high",
                    "title": "Common/Stock Phrase Detected",
                    "description": f"This phrase may be too generic or commonly used: '{phrase[:50]}'",
                    "affected_text": phrase[:100] if isinstance(phrase, str) else "",
                    "location": "Document",
                    "suggested_fix": "Rephrase in your own words",
                    "rewrite_ready": True
                }
                rewrite_opportunities.append(opp)
    
    # 3. Extract issues from citation data
    if citation_data:
        # Unsupported claims
        if citation_data.get("unsupported_claims"):
            for claim in citation_data["unsupported_claims"][:5]:
                claim_text = claim.get("claim", "") if isinstance(claim, dict) else str(claim)
                opp = {
                    "id": f"citation_{len(rewrite_opportunities)}",
                    "source": "citation",
                    "type": "unsupported_claim",
                    "severity": "major",
                    "priority": "high",
                    "title": "Claim Needs Citation",
                    "description": f"This claim may need a supporting citation",
                    "affected_text": claim_text[:150],
                    "location": claim.get("location", "Document") if isinstance(claim, dict) else "Document",
                    "suggested_fix": "Add appropriate citation or rephrase as opinion",
                    "rewrite_ready": True
                }
                rewrite_opportunities.append(opp)
        
        # Citation issues
        if citation_data.get("issues"):
            for issue in citation_data["issues"][:5]:
                opp = {
                    "id": f"citissue_{len(rewrite_opportunities)}",
                    "source": "citation",
                    "type": "citation_format",
                    "severity": issue.get("severity", "minor"),
                    "priority": "medium",
                    "title": issue.get("title", "Citation Issue"),
                    "description": issue.get("description", ""),
                    "affected_text": "",
                    "location": issue.get("location", "References"),
                    "suggested_fix": issue.get("fix", "Review citation formatting"),
                    "rewrite_ready": False
                }
                rewrite_opportunities.append(opp)
    
    # 4. Section-specific improvements
    section_improvements = []
    
    if 'abstract' in sections:
        abstract = sections['abstract']
        word_count = len(abstract.split())
        if word_count < 100:
            section_improvements.append({
                "section": "abstract",
                "issue": "Abstract is too short",
                "suggestion": "Expand to include motivation, method, results, and significance"
            })
        elif word_count > 350:
            section_improvements.append({
                "section": "abstract",
                "issue": "Abstract is too long",
                "suggestion": "Condense to 150-300 words"
            })
    
    if 'introduction' in sections:
        intro = sections['introduction']
        # Check for research gap statement
        if not re.search(r'(?i)(gap|lacking|missing|however|yet|but|limited)', intro):
            section_improvements.append({
                "section": "introduction",
                "issue": "Research gap may not be clearly stated",
                "suggestion": "Add explicit statement of what is missing in existing work"
            })
    
    if 'conclusion' in sections:
        conclusion = sections['conclusion']
        if len(conclusion.split()) < 50:
            section_improvements.append({
                "section": "conclusion",
                "issue": "Conclusion is too brief",
                "suggestion": "Expand to summarize findings and discuss implications"
            })
    
    # Convert section improvements to opportunities
    for imp in section_improvements:
        opp = {
            "id": f"section_{len(rewrite_opportunities)}",
            "source": "structure",
            "type": "section_improvement",
            "severity": "major",
            "priority": "high",
            "title": f"{imp['section'].title()}: {imp['issue']}",
            "description": imp['suggestion'],
            "affected_text": sections.get(imp['section'], "")[:200],
            "location": imp['section'].title(),
            "suggested_fix": imp['suggestion'],
            "section_name": imp['section'],
            "full_section_text": sections.get(imp['section'], ""),
            "rewrite_ready": True
        }
        rewrite_opportunities.append(opp)
    
    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    rewrite_opportunities.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))
    
    # Calculate summary stats
    stats = {
        "total_opportunities": len(rewrite_opportunities),
        "critical_count": sum(1 for o in rewrite_opportunities if o.get("priority") == "critical"),
        "high_count": sum(1 for o in rewrite_opportunities if o.get("priority") == "high"),
        "medium_count": sum(1 for o in rewrite_opportunities if o.get("priority") == "medium"),
        "low_count": sum(1 for o in rewrite_opportunities if o.get("priority") == "low"),
        "rewrite_ready_count": sum(1 for o in rewrite_opportunities if o.get("rewrite_ready")),
        "sections_identified": list(sections.keys())
    }
    
    return {
        "opportunities": rewrite_opportunities[:20],  # Limit to top 20
        "stats": stats,
        "sections": {k: v[:500] for k, v in sections.items()},  # Truncate section previews
        "full_sections": sections  # Keep full sections for rewriting
    }


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