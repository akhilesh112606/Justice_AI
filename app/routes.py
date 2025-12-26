import base64
import io
import json
import os
import smtplib
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

main = Blueprint("main", __name__)
load_dotenv()


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


def _debug(label: str, payload):
    """Lightweight debug printer to aid troubleshooting without leaking secrets."""
    try:
        print(f"[DEBUG] {label}: {payload}")
    except Exception:
        print(f"[DEBUG] {label}: <unprintable>")


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
    file = request.files.get("file")
    if not file or file.filename.strip() == "":
        flash("Please choose an image to upload.", "error")
        return redirect(url_for("main.index"))

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
        flash("Unsupported file type. Please upload an image.", "error")
        return redirect(url_for("main.index"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    text = ""
    try:
        text = extract_text(temp_path)
    except Exception as exc:  # noqa: BLE001
        flash(f"OCR failed: {exc}", "error")
        return redirect(url_for("main.index"))
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    cleaned_text = text.strip() if text else "No text detected."

    formatted_text = format_text(cleaned_text)

    _debug("upload.cleaned_text_len", len(cleaned_text))
    _debug("upload.formatted_text_len", len(formatted_text))

    characters, general_questions = build_questions(formatted_text)
    character_profiles = extract_character_profiles(formatted_text)
    roadmap_steps = build_roadmap(formatted_text, characters, general_questions)
    locations = extract_locations(formatted_text)
    legal_sections, legal_audit = build_judicial_sections(formatted_text)

    _debug("upload.characters_count", len(characters))
    _debug("upload.general_questions_count", len(general_questions))
    _debug("upload.profiles_count", len(character_profiles))
    _debug("upload.character_profiles_data", character_profiles)
    _debug("upload.roadmap_steps_count", len(roadmap_steps))
    _debug("upload.locations_count", len(locations))
    _debug("upload.legal_sections_count", len(legal_sections))
    _debug("upload.legal_audit", legal_audit)

    return render_template(
        "dashboard.html",
        extracted_text=formatted_text,
        character_questions=characters,
        character_profiles=character_profiles,
        general_questions=general_questions,
        roadmap_steps=roadmap_steps,
        locations=locations,
        legal_sections=legal_sections,
        legal_audit=legal_audit,
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
    if ext not in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}:
        return jsonify({"error": "Unsupported audio type. Use wav/mp3/m4a/aac/ogg/flac/webm."}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        audio_file.save(tmp.name)
        temp_path = tmp.name

    transcript = None
    try:
        transcript, err = _transcribe_audio(temp_path)
        if err or not transcript:
            return jsonify({"error": f"Transcription failed: {err or 'no text'}"}), 500

        insights = _analyze_audio_insights(transcript, character, fir_context) or {}
        payload = {
            "transcript": transcript,
            "insights": insights,
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