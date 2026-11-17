"""The "Scan my mood" button: grabs one webcam frame in the browser, with no preview.

Clicking it opens the camera, waits a moment for auto-exposure to settle, draws
one frame to an off-screen canvas, stops the camera and sends the JPEG to Python.
The browser still asks for camera permission the first time, and only allows the
camera on localhost or over HTTPS.
"""

from __future__ import annotations

import base64
import binascii
from typing import Optional, Tuple

import streamlit as st

_HTML = """
<button type="button" class="mood-scan-btn">
  <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
    <circle cx="12" cy="12" r="3.2" fill="currentColor"/>
    <path fill="currentColor" d="M9 2 7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2h-3.17L15 2H9zm3 15c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"/>
  </svg>
  <span class="mood-scan-label"></span>
</button>
"""

# Scoped by class: styles are not isolated, so the button can use the app's colour tokens.
_CSS = """
.mood-scan-btn {
    display: inline-flex; align-items: center; gap: 8px; min-height: 44px; padding: 0 24px;
    border-radius: var(--r-pill, 999px); border: 1px solid var(--line-strong, rgba(255,255,255,.22));
    background: transparent; color: var(--ink, #fff); cursor: pointer; white-space: nowrap;
    font-family: var(--font-text, system-ui, sans-serif); font-size: 0.95rem; font-weight: 500;
    touch-action: manipulation;
    transition: transform .18s var(--ease, ease), background-color .18s var(--ease, ease), border-color .18s var(--ease, ease);
}
.mood-scan-btn:hover:not(:disabled) { border-color: var(--ink, #fff); background: var(--surface, #1b0e17); transform: translateY(-1px); }
.mood-scan-btn:active:not(:disabled) { transform: translateY(1px) scale(.98); }
.mood-scan-btn:focus-visible { outline: 2px solid var(--accent, #ff488b); outline-offset: 3px; }
.mood-scan-btn:disabled { cursor: progress; opacity: .7; }
"""

_JS = """
export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  const btn = parentElement.querySelector('.mood-scan-btn');
  const label = btn.querySelector('.mood-scan-label');
  // Called again on every rerun with fresh data: reset the button each time.
  label.textContent = data.label;
  btn.title = 'Read your mood from one webcam photo. The photo is not saved.';
  btn.disabled = false;

  btn.onclick = async () => {
    btn.disabled = true;
    label.textContent = 'Reading your mood…';
    let stream = null;
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw { name: 'InsecureContext' };
      }
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      const video = document.createElement('video');  // never attached to the page
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play();
      await new Promise((r) => setTimeout(r, 700));  // let auto-exposure settle
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      setTriggerValue('photo', canvas.toDataURL('image/jpeg', 0.9));
    } catch (err) {
      setTriggerValue('error', (err && err.name) || 'Error');
    } finally {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    }
  };
}
"""

_component = st.components.v2.component(
    "mood_scan_button", html=_HTML, css=_CSS, js=_JS, isolate_styles=False
)

# Browser error names (getUserMedia) to what the viewer can do about them.
CAMERA_ERRORS = {
    "NotAllowedError": "Camera access is blocked. Allow it in your browser’s site settings, or pick a mood.",
    "SecurityError": "Camera access is blocked. Allow it in your browser’s site settings, or pick a mood.",
    "NotFoundError": "No camera was found. Pick a mood instead.",
    "OverconstrainedError": "No camera was found. Pick a mood instead.",
    "NotReadableError": "Your camera is in use by another app. Close it and scan again, or pick a mood.",
    "InsecureContext": "The camera only works over HTTPS or on localhost. Pick a mood instead.",
}
DEFAULT_CAMERA_ERROR = "The camera couldn’t be opened. Scan again, or pick a mood."


def mood_scan_button(label: str, nonce: int, key: str = "mood-scan") -> Tuple[Optional[bytes], Optional[str]]:
    """Render the button. Returns (photo JPEG bytes, error message); both None unless just clicked.

    `nonce` must change after each scan so the button resets even when its label doesn't.
    """
    result = _component(
        key=key,
        data={"label": label, "nonce": nonce},
        width="content",
        on_photo_change=lambda: None,
        on_error_change=lambda: None,
    )
    if result.error:
        return None, CAMERA_ERRORS.get(str(result.error), DEFAULT_CAMERA_ERROR)
    if result.photo:
        try:
            return base64.b64decode(str(result.photo).split(",", 1)[-1], validate=True), None
        except (binascii.Error, ValueError):
            return None, "That photo couldn’t be read. Scan again, or pick a mood."
    return None, None
