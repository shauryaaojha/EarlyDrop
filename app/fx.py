"""Injects the FX engine (fx/edfx.js) and the global stylesheet (fx/edfx.css) into the Streamlit page.

Streamlit renders st.iframe HTML in a same-origin iframe, so a 1px frame can upsert a <style> and a
<script> into the parent document. The engine guards itself, so reruns are cheap. Both files are read
on every run, so edits apply without restarting the server.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent


def inject(page_key: str, enabled: bool = True) -> None:
    """Upsert the stylesheet into the parent <head> (st.html drops it) and route the FX engine."""
    css = (HERE / "fx" / "edfx.css").read_text(encoding="utf8")      # read per run: edits apply live
    engine = (HERE / "fx" / "edfx.js").read_text(encoding="utf8")
    cfg = json.dumps({"page": page_key, "fx": enabled, "loader": True})
    with st.container(key="edfx"):
        st.iframe(f"""<script>
    (function(){{
      const W = window.parent, D = W.document;
      let st = D.getElementById('ed-css');
      if (!st) {{ st = D.createElement('style'); st.id = 'ed-css'; D.head.appendChild(st); }}
      const css = {json.dumps(css)};
      if (st.textContent !== css) st.textContent = css;
      if (!W.EDFX) {{ const s = D.createElement('script'); s.id = 'ed-fx-engine'; s.textContent = {json.dumps(engine)}; D.head.appendChild(s); }}
      W.EDFX && W.EDFX.route({cfg});
    }})();
    </script>""", height=1)
