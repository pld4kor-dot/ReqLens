"""ReqInOne 2.0 – Streamlit Demo UI.

Seven screens:
  1. Project Upload
  2. Pipeline Runner    ← one HTTP request per agent, spinner per row
  3. Requirement Review Queue
  4. Evidence Viewer
  5. Knowledge Graph
  6. Change Impact Analysis
  7. SRS Export

Screen 3 – Review Queue colour coding
--------------------------------------
All requirements are shown in a single flat list, each inside a
colour-coded card that matches its review_status:

  ✅ accepted       → light green  (#f0fff4 / #a8d5b5 border)
  ❌ rejected       → light red    (#fff0f0 / #f5b8b8 border)
                       text shown with strikethrough
  ⏳ pending        → light yellow (#fffde7 / #ffe082 border)
                       text is EDITABLE via st.text_area
                       💾 Save button → PATCH /requirements/{id}
  ✍️ needs_revision → same yellow card as pending, also editable
  ⏭️ deferred       → light grey   (#f8f8f8 / #cccccc border)

Every card has Accept / Reject / Needs Revision decision buttons.
Accepted and rejected cards collapse the decision buttons inside an
expander so the list stays clean but the option is still there.
"""

from __future__ import annotations

import json
import time

import httpx
import streamlit as st
import base64
from PIL import Image
import io

def image_to_base64(image_path: str, format: str = "PNG") -> str:
    """Load an image with PIL and return it as a base64 string."""
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8081"

st.set_page_config(
    page_title="ReqLens",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

# ── Logo (sidebar + top-left corner of every page) ───────────────────────────
# The logo PNG is embedded as a base64 data-URI so no external file path is
# needed — the app is fully self-contained regardless of working directory.
_LOGO_B64 = image_to_base64("src/reqlens/ui/reqlens_logo.png")
_LOGO_URI = f"data:image/png;base64,{_LOGO_B64}"

# Sidebar
st.sidebar.markdown(
    "> *LLMs propose. Evidence gates. Graph validates. Humans approve.*"
)

# Top-left corner logo – fixed-position, rendered on every page.
# Scaled to 160px wide; sits above Streamlit's own header bar.
st.markdown(
    f"""
    <style>
    #reqinone-corner-logo {{
        position: fixed;
        top: 8px;
        left: 62px;
        z-index: 999999;
        pointer-events: none;
        opacity: 0.93;
    }}
    @media (max-width: 900px) {{
        #reqinone-corner-logo {{ left: 10px; }}
    }}
    </style>
    <div id="reqinone-corner-logo">
      <img src="{_LOGO_URI}" width="160"
           style="display:block;height:auto;filter:drop-shadow(0 1px 3px rgba(0,0,0,0.18));"/>
    </div>
    """,
    unsafe_allow_html=True,
)

screen = st.sidebar.radio(
    "Navigate",
    [
        "1. Project Upload or Select",
        "2. Pipeline Runner",
        "3. Review Queue",
        "4. Evidence Viewer",
        "5. Knowledge Graph",
        "6. Impact Analysis",
        "7. SRS Export",
    ],
)

# ── Shared API helpers ────────────────────────────────────────────────────────

def api_get(path: str):
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=1000)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, **kwargs):
    try:
        r = httpx.post(f"{API_BASE}{path}", timeout=6000, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_patch(path: str, **kwargs):
    """PATCH – used to save edited requirement text back to the DB."""
    try:
        r = httpx.patch(f"{API_BASE}{path}", timeout=6000, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_get_text(path: str):
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=600)
        r.raise_for_status()
        return r.text
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Screen 1 – Project Upload
# ─────────────────────────────────────────────────────────────────────────────
if screen == "1. Project Upload or Select":
    st.header("📁 Project Upload or Select")

    # ── Active project banner ──────────────────────────────────────────────────
    # Show which project is currently loaded (name only, no raw ID shown).
    active_id   = st.session_state.get("project_id", "")
    active_name = st.session_state.get("project_name", "")
    if active_id and active_name:
        st.success(f"✅ Active project: **{active_name}**  ·  documents uploaded here go to this project")
    elif active_id:
        st.info(f"✅ A project is loaded. Upload documents below.")

    st.markdown("---")

    # ── Two columns: left = project details, right = document upload ───
    left, right = st.columns(2, gap="large")

    # ── LEFT: create a new project ─────────────────────────────────────────────
    with left:
        st.subheader("Create New Project")
        project_name = st.text_input(
            "Project name",
            placeholder="e.g. University Event Management System",
        )
        project_desc = st.text_area(
            "Description",
            placeholder="Brief description of the project scope and goals…",
            height=120,
        )
        if st.button("➕ Create Project", type="primary", disabled=not project_name.strip()):
            result = api_post(
                "/projects",
                json={"name": project_name.strip(), "description": project_desc.strip()},
            )
            if result:
                # Store ID silently; display only the name to the user
                st.session_state["project_id"]   = result["id"]
                st.session_state["project_name"] = result["name"]
                st.success(f"Project **{result['name']}** created. You can now upload documents →")
                st.rerun()

    # ── RIGHT: upload documents to the active project ──────────────────────────
    with right:
        st.subheader("Upload Documents")

        # Guard: need an active project before uploading
        if not active_id:
            st.info("Create or select a project on the left (or from the list below) to enable uploads.")
        else:
            uploaded_files = st.file_uploader(
                f"Add source documents to **{active_name or active_id}**",
                type=["txt", "md", "csv", "pdf"],
                accept_multiple_files=True,
                help="Supported: plain text, Markdown, CSV, PDF",
            )
            if st.button(
                "⬆️ Upload",
                disabled=not uploaded_files,
                type="primary",
            ):
                success_count = 0
                for f in uploaded_files:
                    result = api_post(
                        f"/projects/{active_id}/documents",
                        files={"file": (f.name, f.getvalue())},
                    )
                    if result:
                        success_count += 1

                if success_count:
                    st.success(
                        f"Uploaded {success_count} file(s) to **{active_name or active_id}**."
                    )
                    # Show already-uploaded docs for this project
                    docs = api_get(f"/projects/{active_id}/documents")
                    if docs:
                        st.caption(f"{len(docs)} document(s) in this project:")
                        for d in docs:
                            st.markdown(f"  • `{d['filename']}`")

    st.markdown("---")

    # ── Existing projects list – clickable, no raw IDs shown ──────────────────
    st.subheader("Existing Projects")
    projects = api_get("/projects")

    if not projects:
        st.info("No projects yet. Create your first project above.")
    else:
        # Sort newest-first (API already does this, but be explicit)
        for p in projects:
            pid   = p["id"]
            pname = p["name"]
            pdesc = p.get("description", "")
            # Parse date for display
            try:
                from datetime import datetime
                created = datetime.fromisoformat(p["created_at"]).strftime("%d %b %Y, %H:%M")
            except Exception:
                created = p.get("created_at", "")

            is_active = (pid == active_id)

            # Card background: highlight the currently-loaded project
            card_bg     = "#f0fff4" if is_active else "#fafafa"
            card_border = "#a8d5b5" if is_active else "#e0e0e0"
            active_tag  = (
                "  <span style='background:#a8d5b5;color:#1a5c35;"
                "padding:1px 8px;border-radius:8px;font-size:0.72em;"
                "font-weight:600'>● Active</span>"
                if is_active else ""
            )

            st.markdown(
                f"""
                <div style="
                    background:{card_bg};
                    border:1.5px solid {card_border};
                    border-radius:8px;
                    padding:10px 16px;
                    margin-bottom:6px;
                ">
                    <div style="font-weight:600;font-size:1em">
                        {pname}{active_tag}
                    </div>
                    <div style="color:#666;font-size:0.83em;margin-top:2px">
                        {pdesc if pdesc else '<i>No description</i>'}
                    </div>
                    <div style="color:#aaa;font-size:0.75em;margin-top:4px">
                        Created {created}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Load + Delete buttons on the same row.
            # Delete uses a 2-step confirmation: first click arms a confirm
            # state in session_state; second click within that pid actually
            # calls DELETE /projects/{pid}.
            confirm_key = f"confirm_delete_{pid}"

            b_load, b_del, b_msg = st.columns([1.2, 1.2, 4])

            with b_load:
                load_label = "✅ Loaded" if is_active else "Load →"
                if st.button(load_label, key=f"load_{pid}", disabled=is_active):
                    st.session_state["project_id"]   = pid
                    st.session_state["project_name"] = pname
                    st.rerun()

            with b_del:
                # Two-step confirmation pattern
                if st.session_state.get(confirm_key, False):
                    if st.button(
                        "⚠️ Confirm delete",
                        key=f"confirm_del_{pid}",
                        type="primary",
                    ):
                        try:
                            import httpx
                            r = httpx.delete(f"{API_BASE}/projects/{pid}", timeout=60)
                            r.raise_for_status()
                            result = r.json()
                            st.session_state[confirm_key] = False
                            st.success(
                                f"Deleted **{pname}** "
                                f"({result.get('total_rows', '?')} rows removed)."
                            )
                            # Clear active project if we just deleted it
                            if pid == active_id:
                                st.session_state.pop("project_id",   None)
                                st.session_state.pop("project_name", None)
                            st.rerun()
                        except Exception as e:
                            st.session_state[confirm_key] = False
                            st.error(f"Delete failed: {e}")
                else:
                    if st.button("🗑️ Delete", key=f"del_{pid}"):
                        st.session_state[confirm_key] = True
                        st.rerun()

            with b_msg:
                if st.session_state.get(confirm_key, False):
                    st.warning(
                        "This permanently removes the project and **all** of its "
                        "documents, requirements, review decisions, and graph data. "
                        "Click **Confirm delete** to proceed."
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Screen 2 – Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────
elif screen == "2. Pipeline Runner":
    st.header(" Pipeline Runner")

    project_id   = st.session_state.get("project_id", "")
    project_name = st.session_state.get("project_name", "")
    if not project_id:
        st.warning("No project loaded. Go to **1. Project Upload** and create or select a project.")
        st.stop()
    st.markdown(f"<span style='font-size:1.05em;font-weight:600;color:#444'>📁 Selected Project: {project_name or project_id}</span>", unsafe_allow_html=True)

    # ── Reset state when the loaded project changes ──────────────────────────
    if "pipeline_project_id" not in st.session_state:
        st.session_state["pipeline_project_id"] = project_id

    if st.session_state["pipeline_project_id"] != project_id:
        st.session_state.agent_results = {}
        st.session_state.pipeline_ran = False
        st.session_state["pipeline_project_id"] = project_id

    AGENTS = [
        {"step": 1, "name": "Extraction Agent",    "icon": "🔎", "path": "extraction",
         "desc": "Scan source spans for requirement candidates"},
        {"step": 2, "name": "Evidence Agent",       "icon": "🧪", "path": "evidence",
         "desc": "Gate candidates against source evidence"},
        {"step": 3, "name": "Classification Agent", "icon": "🏷️", "path": "classification",
         "desc": "Label each requirement as FR / NFR / constraint…"},
        {"step": 4, "name": "Ambiguity Agent",      "icon": "❓", "path": "ambiguity",
         "desc": "Flag vague or unmeasurable phrasing"},
        {"step": 5, "name": "Dependency Agent",     "icon": "🔗", "path": "dependency",
         "desc": "Map dependency edges in the requirement graph"},
        {"step": 6, "name": "Consistency Agent",    "icon": "⚖️", "path": "consistency",
         "desc": "Detect contradictions and duplicates"},
        {"step": 7, "name": "Traceability Agent",   "icon": "🗺️", "path": "traceability",
         "desc": "Link requirements back to source spans"},
        {"step": 8, "name": "Elicitation Agent",    "icon": "🗣️", "path": "elicitation",
         "desc": "Turn unresolved questions into stakeholder questions"},
        {"step": 9, "name": "Ingestion Agent",      "icon": "💾", "path": "ingestion",
         "desc": "Persist accepted requirements to the database"},
        {"step": 10, "name": "Composer Agent",      "icon": "✍️", "path": "composer",
         "desc": "Compose the final SRS document"},
    ]

    if "agent_results" not in st.session_state:
        st.session_state.agent_results = {}
    if "pipeline_ran" not in st.session_state:
        st.session_state.pipeline_ran = False

    # ── Check SRS cache to determine initial state ───────────────────────────
    # Do this once per project load (keyed by project_id so switching projects
    # re-checks). If an SRS already exists we pre-populate agent_results so
    # every row shows ✅ Done, and set pipeline_ran = True.
    _cache_key = f"srs_checked_{project_id}"
    if not st.session_state.get(_cache_key):
        srs_status_check = api_get(f"/projects/{project_id}/export/srs/status") or {}
        _srs_already_ready = bool(srs_status_check.get("ready", False))
        if _srs_already_ready and not st.session_state.agent_results:
            # Pre-fill all agents as completed so the table shows ✅ Done
            for _a in AGENTS:
                st.session_state.agent_results[_a["path"]] = {
                    "status": "completed",
                    "summary": "Previously completed",
                    "elapsed_s": 0.0,
                    "warnings": [],
                    "errors": [],
                }
            st.session_state.pipeline_ran = True
            st.session_state["_srs_was_preloaded"] = True
        st.session_state[_cache_key] = True

    _preloaded = st.session_state.get("_srs_was_preloaded", False)

    # ── Run / Re-run button ───────────────────────────────────────────────────
    _btn_label = "🔄 Re-run Full Pipeline" if st.session_state.pipeline_ran else "▶️  Run Full Pipeline"
    run_clicked = st.button(
        _btn_label,
        disabled=not project_id,
        type="primary",
    )

    if run_clicked:
        # Clear preload flag so after a real run we show live results
        st.session_state["_srs_was_preloaded"] = False

    n_done = sum(
        1 for a in AGENTS
        if st.session_state.agent_results.get(a["path"]) is not None
    )
    progress_bar = st.progress(n_done / len(AGENTS))

    st.markdown("---")

    hdr = st.columns([0.4, 0.4, 2.4, 1.8, 1.2, 4.4])
    for col, label in zip(hdr, ["**#**", "", "**Agent**", "**Status**", "**Time**", "**Output**"]):
        col.markdown(label)
    st.markdown(
        "<hr style='margin:4px 0 10px 0; border-color:#e0e0e0;'>",
        unsafe_allow_html=True,
    )

    status_cells: dict[str, tuple] = {}

    for agent in AGENTS:
        cols = st.columns([0.4, 0.4, 2.4, 1.8, 1.2, 4.4])
        cols[0].markdown(f"`{agent['step']}`")
        cols[1].markdown(agent["icon"])
        cols[2].markdown(f"**{agent['name']}**  \n"
                         f"<small style='color:#888'>{agent['desc']}</small>",
                         unsafe_allow_html=True)

        status_ph = cols[3].empty()
        time_ph   = cols[4].empty()
        output_ph = cols[5].empty()

        prev = st.session_state.agent_results.get(agent["path"])
        if prev is None:
            status_ph.markdown(
                "<span style='background:#f0f0f0;padding:3px 10px;"
                "border-radius:12px;font-size:0.82em'>⬜ Pending</span>",
                unsafe_allow_html=True,
            )
        else:
            _status   = prev.get("status", "completed")
            _elapsed  = prev.get("elapsed_s", 0.0)
            _summary  = prev.get("summary", "")
            _warnings = prev.get("warnings", [])
            _errors   = prev.get("errors", [])

            if _status == "completed":
                badge_html = ("<span style='background:#f0fff4;padding:3px 10px;"
                              "border-radius:12px;font-size:0.82em'>✅ Done</span>")
            elif _status == "skipped":
                badge_html = ("<span style='background:#f8f8f8;padding:3px 10px;"
                              "border-radius:12px;font-size:0.82em'>⏭️ Skipped</span>")
            else:
                badge_html = ("<span style='background:#fff0f0;padding:3px 10px;"
                              "border-radius:12px;font-size:0.82em'>❌ Failed</span>")

            status_ph.markdown(badge_html, unsafe_allow_html=True)
            if _elapsed:
                time_ph.markdown(
                    f"<span style='font-size:0.82em;color:#555'>🕒 {_elapsed:.1f}s</span>",
                    unsafe_allow_html=True,
                )
            parts = [_summary] if _summary else []
            if _warnings:
                parts.append(f"⚠️ {len(_warnings)} warning(s)")
            if _errors:
                parts.append(f"🔴 {'; '.join(_errors[:2])}")
            output_ph.markdown("  ·  ".join(parts))

        status_cells[agent["path"]] = (status_ph, time_ph, output_ph)


    if run_clicked and project_id:
        st.session_state.agent_results = {}
        st.session_state.pipeline_ran  = False
        for a in AGENTS:
            s_ph, t_ph, o_ph = status_cells[a["path"]]
            s_ph.markdown(
                "<span style='background:#f0f0f0;padding:3px 10px;"
                "border-radius:12px;font-size:0.82em'>⬜ Pending</span>",
                unsafe_allow_html=True,
            )
            t_ph.markdown("")
            o_ph.markdown("")
        progress_bar.progress(0.0)

        pipeline_ok = True

        with st.spinner("Preparing source spans…"):
            prep = api_post(f"/projects/{project_id}/agents/prepare")
        if prep is None:
            st.error("Prepare step failed — cannot continue.")
            pipeline_ok = False

        if pipeline_ok:
            for idx, agent in enumerate(AGENTS):
                path = agent["path"]
                s_ph, t_ph, o_ph = status_cells[path]

                s_ph.markdown(
                    "<span style='background:#fffbe6;padding:3px 10px;"
                    "border-radius:12px;font-size:0.82em'>🔄 Running…</span>",
                    unsafe_allow_html=True,
                )
                o_ph.markdown(
                    f"<span style='color:#888;font-size:0.9em'>{agent['desc']}</span>",
                    unsafe_allow_html=True,
                )

                t_start = time.perf_counter()
                result  = api_post(f"/projects/{project_id}/agents/{path}/run", json={})
                elapsed = time.perf_counter() - t_start

                if result is None:
                    result = {
                        "status": "failed", "summary": "Network or server error",
                        "elapsed_s": elapsed, "warnings": [], "errors": ["HTTP request failed"],
                    }

                st.session_state.agent_results[path] = result

                _status  = result.get("status", "completed")
                _elapsed = result.get("elapsed_s", elapsed)
                _summary = result.get("summary", "")
                _warns   = result.get("warnings", [])
                _errors  = result.get("errors", [])

                if _status == "completed":
                    badge = ("<span style='background:#f0fff4;padding:3px 10px;"
                             "border-radius:12px;font-size:0.82em'>✅ Done</span>")
                elif _status == "skipped":
                    badge = ("<span style='background:#f8f8f8;padding:3px 10px;"
                             "border-radius:12px;font-size:0.82em'>⏭️ Skipped</span>")
                else:
                    badge = ("<span style='background:#fff0f0;padding:3px 10px;"
                             "border-radius:12px;font-size:0.82em'>❌ Failed</span>")
                    pipeline_ok = False

                s_ph.markdown(badge, unsafe_allow_html=True)
                t_ph.markdown(
                    f"<span style='font-size:0.82em;color:#555'>🕒 {_elapsed:.1f}s</span>",
                    unsafe_allow_html=True,
                )
                parts = [_summary] if _summary else []
                if _warns:
                    parts.append(f"⚠️ {len(_warns)} warning(s)")
                if _errors:
                    parts.append(f"🔴 {'; '.join(_errors[:2])}")
                o_ph.markdown("  ·  ".join(parts))
                progress_bar.progress((idx + 1) / len(AGENTS))

                if not pipeline_ok and agent["step"] <= 2:
                    for rem in AGENTS[idx + 1:]:
                        r_s, r_t, r_o = status_cells[rem["path"]]
                        r_s.markdown(
                            "<span style='background:#f8f8f8;padding:3px 10px;"
                            "border-radius:12px;font-size:0.82em'>⏭️ Skipped</span>",
                            unsafe_allow_html=True,
                        )
                        r_o.markdown(
                            "<span style='color:#aaa;font-size:0.9em'>"
                            "Skipped — earlier step failed</span>",
                            unsafe_allow_html=True,
                        )
                    progress_bar.progress(1.0)
                    break

        st.session_state.pipeline_ran = True
        # Invalidate the SRS-check cache so a fresh check runs next visit
        st.session_state.pop(_cache_key, None)

    st.markdown("---")
    if st.session_state.pipeline_ran and project_id:
        results     = st.session_state.agent_results
        n_completed = sum(1 for r in results.values() if r.get("status") == "completed")
        n_failed    = sum(1 for r in results.values() if r.get("status") == "failed")
        if _preloaded and not run_clicked:
            st.info(
                "✅ Pipeline previously completed for this project. "
                "All agents show **Done**. Click **🔄 Re-run Full Pipeline** to run again."
            )
        elif n_failed == 0:
            st.success(f"✅ Pipeline complete — {n_completed}/{len(AGENTS)} agent(s) succeeded")
        else:
            st.warning(f"Pipeline finished with {n_failed} failure(s) — {n_completed} agent(s) succeeded")
        reqs = api_get(f"/projects/{project_id}/requirements")
        if reqs is not None:
            accepted = [r for r in reqs if r.get("status") == "accepted"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Requirements",   len(reqs))
            c2.metric("Accepted (SRS-Ready)", len(accepted))
            c3.metric("Pending Review",       len(reqs) - len(accepted))
    else:
        st.info(
            "Enter a Project ID and click **▶️ Run Full Pipeline**. "
            "Each agent row will show a spinner while running and update to ✅ or ❌ when done."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Screen 3 – Review Queue
# ─────────────────────────────────────────────────────────────────────────────
elif screen == "3. Review Queue":
    st.header("📋 Requirement Review Queue")

    project_id   = st.session_state.get("project_id", "")
    project_name = st.session_state.get("project_name", "")
    if not project_id:
        st.warning("No project loaded. Go to **1. Project Upload** and create or select a project.")
        st.stop()
    # Project header + search-by-id bar on the same row.
    # Keyed per-project so switching projects does not leak the query.
    hdr_left, hdr_right = st.columns([0.62, 0.38])
    with hdr_left:
        st.markdown(
            f"<span style='font-size:1.05em;font-weight:600;color:#444'>"
            f"📁 Selected Project: {project_name or project_id}</span>",
            unsafe_allow_html=True,
        )
    with hdr_right:
        search_id = st.text_input(
            "Search by requirement id",
            key=f"review_search_{project_id}",
            placeholder="Search by requirement id (e.g. REQ_abc123)",
            label_visibility="collapsed",
        ).strip()

    # -- Colour palette (shared by tabs and cards) ---
    # Each status - (card_bg, border, emoji, tab_label, text_color, tab_bg_css)
    STATUS_STYLE = {
        "accepted":      ("#f0fff4", "#a8d5b5", "✅", "Accepted",       "#1a5c35", "#d4edda"),
        "rejected":      ("#fff5f5", "#f5b8b8", "❌", "Rejected",       "#7f1d1d", "#f8d7da"),
        "pending":       ("#fffde7", "#ffe082", "⏳", "Pending",        "#78600a", "#fff9c4"),
        "needs_revision":("#fffde7", "#ffe082", "✍️", "Needs Revision", "#78600a", "#fff9c4"),
        "deferred":      ("#f8f8f8", "#cccccc", "⏭️", "Deferred",      "#555555", "#eeeeee"),
    }
    DEFAULT_STYLE = ("#f8f8f8", "#cccccc", "❓", "Unknown", "#333333", "#eeeeee")

    # -- Fetch requirements ---
    reqs = api_get(f"/projects/{project_id}/requirements") or []

    # Narrow the working set by the search-by-id query, if any. The
    # tab counts further down are computed from this filtered list so
    # search affects both the visible cards and the per-tab badge counts.
    if search_id:
        needle = search_id.lower()
        reqs = [r for r in reqs if needle in str(r.get("text","")).lower() or needle in str(r.get("id","")).lower()]
        if not reqs:
            st.info(
                f"No requirements match id '{search_id}'. "
                "Clear the search box to see all requirements."
            )
            st.stop()
        st.caption(f"Showing requirements matching id '{search_id}'.")

    if not reqs:
        st.info("No requirements found for this project. Run the pipeline first.")
        st.stop()

    # -- Count per status for tab labels ---
    counts: dict[str, int] = {}
    for r in reqs:
        k = r.get("review_status", "unknown")
        counts[k] = counts.get(k, 0) + 1

    # -- Shared decision-button helper ---
    def _decision_buttons(req_id: str, key_sfx: str) -> None:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✅ Accept", key=f"acc_{key_sfx}", use_container_width=True):
                if api_post("/review-decisions",
                            json={"requirement_id": req_id, "decision": "accepted"}):
                    st.rerun()
        with c2:
            if st.button("❌ Reject", key=f"rej_{key_sfx}", use_container_width=True):
                if api_post("/review-decisions",
                            json={"requirement_id": req_id, "decision": "rejected"}):
                    st.rerun()
        with c3:
            if st.button("✍️ Needs Revision", key=f"rev_{key_sfx}", use_container_width=True):
                if api_post("/review-decisions",
                            json={"requirement_id": req_id, "decision": "needs_revision"}):
                    st.rerun()

    # -- Card renderer (used inside each tab) ---
    def _render_card(req: dict, idx: int, tab_status: str) -> None:
        """Render one colour-coded requirement card with appropriate body."""
        req_id       = req["id"]
        rev_status   = req.get("review_status", "unknown")
        kind         = req.get("kind", "—")
        nfr          = req.get("nfr_subtype", "—")
        spans        = req.get("source_span_ids", [])
        current_text = req.get("text", "")

        bg, border, emoji, label, txt_color, _ = STATUS_STYLE.get(rev_status, DEFAULT_STYLE)

        # Coloured card header
        st.markdown(
            f"""
            <div style="
                background:{bg};
                border:1.5px solid {border};
                border-radius:10px 10px 0 0;
                padding:10px 16px 8px 16px;
                margin-top:10px;
                margin-bottom:0;
            ">
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                    <span style="font-size:1.1em">{emoji}</span>
                    <code style="font-size:0.8em;color:#555">{req_id}</code>
                    <span style="
                        background:{border};color:{txt_color};
                        padding:2px 9px;border-radius:8px;
                        font-size:0.75em;font-weight:600;
                    ">{label}</span>
                    <span style="color:#888;font-size:0.76em;margin-left:auto;">
                        {kind}&nbsp;\u00c2\u00b7&nbsp;{nfr}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Card body border (matching colour)
        st.markdown(
            f"""<div style="
                background:{bg};
                border-left:1.5px solid {border};
                border-right:1.5px solid {border};
                border-bottom:1.5px solid {border};
                border-radius:0 0 10px 10px;
                padding:10px 16px 14px 16px;
                margin-bottom:4px;
            "></div>""",
            unsafe_allow_html=True,
        )

        key_sfx = f"{tab_status}_{idx}_{req_id}"

        # -- YELLOW tab (pending / needs_revision): editable text ---
        if rev_status in ("pending", "needs_revision"):
            if rev_status == "needs_revision":
                st.markdown(
                    "<p style='color:#92400e;font-size:0.82em;margin:0 0 6px 0'>"
                    "⚠️ Flagged for revision — edit the text below then decide.</p>",
                    unsafe_allow_html=True,
                )
            edited = st.text_area(
                "Requirement text",
                value=current_text,
                key=f"txt_{key_sfx}",
                height=88,
                label_visibility="collapsed",
            )
            save_c, _ = st.columns([0.16, 0.84])
            with save_c:
                if st.button("💾 Save", key=f"save_{key_sfx}", use_container_width=True):
                    if edited.strip() and edited.strip() != current_text.strip():
                        saved = api_patch(f"/requirements/{req_id}",
                                          json={"text": edited.strip()})
                        if saved:
                            st.success("Text saved to database.")
                            st.rerun()
                    else:
                        st.info("No changes detected.")
            if spans:
                st.caption(f"📌 Source spans: {', '.join(spans)}")
            _decision_buttons(req_id, key_sfx)

        # -- GREEN tab (accepted): read-only ---
        elif rev_status == "accepted":
            st.markdown(
                f"<p style='margin:0 0 8px 0;font-size:0.94em;"
                f"color:{txt_color};line-height:1.5'>{current_text}</p>",
                unsafe_allow_html=True,
            )
            if spans:
                st.caption(f"📌 Source spans: {', '.join(spans)}")
            with st.expander("Change decision"):
                _decision_buttons(req_id, f"chg_{key_sfx}")

        # -- RED tab (rejected): strikethrough ---
        elif rev_status == "rejected":
            st.markdown(
                f"<p style='margin:0 0 8px 0;font-size:0.94em;"
                f"color:#aaa;text-decoration:line-through;line-height:1.5'>"
                f"{current_text}</p>",
                unsafe_allow_html=True,
            )
            if spans:
                st.caption(f"📌 Source spans: {', '.join(spans)}")
            with st.expander("Change decision"):
                _decision_buttons(req_id, f"chg_{key_sfx}")

        # -- Any other status ---
        else:
            st.markdown(
                f"<p style='margin:0 0 8px 0;font-size:0.94em;"
                f"color:#555;line-height:1.5'>{current_text}</p>",
                unsafe_allow_html=True,
            )
            if spans:
                st.caption(f"📌 Source spans: {', '.join(spans)}")
            _decision_buttons(req_id, key_sfx)

    # -- Sync Graph button ---
    st.divider()
    sync_col, info_col = st.columns([0.32, 0.68])
    with sync_col:
        if st.button(
            "🔄 Sync Knowledge Graph with Decisions",
            key="btn_sync_graph",
            help=(
                "Re-scans every requirement's review decision and updates the "
                "knowledge graph: accepted nodes are ensured present, rejected "
                "nodes are removed (along with their edges)."
            ),
        ):
            with st.spinner("Syncing knowledge graph…"):
                result = api_post(f"/projects/{project_id}/graph/sync-decisions")
            if result:
                st.success(
                    f"✅ Graph synced — {result.get('synced', 0)} requirements processed "
                    f"({result.get('removed', 0)} removed, {result.get('updated', 0)} updated"
                    + (f", ⚠️ {result.get('errors', 0)} errors" if result.get("errors") else "")
                    + ")"
                )
    with info_col:
        n_accepted_total = counts.get("accepted", 0)
        n_rejected_total = counts.get("rejected", 0)
        st.markdown(
            f"<span style='font-size:0.87em;color:#666'>"
            f"Graph will keep <b>{n_accepted_total}</b> accepted node(s) and "
            f"remove <b>{n_rejected_total}</b> rejected node(s).</span>",
            unsafe_allow_html=True,
        )

    # -- Inject CSS to tint each Streamlit tab button ---
    # Streamlit's st.tabs() renders tab buttons as <button> elements with a
    # data-baseweb attribute. We target each tab's nth button via CSS and apply
    # a background tint matching its status colour so the tab itself is coloured.
    st.markdown("""
        <style>
        /* Tab bar colour each button by position */
        div[data-testid="stTabs"] > div > div > div:nth-child(1) button {
            background: #d4edda !important;
            color: #1a5c35 !important;
            border-radius: 6px 6px 0 0;
            font-weight: 600;
        }
        div[data-testid="stTabs"] > div > div > div:nth-child(2) button {
            background: #f8d7da !important;
            color: #7f1d1d !important;
            border-radius: 6px 6px 0 0;
            font-weight: 600;
        }
        div[data-testid="stTabs"] > div > div > div:nth-child(3) button {
            background: #fff9c4 !important;
            color: #78600a !important;
            border-radius: 6px 6px 0 0;
            font-weight: 600;
        }
        div[data-testid="stTabs"] > div > div > div:nth-child(4) button {
            background: #fff9c4 !important;
            color: #78600a !important;
            border-radius: 6px 6px 0 0;
            font-weight: 600;
        }
        div[data-testid="stTabs"] > div > div > div:nth-child(5) button {
            background: #eeeeee !important;
            color: #555555 !important;
            border-radius: 6px 6px 0 0;
            font-weight: 600;
        }
        /* Active / selected tab stays same colour, just slightly brighter border */
        div[data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom: 3px solid currentColor !important;
            opacity: 1 !important;
        }
        /* Inactive tabs slightly dimmed */
        div[data-testid="stTabs"] button[aria-selected="false"] {
            opacity: 0.72;
        }
        </style>
    """, unsafe_allow_html=True)

    # -- Build tabs ---
    n_acc = counts.get("accepted", 0)
    n_rej = counts.get("rejected", 0)
    n_pen = counts.get("pending", 0)
    n_rev = counts.get("needs_revision", 0)
    n_def = counts.get("deferred", 0)

    tab_acc, tab_rej, tab_pen, tab_nr, tab_def = st.tabs([
        f"✅ Accepted ({n_acc})",
        f"❌ Rejected ({n_rej})",
        f"⏳ Pending ({n_pen})",
        f"✍️ Needs Revision ({n_rev})",
        f"⏭️ Deferred ({n_def})",
    ])

    with tab_acc:
        group = [r for r in reqs if r.get("review_status") == "accepted"]
        if not group:
            st.info("No accepted requirements yet.")
        for idx, req in enumerate(group):
            _render_card(req, idx, "acc")

    with tab_rej:
        group = [r for r in reqs if r.get("review_status") == "rejected"]
        if not group:
            st.info("No rejected requirements yet.")
        for idx, req in enumerate(group):
            _render_card(req, idx, "rej")

    with tab_pen:
        group = [r for r in reqs if r.get("review_status") == "pending"]
        if not group:
            st.info("No requirements pending review.")
        for idx, req in enumerate(group):
            _render_card(req, idx, "pen")

    with tab_nr:
        group = [r for r in reqs if r.get("review_status") == "needs_revision"]
        if not group:
            st.info("No requirements flagged for revision.")
        for idx, req in enumerate(group):
            _render_card(req, idx, "nr")

    with tab_def:
        group = [r for r in reqs if r.get("review_status") == "deferred"]
        if not group:
            st.info("No deferred requirements.")
        for idx, req in enumerate(group):
            _render_card(req, idx, "def")


# ---------------------------------------------------------------------------
# Screen 4 - Evidence Viewer
# ---------------------------------------------------------------------------
elif screen == "4. Evidence Viewer":
    st.header("Evidence Viewer")
    project_id   = st.session_state.get("project_id", "")
    project_name = st.session_state.get("project_name", "")
    if not project_id:
        st.warning("No project loaded. Go to **1. Project Upload** and create or select a project.")
        st.stop()
    # Project header + search-by-id bar on the same row.
    # Keyed per-project so switching projects does not leak the query.
    hdr_left, hdr_right = st.columns([0.62, 0.38])
    with hdr_left:
        st.markdown(
            f"<span style='font-size:1.05em;font-weight:600;color:#444'>"
            f"📁 Selected Project: {project_name or project_id}</span>",
            unsafe_allow_html=True,
        )
    with hdr_right:
        search_id = st.text_input(
            "Search by requirement id",
            key=f"evidence_search_{project_id}",
            placeholder="Search by requirement id (e.g. REQ_abc123)",
            label_visibility="collapsed",
        ).strip()

    # -- Reset state when the loaded project changes ---
    # The selectbox below otherwise retains its previous selection across reruns,
    # which can leave the previous project's requirement on screen after switching.
    if "evidence_project_id" not in st.session_state:
        st.session_state["evidence_project_id"] = project_id
    if st.session_state["evidence_project_id"] != project_id:
        # Drop the per-project selectbox state from any previous project so its
        # cached selection can't leak in if the user switches back later.
        prev_key = f"evidence_selected_{st.session_state['evidence_project_id']}"
        st.session_state.pop(prev_key, None)
        st.session_state["evidence_project_id"] = project_id

    reqs  = api_get(f"/projects/{project_id}/requirements")
    spans = api_get(f"/projects/{project_id}/source-spans")
    docs  = api_get(f"/projects/{project_id}/documents")

    # Narrow `reqs` by the search-by-id query before the selectbox is built,
    # so the dropdown only lists matching requirements.
    if reqs and search_id:
        needle = search_id.lower()
        reqs = [r for r in reqs if needle in str(r.get("id", "")).lower() or needle in str(r.get("text","")).lower()]
        if not reqs:
            st.info(
                f"No requirements match id '{search_id}'. "
                "Clear the search box to see all requirements."
            )
            st.stop()

    if reqs and spans:
        span_map = {s["id"]: s for s in spans}
        # document_id - filename, so we can show *which* uploaded document each
        # supporting span came from (e.g. "meeting_transcript.txt") instead of
        # the opaque internal span id alone.
        doc_name_map: dict[str, str] = {
            d["id"]: d.get("filename", "") for d in (docs or [])
        }
        selected_req = st.selectbox(
            "Select requirement",
            options=[f"{r['id']}: {r['text'][:60]}..." for r in reqs],
            key=f"evidence_selected_{project_id}",
        )
        if selected_req:
            req_id = selected_req.split(":")[0]
            req = next((r for r in reqs if r["id"] == req_id), None)
            if req:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Requirement")
                    st.markdown(f"**{req['id']}**")
                    st.markdown(req["text"])
                    st.markdown(f"*Kind:* {req['kind']} | *NFR:* {req['nfr_subtype']}")
                with col2:
                    st.subheader("Supporting Source Spans")
                    for span_id in req.get("source_span_ids", []):
                        span = span_map.get(span_id)
                        if span:
                            doc_filename = doc_name_map.get(
                                span.get("document_id", ""), ""
                            ) or "unknown document"
                            section = span.get("section_title") or ""
                            speaker = span.get("speaker") or ""
                            # Build a one-line provenance header: filename, then
                            # optional section title or speaker if the span has them.
                            meta_bits = [f"\U0001f4c4 **{doc_filename}**"]
                            if section:
                                meta_bits.append(f"\u00a7 {section}")
                            if speaker:
                                meta_bits.append(f"\U0001f5e3 {speaker}")
                            meta_line = "  \u00b7  ".join(meta_bits)

                            st.info(
                                f"{meta_line}\n\n"
                                f"**[{span_id}]** {span['text'][:200]}..."
                            )
                        else:
                            st.warning(f"Span {span_id} not found.")
    elif reqs is not None and spans is not None:
        # API returned successfully but the new project has no data yet.
        st.info("This project has no requirements or source spans yet. Run the pipeline first.")

# ─────────────────────────────────────────────────────────────────────────────
# Screen 5 – Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────
elif screen == "5. Knowledge Graph":
    st.header("🕸️ Knowledge Graph")

    project_id   = st.session_state.get("project_id", "")
    project_name = st.session_state.get("project_name", "")
    if not project_id:
        st.warning("No project loaded. Go to **1. Project Upload** and create or select a project.")
        st.stop()
    st.markdown(
        f"<span style='font-size:1.05em;font-weight:600;color:#444'>📁 Selected Project: {project_name or project_id}</span>",
        unsafe_allow_html=True,
    )

    import streamlit.components.v1 as components

    # ── Fetch graph data ─────────────────────────────────────────────────────
    graph = api_get(f"/projects/{project_id}/graph")
    stats = api_get(f"/projects/{project_id}/graph/stats")

    if not graph or (not graph.get("nodes") and not graph.get("edges")):
        st.info("No graph data yet. Run the pipeline first to build the knowledge graph.")
        st.stop()

    nodes_raw_all = graph.get("nodes", [])
    edges_raw_all = graph.get("edges", [])

    # ── Filter: only show accepted requirements (and their connected source spans) ──
    # Keep requirement nodes only if their status == "accepted".
    # Keep source_span nodes only if they are referenced by an edge that touches
    # a kept (accepted) requirement, so the graph stays connected and meaningful.
    accepted_req_ids = {
        n.get("id")
        for n in nodes_raw_all
        if n.get("node_type") == "requirement" and n.get("status") == "accepted"
    }

    # Find source_span ids that are connected (in either direction) to an accepted requirement
    connected_span_ids: set = set()
    span_ids_all = {
        n.get("id") for n in nodes_raw_all if n.get("node_type") == "source_span"
    }
    for e in edges_raw_all:
        s = e.get("source", "")
        t = e.get("target", "")
        if s in accepted_req_ids and t in span_ids_all:
            connected_span_ids.add(t)
        if t in accepted_req_ids and s in span_ids_all:
            connected_span_ids.add(s)

    kept_ids = accepted_req_ids | connected_span_ids

    nodes_raw = [n for n in nodes_raw_all if n.get("id") in kept_ids]
    edges_raw = [
        e for e in edges_raw_all
        if e.get("source") in kept_ids and e.get("target") in kept_ids
    ]

    # Counts for both filtered and total (for context)
    n_req_total = sum(1 for n in nodes_raw_all if n.get("node_type") == "requirement")
    n_req = len(accepted_req_ids)
    n_span = sum(1 for n in nodes_raw if n.get("node_type") == "source_span")
    n_edge = len(edges_raw)

    st.caption(
        f"Showing only **accepted** requirements ({n_req} of {n_req_total}). "
        "Requirements with status *proposed*, *needs revision*, *rejected*, or *deferred* are hidden."
    )

    if n_req == 0:
        st.info(
            "No requirements have been accepted yet. "
            "Go to **3. Review Queue** to accept requirements, then return here to see the graph."
        )
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Nodes",       len(nodes_raw))
    m2.metric("Requirements",      n_req)
    m3.metric("Source Spans",      n_span)
    m4.metric("Edges",             n_edge)

    st.markdown("---")

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        show_spans = st.toggle("Show source span nodes", value=True)
    with ctrl2:
        show_labels = st.toggle("Show edge labels", value=True)
    with ctrl3:
        layout_choice = st.selectbox(
            "Layout",
            ["Barnes-Hut (force-directed)", "Hierarchical (top-down)"],
            index=0,
        )

    # ── Edge-type colour legend ──────────────────────────────────────────────
    EDGE_COLOURS: dict[str, str] = {
        "derived_from":    "#3498db",   # blue
        "requires":        "#e67e22",   # orange
        "conflicts_with":  "#e74c3c",   # red
        "duplicates":      "#9b59b6",   # purple
        "refines":         "#1abc9c",   # teal
        "constrains":      "#f39c12",   # amber
        "tested_by":       "#2ecc71",   # green
        "affected_by":     "#95a5a6",   # grey
        "realized_by":     "#34495e",   # dark
        "owned_by":        "#7f8c8d",   # mid-grey
    }
    DEFAULT_EDGE_COLOUR = "#aaaaaa"

    # Truncate long text for node tooltips
    def _trunc(text: str, n: int = 120) -> str:
        return text[:n] + "…" if len(text) > n else text

    # ── Build Pyvis network ──────────────────────────────────────────────────
    try:
        from pyvis.network import Network
    except ImportError:
        st.error("pyvis is not installed. Run: pip install pyvis")
        st.stop()

    net = Network(
        height="620px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#333333",
    )

    # Physics options
    if "Hierarchical" in layout_choice:
        net.set_options("""
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed",
              "levelSeparation": 220,
              "nodeSpacing": 120
            }
          },
          "physics": { "enabled": false },
          "edges": { "smooth": { "type": "cubicBezier", "forceDirection": "horizontal" } }
        }
        """)
    else:
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 150,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "minVelocity": 0.75
          },
          "edges": { "smooth": { "type": "dynamic" } }
        }
        """)

    # Add nodes
    for node in nodes_raw:
        nid       = node.get("id", "?")
        ntype     = node.get("node_type", "")
        ntext     = node.get("text", "")
        nkind     = node.get("kind", "")
        nstatus   = node.get("status", "")

        if ntype == "requirement":
            # Colour by kind
            kind_colours = {
                "functional":        "#4a90d9",
                "non_functional":    "#8e44ad",
                "constraint":        "#e67e22",
                "domain_assumption": "#16a085",
                "business_rule":     "#c0392b",
            }
            colour = kind_colours.get(nkind, "#4a90d9")
            short_label = nid.split("-")[-1] if "-" in nid else nid[:8]
            tooltip = (
                f"{ntext}"
            )
            net.add_node(
                nid,
                label=short_label,
                color=colour,
                shape="dot",
                size=22,
                title=tooltip,
                font={"size": 13, "color": "#ffffff", "bold": True},
                borderWidth=2,
                borderWidthSelected=4,
            )

        elif ntype == "source_span":
            if not show_spans:
                continue
            doc_id = node.get("document_id", "")
            tooltip = (
                f"{_trunc(ntext, 160)}"
            )
            net.add_node(
                nid,
                label="◆",
                color={"background": "#27ae60", "border": "#1e8449"},
                shape="diamond",
                size=14,
                title=tooltip,
                font={"size": 12, "color": "#ffffff"},
            )

    # Collect IDs actually added to the network
    added_ids = set(net.get_nodes())

    # Add edges (only between nodes that were added)
    for edge in edges_raw:
        src   = edge.get("source", "")
        tgt   = edge.get("target", "")
        etype = edge.get("edge_type", "")
        conf  = edge.get("confidence", None)

        if src not in added_ids or tgt not in added_ids:
            continue

        colour   = EDGE_COLOURS.get(etype, DEFAULT_EDGE_COLOUR)
        edge_lbl = etype.replace("_", " ") if show_labels else ""
        conf_str = f"  conf: {conf:.2f}" if conf is not None else ""

        net.add_edge(
            src, tgt,
            label=edge_lbl,
            color=colour,
            arrows="to",
            title=f"{etype}{conf_str}",
            font={"size": 10, "color": "#555"},
            width=1.8,
            selectionWidth=3,
        )

    # ── Render ───────────────────────────────────────────────────────────────
    html_str = net.generate_html()
    components.html(html_str, height=640, scrolling=False)

    # ── Legend ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Node types**")
    leg1, leg2, leg3 = st.columns(3)
    with leg1:
        st.markdown(
            "<span style='display:inline-block;width:14px;height:14px;"
            "border-radius:50%;background:#4a90d9;vertical-align:middle'></span>"
            " &nbsp;Functional requirement",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span style='display:inline-block;width:14px;height:14px;"
            "border-radius:50%;background:#8e44ad;vertical-align:middle'></span>"
            " &nbsp;Non-functional requirement",
            unsafe_allow_html=True,
        )
    with leg2:
        st.markdown(
            "<span style='display:inline-block;width:14px;height:14px;"
            "border-radius:50%;background:#e67e22;vertical-align:middle'></span>"
            " &nbsp;Constraint",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span style='display:inline-block;width:14px;height:14px;"
            "border-radius:50%;background:#27ae60;transform:rotate(45deg);"
            "vertical-align:middle'></span>"
            " &nbsp;Source span",
            unsafe_allow_html=True,
        )
    with leg3:
        st.markdown(
            "<span style='display:inline-block;width:14px;height:14px;"
            "border-radius:50%;background:#16a085;vertical-align:middle'></span>"
            " &nbsp;Domain assumption",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span style='display:inline-block;width:14px;height:14px;"
            "border-radius:50%;background:#c0392b;vertical-align:middle'></span>"
            " &nbsp;Business rule",
            unsafe_allow_html=True,
        )

    st.markdown("**Edge types**")
    edge_rows = list(EDGE_COLOURS.items())
    col_a, col_b = st.columns(2)
    half = len(edge_rows) // 2 + len(edge_rows) % 2
    for etype, ecolour in edge_rows[:half]:
        col_a.markdown(
            f"<span style='display:inline-block;width:24px;height:3px;"
            f"background:{ecolour};vertical-align:middle;margin-right:6px'></span>"
            f"{etype.replace('_', ' ')}",
            unsafe_allow_html=True,
        )
    for etype, ecolour in edge_rows[half:]:
        col_b.markdown(
            f"<span style='display:inline-block;width:24px;height:3px;"
            f"background:{ecolour};vertical-align:middle;margin-right:6px'></span>"
            f"{etype.replace('_', ' ')}",
            unsafe_allow_html=True,
        )

    # ── Conflicts panel ──────────────────────────────────────────────────────
    conflicts = api_get(f"/projects/{project_id}/export/conflicts")
    if conflicts:
        try:
            conflict_list = json.loads(conflicts) if isinstance(conflicts, str) else conflicts
            if conflict_list:
                st.markdown("---")
                st.subheader(f"⚠️ Conflicts ({len(conflict_list)})")
                for c in conflict_list:
                    st.error(
                        f"**{c.get('conflict_type', '?')}** | "
                        f"Reqs: {c.get('involved_requirement_ids', [])} | "
                        f"{c.get('explanation', '')}"
                    )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Screen 6 – Change Impact Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif screen == "6. Impact Analysis":
    st.header("⚡ Change Impact Analysis")

    project_id   = st.session_state.get("project_id", "")
    project_name = st.session_state.get("project_name", "")
    if not project_id:
        st.warning("No project loaded. Go to **1. Project Upload** and create or select a project.")
        st.stop()
    st.markdown(
        f"<span style='font-size:1.05em;font-weight:600;color:#444'>📁 Selected Project: {project_name or project_id}</span>",
        unsafe_allow_html=True,
    )

    # Persist impact result across reruns so the Enforce section stays visible
    if "impact_result" not in st.session_state:
        st.session_state.impact_result = None
    if "impact_change" not in st.session_state:
        st.session_state.impact_change = ""
    if "enforce_result" not in st.session_state:
        st.session_state.enforce_result = None

    # ── Reset state when the loaded project changes ──────────────────────────
    # Otherwise switching projects would leave the previous project's impact
    # analysis (change summary, direct/indirect impacts, enforce result) on
    # screen, attached to the wrong project.
    if "impact_project_id" not in st.session_state:
        st.session_state["impact_project_id"] = project_id
    if st.session_state["impact_project_id"] != project_id:
        st.session_state.impact_result   = None
        st.session_state.impact_change   = ""
        st.session_state.enforce_result  = None
        st.session_state["impact_project_id"] = project_id

    change_text = st.text_area(
        "Describe the change request",
        value=st.session_state.impact_change or "Replace Google login with institutional SSO.",
        height=90,
    )

    if st.button("🔎 Analyse Impact", type="primary") and change_text.strip():
        st.session_state.impact_change  = change_text.strip()
        st.session_state.enforce_result = None
        with st.spinner("Running Impact Agent — analysing dependency graph…"):
            result = api_post(
                f"/projects/{project_id}/impact",
                json={"change_request": change_text.strip()},
            )
        st.session_state.impact_result = result

    result = st.session_state.impact_result
    if not result:
        st.info("Enter a change request above and click **🔎 Analyse Impact** to begin.")
        st.stop()

    directly   = result.get("directly_affected",   [])
    indirectly = result.get("indirectly_affected", [])
    tasks      = result.get("suggested_review_tasks", [])

    st.success("Impact analysis complete.")
    st.subheader("Change Summary")
    st.info(result.get("change_summary", ""))

    m1, m2, m3 = st.columns(3)
    m1.metric("Directly Affected",   len(directly))
    m2.metric("Indirectly Affected", len(indirectly))
    m3.metric("Review Tasks",        len(tasks))

    st.divider()

    # Every impacted node shown in the dropdowns below is treated as enforceable.
    # (The impact agent's `node_type` is a free-form LLM string — e.g. "requirement",
    # "Requirement", "FunctionalRequirement" — so filtering on an exact value was
    # silently dropping everything and breaking the Enforce section.)
    direct_req_ids   = [n["node_id"] for n in directly]
    indirect_req_ids = [n["node_id"] for n in indirectly]
    all_req_ids      = list(dict.fromkeys(direct_req_ids + indirect_req_ids))

    if directly:
        with st.expander(f"Direct impacts ({len(directly)})", expanded=True):
            for node in directly:
                st.error(
                    "**" + node["node_id"] + "** `" + node["node_type"] + "`  \n"
                    + node["explanation"]
                )
    if indirectly:
        with st.expander(f"Indirect impacts ({len(indirectly)})", expanded=False):
            for node in indirectly:
                st.warning(
                    "**" + node["node_id"] + "** `" + node["node_type"] + "`  \n"
                    + node["explanation"]
                )
    if tasks:
        st.divider()
        st.subheader("Suggested Review Tasks")
        for i, task in enumerate(tasks, 1):
            st.markdown(f"{i}. {task}")

    # ── ENFORCE section ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("🚨 Enforce Change Request")

    if not all_req_ids:
        st.info("No requirement nodes were identified — nothing to enforce.")
    else:
        n_reqs = len(all_req_ids)
        st.markdown(
            f"Enforcement will create a **new version** of this project and use the LLM to "
            f"rewrite each of the **{n_reqs} impacted requirement(s)** so they incorporate "
            f"the change request directly.  The current project stays untouched."
        )

        c_direct, c_indirect = st.columns(2)
        with c_direct:
            enforce_direct = st.checkbox(
                f"Include direct impacts ({len(direct_req_ids)})",
                value=True,
            )
        with c_indirect:
            enforce_indirect = st.checkbox(
                f"Include indirect impacts ({len(indirect_req_ids)})",
                value=False,
            )

        version_name = st.text_input(
            "New version name (optional)",
            value="",
            placeholder=f"Default: {project_name or project_id} (rev N)",
        )

        target_direct   = direct_req_ids   if enforce_direct   else []
        target_indirect = indirect_req_ids if enforce_indirect else []
        total_targets   = len(target_direct) + len(target_indirect)

        enforce_btn = st.button(
            f"⚡ Create new version and enforce {total_targets} requirement(s)",
            type="primary",
            disabled=(total_targets == 0),
        )

        if enforce_btn:
            with st.spinner(
                f"Rewriting {total_targets} requirement(s) with LLM and "
                "cloning project — this may take a minute…"
            ):
                payload = {
                    "change_request": st.session_state.impact_change,
                    "direct_ids":     target_direct,
                    "indirect_ids":   target_indirect,
                }
                if version_name.strip():
                    payload["new_project_name"] = version_name.strip()
                resp = api_post(
                    f"/projects/{project_id}/impact/enforce",
                    json=payload,
                )
            if resp:
                st.session_state.enforce_result = resp

        er = st.session_state.enforce_result
        if er:
            st.success(
                f"✅ New version created: **{er['new_project_name']}**  \n"
                f"Rewrote {er['rewritten_count']} of {er['impacted_count']} "
                f"impacted requirement(s)."
            )
            if er.get("failed_ids"):
                st.warning(
                    f"⚠️ LLM rewrite failed for {len(er['failed_ids'])} requirement(s); "
                    f"their original text was kept in the new version."
                )
            colA, colB = st.columns(2)
            with colA:
                if st.button("Open new version (switch project)"):
                    st.session_state["project_id"]   = er["new_project_id"]
                    st.session_state["project_name"] = er["new_project_name"]
                    st.rerun()
            with colB:
                st.markdown(
                    "Or go to **7. SRS Export** to download the SRS for "
                    "the new version."
                )


# Screen 7 – SRS Export
# ─────────────────────────────────────────────────────────────────────────────
elif screen == "7. SRS Export":
    st.header("📥 SRS Export")

    project_id   = st.session_state.get("project_id", "")
    project_name = st.session_state.get("project_name", "")
    if not project_id:
        st.warning("No project loaded. Go to **1. Project Upload** and create or select a project.")
        st.stop()
    st.markdown(
        f"<span style='font-size:1.05em;font-weight:600;color:#444'>📁 Selected Project: {project_name or project_id}</span>",
        unsafe_allow_html=True,
    )

    # ── Standard exports for the currently-loaded project ────────────────────
    st.subheader("Current Project Exports")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Check whether the pipeline has already composed an SRS for this
        # project.  If yes, surface a direct download button immediately —
        # no need for the user to click "Generate" first.
        srs_status = api_get(f"/projects/{project_id}/export/srs/status") or {}
        srs_ready = bool(srs_status.get("ready", False))

        if srs_ready:
            # Pre-fetch the cached markdown (cheap server-side file read,
            # no LLM involved) so the download button is wired up on first
            # render of this screen.
            md_cached = st.session_state.get(f"current_srs_md_{project_id}")
            if md_cached is None:
                md_cached = api_get_text(
                    f"/projects/{project_id}/export/srs/markdown"
                )
                if md_cached:
                    st.session_state[f"current_srs_md_{project_id}"] = md_cached

            if md_cached:
                st.success("✅ SRS ready (composed during pipeline run)")
                # Build a base64 data-URI so the markdown opens in a new browser
                # tab without needing a file server.
                import base64 as _b64
                _md_b64 = _b64.b64encode(md_cached.encode()).decode()
                _preview_href = f"data:text/plain;charset=utf-8;base64,{_md_b64}"
                dl_col, icon_col = st.columns([0.6, 0.4])
                with dl_col:
                    st.download_button(
                        label="⬇️ Download SRS.md",
                        data=md_cached,
                        file_name="srs.md",
                        mime="text/markdown",
                        key="dl_srs_md_ready",
                    )
                with icon_col:
                    st.markdown(
                        f"<a href='{_preview_href}' target='_blank' "
                        f"title='Preview SRS.md in new tab' "
                        f"style='font-size:1em;text-decoration:none;'>Preview</a>",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("SRS reported ready but could not be loaded.")
        else:
            # Fallback: pipeline hasn't run yet, or composer failed.
            # Keep the original on-demand Generate flow.
            if st.button("📄 Generate SRS Document", key="btn_srs_md"):
                with st.spinner("Composing SRS from accepted requirements..."):
                    md_text = api_get_text(f"/projects/{project_id}/export/srs/markdown")
                if md_text:
                    st.success("SRS composed successfully!")
                    import base64 as _b64
                    _md_b64 = _b64.b64encode(md_text.encode()).decode()
                    _preview_href = f"data:text/plain;charset=utf-8;base64,{_md_b64}"
                    dl_col2, icon_col2 = st.columns([0.6, 0.4])
                    with dl_col2:
                        st.download_button("Download SRS.md", md_text,
                                           "srs.md", "text/markdown", key="dl_srs_md")
                    with icon_col2:
                        st.markdown(
                            f"<a href='{_preview_href}' target='_blank' "
                            f"title='Preview SRS.md in new tab' "
                            f"style='font-size:1.6em;text-decoration:none;'>🔍</a>",
                            unsafe_allow_html=True,
                        )
    with col2:
        if st.button("📊 Export Traceability (CSV)"):
            data = api_get_text(f"/projects/{project_id}/export/traceability")
            if data:
                st.download_button("Download CSV", data, "traceability.csv", "text/csv")
    with col3:
        if st.button("🔗 Export Graph (GraphML)"):
            data = api_get_text(f"/projects/{project_id}/export/graph/graphml")
            if data:
                st.download_button("Download GraphML", data, "graph.graphml", "application/xml")
    with col4:
        if st.button("📈 Export Graph (JSON)"):
            data = api_get(f"/projects/{project_id}/export/graph/json")
            if data:
                st.download_button("Download JSON", json.dumps(data, indent=2),
                                   "graph.json", "application/json")

    # ── Project versions section ─────────────────────────────────────────────
    st.divider()
    st.subheader("📚 Project Versions")
    st.caption(
        "Each enforcement of a change request creates a new project version. "
        "Download the SRS for any version below to compare how requirements evolved."
    )

    versions = api_get(f"/projects/{project_id}/versions")
    if not versions:
        st.info("No version history available for this project.")
    else:
        # Sort by created_at ascending so v1 is at the top
        versions_sorted = sorted(versions, key=lambda v: v.get("created_at", ""))

        for idx, v in enumerate(versions_sorted):
            v_id     = v["id"]
            v_name   = v["name"]
            v_desc   = v.get("description", "")
            v_parent = v.get("parent_project_id")
            v_created = v.get("created_at", "")
            try:
                from datetime import datetime
                v_created_human = datetime.fromisoformat(v_created).strftime("%d %b %Y, %H:%M")
            except Exception:
                v_created_human = v_created

            is_root    = (v_parent is None)
            is_current = (v_id == project_id)

            if is_current:
                bg, border, badge, badge_bg = "#f0fff4", "#a8d5b5", "● Current", "#a8d5b5"
            elif is_root:
                bg, border, badge, badge_bg = "#eef5ff", "#9bc4f5", "Original", "#9bc4f5"
            else:
                bg, border, badge, badge_bg = "#fafafa", "#dcdcdc", f"v{idx + 1}", "#dcdcdc"

            st.markdown(
                f"""
                <div style="
                    background:{bg};
                    border:1.5px solid {border};
                    border-radius:8px;
                    padding:10px 16px;
                    margin-bottom:6px;
                ">
                    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                        <span style="font-weight:600;font-size:1em">{v_name}</span>
                        <span style="
                            background:{badge_bg};color:#1a1a1a;
                            padding:1px 8px;border-radius:8px;
                            font-size:0.72em;font-weight:600;
                        ">{badge}</span>
                        <span style="color:#888;font-size:0.78em;margin-left:auto;">
                            Created {v_created_human}
                        </span>
                    </div>
                    <div style="color:#666;font-size:0.82em;margin-top:4px">
                        {v_desc if v_desc else "<i>No description</i>"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_cols = st.columns([1, 1, 4])
            with btn_cols[0]:
                gen_key = f"gen_srs_{idx}_{v_id}"
                if st.button("📄 Generate SRS", key=gen_key):
                    with st.spinner(f"Composing SRS for {v_name}…"):
                        md = api_get_text(f"/projects/{v_id}/export/srs/markdown")
                    if md:
                        st.session_state[f"srs_md_{v_id}"] = md
                        st.success(f"SRS for **{v_name}** is ready.")

            with btn_cols[1]:
                md_cached = st.session_state.get(f"srs_md_{v_id}")
                if md_cached:
                    safe_name = "".join(
                        c if c.isalnum() or c in "-_" else "_" for c in v_name
                    )
                    st.download_button(
                        "⬇️ Download .md",
                        data=md_cached,
                        file_name=f"srs_{safe_name}.md",
                        mime="text/markdown",
                        key=f"dl_srs_{idx}_{v_id}",
                    )

            with btn_cols[2]:
                if not is_current:
                    if st.button(
                        "Switch to this version",
                        key=f"switch_{idx}_{v_id}",
                    ):
                        st.session_state["project_id"]   = v_id
                        st.session_state["project_name"] = v_name
                        st.rerun()