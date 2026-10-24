import json
import os
import time
from datetime import date, datetime, timedelta, timezone

import streamlit as st
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.8-flash"
# Backup models, tried in order if the main one is busy.
# Copy the exact names from the model list in Google AI Studio. Names that don't exist are skipped.
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
PLACEHOLDER = "Select..."

st.set_page_config(page_title="THUTHUKA-AI", page_icon="🧭")

# What the student has saved, applied for or snoozed (kept for this visit)
st.session_state.setdefault("interested", [])
st.session_state.setdefault("applied", [])
st.session_state.setdefault("snoozed", {})

# Load the vetted programme list
with open("programmes.json", encoding="utf-8") as f:
    programmes = json.load(f)
programmes_by_id = {p["id"]: p for p in programmes}

# Load the employer-demand skills
with open("skills_data.json", encoding="utf-8") as f:
    skills_data = json.load(f)

# Connection to Gemini
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ---------- Answer options ----------
AGE_BANDS = [PLACEHOLDER, "Under 18", "18-24", "25-35", "36 or older"]
STAGES = [
    PLACEHOLDER,
    "Grade 11 or 12",
    "Matric completed",
    "TVET student",
    "University student",
    "Graduate",
    "Looking for work",
]
PROVINCES = [
    PLACEHOLDER,
    "Eastern Cape",
    "Free State",
    "Gauteng",
    "KwaZulu-Natal",
    "Limpopo",
    "Mpumalanga",
    "Northern Cape",
    "North West",
    "Western Cape",
]
MATHS = [PLACEHOLDER, "Mathematics", "Maths Literacy",
         "Not applicable yet", "Prefer not to say"]
INTERESTS = [
    "Building apps or websites",
    "Data and AI",
    "Networking or cybersecurity",
    "Cloud computing",
    "Design",
    "I'm not sure yet",
]
TECH_SKILLS = [
    "None yet",
    "Basic computer skills",
    "Some coding",
    "I have done a course or certificate",
]
# Soft skills come from skills_data.json, so the options are backed by the research
SOFT_SKILLS = [s["skill"] for s in skills_data if s["type"] == "soft"]
PROOF_OPTIONS = [
    "Not yet",
    "A certificate or course",
    "A leadership role or volunteering (school, club, church, community)",
    "A project I built or worked on",
]
DEVICES = ["Smartphone", "Shared computer", "Own laptop"]
DATA_ACCESS = [
    PLACEHOLDER,
    "Regular data or WiFi",
    "Limited data",
    "Mostly WiFi at a school or hub",
]
MOBILITY = [
    PLACEHOLDER,
    "I can study full-time and move to another city",
    "I can study full-time but cannot move",
    "I need to stay home (part-time or online only)",
    "Not sure",
]
COST = [
    PLACEHOLDER,
    "Free only",
    "A stipend would help",
    "I can pay for some things",
]
# The question widgets whose values we remember when the student changes screen
ANSWER_KEYS = [
    "age", "stage", "province", "maths", "interests", "tech_skills",
    "soft_skills", "devices", "data_access", "mobility", "cost",
]

# ---------- What the AI is allowed to see ----------
# Only the fields it needs. Notes, links and check dates stay out.
AI_PROGRAMME_FIELDS = [
    "id", "name", "provider", "type", "field", "skills_built", "audience",
    "requirements", "cost", "pays_learner", "delivery", "location", "timing",
]
AI_SKILL_FIELDS = ["skill", "type",
                   "why_it_matters", "employer_signal", "sa_signal"]

SYSTEM_PROMPT = """You are THUTHUKA-AI, a career guide for young South Africans, especially students in underserved communities. You receive a student's profile, a list of vetted programmes and a list of employer-demand skills with their evidence. Build a personalised career pathway.

Rules:
1. Recommend ONLY programmes from the provided programme list, and refer to each one by its exact id. Never invent programmes, dates, requirements or links.
2. If a requirement, date or cost is blank or unclear in the data, tell the student to check the programme's official page. Never guess.
3. Base skills advice on the provided skills list. When you make a claim about what employers want, stay within what the employer_signal and sa_signal fields say.
4. Respect the student's situation: age group, matric maths, province, devices, data access, whether they can move, and whether they need free options. Put options that fit their situation first, and say plainly when something does not fit (for example a campus programme far from home, or a minimum qualification they do not have yet).
5. Include at least one alternative to the traditional university route when it fits the student. If they need free or funded options, include a funding route such as a bursary when one is in the list.
6. If the student is under 18, mention that some programmes need a parent or guardian's consent or have a minimum age.
7. If the student is not sure what they want, suggest one or two directions based on their interests and skills, and explain why in simple words.
8. For each soft skill the student ticked, say whether they have proof. Proof is self-reported. If they have no proof yet, suggest how one of the recommended programmes could help them earn it.
9. Write in warm, plain English with short sentences. No jargon. Speak directly to the student.
10. Respond with JSON only, in exactly the structure the user message gives you."""

OUTPUT_FORMAT = """{
  "summary": "2-3 sentences speaking to the student",
  "direction": {"title": "a career direction", "why": "why it suits them"},
  "skills_you_have": [{"skill": "...", "note": "..."}],
  "skills_to_build": [{"skill": "...", "why": "why employers want it", "how": "how to start"}],
  "soft_skills": [{"skill": "...", "proof_status": "has proof | no proof yet", "advice": "..."}],
  "programmes": [{"id": "exact id from the list", "route": "short label such as Online learning, Funded study or Work experience", "why_it_fits": "...", "what_you_meet": "...", "what_to_check": "..."}],
  "next_steps": ["...", "...", "..."]
}"""


# ---------- AI ----------
def parse_ai_json(text):
    """Turns the AI's reply into a Python dictionary, even if it adds code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def make_config(low_thinking):
    """Settings for the AI call. Low thinking makes Gemini 3 models answer faster."""
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(
            thinking_level="LOW") if low_thinking else None,
    )


def build_pathway(profile):
    """Sends the student's profile, the vetted programmes and the skills to Gemini."""
    programmes_for_ai = [
        {k: p.get(k, "") for k in AI_PROGRAMME_FIELDS} for p in programmes]
    skills_for_ai = [{k: s.get(k, "") for k in AI_SKILL_FIELDS}
                     for s in skills_data]
    compact = {"separators": (",", ":"), "ensure_ascii": False}
    request = (
        "STUDENT PROFILE:\n" + json.dumps(profile, **compact)
        + "\n\nVETTED PROGRAMMES:\n" + json.dumps(programmes_for_ai, **compact)
        + "\n\nEMPLOYER-DEMAND SKILLS:\n" +
        json.dumps(skills_for_ai, **compact)
        + "\n\nRespond with JSON in exactly this structure. Choose 3 to 5 programmes:\n"
        + OUTPUT_FORMAT
    )

    last_error = None
    for model in [MODEL_NAME] + FALLBACK_MODELS:
        low_thinking = True
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=request,
                    config=make_config(low_thinking),
                )
                return parse_ai_json(response.text)
            except Exception as error:
                last_error = error
                if "thinking" in str(error).lower():
                    # This model doesn't accept the thinking setting, so try again without it
                    low_thinking = False
                else:
                    time.sleep(3 * (attempt + 1))
    raise last_error


# ---------- Dates and reminders ----------
def sa_today():
    """Today's date in South Africa (UTC+2, no daylight saving)."""
    return datetime.now(timezone(timedelta(hours=2))).date()


def nice_date(iso_date):
    """Turns 2026-10-31 into '31 October 2026'."""
    return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d %B %Y").lstrip("0")


def to_date(iso_date):
    return datetime.strptime(iso_date, "%Y-%m-%d").date() if iso_date else None


def days_left(p):
    """Days until a REAL deadline. Returns None if the date is a sample or missing."""
    if p.get("dates_are_sample") or not p.get("demo_deadline"):
        return None
    try:
        deadline = datetime.strptime(p["demo_deadline"], "%Y-%m-%d").date()
    except ValueError:
        return None
    return (deadline - sa_today()).days


def show_deadline(p):
    """Shows a clear deadline line on a programme card (real dates only)."""
    days = days_left(p)
    if days is None:
        return
    when = nice_date(p["demo_deadline"])
    if days < 0:
        st.info(
            f"This deadline passed on {when}. Check the official page for the next intake.")
    elif days == 0:
        st.warning(f"**Closes today ({when}).**")
    elif days <= 14:
        st.warning(
            f"**Closes {when}: {days} day{'s' if days != 1 else ''} left.**")
    else:
        st.caption(f"Closes {when} ({days} days left).")


def current_demo_date():
    """The date the demo slider is set to (today if it hasn't been used yet)."""
    value = st.session_state.get("demo_date")
    return value if value else sa_today()


def reminder_status(p, demo_date, applied, snoozed_until):
    """Works out which notification a saved programme should show on the demo date.
    Returns (state, days_until_deadline)."""
    opens = to_date(p.get("demo_opens"))
    deadline = to_date(p.get("demo_deadline"))
    if not opens or not deadline:
        return "always_open", None
    days = (deadline - demo_date).days
    if demo_date < opens:
        return "not_open", days
    if days < 0:
        return ("applied" if applied else "closed"), days
    if applied:
        return "applied", days
    if days <= 10:
        if snoozed_until and demo_date < snoozed_until:
            return "snoozed", days
        return "ask", days
    return "open", days


def status_label(state, days):
    """A short status for the Saved tab."""
    if state == "always_open":
        return "Open all year"
    if state == "not_open":
        return "Not open yet"
    if state == "open":
        return f"Open. Closes in {days} days"
    if state in ("ask", "snoozed"):
        return f"Closing soon: {days} day{'s' if days != 1 else ''} left"
    if state == "closed":
        return "Closed"
    return "Applied"


# ---------- Buttons that change what the student has saved ----------
def toggle_interest(pid):
    """Saves a programme to 'Saved', or removes it."""
    if pid in st.session_state["interested"]:
        st.session_state["interested"].remove(pid)
    else:
        st.session_state["interested"].append(pid)


def mark_applied(pid):
    """The student says they applied, so the reminders stop."""
    if pid not in st.session_state["applied"]:
        st.session_state["applied"].append(pid)


def unmark_applied(pid):
    """Undo, in case 'applied' was tapped by mistake."""
    if pid in st.session_state["applied"]:
        st.session_state["applied"].remove(pid)


def snooze_reminder(pid):
    """The student says 'not yet', so we ask again in 3 days."""
    st.session_state["snoozed"][pid] = current_demo_date() + timedelta(days=3)


def remove_saved(pid):
    if pid in st.session_state["interested"]:
        st.session_state["interested"].remove(pid)


# ---------- Remembering answers when the student changes screen ----------
def keep(extra_keys=()):
    """Remembers widget values so they survive a change of screen."""
    wanted = set(ANSWER_KEYS) | set(extra_keys)
    for key in list(st.session_state.keys()):
        if key in wanted or key.startswith("proof_"):
            st.session_state["_saved_" + key] = st.session_state[key]


def restore():
    """Puts remembered values back before the widgets are drawn."""
    for key in list(st.session_state.keys()):
        if key.startswith("_saved_"):
            real_key = key[len("_saved_"):]
            if real_key not in st.session_state:
                st.session_state[real_key] = st.session_state[key]


# ---------- Demo persona ----------
def load_demo_persona():
    """Fills the form with Thando, the demo learner from Mandeni."""
    st.session_state["age"] = "18-24"
    st.session_state["stage"] = "Grade 11 or 12"
    st.session_state["province"] = "KwaZulu-Natal"
    st.session_state["maths"] = "Maths Literacy"
    st.session_state["interests"] = ["I'm not sure yet"]
    st.session_state["tech_skills"] = "Basic computer skills"
    st.session_state["soft_skills"] = [
        "Leadership and social influence",
        "Empathy and active listening",
        "Curiosity and lifelong learning",
    ]
    st.session_state["proof_Leadership and social influence"] = (
        "A leadership role or volunteering (school, club, church, community)"
    )
    st.session_state["proof_Empathy and active listening"] = "Not yet"
    st.session_state["proof_Curiosity and lifelong learning"] = "Not yet"
    st.session_state["devices"] = ["Smartphone"]
    st.session_state["data_access"] = "Limited data"
    st.session_state["mobility"] = "I can study full-time but cannot move"
    st.session_state["cost"] = "Free only"


# ---------- Showing the pathway ----------
def render_pathway(pathway):
    """Shows the AI's pathway. Programme details come from our own data, not the AI."""
    st.header("Your pathway")
    st.write(pathway.get("summary", ""))

    direction = pathway.get("direction") or {}
    if direction.get("title"):
        st.subheader("Where this could take you")
        st.markdown(f"**{direction['title']}**")
        st.write(direction.get("why", ""))

    st.subheader("Your skills")
    col_have, col_build = st.columns(2)
    with col_have:
        st.markdown("**You already have**")
        for item in pathway.get("skills_you_have", []):
            st.markdown(
                f"- **{item.get('skill', '')}**: {item.get('note', '')}")
    with col_build:
        st.markdown("**Skills to build**")
        for item in pathway.get("skills_to_build", []):
            st.markdown(
                f"- **{item.get('skill', '')}**: {item.get('why', '')} "
                f"*How to start:* {item.get('how', '')}"
            )

    soft = pathway.get("soft_skills", [])
    if soft:
        st.subheader("Your soft skills and proof")
        st.caption(
            "Proof is self-reported in this demo. Verification is on the roadmap.")
        for item in soft:
            st.markdown(
                f"- **{item.get('skill', '')}** ({item.get('proof_status', '')}): "
                f"{item.get('advice', '')}"
            )

    st.subheader("Programmes for you")
    matched = [(item, programmes_by_id.get(item.get("id")))
               for item in pathway.get("programmes", [])]

    closing_soon = []
    for item, p in matched:
        if p:
            days = days_left(p)
            if days is not None and 0 <= days <= 14:
                closing_soon.append(
                    f"{p['name']} ({days} day{'s' if days != 1 else ''} left)")
    if closing_soon:
        st.warning("**Closing soon:** " + "; ".join(closing_soon))

    dropped = 0
    for item, p in matched:
        if not p:
            dropped += 1
            continue
        with st.container(border=True):
            st.markdown(f"**{p['name']}**")
            st.caption(
                f"{p['provider']} · {p['type']} · {item.get('route', '')}")
            show_deadline(p)
            st.write(item.get("why_it_fits", ""))
            if item.get("what_you_meet"):
                st.markdown(
                    f"**What you already meet:** {item['what_you_meet']}")
            if item.get("what_to_check"):
                st.markdown(f"**What to check:** {item['what_to_check']}")
            if p.get("timing"):
                st.markdown(f"**Timing:** {p['timing']}")
            if p.get("cost") == "free":
                st.markdown("**Cost to you:** free")
            if p.get("official_site_confirmed"):
                st.caption("Details taken from the programme's official page.")
            else:
                st.caption(
                    "Details from a government source. Confirm on the official page.")
            if p.get("link"):
                st.link_button("Open official page", p["link"])
            saved = p["id"] in st.session_state["interested"]
            st.button(
                "Remove from saved" if saved else "Save this programme",
                key=f"interest_{p['id']}",
                on_click=toggle_interest,
                args=(p["id"],),
            )
    if dropped:
        st.caption(
            f"{dropped} suggestion(s) were removed because they weren't in our vetted list.")

    steps = pathway.get("next_steps", [])
    if steps:
        st.subheader("Your next steps")
        for i, step in enumerate(steps, start=1):
            st.markdown(f"{i}. {step}")

    st.caption(
        "Always confirm dates and requirements on the official page before you apply.")


# ---------- Screen 1: the questions ----------
def show_questions():
    restore()
    st.caption("Step 1 of 3: Your answers")
    st.write("Answer a few quick questions and get a personalised career pathway.")
    st.caption(
        "Demo only: please don't enter real personal details. "
        "We don't ask for your name, ID number or contact details."
    )
    st.caption(
        f"{len(programmes)} vetted programmes and {len(skills_data)} skills loaded.")

    if st.session_state.get("pathway"):
        if st.button("Skip to my pathway →", key="skip_to_pathway"):
            st.switch_page(PAGES["pathway"])

    st.button("Fill with demo learner (Thando)", on_click=load_demo_persona)

    st.subheader("About you")
    st.selectbox("1. Which age group are you in?", AGE_BANDS, key="age")
    if st.session_state.get("age") == "Under 18":
        st.info(
            "Some programmes need a parent or guardian's consent, "
            "and some have a minimum age."
        )
    st.selectbox("2. Where are you right now?", STAGES, key="stage")
    st.selectbox("3. Which province are you in?", PROVINCES, key="province")
    st.selectbox("4. What is your matric maths?", MATHS, key="maths")

    st.subheader("Your interests and skills")
    st.multiselect("5. What interests you? (pick any)",
                   INTERESTS, key="interests")
    st.radio("6. What tech skills do you have so far?",
             TECH_SKILLS, key="tech_skills")
    st.multiselect(
        "7. Which of these soft skills do you have? (pick any)",
        SOFT_SKILLS,
        key="soft_skills",
    )

    selected_soft_skills = st.session_state.get("soft_skills", [])
    if selected_soft_skills:
        st.markdown("**8. Do you have proof for these skills?**")
        st.caption(
            "'Not yet' is a fine answer. We'll help you find ways to earn proof.")
        for skill in selected_soft_skills:
            st.selectbox(f"Proof for: {skill}",
                         PROOF_OPTIONS, key=f"proof_{skill}")

    st.subheader("Your situation")
    st.multiselect("9a. What do you have access to?", DEVICES, key="devices")
    st.selectbox("9b. How do you get online?", DATA_ACCESS, key="data_access")
    st.selectbox("10a. Could you study full-time or move?",
                 MOBILITY, key="mobility")
    st.selectbox("10b. Do you need free or funded options?", COST, key="cost")

    # Remember everything typed so far, in case the student changes screen
    keep()

    st.caption("Your answers are sent to an AI service to build your pathway.")

    if st.button("Build my pathway", type="primary"):
        required = {
            "age": "age group",
            "stage": "where you are right now",
            "province": "province",
            "maths": "matric maths",
            "data_access": "how you get online",
            "mobility": "study and moving",
            "cost": "free or funded options",
        }
        missing = [
            label
            for key, label in required.items()
            if st.session_state.get(key) in (None, PLACEHOLDER)
        ]
        if not st.session_state.get("interests"):
            missing.append("interests")
        if not st.session_state.get("devices"):
            missing.append("what you have access to")

        if missing:
            st.warning("Please answer: " + ", ".join(missing) + ".")
        else:
            profile = {
                "age_band": st.session_state["age"],
                "stage": st.session_state["stage"],
                "province": st.session_state["province"],
                "matric_maths": st.session_state["maths"],
                "interests": st.session_state["interests"],
                "tech_skills": st.session_state["tech_skills"],
                "soft_skills": [
                    {"skill": s, "proof": st.session_state.get(
                        f"proof_{s}", "Not yet")}
                    for s in selected_soft_skills
                ],
                "devices": st.session_state["devices"],
                "data_access": st.session_state["data_access"],
                "mobility": st.session_state["mobility"],
                "cost_preference": st.session_state["cost"],
            }
            st.session_state["profile"] = profile
            st.session_state.pop("pathway", None)
            st.session_state.pop("pathway_error", None)
            st.session_state.pop("pathway_is_saved", None)
            with st.spinner("Matching you with verified programmes. This usually takes 30 to 60 seconds..."):
                try:
                    st.session_state["pathway"] = build_pathway(profile)
                except Exception as error:
                    st.session_state["pathway_error"] = str(error)
            if st.session_state.get("pathway"):
                st.switch_page(PAGES["pathway"])

    if st.session_state.get("pathway_error"):
        st.error(
            "We couldn't build your pathway just now. Please try again in a moment.")
        with st.expander("Technical details"):
            st.write(st.session_state["pathway_error"])
        if os.path.exists("demo_pathway.json"):
            if st.button("Show a saved example instead"):
                with open("demo_pathway.json", encoding="utf-8") as f:
                    st.session_state["pathway"] = json.load(f)
                st.session_state["pathway_is_saved"] = True
                st.session_state.pop("pathway_error", None)
                st.switch_page(PAGES["pathway"])


# ---------- Screen 2: the pathway ----------
def show_pathway():
    st.caption("Step 2 of 3: Your pathway")
    pathway = st.session_state.get("pathway")
    if not pathway:
        st.info("You don't have a pathway yet. Answer the questions first.")
        if st.button("← Back to my answers", key="p_back_none"):
            st.switch_page(PAGES["questions"])
        return

    col_back, col_go = st.columns(2)
    if col_back.button("← Back to my answers", key="p_back_top"):
        st.switch_page(PAGES["questions"])
    if col_go.button("Go to my app →", key="p_go_top", type="primary"):
        st.switch_page(PAGES["app"])

    if st.session_state.get("pathway_is_saved"):
        st.info("This is a saved example for our demo learner, not a live result.")
    render_pathway(pathway)

    st.divider()
    st.write(
        "Happy with your pathway? Save the programmes you like, then open your app.")
    col_back2, col_go2 = st.columns(2)
    if col_back2.button("← Back to my answers", key="p_back_bottom"):
        st.switch_page(PAGES["questions"])
    if col_go2.button("Go to my app →", key="p_go_bottom", type="primary"):
        st.switch_page(PAGES["app"])

    if not st.session_state.get("pathway_is_saved"):
        with st.expander("Team tools"):
            st.caption("Save a good result as the demo fallback.")
            st.download_button(
                "Download this pathway as demo_pathway.json",
                data=json.dumps(pathway, indent=2, ensure_ascii=False),
                file_name="demo_pathway.json",
                mime="application/json",
            )
    if "profile" in st.session_state:
        with st.expander("See your profile"):
            st.json(st.session_state["profile"])


# ---------- Screen 3: the app ----------
def show_app():
    restore()
    today = sa_today()
    if st.session_state.get("demo_date") and st.session_state["demo_date"] < today:
        st.session_state["demo_date"] = today

    st.caption("Step 3 of 3: Your app")
    slider_options = {
        "min_value": today,
        "max_value": date(2027, 8, 31),
        "format": "D MMM YYYY",
        "key": "demo_date",
    }
    if "demo_date" not in st.session_state:
        slider_options["value"] = today
    demo_date = st.slider(
        "Demo date: move it forward to see how reminders work", **slider_options)
    keep(["demo_date"])

    saved_ids = [pid for pid in st.session_state["interested"]
                 if pid in programmes_by_id]
    applied_ids = [pid for pid in st.session_state["applied"]
                   if pid in programmes_by_id]
    waiting_ids = [pid for pid in saved_ids if pid not in applied_ids]

    tab_pathway, tab_saved, tab_reminders, tab_applied = st.tabs([
        "My pathway",
        f"Saved ({len(waiting_ids)})",
        "Reminders",
        f"Applied ({len(applied_ids)})",
    ])

    with tab_pathway:
        pathway = st.session_state.get("pathway")
        if pathway:
            render_pathway(pathway)
        else:
            st.info("No pathway yet.")
        if st.button("Change my answers", key="app_change_answers"):
            st.switch_page(PAGES["questions"])

    with tab_saved:
        if not waiting_ids:
            st.caption(
                "Nothing saved yet. Save programmes from your pathway and they'll appear here.")
        for pid in waiting_ids:
            p = programmes_by_id[pid]
            state, days = reminder_status(
                p, demo_date, False, st.session_state["snoozed"].get(pid)
            )
            with st.container(border=True):
                st.markdown(f"**{p['name']}**")
                st.caption(f"{p['provider']} · {p['type']}")
                st.write(status_label(state, days))
                col_a, col_b = st.columns(2)
                col_a.button(
                    "I've applied", key=f"applied_{pid}", on_click=mark_applied, args=(pid,))
                col_b.button(
                    "Remove", key=f"remove_{pid}", on_click=remove_saved, args=(pid,))
                if p.get("link"):
                    st.markdown(f"[Open official page]({p['link']})")

    with tab_reminders:
        st.caption(
            f"Reminders as they would appear on {nice_date(demo_date.isoformat())}.")
        shown = 0
        for pid in waiting_ids:
            p = programmes_by_id[pid]
            snoozed_until = st.session_state["snoozed"].get(pid)
            state, days = reminder_status(p, demo_date, False, snoozed_until)
            if state == "always_open":
                continue
            shown += 1
            with st.container(border=True):
                st.markdown(f"**{p['name']}**")
                if p.get("dates_are_sample"):
                    st.caption(
                        "Sample dates for the demo. Real dates are on the official page.")
                opens = nice_date(p["demo_opens"])
                closes = nice_date(p["demo_deadline"])
                if state == "not_open":
                    st.info(
                        f"Not open yet. We'll tell you when applications open (expected {opens}).")
                elif state == "open":
                    st.success(
                        f"Applications are now open! They close on {closes}.")
                elif state == "ask":
                    st.warning(
                        f"**{days} day{'s' if days != 1 else ''} left** to apply "
                        f"(closes {closes}). Have you applied?"
                    )
                    col_yes, col_no = st.columns(2)
                    col_yes.button(
                        "Yes, I applied", key=f"yes_{pid}", on_click=mark_applied, args=(pid,)
                    )
                    col_no.button(
                        "Not yet", key=f"no_{pid}", on_click=snooze_reminder, args=(pid,)
                    )
                elif state == "snoozed":
                    again = snoozed_until.strftime("%d %B %Y").lstrip("0")
                    st.info(
                        f"{days} day{'s' if days != 1 else ''} left. "
                        f"You said not yet, so we'll remind you again on {again}."
                    )
                elif state == "closed":
                    st.info(
                        f"This deadline passed on {closes}. Check the official page for the next intake.")
                if p.get("link"):
                    st.markdown(f"[Open official page]({p['link']})")
        if not shown:
            st.caption(
                "No reminders right now. Save programmes with deadlines to get reminders.")

    with tab_applied:
        if not applied_ids:
            st.caption("Programmes you've applied for will appear here.")
        for pid in applied_ids:
            p = programmes_by_id[pid]
            with st.container(border=True):
                st.markdown(f"**{p['name']}**")
                st.caption(f"{p['provider']} · {p['type']}")
                st.success("You told us you've applied. Good luck!")
                st.button("Undo", key=f"undo_{pid}",
                          on_click=unmark_applied, args=(pid,))
                if p.get("link"):
                    st.markdown(f"[Open official page]({p['link']})")


# ---------- Pages ----------
# Each step is its own Streamlit page. Switching page opens the new step at the top.
PAGES = {
    "questions": st.Page(show_questions, title="Your answers", default=True),
    "pathway": st.Page(show_pathway, title="Your pathway", url_path="pathway"),
    "app": st.Page(show_app, title="Your app", url_path="app"),
}

st.title("THUTHUKA-AI")

try:
    navigation = st.navigation(list(PAGES.values()), position="hidden")
except Exception:
    navigation = st.navigation(list(PAGES.values()))
navigation.run()
