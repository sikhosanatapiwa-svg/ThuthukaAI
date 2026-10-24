import json
import os
import time
from datetime import datetime, timedelta, timezone
import streamlit as st
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.8-flash"
# Backup models, tried in order if the main one is busy.
# Copy the exact names from the model list in Google AI Studio. Names that don't exist are skipped.
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
PLACEHOLDER = "Select..."

st.set_page_config(page_title="THUTHUKA-AI", page_icon="🧭")

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


def sa_today():
    """Today's date in South Africa (UTC+2, no daylight saving)."""
    return datetime.now(timezone(timedelta(hours=2))).date()


def nice_date(iso_date):
    """Turns 2026-10-31 into '31 October 2026'."""
    return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d %B %Y").lstrip("0")


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


def render_pathway(pathway):
    """Shows the AI's pathway on the page. Programme details come from our own data."""
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


# ---------- Page ----------
st.title("THUTHUKA-AI")
st.write("Answer a few quick questions and get a personalised career pathway.")
st.caption(
    "Demo only: please don't enter real personal details. "
    "We don't ask for your name, ID number or contact details."
)
st.caption(
    f"{len(programmes)} vetted programmes and {len(skills_data)} skills loaded.")

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
st.multiselect("5. What interests you? (pick any)", INTERESTS, key="interests")
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

st.caption("Your answers are sent to an AI service to build your pathway.")

# ---------- Submit ----------
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
        with st.spinner("Building your pathway..."):
            try:
                st.session_state["pathway"] = build_pathway(profile)
            except Exception as error:
                st.session_state["pathway_error"] = str(error)

# ---------- Result ----------
if st.session_state.get("pathway_error"):
    st.error("We couldn't build your pathway just now. Please try again in a moment.")
    with st.expander("Technical details"):
        st.write(st.session_state["pathway_error"])
    if os.path.exists("demo_pathway.json"):
        if st.button("Show a saved example instead"):
            with open("demo_pathway.json", encoding="utf-8") as f:
                st.session_state["pathway"] = json.load(f)
            st.session_state["pathway_is_saved"] = True
            st.session_state.pop("pathway_error", None)
            st.rerun()

if st.session_state.get("pathway"):
    if st.session_state.get("pathway_is_saved"):
        st.info("This is a saved example for our demo learner, not a live result.")
    render_pathway(st.session_state["pathway"])
    if not st.session_state.get("pathway_is_saved"):
        with st.expander("Team tools"):
            st.caption("Save a good result as the demo fallback.")
            st.download_button(
                "Download this pathway as demo_pathway.json",
                data=json.dumps(
                    st.session_state["pathway"], indent=2, ensure_ascii=False),
                file_name="demo_pathway.json",
                mime="application/json",
            )

if "profile" in st.session_state:
    with st.expander("See your profile"):
        st.json(st.session_state["profile"])
