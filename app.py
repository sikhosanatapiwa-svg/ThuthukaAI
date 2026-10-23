import json
import streamlit as st
from google import genai

MODEL_NAME = "gemini-3.8-flash"
PLACEHOLDER = "Select..."

st.set_page_config(page_title="THUTHUKA-AI", page_icon="🧭")

# Load the vetted programme list (used in the next step to build the pathway)
with open("programmes.json", encoding="utf-8") as f:
    programmes = json.load(f)

# Load the employer-demand skills (used for the soft-skills question and, next, the pathway)
with open("skills_data.json", encoding="utf-8") as f:
    skills_data = json.load(f)

# Connection to Gemini (used in the next step)
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
        st.session_state["profile"] = {
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

# ---------- Result ----------
if "profile" in st.session_state:
    st.success(
        "Your profile is ready. The pathway is built from this in the next step.")
    with st.expander("See your profile"):
        st.json(st.session_state["profile"])
