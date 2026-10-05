# -*- coding: utf-8 -*-
"""
NAVIGATE
Customer Retention Intelligence
"""

import streamlit as st
import pandas as pd
import joblib
import html
import textwrap
import matplotlib.pyplot as plt


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="NAVIGATE | Customer Retention",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "page" not in st.session_state:
    st.session_state.page = "home"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def render_html(content):
    st.html(textwrap.dedent(content))


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("churn_model.pkl")
impact_scaler = joblib.load("impact_scaler.pkl")
priority_thresholds = joblib.load("priority_thresholds.pkl")


# ============================================================
# FUNCTIONS
# ============================================================

def get_risk_level(score):
    if score < 30:
        return "Low"
    if score < 50:
        return "Medium"
    if score < 75:
        return "High"
    return "Critical"


def get_priority_level(score):
    if score < priority_thresholds["priority_25"]:
        return "Low"
    if score < priority_thresholds["priority_50"]:
        return "Medium"
    if score < priority_thresholds["priority_75"]:
        return "High"
    return "Critical"


def get_badge_class(level):
    return {
        "Low": "low",
        "Medium": "medium",
        "High": "high",
        "Critical": "critical"
    }.get(level, "low")


def get_risk_explanation(level, score):
    """Explain what the model-estimated churn level means."""
    if level == "Critical":
        return (
            f"A churn probability of {score:.0f}% places this customer in the critical-risk range. "
            "The customer requires immediate retention attention because the model estimates a very high likelihood of churn."
        )
    if level == "High":
        return (
            f"A churn probability of {score:.0f}% places this customer in the high-risk range. "
            "The customer shows a strong likelihood of churn and should be reviewed proactively."
        )
    if level == "Medium":
        return (
            f"A churn probability of {score:.0f}% places this customer in the medium-risk range. "
            "The customer is not currently critical, but the account should be monitored for increasing risk."
        )
    return (
        f"A churn probability of {score:.0f}% places this customer in the low-risk range. "
        "The model currently estimates a relatively low likelihood of churn."
    )


def get_decision_summary(priority_level, risk_level, churn_risk):
    if priority_level == "Critical":
        return (
            "Immediate retention attention recommended",
            f"This customer has {risk_level.lower()} churn risk ({churn_risk:.0f}%) and falls in the highest "
            "retention-priority group. Review the strongest customer signals and begin targeted retention action promptly."
        )

    if priority_level == "High":
        return (
            "High retention priority",
            f"This customer has {risk_level.lower()} churn risk ({churn_risk:.0f}%) and should receive proactive "
            "retention attention. Address the most relevant account signals before risk increases."
        )

    if priority_level == "Medium":
        return (
            "Moderate retention priority",
            f"This customer currently has {risk_level.lower()} churn risk ({churn_risk:.0f}%). "
            "Monitor the account and use targeted engagement where appropriate."
        )

    return (
        "Low retention priority",
        f"This customer currently has {risk_level.lower()} churn risk ({churn_risk:.0f}%). "
        "Maintain normal engagement and continue monitoring for meaningful changes."
    )


def get_factor_action_pairs(
    tenure,
    contract,
    payment_method,
    paperless_billing,
    monthly_charges,
    total_charges,
    churn_risk,
    business_impact,
):
    """
    Return reviewable customer signals and a practical action for each signal.
    These are rule-based decision-support recommendations, not causal claims.
    """
    pairs = []

    if contract == "Month-to-month":
        pairs.append((
            "Month-to-month contract",
            "The customer has a flexible contract and can leave without a long commitment.",
            "Consider offering a suitable incentive for a longer-term contract."
        ))

    if payment_method == "Electronic check":
        pairs.append((
            "Electronic check payment",
            "The customer uses a manual electronic payment method.",
            "Offer an automatic payment option if it is suitable for the customer."
        ))

    if tenure < 12:
        pairs.append((
            "Short customer tenure",
            f"The customer has been with the company for only {tenure} month(s).",
            "Strengthen onboarding, early engagement, and first-year retention support."
        ))
    elif tenure < 24:
        pairs.append((
            "Developing customer relationship",
            f"The customer has been with the company for {tenure} months.",
            "Use a proactive check-in to strengthen the customer relationship."
        ))

    # Monthly charges are used carefully: we describe the entered amount,
    # without claiming it caused churn.
    if monthly_charges >= 80:
        pairs.append((
            "High monthly payment amount",
            f"The customer currently pays ${monthly_charges:,.2f} per month.",
            "Review the current plan, service fit, and any eligible retention offers."
        ))
    elif monthly_charges >= 60:
        pairs.append((
            "Moderate-to-high monthly payment amount",
            f"The customer currently pays ${monthly_charges:,.2f} per month.",
            "Confirm that the current plan still matches the customer's needs and perceived value."
        ))

    if paperless_billing == "Yes":
        pairs.append((
            "Paperless billing",
            "The customer receives bills digitally.",
            "Use digital channels for timely, personalized retention communication."
        ))

    if business_impact >= 75:
        pairs.append((
            "High relative customer value",
            f"The customer's relative business-impact score is {business_impact:.0f}/100.",
            "Prioritize a personalized response that reflects the customer's relative value."
        ))
    elif business_impact >= 50:
        pairs.append((
            "Meaningful relative customer value",
            f"The customer's relative business-impact score is {business_impact:.0f}/100.",
            "Consider the customer's value when selecting the retention response."
        ))

    if churn_risk >= 75:
        pairs.insert(0, (
            "Critical model-estimated churn risk",
            f"The model estimates a {churn_risk:.0f}% probability of churn.",
            "Begin retention outreach promptly and review the account before other lower-risk cases."
        ))
    elif churn_risk >= 50:
        pairs.insert(0, (
            "Elevated model-estimated churn risk",
            f"The model estimates a {churn_risk:.0f}% probability of churn.",
            "Proactively review the account and contact the customer when appropriate."
        ))
    elif churn_risk >= 30:
        pairs.insert(0, (
            "Moderate model-estimated churn risk",
            f"The model estimates a {churn_risk:.0f}% probability of churn.",
            "Monitor the account and watch for additional risk signals."
        ))

    if not pairs:
        pairs.append((
            "No major rule-based warning signal",
            "The entered customer profile does not trigger the main review rules used by this prototype.",
            "Maintain normal engagement and continue monitoring the model-estimated churn risk."
        ))

    return pairs


def get_recommended_actions(factor_action_pairs, priority_level):
    """Create a larger, ordered action plan from the customer-specific signals."""
    actions = []

    if priority_level == "Critical":
        actions.extend([
            ("Immediate", "Contact the customer promptly for a retention-focused conversation."),
            ("Immediate", "Review recent interactions, complaints, or service concerns before outreach."),
        ])
    elif priority_level == "High":
        actions.extend([
            ("Immediate", "Proactively contact the customer and review current needs or concerns."),
        ])
    elif priority_level == "Medium":
        actions.extend([
            ("Priority", "Schedule a proactive customer check-in and continue monitoring risk."),
        ])
    else:
        actions.extend([
            ("Maintain", "Maintain regular engagement and continue monitoring retention risk."),
        ])

    # Add customer-specific actions.
    for _, _, action in factor_action_pairs:
        if action not in [a[1] for a in actions]:
            actions.append(("Targeted", action))

    # Add follow-up for customers needing attention.
    if priority_level in ("Critical", "High"):
        actions.append((
            "Follow-up",
            "Reassess the customer after the retention action to determine whether further intervention is needed."
        ))
    elif priority_level == "Medium":
        actions.append((
            "Follow-up",
            "Reassess the account if the customer's profile or model-estimated risk changes."
        ))

    # Remove accidental duplicates while preserving order.
    unique = []
    seen = set()
    for label, action in actions:
        if action not in seen:
            unique.append((label, action))
            seen.add(action)

    return unique


def get_next_step(priority_level, actions):
    if not actions:
        return "Continue monitoring the customer."

    first_action = actions[0][1]

    if priority_level in ("Critical", "High"):
        return f"Next step: {first_action}"
    if priority_level == "Medium":
        return f"Next step: {first_action}"
    return "Next step: Maintain normal engagement and monitor for meaningful changes."


def get_result_interpretation(risk_level, churn_risk, factor_action_pairs):
    """Create a concise executive interpretation without claiming causality."""
    factor_names = [
        factor for factor, _, _ in factor_action_pairs
        if "model-estimated churn risk" not in factor.lower()
        and "customer value" not in factor.lower()
        and "paperless billing" not in factor.lower()
    ]
    shown = factor_names[:2]

    if shown:
        signal_text = " Review signals include " + " and ".join(shown) + "."
    else:
        signal_text = " No additional major rule-based warning signal was triggered by the entered profile."

    if risk_level == "Critical":
        result = f"Critical Churn Risk — {churn_risk:.0f}%"
        why = (
            "The model estimates a very high likelihood of churn, so this account requires prompt retention attention."
            + signal_text
        )
        action = "Prioritize retention outreach, review the customer's current needs, and apply the most relevant targeted retention actions."
    elif risk_level == "High":
        result = f"High Churn Risk — {churn_risk:.0f}%"
        why = (
            "The model estimates an elevated likelihood of churn, making proactive review important before the risk increases."
            + signal_text
        )
        action = "Contact the customer proactively, review the strongest account signals, and select a targeted retention response."
    elif risk_level == "Medium":
        result = f"Medium Churn Risk — {churn_risk:.0f}%"
        why = (
            "The customer is not in the highest-risk range, but the model indicates enough churn risk to justify monitoring and targeted engagement."
            + signal_text
        )
        action = "Monitor the account, use a proactive check-in where appropriate, and reassess if the customer profile changes."
    else:
        result = f"Low Churn Risk — {churn_risk:.0f}%"
        why = (
            "The model currently estimates a relatively low likelihood of churn, so urgent retention intervention is not indicated."
            + signal_text
        )
        action = "Maintain normal engagement and continue monitoring for meaningful changes."

    return result, why, action


# ============================================================
# CSS
# ============================================================

render_html("""
<style>

/* ---------- APP ---------- */

.stApp {
    background:
        radial-gradient(circle at 8% 12%,
            rgba(255,255,255,.95) 0%,
            rgba(255,255,255,0) 28%),

        radial-gradient(circle at 88% 18%,
            rgba(235,122,148,.22) 0%,
            rgba(235,122,148,0) 34%),

        radial-gradient(circle at 15% 85%,
            rgba(229,111,139,.20) 0%,
            rgba(229,111,139,0) 32%),

        linear-gradient(
            135deg,
            #fffafb 0%,
            #fbeef1 42%,
            #f7dce3 100%
        );

    color: #701426;
    min-height: 100vh;
}

.block-container {
    max-width: 1450px;
    padding: 1.7rem 3.2rem 4rem;
}

#MainMenu,
footer,
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

div[data-testid="stToolbar"] {
    visibility: hidden;
}

/* Remove number-input helper text such as
   "Press Enter to submit form" */

[data-testid="InputInstructions"] {
    display: none !important;
}


/* ---------- TOPBAR ---------- */

.topbar {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding: 7px 0 20px;
    margin-bottom: 28px;
    border-bottom:1px solid rgba(133,37,55,.13);
}

.brand-wrap {
    display:flex;
    align-items:center;
    gap:11px;
}

.brand-logo {
    width:40px;
    height:40px;
    border-radius:12px;

    background:
        linear-gradient(
            145deg,
            #7b1025,
            #c82f50 55%,
            #ed8299
        );

    display:flex;
    align-items:center;
    justify-content:center;

    color:white;
    font-size:20px;
    font-weight:900;

    box-shadow:
        0 9px 22px rgba(134,25,48,.18);
}

.brand-name {
    color:#741426;
    font-weight:850;
    letter-spacing:.17em;
    font-size:.88rem;
}

.brand-sub {
    color:#a0747d;
    font-size:.61rem;
    margin-top:1px;
}

.topbar-right {
    color:#a06470;
    font-size:.65rem;
    font-weight:750;
    letter-spacing:.13em;
    text-transform:uppercase;
}


/* ---------- PROGRESS ---------- */

.steps {
    display:flex;
    justify-content:center;
    align-items:center;
    gap:13px;
    margin: 3px auto 33px;
}

.step {
    display:flex;
    align-items:center;
    gap:8px;
    color:#a7888e;
    font-size:.68rem;
    font-weight:650;
}

.step-circle {
    width:26px;
    height:26px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    border:1px solid #e6cbd1;
    background:rgba(255,255,255,.68);
    font-size:.65rem;
    font-weight:800;
}

.step.active {
    color:#8d1730;
}

.step.active .step-circle {
    color:#fff;
    background:linear-gradient(135deg,#a31836,#dc4a68);
    border-color:#c52d4c;
    box-shadow:0 5px 13px rgba(171,31,61,.20);
}

.step-line {
    width:65px;
    height:1px;
    background:#e4c9cf;
}


/* ---------- HERO ---------- */

.hero-grid {
    display:grid;
    grid-template-columns: 1.02fr .98fr;
    align-items:center;
    gap:45px;
    min-height:490px;
}

.hero-copy {
    padding:35px 0 45px;
}

.eyebrow {
    color:#b44359;
    font-size:.70rem;
    font-weight:850;
    letter-spacing:.19em;
    text-transform:uppercase;
    margin-bottom:19px;
}

.hero-title {
    color:#741426;
    font-size:clamp(3rem,5.3vw,5.3rem);
    font-weight:850;
    line-height:.96;
    letter-spacing:-.06em;
    margin-bottom:24px;
}

.hero-title span {
    background:linear-gradient(90deg,#b51f42,#e25c78);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero-description {
    color:#88666d;
    max-width:610px;
    font-size:1rem;
    line-height:1.75;
}


/* ---------- HERO VISUAL ---------- */

.hero-visual {
    position:relative;
    min-height:420px;
}

.orb {
    position:absolute;
    border-radius:50%;
    filter:blur(1px);
}

@keyframes floatOrbOne {
    0%,100% { transform:translate3d(0,0,0) scale(1); }
    35% { transform:translate3d(-18px,14px,0) scale(1.035); }
    70% { transform:translate3d(12px,-12px,0) scale(.985); }
}

@keyframes floatOrbTwo {
    0%,100% { transform:translate3d(0,0,0) scale(1); }
    40% { transform:translate3d(20px,-16px,0) scale(.97); }
    75% { transform:translate3d(-12px,12px,0) scale(1.04); }
}

.orb-one {
    width:330px;
    height:330px;
    right:25px;
    top:25px;
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.82),
            rgba(234,105,137,.28)
        );
    animation:floatOrbOne 9s ease-in-out infinite;
}

.orb-two {
    width:230px;
    height:230px;
    right:190px;
    top:145px;
    background:rgba(240,133,157,.17);
    animation:floatOrbTwo 11s ease-in-out infinite;
}

.visual-card {
    position:absolute;
    width:390px;
    right:40px;
    top:105px;

    padding:30px;

    border:1px solid rgba(255,255,255,.80);
    border-radius:27px;

    background:rgba(255,255,255,.52);
    backdrop-filter:blur(18px);

    box-shadow:
        0 28px 70px rgba(130,28,53,.15);

    transform:rotate(3deg);
}

.visual-mini {
    background:rgba(255,255,255,.70);
    border:1px solid rgba(229,186,196,.7);
    border-radius:17px;
    padding:17px;
    margin-top:16px;
}

.visual-row {
    display:flex;
    align-items:center;
    gap:13px;
}

.visual-icon {
    width:48px;
    height:48px;
    border-radius:15px;

    display:flex;
    align-items:center;
    justify-content:center;

    background:linear-gradient(145deg,#f9d9e1,#ffffff);
    color:#a51e3c;
    font-size:22px;
}

.fake-lines {
    flex:1;
}

.fake-line {
    height:9px;
    border-radius:99px;
    background:#e9aeba;
    margin:7px 0;
}

.fake-line.small {
    width:60%;
    background:#f0ccd3;
}


/* ---------- GLASS PANELS ---------- */

.glass-panel {
    background:rgba(255,255,255,.69);
    backdrop-filter:blur(18px);

    border:1px solid rgba(225,179,189,.58);
    border-radius:28px;

    box-shadow:
        0 22px 60px rgba(119,30,49,.08);

    padding:38px;
}


/* ---------- PAGE HEADER ---------- */

.page-kicker {
    color:#b14a5e;
    font-size:.66rem;
    font-weight:850;
    letter-spacing:.18em;
    text-transform:uppercase;
    margin-bottom:11px;
}

.page-title {
    color:#741426;
    font-size:2.35rem;
    font-weight:850;
    letter-spacing:-.045em;
    line-height:1.04;
}

.page-copy {
    color:#957078;
    font-size:.83rem;
    line-height:1.65;
    margin-top:10px;
    max-width:630px;
}


/* ---------- INFO BOX ---------- */

.quick-card {
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.82),
            rgba(255,236,241,.78)
        );

    border:1px solid #edcfd6;
    border-radius:18px;

    padding:18px 20px;

    display:flex;
    gap:15px;
    align-items:center;

    margin-top:28px;
}

.quick-icon {
    width:43px;
    height:43px;
    border-radius:13px;

    background:#f9d9e1;
    color:#a91e3e;

    display:flex;
    align-items:center;
    justify-content:center;

    font-size:20px;
}

.quick-title {
    color:#7c1a2e;
    font-size:.78rem;
    font-weight:800;
}

.quick-copy {
    color:#9c747c;
    font-size:.67rem;
    margin-top:3px;
}


/* ---------- FORM ---------- */

div[data-testid="stForm"] {
    background:rgba(255,255,255,.73);
    backdrop-filter:blur(20px);

    border:1px solid rgba(222,174,185,.62);
    border-radius:25px;

    padding:30px 32px 27px;

    box-shadow:
        0 18px 55px rgba(121,29,50,.07);
}

.form-heading {
    display:flex;
    align-items:center;
    gap:11px;
    color:#79172b;
    font-size:.9rem;
    font-weight:800;
    margin-bottom:23px;
}

.form-heading-icon {
    width:35px;
    height:35px;
    border-radius:11px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:#f9dce3;
    color:#b52747;
    font-size:17px;
}

label {
    color:#7a172b !important;
    font-size:.76rem !important;
    font-weight:760 !important;
}

div[data-baseweb="select"] > div {
    background:rgba(255,255,255,.90) !important;
    border:1px solid #e1bdc6 !important;
    border-radius:11px !important;
    min-height:46px;
    box-shadow:none !important;
}

div[data-testid="stNumberInput"] input {
    background:rgba(255,255,255,.90) !important;
    color:#77172b !important;
    min-height:46px;
}

div[data-testid="stNumberInput"] {
    border-radius:11px !important;
}

.field-help {
    color:#aa848b;
    font-size:.64rem;
    line-height:1.45;
    margin-top:-7px;
    margin-bottom:15px;
}

.required-note {
    display:flex;
    align-items:center;
    gap:8px;
    color:#9b737b;
    font-size:.67rem;
    margin-top:12px;
}


/* ---------- BUTTONS ---------- */

.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    border:none !important;
    border-radius:13px !important;

    min-height:49px;

    color:#fff !important;
    font-size:.80rem !important;
    font-weight:800 !important;

    background:
        linear-gradient(
            110deg,
            #8d1731 0%,
            #b72243 50%,
            #d83e5e 100%
        ) !important;

    box-shadow:
        0 10px 24px rgba(158,28,57,.20) !important;

    transition:
        transform .18s ease,
        box-shadow .18s ease !important;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    transform:translateY(-2px);
    box-shadow:
        0 14px 30px rgba(158,28,57,.28) !important;
}


/* ---------- ALERT ---------- */

div[data-testid="stAlert"] {
    background:rgba(255,239,243,.92) !important;
    border:1px solid #e6b9c3 !important;
    color:#7c1b30 !important;
    border-radius:14px !important;
}


/* ---------- RESULT CARDS ---------- */

.metric-grid-card {
    background:rgba(255,255,255,.76);
    backdrop-filter:blur(18px);

    border:1px solid rgba(224,180,190,.62);
    border-radius:20px;

    padding:25px;

    min-height:205px;

    box-shadow:
        0 15px 42px rgba(118,29,49,.07);
}

.metric-top {
    display:flex;
    justify-content:space-between;
    gap:15px;
}

.metric-heading {
    display:flex;
    gap:12px;
    align-items:center;
}

.metric-icon {
    width:44px;
    height:44px;
    border-radius:14px;

    background:
        linear-gradient(145deg,#f8d8e0,#fff);

    border:1px solid #f0cdd5;

    display:flex;
    align-items:center;
    justify-content:center;

    color:#ad2342;
    font-size:20px;
}

.metric-label {
    color:#79555c;
    font-size:.68rem;
    font-weight:750;
}

.metric-value {
    color:#78172b;
    font-size:2.45rem;
    font-weight:850;
    letter-spacing:-.05em;
    margin-top:12px;
}

.metric-unit {
    font-size:.72rem;
    color:#a27981;
    font-weight:600;
}

.metric-note {
    color:#9b757d;
    font-size:.68rem;
    line-height:1.55;
    margin-top:14px;
}

.progress-track {
    height:7px;
    background:#f0d9de;
    border-radius:999px;
    overflow:hidden;
    margin-top:14px;
}

.progress-fill {
    height:100%;
    border-radius:999px;
    background:
        linear-gradient(
            90deg,
            #991c39,
            #dd4867
        );
}


/* ---------- BADGES ---------- */

.badge {
    display:inline-flex;
    align-items:center;
    gap:5px;
    padding:6px 10px;
    border-radius:999px;
    font-size:.61rem;
    font-weight:850;
}

.badge-low {
    color:#6f5548;
    background:#f4e8de;
}

.badge-medium {
    color:#a05b12;
    background:#fff0d9;
}

.badge-high {
    color:#9d1f3a;
    background:#fbe0e6;
}

.badge-critical {
    color:#80142a;
    background:#f5cdd6;
}


/* ---------- DECISION ---------- */

.decision-card {
    margin-top:20px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.82),
            rgba(252,226,233,.72)
        );

    border:1px solid #e7c5cd;
    border-radius:20px;

    padding:25px;

    box-shadow:
        0 15px 40px rgba(117,29,48,.06);
}

.decision-head {
    display:flex;
    align-items:center;
    gap:12px;
}

.decision-icon {
    width:40px;
    height:40px;
    border-radius:13px;
    background:#f7d8e0;
    display:flex;
    align-items:center;
    justify-content:center;
    color:#aa2341;
    font-size:18px;
}

.decision-label {
    color:#a24a5b;
    font-size:.62rem;
    font-weight:850;
    letter-spacing:.12em;
    text-transform:uppercase;
}

.decision-title {
    color:#79182c;
    font-size:1rem;
    font-weight:850;
    margin-top:11px;
}

.decision-text {
    color:#966f77;
    font-size:.72rem;
    line-height:1.6;
    margin-top:5px;
}


/* ---------- DETAILS ---------- */

.detail-card {
    background:rgba(255,255,255,.75);
    border:1px solid rgba(225,182,191,.62);
    border-radius:20px;
    padding:25px;
    min-height:270px;
    box-shadow:0 15px 40px rgba(117,29,48,.06);
}

.detail-header {
    display:flex;
    align-items:center;
    gap:11px;
    margin-bottom:7px;
}

.detail-icon {
    width:37px;
    height:37px;
    border-radius:12px;
    background:#f8dce3;
    color:#ad2443;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:17px;
}

.detail-title {
    color:#79172b;
    font-size:.88rem;
    font-weight:850;
}

.detail-copy {
    color:#9d777e;
    font-size:.67rem;
    line-height:1.55;
    margin:8px 0 17px;
}

.risk-item {
    display:flex;
    align-items:center;
    gap:10px;
    padding:10px 0;
    border-bottom:1px solid #f0dde1;
    color:#835761;
    font-size:.73rem;
}

.risk-item:last-child {
    border-bottom:none;
}

.risk-dot {
    width:7px;
    height:7px;
    border-radius:50%;
    background:#af2847;
}

.action-item {
    display:grid;
    grid-template-columns:30px 1fr;
    gap:10px;
    padding:10px 0;
    border-bottom:1px solid #f0dde1;
}

.action-item:last-child {
    border-bottom:none;
}

.action-number {
    width:27px;
    height:27px;
    border-radius:9px;
    background:#f8dce3;
    color:#a72342;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:.60rem;
    font-weight:850;
}

.action-text {
    color:#835761;
    font-size:.73rem;
    line-height:1.5;
    padding-top:4px;
}

.note {
    color:#a17e85;
    font-size:.63rem;
    line-height:1.6;
    margin-top:19px;
}


/* ---------- ANALYTICS CHARTS ---------- */

.chart-card {
    background:rgba(255,255,255,.76);
    border:1px solid rgba(224,180,190,.62);
    border-radius:20px;
    padding:20px 22px 14px;
    box-shadow:0 15px 42px rgba(118,29,49,.07);
    margin-top:10px;
}

.chart-title {
    color:#79172b;
    font-size:.84rem;
    font-weight:850;
    margin-bottom:2px;
}

.chart-subtitle {
    color:#9b757d;
    font-size:.65rem;
    line-height:1.5;
    margin-bottom:8px;
}

/* ---------- EXPANDABLE RESULT CARDS ---------- */

div[data-testid="stExpander"] {
    background:rgba(255,255,255,.72);
    border:1px solid rgba(224,180,190,.62) !important;
    border-radius:20px !important;
    box-shadow:0 15px 42px rgba(118,29,49,.06);
    overflow:hidden;
    margin-top:14px;
}

div[data-testid="stExpander"] details {
    border:none !important;
}

div[data-testid="stExpander"] summary {
    padding:16px 19px !important;
    color:#79172b !important;
    font-weight:800 !important;
    font-size:.82rem !important;
}

div[data-testid="stExpander"] summary:hover {
    background:rgba(248,220,227,.32);
}

div[data-testid="stExpanderDetails"] {
    padding:0 19px 18px !important;
}

/* ---------- MOBILE ---------- */

@media(max-width:900px){

    .block-container {
        padding-left:1rem;
        padding-right:1rem;
    }

    .hero-grid {
        grid-template-columns:1fr;
    }

    .hero-visual {
        min-height:330px;
    }

    .visual-card {
        width:85%;
        right:7%;
    }

    .step-line {
        width:25px;
    }

    .step span:last-child {
        display:none;
    }

    .hero-title {
        font-size:3rem;
    }
}


/* ---------- FLOATING ANALYTICS BACKGROUND ---------- */

.block-container {
    position:relative;
    z-index:2;
}

.analytics-bg {
    position:fixed;
    inset:0;
    overflow:hidden;
    pointer-events:none;
    z-index:1;
}

.analytics-symbol {
    position:absolute;
    width:34px;
    height:34px;
    color:#9e2944;
    opacity:.075;
    filter:blur(.15px);
    will-change:transform;
}

.analytics-symbol svg {
    width:100%;
    height:100%;
    display:block;
    fill:none;
    stroke:currentColor;
    stroke-width:1.5;
    stroke-linecap:round;
    stroke-linejoin:round;
}

.analytics-symbol.s1 { left:4%; top:18%; width:28px; height:28px; animation:bgFloatA 18s ease-in-out infinite; }
.analytics-symbol.s2 { left:11%; top:70%; width:42px; height:42px; opacity:.055; animation:bgFloatB 23s ease-in-out infinite; }
.analytics-symbol.s3 { left:28%; top:9%; width:31px; height:31px; animation:bgFloatC 20s ease-in-out infinite; }
.analytics-symbol.s4 { left:45%; top:78%; width:36px; height:36px; opacity:.06; animation:bgFloatA 25s ease-in-out infinite reverse; }
.analytics-symbol.s5 { right:31%; top:19%; width:27px; height:27px; opacity:.06; animation:bgFloatB 21s ease-in-out infinite; }
.analytics-symbol.s6 { right:18%; top:65%; width:40px; height:40px; opacity:.05; animation:bgFloatC 26s ease-in-out infinite; }
.analytics-symbol.s7 { right:5%; top:31%; width:32px; height:32px; animation:bgFloatA 22s ease-in-out infinite; }
.analytics-symbol.s8 { right:8%; bottom:7%; width:25px; height:25px; opacity:.055; animation:bgFloatB 19s ease-in-out infinite reverse; }
.analytics-symbol.s9 { left:20%; bottom:5%; width:30px; height:30px; opacity:.05; animation:bgFloatC 24s ease-in-out infinite reverse; }
.analytics-symbol.s10 { left:58%; top:7%; width:24px; height:24px; opacity:.05; animation:bgFloatA 20s ease-in-out infinite reverse; }

@keyframes bgFloatA {
    0%,100% { transform:translate3d(0,0,0) rotate(0deg); }
    35% { transform:translate3d(12px,-16px,0) rotate(5deg); }
    70% { transform:translate3d(-8px,10px,0) rotate(-4deg); }
}

@keyframes bgFloatB {
    0%,100% { transform:translate3d(0,0,0) rotate(0deg); }
    40% { transform:translate3d(-15px,-9px,0) rotate(-6deg); }
    75% { transform:translate3d(10px,14px,0) rotate(4deg); }
}

@keyframes bgFloatC {
    0%,100% { transform:translate3d(0,0,0) scale(1); }
    50% { transform:translate3d(7px,-18px,0) scale(1.06); }
}

@media(max-width:900px) {
    .analytics-symbol { opacity:.045; }
    .analytics-symbol.s3,
    .analytics-symbol.s5,
    .analytics-symbol.s9 { display:none; }
}

@media (prefers-reduced-motion: reduce) {
    .analytics-symbol,
    .orb-one,
    .orb-two { animation:none !important; }
}

</style>
""")



# ============================================================
# SUBTLE FLOATING ANALYTICS BACKGROUND
# ============================================================

render_html("""
<div class="analytics-bg" aria-hidden="true">

    <div class="analytics-symbol s1">
        <svg viewBox="0 0 32 32"><path d="M5 26V16M12 26V10M19 26V19M26 26V6"/><path d="M3 27.5H29"/></svg>
    </div>

    <div class="analytics-symbol s2">
        <svg viewBox="0 0 32 32"><path d="M4 24L10 18L15 20L22 11L28 7"/><circle cx="4" cy="24" r="1.5"/><circle cx="10" cy="18" r="1.5"/><circle cx="15" cy="20" r="1.5"/><circle cx="22" cy="11" r="1.5"/><circle cx="28" cy="7" r="1.5"/></svg>
    </div>

    <div class="analytics-symbol s3">
        <svg viewBox="0 0 32 32"><ellipse cx="16" cy="8" rx="10" ry="4"/><path d="M6 8V16C6 18.2 10.5 20 16 20S26 18.2 26 16V8M6 16V24C6 26.2 10.5 28 16 28S26 26.2 26 24V16"/></svg>
    </div>

    <div class="analytics-symbol s4">
        <svg viewBox="0 0 32 32"><circle cx="16" cy="16" r="11"/><circle cx="16" cy="16" r="6"/><circle cx="16" cy="16" r="1.5"/><path d="M24 8L28 4M24 4H28V8"/></svg>
    </div>

    <div class="analytics-symbol s5">
        <svg viewBox="0 0 32 32"><circle cx="16" cy="11" r="5"/><path d="M7 27C8 21 11 18 16 18S24 21 25 27"/></svg>
    </div>

    <div class="analytics-symbol s6">
        <svg viewBox="0 0 32 32"><circle cx="7" cy="16" r="2.5"/><circle cx="16" cy="7" r="2.5"/><circle cx="25" cy="14" r="2.5"/><circle cx="20" cy="25" r="2.5"/><path d="M9 14L14 9M18.5 8L23 12M24 16.5L21 22.5M18 24L9 17"/></svg>
    </div>

    <div class="analytics-symbol s7">
        <svg viewBox="0 0 32 32"><path d="M5 25V18H10V25M13 25V12H18V25M21 25V7H26V25"/><path d="M4 27H28"/></svg>
    </div>

    <div class="analytics-symbol s8">
        <svg viewBox="0 0 32 32"><path d="M6 24L13 17L18 20L27 9"/><path d="M21 9H27V15"/></svg>
    </div>

    <div class="analytics-symbol s9">
        <svg viewBox="0 0 32 32"><rect x="5" y="5" width="8" height="8" rx="2"/><rect x="19" y="5" width="8" height="8" rx="2"/><rect x="5" y="19" width="8" height="8" rx="2"/><rect x="19" y="19" width="8" height="8" rx="2"/></svg>
    </div>

    <div class="analytics-symbol s10">
        <svg viewBox="0 0 32 32"><path d="M6 23C10 18 12 12 16 12C20 12 21 18 26 7"/><circle cx="6" cy="23" r="1.5"/><circle cx="16" cy="12" r="1.5"/><circle cx="26" cy="7" r="1.5"/></svg>
    </div>

</div>
""")

# ============================================================
# TOP BAR
# ============================================================

render_html("""
<div class="topbar">

    <div class="brand-wrap">

        <div class="brand-logo">
            NA
        </div>

        <div>
            <div class="brand-name">NAVIGATE</div>
            <div class="brand-sub">
                Customer Retention Intelligence
            </div>
        </div>

    </div>

    <div class="topbar-right">
        Turn insights into longer relationships
    </div>

</div>
""")



# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":

    render_html("""
    <div class="hero-grid">

        <div class="hero-copy">

            <div class="eyebrow">
                Retention Decision Support
            </div>

            <div class="hero-title">
                Understand risk.<br>
                <span>Prioritize retention.</span>
            </div>

            <div class="hero-description">

                A customer intelligence system that estimates
                churn risk and helps businesses identify which
                customers may need more attention.

            </div>

        </div>


        <div class="hero-visual">

            <div class="orb orb-one"></div>
            <div class="orb orb-two"></div>

            <div class="visual-card" style="transform:rotate(2deg);">

                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:22px;">
                    <div>
                        <div style="color:#861b32;font-weight:850;font-size:.88rem;">Customer Analytics</div>
                        <div style="color:#b57b87;font-size:.61rem;margin-top:3px;">From customer data to retention decisions</div>
                    </div>
                    <div style="width:39px;height:39px;border-radius:13px;background:linear-gradient(145deg,#f8d8e0,#ffffff);border:1px solid #efcbd4;display:flex;align-items:center;justify-content:center;color:#a91e3e;font-size:15px;font-weight:900;letter-spacing:-.04em;">NA</div>
                </div>

                <div style="background:rgba(255,255,255,.66);border:1px solid rgba(229,186,196,.72);border-radius:18px;padding:20px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:17px;">
                        <div style="color:#8a3246;font-size:.66rem;font-weight:800;">Retention Intelligence</div>
                        <div style="color:#b7828d;font-size:.56rem;">ANALYSIS</div>
                    </div>

                    <div style="height:105px;display:flex;align-items:flex-end;gap:11px;padding:0 7px 10px;border-bottom:1px solid #efd7dc;">
                        <div style="flex:1;height:38%;border-radius:8px 8px 3px 3px;background:linear-gradient(180deg,#ed9daf,#f5c7d1);"></div>
                        <div style="flex:1;height:58%;border-radius:8px 8px 3px 3px;background:linear-gradient(180deg,#dc6f89,#efa9b9);"></div>
                        <div style="flex:1;height:47%;border-radius:8px 8px 3px 3px;background:linear-gradient(180deg,#e58ca1,#f3bbc7);"></div>
                        <div style="flex:1;height:78%;border-radius:8px 8px 3px 3px;background:linear-gradient(180deg,#b92b4c,#df7189);"></div>
                        <div style="flex:1;height:66%;border-radius:8px 8px 3px 3px;background:linear-gradient(180deg,#ca4765,#e98fa3);"></div>
                        <div style="flex:1;height:90%;border-radius:8px 8px 3px 3px;background:linear-gradient(180deg,#8d1731,#c93b5b);"></div>
                    </div>

                    <div style="display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:7px;margin-top:17px;">
                        <div style="text-align:center;">
                            <div style="color:#a22643;font-size:.75rem;font-weight:850;">◉</div>
                            <div style="color:#8f5965;font-size:.57rem;font-weight:750;margin-top:4px;">Customer Data</div>
                        </div>
                        <div style="color:#d28a9a;font-size:.8rem;">→</div>
                        <div style="text-align:center;">
                            <div style="color:#a22643;font-size:.75rem;font-weight:850;">◇</div>
                            <div style="color:#8f5965;font-size:.57rem;font-weight:750;margin-top:4px;">Risk Analysis</div>
                        </div>
                        <div style="color:#d28a9a;font-size:.8rem;">→</div>
                        <div style="text-align:center;">
                            <div style="color:#a22643;font-size:.75rem;font-weight:850;">◎</div>
                            <div style="color:#8f5965;font-size:.57rem;font-weight:750;margin-top:4px;">Retention</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </div>
    """)

    left, center, right = st.columns([1, 1.3, 2.8])

    with center:
        if st.button(
            "Start Customer Analysis  →",
            use_container_width=True
        ):
            st.session_state.page = "form"
            st.rerun()


# ============================================================
# FORM PAGE
# ============================================================

elif st.session_state.page == "form":

    render_html("""
    <div class="steps">

        <div class="step active">
            <div class="step-circle">1</div>
            <span>Customer Information</span>
        </div>

        <div class="step-line"></div>

        <div class="step">
            <div class="step-circle">2</div>
            <span>Analysis</span>
        </div>

        <div class="step-line"></div>

        <div class="step">
            <div class="step-circle">3</div>
            <span>Results</span>
        </div>

    </div>
    """)

    intro_col, form_col = st.columns(
        [0.72, 1.85],
        gap="large"
    )

    with intro_col:

        render_html("""
        <div style="padding:22px 5px;">

            <div class="page-kicker">
                Customer Analysis / 01
            </div>

            <div class="page-title">
                Customer<br>information
            </div>

            <div class="page-copy">
                Enter the customer's basic account and
                payment information. All fields are required
                to run the analysis.
            </div>

            <div class="quick-card">

                <div class="quick-icon">
                    ▣
                </div>

                <div>
                    <div class="quick-title">
                        Quick and easy
                    </div>

                    <div class="quick-copy">
                        Just 6 key details to get started.
                    </div>
                </div>

            </div>

        </div>
        """)

        if st.button(
            "← Back to Home",
            use_container_width=True
        ):
            # Going to Home starts a fresh customer analysis.
            st.session_state.page = "home"
            st.session_state.analysis_result = None
            st.rerun()


    with form_col:

        saved_inputs = {}
        if st.session_state.analysis_result is not None:
            saved_inputs = st.session_state.analysis_result.get("customer_inputs", {})

        saved_tenure = int(saved_inputs.get("tenure", 0))
        saved_contract = saved_inputs.get("contract", "Select...")
        saved_payment = saved_inputs.get("payment_method", "Select...")
        saved_billing_raw = saved_inputs.get("paperless_billing", None)
        saved_billing = (
            "Digital / Paperless" if saved_billing_raw == "Yes"
            else "Paper / Standard" if saved_billing_raw == "No"
            else "Select..."
        )
        saved_monthly = float(saved_inputs.get("monthly_charges", 0.0))
        saved_total = float(saved_inputs.get("total_charges", 0.0))

        contract_options = ["Select...", "Month-to-month", "One year", "Two year"]
        payment_options = [
            "Select...",
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
        billing_options = ["Select...", "Digital / Paperless", "Paper / Standard"]

        with st.form(
            "customer_form",
            clear_on_submit=False,
            enter_to_submit=False
        ):

            render_html("""
            <div class="form-heading">

                <div class="form-heading-icon">
                    ♙
                </div>

                <div>
                    Customer Details
                </div>

            </div>
            """)

            col1, col2 = st.columns(
                2,
                gap="large"
            )


            # -----------------------------------------------
            # LEFT
            # -----------------------------------------------

            with col1:

                tenure = st.number_input(
                    "How long has the customer been with the company? *",
                    min_value=0,
                    max_value=72,
                    value=saved_tenure,
                    step=1
                )

                render_html("""
                <div class="field-help">
                    Length of the customer relationship in months.
                    Example: 12 means one year.
                </div>
                """)


                contract = st.selectbox(
                    "Customer contract type *",
                    contract_options,
                    index=contract_options.index(saved_contract) if saved_contract in contract_options else 0
                )

                render_html("""
                <div class="field-help">
                    How long the customer's current contract
                    or plan lasts.
                </div>
                """)


                payment_method = st.selectbox(
                    "How does the customer make payments? *",
                    payment_options,
                    index=payment_options.index(saved_payment) if saved_payment in payment_options else 0
                )

                render_html("""
                <div class="field-help">
                    The customer's usual payment method.
                </div>
                """)


            # -----------------------------------------------
            # RIGHT
            # -----------------------------------------------

            with col2:

                paperless_billing = st.selectbox(
                    "How does the customer receive bills? *",
                    billing_options,
                    index=billing_options.index(saved_billing) if saved_billing in billing_options else 0
                )

                render_html("""
                <div class="field-help">
                    Whether bills are received digitally
                    or as standard paper bills.
                </div>
                """)


                monthly_charges = st.number_input(
                    "Customer's monthly payment amount *",
                    min_value=0.0,
                    value=saved_monthly,
                    step=1.0,
                    format="%.2f"
                )

                render_html("""
                <div class="field-help">
                    Amount the customer normally pays
                    to the company each month.
                </div>
                """)


                total_charges = st.number_input(
                    "Customer's total payments to date *",
                    min_value=0.0,
                    value=saved_total,
                    step=10.0,
                    format="%.2f"
                )

                render_html("""
                <div class="field-help">
                    Total amount paid since becoming
                    a customer.
                </div>
                """)


            render_html("""
            <div class="required-note">
                ⓘ &nbsp; All fields are required before the analysis can run.
            </div>
            """)

            analyze = st.form_submit_button(
                "Run Analysis  →",
                use_container_width=True
            )


        # -----------------------------------------------
        # VALIDATION
        # -----------------------------------------------

        if analyze:

            missing = []

            if tenure <= 0:
                missing.append("Customer relationship length")

            if contract == "Select...":
                missing.append("Contract type")

            if payment_method == "Select...":
                missing.append("Payment method")

            if paperless_billing == "Select...":
                missing.append("Billing method")

            if monthly_charges <= 0:
                missing.append("Monthly payment amount")

            if total_charges <= 0:
                missing.append("Total payments")


            if missing:

                st.error(
                    "Please complete all required fields with valid values before running the analysis."
                )

            else:

                billing_value = (
                    "Yes"
                    if paperless_billing == "Digital / Paperless"
                    else "No"
                )


                customer_data = pd.DataFrame({
                    "tenure": [tenure],
                    "Contract": [contract],
                    "PaperlessBilling": [billing_value],
                    "PaymentMethod": [payment_method],
                    "MonthlyCharges": [monthly_charges],
                    "TotalCharges": [total_charges]
                })


                # -------------------------------------------
                # CHURN RISK
                # -------------------------------------------

                churn_probability = model.predict_proba(
                    customer_data
                )[0, 1]

                churn_risk = round(
                    churn_probability * 100,
                    2
                )

                risk_level = get_risk_level(
                    churn_risk
                )


                # -------------------------------------------
                # BUSINESS IMPACT
                # -------------------------------------------

                impact_values = impact_scaler.transform(
                    customer_data[
                        [
                            "MonthlyCharges",
                            "TotalCharges"
                        ]
                    ]
                )

                monthly_score = impact_values[0, 0] * 100
                total_score = impact_values[0, 1] * 100

                business_impact = round(
                    monthly_score * .60
                    +
                    total_score * .40,
                    2
                )

                business_impact_display = max(
                    0,
                    min(100, business_impact)
                )


                # -------------------------------------------
                # PRIORITY
                # -------------------------------------------

                retention_priority = round(
                    churn_risk * .70
                    +
                    business_impact * .30,
                    2
                )

                retention_priority_display = max(
                    0,
                    min(100, retention_priority)
                )

                priority_level = get_priority_level(
                    retention_priority
                )


                factor_action_pairs = get_factor_action_pairs(
                    tenure=tenure,
                    contract=contract,
                    payment_method=payment_method,
                    paperless_billing=billing_value,
                    monthly_charges=monthly_charges,
                    total_charges=total_charges,
                    churn_risk=churn_risk,
                    business_impact=business_impact_display,
                )

                actions = get_recommended_actions(
                    factor_action_pairs,
                    priority_level
                )

                risk_explanation = get_risk_explanation(
                    risk_level,
                    churn_risk
                )

                decision_title, decision_text = get_decision_summary(
                    priority_level,
                    risk_level,
                    churn_risk
                )

                next_step = get_next_step(
                    priority_level,
                    actions
                )


                st.session_state.analysis_result = {
                    "churn_risk": churn_risk,
                    "risk_level": risk_level,
                    "business_impact": business_impact_display,
                    "retention_priority": retention_priority_display,
                    "priority_level": priority_level,
                    "factor_action_pairs": factor_action_pairs,
                    "actions": actions,
                    "risk_explanation": risk_explanation,
                    "decision_title": decision_title,
                    "decision_text": decision_text,
                    "next_step": next_step,
                    "customer_inputs": {
                        "tenure": tenure,
                        "contract": contract,
                        "payment_method": payment_method,
                        "paperless_billing": billing_value,
                        "monthly_charges": monthly_charges,
                        "total_charges": total_charges,
                    }
                }


                # Separate results page
                st.session_state.page = "results"
                st.rerun()


# ============================================================
# RESULTS PAGE
# ============================================================

elif st.session_state.page == "results":

    if st.session_state.analysis_result is None:
        st.session_state.page = "form"
        st.rerun()


    # --------------------------------------------------------
    # HOME BUTTON - RESULTS PAGE ONLY
    # --------------------------------------------------------

    home_col, empty_col = st.columns([1, 6])

    with home_col:

        if st.button(
            "⌂  Home",
            use_container_width=True,
            key="results_home_button"
        ):
            st.session_state.page = "home"
            st.session_state.analysis_result = None
            st.rerun()


    result = st.session_state.analysis_result

    churn_risk = result["churn_risk"]
    risk_level = result["risk_level"]

    business_impact = result["business_impact"]

    retention_priority = result[
        "retention_priority"
    ]

    priority_level = result[
        "priority_level"
    ]

    factor_action_pairs = result["factor_action_pairs"]
    actions = result["actions"]
    risk_explanation = result["risk_explanation"]
    decision_title = result["decision_title"]
    decision_text = result["decision_text"]
    next_step = result["next_step"]
    customer_inputs = result["customer_inputs"]


    render_html("""
    <div class="steps">

        <div class="step">
            <div class="step-circle">1</div>
            <span>Customer Information</span>
        </div>

        <div class="step-line"></div>

        <div class="step">
            <div class="step-circle">2</div>
            <span>Analysis</span>
        </div>

        <div class="step-line"></div>

        <div class="step active">
            <div class="step-circle">3</div>
            <span>Results</span>
        </div>

    </div>
    """)


    intro_col, result_col = st.columns(
        [.62, 2.1],
        gap="large"
    )


    # --------------------------------------------------------
    # LEFT INTRO
    # --------------------------------------------------------

    with intro_col:

        render_html("""
        <div style="padding:18px 5px;">

            <div class="page-kicker">
                Customer Analysis / 02
            </div>

            <div class="page-title">
                Retention<br>analysis
            </div>

            <div class="page-copy">

                Review the customer's estimated churn risk,
                business impact, retention priority, and
                suggested retention actions.

            </div>

        </div>
        """)

        if st.button(
            "← Edit Customer Data",
            use_container_width=True
        ):
            st.session_state.page = "form"
            st.rerun()


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    with result_col:

        # ----------------------------------------------------
        # TOP METRICS
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            badge = get_badge_class(priority_level)
            render_html(f"""
            <div class="metric-grid-card">
                <div class="metric-top">
                    <div class="metric-heading">
                        <div class="metric-icon">◎</div>
                        <div class="metric-label">Retention Priority Score</div>
                    </div>
                    <span class="badge badge-{badge}">
                        {html.escape(priority_level)}
                    </span>
                </div>

                <div class="metric-value">
                    {retention_priority:.0f}
                    <span class="metric-unit">/ 100</span>
                </div>

                <div class="progress-track">
                    <div class="progress-fill"
                         style="width:{retention_priority}%"></div>
                </div>

                <div class="metric-note">
                    Indicates how strongly this customer should be prioritized
                    for retention within this prototype.
                </div>
            </div>
            """)

        with c2:
            badge = get_badge_class(risk_level)
            render_html(f"""
            <div class="metric-grid-card">
                <div class="metric-top">
                    <div class="metric-heading">
                        <div class="metric-icon">◇</div>
                        <div class="metric-label">Estimated Churn Risk</div>
                    </div>
                    <span class="badge badge-{badge}">
                        {html.escape(risk_level)}
                    </span>
                </div>

                <div class="metric-value">{churn_risk:.0f}%</div>

                <div class="progress-track">
                    <div class="progress-fill"
                         style="width:{churn_risk}%"></div>
                </div>

                <div class="metric-note">
                    Model-estimated probability that this customer may leave.
                </div>
            </div>
            """)

        with c3:
            render_html(f"""
            <div class="metric-grid-card">
                <div class="metric-top">
                    <div class="metric-heading">
                        <div class="metric-icon">◉</div>
                        <div class="metric-label">Customer Business Impact</div>
                    </div>
                </div>

                <div class="metric-value">
                    {business_impact:.0f}
                    <span class="metric-unit">/ 100</span>
                </div>

                <div class="progress-track">
                    <div class="progress-fill"
                         style="width:{business_impact}%"></div>
                </div>

                <div class="metric-note">
                    Relative customer value based on monthly and total payments.
                </div>
            </div>
            """)

        # ----------------------------------------------------
        # VISUAL ANALYSIS
        # ----------------------------------------------------

        render_html("""
        <div style="margin-top:24px;">
            <div class="page-kicker">Visual Analysis</div>
            <div class="detail-title">Customer score overview</div>
            <div class="detail-copy">
                A focused visual summary of this customer's calculated scores.
            </div>
        </div>
        """)

        chart_labels = ["Churn Risk", "Business Impact", "Retention Priority"]
        chart_values = [churn_risk, business_impact, retention_priority]
        chart_bg = "#fffafb"
        text_color = "#79172b"
        muted_color = "#9b757d"
        accent = "#b72243"
        accent_mid = "#d65a75"
        accent_soft = "#e9aeba"
        grid_color = "#ead8dc"

        # Main comparison — horizontal layout is easier to scan.
        render_html("""
        <div class="chart-card">
            <div class="chart-title">Score Comparison</div>
            <div class="chart-subtitle">
                Compare churn risk, customer impact, and retention priority on one scale.
            </div>
        </div>
        """)
        fig, ax = plt.subplots(figsize=(9.5, 3.0))
        fig.patch.set_facecolor(chart_bg)
        ax.set_facecolor(chart_bg)
        y = [2, 1, 0]
        bars = ax.barh(y, chart_values, height=.46, color=[accent, accent_soft, "#8d1731"])
        ax.set_yticks(y)
        ax.set_yticklabels(chart_labels, color=text_color, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", colors=muted_color, labelsize=8, length=0)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.xaxis.grid(True, color=grid_color, linewidth=.8, alpha=.65)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, chart_values):
            ax.text(min(value + 2, 96), bar.get_y()+bar.get_height()/2, f"{value:.0f}",
                    va="center", ha="left", fontsize=9, fontweight="bold", color=text_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=1.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        chart_left, chart_right = st.columns(2, gap="medium")

        with chart_left:
            render_html("""
            <div class="chart-card">
                <div class="chart-title">Score Profile</div>
                <div class="chart-subtitle">Connected view of the three analysis scores.</div>
            </div>
            """)
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            fig.patch.set_facecolor(chart_bg)
            ax.set_facecolor(chart_bg)
            x = [0, 1, 2]
            ax.plot(x, chart_values, linewidth=2.7, color=accent, zorder=3)
            ax.scatter(x, chart_values, s=58, color=[accent, accent_mid, "#8d1731"], zorder=4)
            ax.fill_between(x, chart_values, 0, color=accent_soft, alpha=.12)
            ax.set_xticks(x)
            ax.set_xticklabels(["Churn", "Impact", "Priority"], color=text_color, fontsize=8)
            ax.set_ylim(0, 105)
            ax.set_yticks([0,25,50,75,100])
            ax.tick_params(axis="both", colors=muted_color, labelsize=8, length=0)
            ax.yaxis.grid(True, color=grid_color, linewidth=.8, alpha=.6)
            ax.set_axisbelow(True)
            for xi, value in zip(x, chart_values):
                ax.text(xi, min(value+5, 101), f"{value:.0f}", ha="center", fontsize=8.5,
                        fontweight="bold", color=text_color)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with chart_right:
            render_html("""
            <div class="chart-card">
                <div class="chart-title">Score Range Distribution</div>
                <div class="chart-subtitle">Where this customer's three scores fall across score bands.</div>
            </div>
            """)
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            fig.patch.set_facecolor(chart_bg)
            ax.set_facecolor(chart_bg)
            bins = [0,20,40,60,80,100]
            counts, edges, patches = ax.hist(chart_values, bins=bins, rwidth=.72, color=accent, alpha=.9)
            ax.set_xlim(0,100)
            ax.set_xticks([0,20,40,60,80,100])
            ax.set_yticks(range(0, int(max(counts))+2))
            ax.tick_params(axis="both", colors=muted_color, labelsize=8, length=0)
            ax.yaxis.grid(True, color=grid_color, linewidth=.8, alpha=.6)
            ax.set_axisbelow(True)
            ax.set_xlabel("Score band", color=muted_color, fontsize=8)
            for patch, count in zip(patches, counts):
                if count:
                    ax.text(patch.get_x()+patch.get_width()/2, count+.05, f"{int(count)}",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=text_color)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ----------------------------------------------------
        # EXPANDABLE RESULT DETAILS
        # ----------------------------------------------------

        interpretation_result, interpretation_why, interpretation_action = (
            get_result_interpretation(
                risk_level,
                churn_risk,
                factor_action_pairs
            )
        )
        interpretation_badge = get_badge_class(risk_level)

        render_html("""
        <div style="margin-top:26px;">
            <div class="page-kicker">Analysis Details</div>
            <div class="detail-title">Explore the customer analysis</div>
            <div class="detail-copy">
                Open a section only when you need more detail.
            </div>
        </div>
        """)

        with st.expander("Result Interpretation", expanded=False):
            render_html(f"""
            <div style="padding:3px 2px 5px;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
                    <span class="badge badge-{interpretation_badge}">
                        {html.escape(interpretation_result)}
                    </span>
                </div>

                <div class="decision-label">Decision Summary</div>
                <div class="decision-title" style="margin-top:6px;">
                    {html.escape(decision_title)}
                </div>
                <div class="decision-text">{html.escape(decision_text)}</div>

                <div style="margin-top:17px;">
                    <div class="decision-label">Why this matters</div>
                    <div class="decision-text" style="margin-top:6px;">
                        {html.escape(interpretation_why)}
                    </div>
                </div>

                <div style="margin-top:17px;padding:14px 16px;border-radius:14px;
                            background:rgba(248,220,227,.55);
                            border:1px solid rgba(225,182,191,.55);">
                    <div class="decision-label">Suggested Action</div>
                    <div class="decision-text" style="margin-top:6px;">
                        {html.escape(interpretation_action)}
                    </div>
                </div>
            </div>
            """)

        with st.expander("Customer Risk Signals", expanded=False):
            render_html("""
            <div class="detail-copy" style="margin-top:2px;">
                Each signal is paired with a practical response. These are review
                signals and do not prove that one characteristic caused the prediction.
            </div>
            """)

            pair_html = ""
            for index, (factor, context, action) in enumerate(factor_action_pairs, start=1):
                pair_html += f"""
                <div style="background:rgba(255,255,255,.66);
                            border:1px solid rgba(225,182,191,.58);
                            border-radius:16px;padding:16px 18px;margin-bottom:10px;">
                    <div style="display:flex;align-items:flex-start;gap:12px;">
                        <div class="action-number">{index:02d}</div>
                        <div style="flex:1;">
                            <div style="color:#79172b;font-size:.80rem;font-weight:850;margin-bottom:5px;">
                                {html.escape(factor)}
                            </div>
                            <div style="color:#956f77;font-size:.69rem;line-height:1.55;margin-bottom:8px;">
                                {html.escape(context)}
                            </div>
                            <div style="color:#7f4551;font-size:.71rem;line-height:1.55;">
                                <strong>Recommended response:</strong> {html.escape(action)}
                            </div>
                        </div>
                    </div>
                </div>
                """
            render_html(pair_html)

        with st.expander("Recommended Action Plan", expanded=False):
            render_html("""
            <div class="detail-copy" style="margin-top:2px;">
                Actions are ordered by urgency and customer-specific relevance.
            </div>
            """)

            action_html = ""
            for i, (label, action) in enumerate(actions, start=1):
                action_html += f"""
                <div class="action-item">
                    <div class="action-number">{i:02d}</div>
                    <div class="action-text">
                        <div style="color:#a24a5b;font-size:.60rem;font-weight:850;
                                    letter-spacing:.08em;text-transform:uppercase;margin-bottom:3px;">
                            {html.escape(label)}
                        </div>
                        {html.escape(action)}
                    </div>
                </div>
                """
            render_html(action_html)

        # ----------------------------------------------------
        # MODEL / PROTOTYPE NOTE
        # ----------------------------------------------------

        render_html("""
        <div class="note">
            NAVIGATE is a decision-support prototype. Churn risk is a
            model-estimated probability, not a guaranteed customer outcome.
            Business impact and retention priority use project scoring
            assumptions. Suggested actions are rule-based recommendations
            derived from the customer information entered and should not be
            interpreted as causal effects.
        </div>
        """)
