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
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


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

if "language" not in st.session_state:
    st.session_state.language = "English"

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


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


def get_decision_summary(level):

    if level == "Critical":
        return (
            "Immediate retention attention recommended",
            "This customer is in the highest retention priority group. "
            "Consider a proactive retention response."
        )

    if level == "High":
        return (
            "High retention priority",
            "This customer shows elevated retention priority. "
            "Review the customer signals and consider proactive action."
        )

    if level == "Medium":
        return (
            "Moderate retention priority",
            "This customer currently shows moderate retention priority. "
            "Continued monitoring and engagement may be appropriate."
        )

    return (
        "Low retention priority",
        "This customer currently ranks in the lower retention priority group. "
        "No immediate targeted retention action is indicated."
    )


def get_risk_factors(
    tenure,
    contract,
    payment_method,
    churn_risk,
    business_impact,
    priority_level
):
    factors = []

    if churn_risk >= 75:
        factors.append("The model estimates a very high likelihood of customer churn")
    elif churn_risk >= 50:
        factors.append("The model estimates an elevated likelihood of customer churn")

    if business_impact >= 75:
        factors.append("The customer has high relative business impact")
    elif business_impact >= 50:
        factors.append("The customer has moderate relative business impact")

    if contract == "Month-to-month":
        factors.append("Customer is currently on a flexible month-to-month contract")

    if payment_method == "Electronic check":
        factors.append("Customer currently uses a manual electronic payment method")

    if tenure < 12:
        factors.append("Customer relationship is less than one year")

    if not factors:
        if priority_level in ["High", "Critical"]:
            factors.append(
                "Overall retention priority is elevated based on the combined analysis"
            )
        else:
            factors.append(
                "No major customer characteristics require immediate review"
            )

    return factors


def get_recommended_actions(
    tenure,
    contract,
    payment_method,
    churn_risk,
    business_impact,
    priority_level
):
    actions = []

    if priority_level == "Critical":
        actions.append("Prioritize this customer for immediate retention outreach")
    elif priority_level == "High":
        actions.append("Consider proactive retention outreach to this customer")
    elif priority_level == "Medium":
        actions.append("Monitor the customer closely and maintain proactive engagement")
    else:
        actions.append("Continue regular customer engagement and monitor for changes")

    if business_impact >= 75 and churn_risk >= 50:
        actions.append(
            "Consider a personalized retention offer based on the customer's value"
        )

    if contract == "Month-to-month":
        actions.append(
            "Consider offering an incentive for a longer-term contract"
        )

    if payment_method == "Electronic check":
        actions.append(
            "Consider encouraging a more convenient automatic payment method"
        )

    if tenure < 12:
        actions.append(
            "Strengthen engagement during the customer's first year"
        )

    if churn_risk >= 75:
        actions.append(
            "Review the customer's experience to identify possible concerns before taking further action"
        )

    return actions


AR = {
    "Home": "الرئيسية",
    "Analyze": "تحليل عميل",
    "Dashboard": "لوحة المعلومات",
    "About": "عن NAVIGATE",
    "Customer Retention Intelligence": "ذكاء الاحتفاظ بالعملاء",
    "Turn insights into longer relationships": "حوّل البيانات إلى علاقات أطول",
    "Retention Decision Support": "دعم قرارات الاحتفاظ بالعملاء",
    "Understand risk.": "افهم المخاطر.",
    "Prioritize retention.": "رتّب أولوية الاحتفاظ.",
    "A customer intelligence system that estimates churn risk and helps businesses identify which customers may need more attention.":
        "نظام ذكي يساعد الشركات على تقدير خطر فقدان العملاء وتحديد العملاء الذين يحتاجون إلى أولوية أكبر للاحتفاظ بهم.",
    "Start Customer Analysis  →": "ابدأ تحليل العميل  ←",
    "Customer Information": "بيانات العميل",
    "Analysis": "التحليل",
    "Results": "النتائج",
    "Customer Analysis / 01": "تحليل العميل / 01",
    "Customer<br>information": "بيانات<br>العميل",
    "Enter the customer's basic account and payment information. All fields are required to run the analysis.":
        "أدخل بيانات العميل الأساسية ومعلومات الدفع. جميع الحقول مطلوبة لإجراء التحليل.",
    "Quick and easy": "سريع وسهل",
    "Just 6 key details to get started.": "6 بيانات أساسية فقط للبدء.",
    "← Back to Home": "العودة للرئيسية →",
    "Customer Details": "تفاصيل العميل",
    "How long has the customer been with the company? *": "منذ كم شهر والعميل يتعامل مع الشركة؟ *",
    "Customer contract type *": "نوع عقد العميل *",
    "How does the customer make payments? *": "كيف يدفع العميل؟ *",
    "How does the customer receive bills? *": "كيف يستلم العميل الفواتير؟ *",
    "Customer's monthly payment amount *": "مبلغ الدفع الشهري للعميل *",
    "Customer's total payments to date *": "إجمالي مدفوعات العميل حتى الآن *",
    "Run Analysis  →": "تشغيل التحليل  ←",
    "Retention analysis": "تحليل الاحتفاظ",
    "Retention<br>analysis": "تحليل<br>الاحتفاظ",
    "Customer Analysis / 02": "تحليل العميل / 02",
    "Review the customer's estimated churn risk, business impact, retention priority, and suggested retention actions.":
        "راجع خطر فقدان العميل، وتأثيره على الأعمال، وأولوية الاحتفاظ به، والإجراءات المقترحة.",
    "← Edit Customer Data": "تعديل بيانات العميل →",
    "Retention Priority Score": "درجة أولوية الاحتفاظ",
    "Estimated Churn Risk": "خطر فقدان العميل المتوقع",
    "Customer Business Impact": "تأثير العميل على الأعمال",
    "Decision Summary": "ملخص القرار",
    "Factors to Review": "عوامل للمراجعة",
    "Suggested Actions": "إجراءات مقترحة",
    "Download Analysis Report": "تحميل تقرير التحليل",
}

def tr(value):
    if st.session_state.language == "العربية":
        return AR.get(value, value)
    return value


def translate_level(level):
    if st.session_state.language != "العربية":
        return level
    return {
        "Low": "منخفض",
        "Medium": "متوسط",
        "High": "مرتفع",
        "Critical": "حرج"
    }.get(level, level)


def translate_factor(text):
    if st.session_state.language != "العربية":
        return text
    mapping = {
        "The model estimates a very high likelihood of customer churn":
            "النموذج يقدّر احتمالًا مرتفعًا جدًا لفقدان العميل",
        "The model estimates an elevated likelihood of customer churn":
            "النموذج يقدّر احتمالًا مرتفعًا لفقدان العميل",
        "The customer has high relative business impact":
            "للعميل تأثير نسبي مرتفع على الأعمال",
        "The customer has moderate relative business impact":
            "للعميل تأثير نسبي متوسط على الأعمال",
        "Customer is currently on a flexible month-to-month contract":
            "العميل يستخدم عقدًا شهريًا مرنًا",
        "Customer currently uses a manual electronic payment method":
            "العميل يستخدم وسيلة دفع إلكترونية يدوية",
        "Customer relationship is less than one year":
            "مدة علاقة العميل بالشركة أقل من سنة",
        "Overall retention priority is elevated based on the combined analysis":
            "أولوية الاحتفاظ بالعميل مرتفعة بناءً على التحليل المجمع",
        "No major customer characteristics require immediate review":
            "لا توجد خصائص رئيسية تتطلب مراجعة فورية"
    }
    return mapping.get(text, text)


def translate_action(text):
    if st.session_state.language != "العربية":
        return text
    mapping = {
        "Prioritize this customer for immediate retention outreach":
            "إعطاء هذا العميل أولوية للتواصل معه بهدف الاحتفاظ به",
        "Consider proactive retention outreach to this customer":
            "النظر في التواصل الاستباقي مع العميل للاحتفاظ به",
        "Monitor the customer closely and maintain proactive engagement":
            "متابعة العميل والحفاظ على تواصل استباقي معه",
        "Continue regular customer engagement and monitor for changes":
            "الاستمرار في التواصل المعتاد مع العميل ومتابعة أي تغيّرات",
        "Consider a personalized retention offer based on the customer's value":
            "النظر في تقديم عرض احتفاظ مخصص يتناسب مع قيمة العميل",
        "Consider offering an incentive for a longer-term contract":
            "النظر في تقديم حافز للانتقال إلى عقد أطول",
        "Consider encouraging a more convenient automatic payment method":
            "اقتراح وسيلة دفع تلقائية أكثر سهولة",
        "Strengthen engagement during the customer's first year":
            "تعزيز التواصل مع العميل خلال سنته الأولى",
        "Review the customer's experience to identify possible concerns before taking further action":
            "مراجعة تجربة العميل لتحديد أي مشكلات محتملة قبل اتخاذ إجراء إضافي"
    }
    return mapping.get(text, text)


def create_pdf_report(result):
    """Generate a clean English PDF report. Arabic UI remains supported;
    English is used in the PDF to ensure reliable rendering on Streamlit Cloud."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setTitle("NAVIGATE Customer Retention Analysis")
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, height - 60, "NAVIGATE")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, height - 78, "Customer Retention Intelligence")

    y = height - 120
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Customer Retention Analysis")
    y -= 34

    rows = [
        ("Retention Priority", f'{result["retention_priority"]:.2f} / 100 ({result["priority_level"]})'),
        ("Estimated Churn Risk", f'{result["churn_risk"]:.2f}% ({result["risk_level"]})'),
        ("Business Impact", f'{result["business_impact"]:.2f} / 100'),
    ]

    pdf.setFont("Helvetica", 11)
    for label, value in rows:
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(50, y, label)
        pdf.setFont("Helvetica", 11)
        pdf.drawString(220, y, value)
        y -= 24

    y -= 12
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Decision Summary")
    y -= 20
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(50, y, result["decision_title"][:85])
    y -= 17
    pdf.setFont("Helvetica", 9)
    for line in textwrap.wrap(result["decision_text"], 90):
        pdf.drawString(50, y, line)
        y -= 14

    y -= 14
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Factors to Review")
    y -= 20
    pdf.setFont("Helvetica", 9)
    for factor in result["risk_factors"]:
        for i, line in enumerate(textwrap.wrap(factor, 88)):
            prefix = "- " if i == 0 else "  "
            pdf.drawString(55, y, prefix + line)
            y -= 14

    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Suggested Actions")
    y -= 20
    pdf.setFont("Helvetica", 9)
    for idx, action in enumerate(result["actions"], 1):
        for i, line in enumerate(textwrap.wrap(action, 84)):
            prefix = f"{idx}. " if i == 0 else "   "
            pdf.drawString(55, y, prefix + line)
            y -= 14
            if y < 70:
                pdf.showPage()
                y = height - 60
                pdf.setFont("Helvetica", 9)

    y -= 18
    pdf.setFont("Helvetica-Oblique", 8)
    note = (
        "NAVIGATE is a decision-support system. Scores depend on the trained model "
        "and project scoring assumptions. Suggested actions do not guarantee a specific outcome."
    )
    for line in textwrap.wrap(note, 100):
        pdf.drawString(50, y, line)
        y -= 12

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


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
}

.orb-two {
    width:230px;
    height:230px;
    right:190px;
    top:145px;
    background:rgba(240,133,157,.17);
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


/* ---------- LANGUAGE SWITCH ---------- */

.language-switch {
    width:100%;
    display:flex;
    align-items:center;
    justify-content:flex-end;
    gap:9px;
    padding:8px 2px;
    background:transparent !important;
}

.language-switch a {
    color:#9a6973;
    text-decoration:none !important;
    font-size:.72rem;
    font-weight:760;
    letter-spacing:.02em;
    padding:3px 2px;
    border-bottom:1.5px solid transparent;
    transition:color .18s ease, border-color .18s ease;
}

.language-switch a:hover {
    color:#861b32;
}

.language-switch a.active {
    color:#861b32;
    border-bottom-color:#b72243;
}

.language-divider {
    color:#d7aeb7;
    font-size:.7rem;
}


/* ---------- ANIMATED INSIGHT SHOWCASE ---------- */

.insight-showcase {
    position:absolute;
    width:430px;
    height:360px;
    right:20px;
    top:42px;
    perspective:1200px;
}

.showcase-glow {
    position:absolute;
    width:330px;
    height:330px;
    border-radius:50%;
    right:30px;
    top:10px;
    background:radial-gradient(
        circle,
        rgba(229,89,123,.20) 0%,
        rgba(237,142,164,.10) 45%,
        rgba(255,255,255,0) 72%
    );
    filter:blur(2px);
}

.showcase-card {
    position:absolute;
    left:50%;
    top:50%;
    width:355px;
    min-height:225px;
    opacity:0;
    transform:translate(-50%,-44%) translateX(55px) scale(.93) rotate(2deg);
    animation:showcaseCycle 16s infinite;
    will-change:transform,opacity;
}

.showcase-card:nth-of-type(2) { animation-delay:0s; }
.showcase-card:nth-of-type(3) { animation-delay:4s; }
.showcase-card:nth-of-type(4) { animation-delay:8s; }
.showcase-card:nth-of-type(5) { animation-delay:12s; }

@keyframes showcaseCycle {
    0% {
        opacity:0;
        transform:translate(-50%,-44%) translateX(65px) scale(.92) rotate(3deg);
        z-index:1;
    }
    7% {
        opacity:1;
        transform:translate(-50%,-50%) translateX(0) scale(1) rotate(0deg);
        z-index:5;
    }
    21% {
        opacity:1;
        transform:translate(-50%,-50%) translateX(0) scale(1) rotate(0deg);
        z-index:5;
    }
    27% {
        opacity:.28;
        transform:translate(-50%,-53%) translateX(-38px) scale(.95) rotate(-2deg);
        z-index:2;
    }
    32%,100% {
        opacity:0;
        transform:translate(-50%,-56%) translateX(-70px) scale(.90) rotate(-3deg);
        z-index:1;
    }
}

.showcase-shell {
    position:relative;
    overflow:hidden;
    border:1px solid rgba(255,255,255,.90);
    background:rgba(255,255,255,.61);
    backdrop-filter:blur(20px);
    -webkit-backdrop-filter:blur(20px);
    box-shadow:0 28px 70px rgba(130,28,53,.14);
}

/* Card 1: circular risk */
.risk-showcase {
    border-radius:34px 34px 34px 12px;
    padding:27px 29px;
}

.showcase-kicker {
    color:#aa4b5e;
    font-size:.61rem;
    font-weight:850;
    letter-spacing:.16em;
    text-transform:uppercase;
}

.risk-layout {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:22px;
    margin-top:20px;
}

.risk-ring {
    width:108px;
    height:108px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    background:
        radial-gradient(circle at center, #fff8fa 54%, transparent 56%),
        conic-gradient(#bd294a 0 72%, #f0d3da 72% 100%);
    box-shadow:inset 0 0 0 1px rgba(177,57,83,.08);
}

.risk-ring strong {
    color:#7a172c;
    font-size:1.45rem;
    letter-spacing:-.04em;
}

.risk-copy {
    flex:1;
}

.showcase-title {
    color:#78172b;
    font-size:1.03rem;
    font-weight:850;
    line-height:1.2;
}

.showcase-small {
    color:#9a747c;
    font-size:.68rem;
    line-height:1.55;
    margin-top:7px;
}

/* Card 2: priority */
.priority-showcase {
    border-radius:18px 42px 18px 42px;
    padding:28px;
    transform-origin:center;
}

.priority-number {
    color:#78172b;
    font-size:3.2rem;
    font-weight:850;
    letter-spacing:-.07em;
    margin-top:15px;
}

.priority-number span {
    font-size:.75rem;
    color:#a47b83;
    letter-spacing:0;
}

.priority-scale {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:7px;
    margin-top:17px;
}

.priority-scale div {
    height:9px;
    border-radius:99px;
    background:#efd5db;
}

.priority-scale div:nth-child(1),
.priority-scale div:nth-child(2),
.priority-scale div:nth-child(3) {
    background:linear-gradient(90deg,#a91e3e,#d74c68);
}

.priority-chip {
    display:inline-flex;
    margin-top:18px;
    padding:7px 11px;
    border-radius:999px;
    background:#f6d8e0;
    color:#8d1730;
    font-size:.62rem;
    font-weight:850;
}

/* Card 3: business impact */
.impact-showcase {
    border-radius:45px 16px 45px 16px;
    padding:28px 30px;
}

.impact-layout {
    display:grid;
    grid-template-columns:1fr 1.05fr;
    align-items:end;
    gap:22px;
    margin-top:18px;
}

.impact-value {
    color:#78172b;
    font-size:2.8rem;
    font-weight:850;
    letter-spacing:-.06em;
}

.impact-bars {
    height:115px;
    display:flex;
    align-items:flex-end;
    justify-content:space-between;
    gap:8px;
    padding:8px 0 2px;
}

.impact-bars span {
    flex:1;
    border-radius:8px 8px 3px 3px;
    background:linear-gradient(180deg,#e8758e,#aa2443);
    opacity:.84;
}

.impact-bars span:nth-child(1){height:35%;}
.impact-bars span:nth-child(2){height:58%;}
.impact-bars span:nth-child(3){height:46%;}
.impact-bars span:nth-child(4){height:78%;}
.impact-bars span:nth-child(5){height:92%;}

/* Card 4: action */
.action-showcase {
    border-radius:28px;
    padding:28px;
    border-left:4px solid #b92344;
}

.action-headline {
    display:flex;
    gap:14px;
    align-items:center;
    margin-top:18px;
}

.action-symbol {
    width:52px;
    height:52px;
    border-radius:16px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(145deg,#f6d4dd,#fff);
    color:#a51e3c;
    font-size:22px;
}

.action-lines {
    margin-top:21px;
    display:grid;
    gap:9px;
}

.action-line {
    height:10px;
    border-radius:99px;
    background:#ecc5ce;
}

.action-line:nth-child(2){width:82%;}
.action-line:nth-child(3){width:62%;background:#f2dce1;}

@media(max-width:900px){
    .insight-showcase {
        width:100%;
        right:0;
        top:15px;
    }

    .showcase-card {
        width:min(355px,88vw);
    }
}

@media(prefers-reduced-motion: reduce){
    .showcase-card {
        animation:none;
        opacity:0;
    }
    .showcase-card:nth-of-type(2){
        opacity:1;
        transform:translate(-50%,-50%);
    }
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

</style>
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
# NAVIGATION + LANGUAGE
# ============================================================

nav1, nav2, nav3, nav4, spacer, lang_col = st.columns(
    [1, 1.2, 1.2, 1, 3.2, 1.25]
)

with nav1:
    if st.button(tr("Home"), key="nav_home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

with nav2:
    if st.button(tr("Analyze"), key="nav_analyze", use_container_width=True):
        st.session_state.page = "form"
        st.rerun()

with nav3:
    if st.button(tr("Dashboard"), key="nav_dashboard", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()

with nav4:
    if st.button(tr("About"), key="nav_about", use_container_width=True):
        st.session_state.page = "about"
        st.rerun()

# Text-only language switch. No white select box.
query_lang = st.query_params.get("lang")

if query_lang == "ar" and st.session_state.language != "العربية":
    st.session_state.language = "العربية"
    st.query_params.clear()
    st.rerun()

if query_lang == "en" and st.session_state.language != "English":
    st.session_state.language = "English"
    st.query_params.clear()
    st.rerun()

with lang_col:
    en_class = "active" if st.session_state.language == "English" else ""
    ar_class = "active" if st.session_state.language == "العربية" else ""

    render_html(f"""
    <div class="language-switch">
        <a class="{en_class}" href="?lang=en" target="_self">EN</a>
        <span class="language-divider">|</span>
        <a class="{ar_class}" href="?lang=ar" target="_self">عربي</a>
    </div>
    """)

if st.session_state.language == "العربية":
    render_html("""
    <style>
        .stApp { direction: rtl; }
        .topbar, .brand-wrap, .metric-top, .detail-header,
        .decision-head, .quick-card, .form-heading {
            direction: rtl;
        }
        input, [data-baseweb="select"] { direction: rtl; }
    </style>
    """)




# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":

    if st.session_state.language == "العربية":
        eyebrow = "دعم قرارات الاحتفاظ بالعملاء"
        title_a = "افهم المخاطر."
        title_b = "رتّب أولوية الاحتفاظ."
        desc = (
            "منصة تساعد الشركات على تقدير خطر فقدان العملاء، "
            "وتحديد أولوية الاحتفاظ بهم، وتحويل النتائج إلى إجراءات قابلة للمراجعة."
        )
        start_label = "ابدأ تحليل العميل  ←"
    else:
        eyebrow = "Retention Decision Support"
        title_a = "Understand risk."
        title_b = "Prioritize retention."
        desc = (
            "A customer intelligence platform that estimates churn risk, "
            "prioritizes retention, and turns analysis into reviewable actions."
        )
        start_label = "Start Customer Analysis  →"

    render_html(f"""
    <div class="hero-grid">
        <div class="hero-copy">
            <div class="eyebrow">{eyebrow}</div>
            <div class="hero-title">
                {title_a}<br>
                <span>{title_b}</span>
            </div>
            <div class="hero-description">{desc}</div>
        </div>

        <div class="hero-visual">
            <div class="orb orb-one"></div>
            <div class="orb orb-two"></div>

            <div class="insight-showcase">
                <div class="showcase-glow"></div>

                <div class="showcase-card">
                    <div class="showcase-shell risk-showcase"
                         style="min-height:225px;display:flex;align-items:center;padding:34px;">
                        <div>
                            <div class="showcase-title" style="font-size:1.55rem;line-height:1.25;">
                                {"اعرف أي العملاء يحتاجون اهتمامك." if st.session_state.language == "العربية" else
                                 "Know who needs your attention."}
                            </div>
                            <div class="showcase-small" style="font-size:.78rem;margin-top:13px;max-width:290px;">
                                {"حدّد العملاء الذين لديهم خطر أعلى لفقدانهم." if st.session_state.language == "العربية" else
                                 "Identify customers with higher estimated churn risk."}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="showcase-card">
                    <div class="showcase-shell priority-showcase"
                         style="min-height:225px;display:flex;align-items:center;padding:34px;">
                        <div>
                            <div class="showcase-title" style="font-size:1.55rem;line-height:1.25;">
                                {"ركّز على العملاء الأكثر أولوية." if st.session_state.language == "العربية" else
                                 "Focus where it matters most."}
                            </div>
                            <div class="showcase-small" style="font-size:.78rem;margin-top:13px;max-width:290px;">
                                {"رتّب أولوية الاحتفاظ بناءً على الخطر والقيمة النسبية للعميل." if st.session_state.language == "العربية" else
                                 "Prioritize retention using risk and relative customer value."}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="showcase-card">
                    <div class="showcase-shell impact-showcase" style="min-height:225px;padding:32px 34px;">
                        <div class="impact-layout" style="margin-top:0;align-items:center;">
                            <div>
                                <div class="showcase-title" style="font-size:1.45rem;line-height:1.25;">
                                    {"شاهد الصورة بشكل أوضح." if st.session_state.language == "العربية" else
                                     "See the bigger picture."}
                                </div>
                                <div class="showcase-small" style="font-size:.76rem;margin-top:12px;">
                                    {"حوّل بيانات العملاء إلى رؤية أسهل للمراجعة." if st.session_state.language == "العربية" else
                                     "Turn customer data into a clearer view for review."}
                                </div>
                            </div>

                            <div class="impact-bars">
                                <span></span>
                                <span></span>
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="showcase-card">
                    <div class="showcase-shell action-showcase"
                         style="min-height:225px;display:flex;align-items:center;padding:34px;">
                        <div>
                            <div class="showcase-title" style="font-size:1.55rem;line-height:1.25;">
                                {"حوّل النتائج إلى خطوات واضحة." if st.session_state.language == "العربية" else
                                 "Turn insight into action."}
                            </div>
                            <div class="showcase-small" style="font-size:.78rem;margin-top:13px;max-width:295px;">
                                {"راجع الإجراءات المقترحة واتخذ قرارات احتفاظ أكثر وضوحًا." if st.session_state.language == "العربية" else
                                 "Review suggested actions and make more informed retention decisions."}
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
        </div>
    </div>
    """)

    left, center, right = st.columns([1, 1.3, 2.8])
    with center:
        if st.button(start_label, use_container_width=True, key="home_start_analysis"):
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
            st.session_state.page = "home"
            st.rerun()


    with form_col:

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
                    value=0,
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
                    [
                        "Select...",
                        "Month-to-month",
                        "One year",
                        "Two year"
                    ]
                )

                render_html("""
                <div class="field-help">
                    How long the customer's current contract
                    or plan lasts.
                </div>
                """)


                payment_method = st.selectbox(
                    "How does the customer make payments? *",
                    [
                        "Select...",
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ]
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
                    [
                        "Select...",
                        "Digital / Paperless",
                        "Paper / Standard"
                    ]
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
                    value=0.0,
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
                    value=0.0,
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


                risk_factors = get_risk_factors(
                    tenure,
                    contract,
                    payment_method,
                    churn_risk,
                    business_impact_display,
                    priority_level
                )

                actions = get_recommended_actions(
                    tenure,
                    contract,
                    payment_method,
                    churn_risk,
                    business_impact_display,
                    priority_level
                )

                decision_title, decision_text = (
                    get_decision_summary(
                        priority_level
                    )
                )


                st.session_state.analysis_result = {
                    "churn_risk": churn_risk,
                    "risk_level": risk_level,
                    "business_impact": business_impact_display,
                    "retention_priority": retention_priority_display,
                    "priority_level": priority_level,
                    "risk_factors": risk_factors,
                    "actions": actions,
                    "decision_title": decision_title,
                    "decision_text": decision_text,
                    "customer_inputs": {
                        "tenure": tenure,
                        "contract": contract,
                        "payment_method": payment_method,
                        "paperless_billing": paperless_billing,
                        "monthly_charges": monthly_charges,
                        "total_charges": total_charges
                    }
                }

                st.session_state.analysis_history.append(
                    st.session_state.analysis_result.copy()
                )


                # Separate results page
                st.session_state.page = "results"
                st.rerun()



# ============================================================
# DASHBOARD PAGE
# ============================================================

elif st.session_state.page == "dashboard":

    ar = st.session_state.language == "العربية"

    render_html(f"""
    <div style="padding:25px 4px 12px;">
        <div class="page-kicker">{"NAVIGATE / لوحة المعلومات" if ar else "NAVIGATE / DASHBOARD"}</div>
        <div class="page-title">{"لوحة المعلومات" if ar else "Analysis dashboard"}</div>
        <div class="page-copy">
            {"ملخص للتحليلات التي أجريتها خلال الجلسة الحالية." if ar else
             "A session-level overview of the customer analyses completed in NAVIGATE."}
        </div>
    </div>
    """)

    history = st.session_state.analysis_history

    if not history:
        st.info(
            "لا توجد تحليلات في هذه الجلسة بعد. ابدأ بتحليل عميل لعرض البيانات هنا."
            if ar else
            "No analyses have been completed in this session yet. Analyze a customer to populate the dashboard."
        )
        if st.button("ابدأ تحليل عميل" if ar else "Analyze a Customer", use_container_width=False):
            st.session_state.page = "form"
            st.rerun()
    else:
        count = len(history)
        avg_risk = sum(x["churn_risk"] for x in history) / count
        avg_priority = sum(x["retention_priority"] for x in history) / count
        critical = sum(1 for x in history if x["priority_level"] == "Critical")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("التحليلات" if ar else "Analyses", count)
        d2.metric("متوسط الخطر" if ar else "Avg. Churn Risk", f"{avg_risk:.1f}%")
        d3.metric("متوسط الأولوية" if ar else "Avg. Priority", f"{avg_priority:.1f}")
        d4.metric("حالات حرجة" if ar else "Critical Cases", critical)

        dashboard_df = pd.DataFrame([
            {
                ("الخطر" if ar else "Churn Risk"): x["churn_risk"],
                ("تأثير الأعمال" if ar else "Business Impact"): x["business_impact"],
                ("الأولوية" if ar else "Retention Priority"): x["retention_priority"],
                ("المستوى" if ar else "Priority Level"): translate_level(x["priority_level"]),
            }
            for x in history
        ])

        st.write("")
        st.dataframe(dashboard_df, use_container_width=True, hide_index=True)

        st.caption(
            "هذه اللوحة تعرض تحليلات الجلسة الحالية فقط ولا تمثل قاعدة بيانات دائمة."
            if ar else
            "This dashboard currently reflects the active session only and is not persistent storage."
        )


# ============================================================
# ABOUT PAGE
# ============================================================

elif st.session_state.page == "about":

    ar = st.session_state.language == "العربية"

    if ar:
        about_title = "عن NAVIGATE"
        about_copy = (
            "NAVIGATE منصة لدعم قرارات الاحتفاظ بالعملاء. "
            "تستخدم نموذج تعلم آلي لتقدير خطر فقدان العميل، ثم تجمع الخطر "
            "مع تأثير العميل النسبي على الأعمال لتحديد أولوية الاحتفاظ."
        )
        how_title = "كيف يعمل؟"
        how_items = [
            "إدخال بيانات العميل الأساسية.",
            "تقدير خطر فقدان العميل باستخدام النموذج.",
            "حساب التأثير النسبي وأولوية الاحتفاظ.",
            "عرض عوامل للمراجعة وإجراءات مقترحة.",
            "إمكانية تحميل تقرير PDF لنتيجة التحليل."
        ]
        tech_title = "التقنيات"
        limitation_title = "مهم"
        limitation = (
            "جودة النتائج تعتمد على بيانات التدريب والنموذج. "
            "NAVIGATE يدعم القرار ولا يضمن سلوك العميل أو نتيجة أي إجراء احتفاظ."
        )
    else:
        about_title = "About NAVIGATE"
        about_copy = (
            "NAVIGATE is a customer-retention decision-support platform. "
            "It uses a machine-learning model to estimate churn risk and combines "
            "that estimate with relative customer business impact to determine retention priority."
        )
        how_title = "How it works"
        how_items = [
            "Enter basic customer information.",
            "Estimate churn risk using the trained model.",
            "Calculate relative business impact and retention priority.",
            "Review relevant factors and suggested actions.",
            "Download a PDF report of the analysis."
        ]
        tech_title = "Technology"
        limitation_title = "Important"
        limitation = (
            "Result quality depends on the training data and model. "
            "NAVIGATE supports decision-making and does not guarantee customer behavior "
            "or the outcome of any retention action."
        )

    render_html(f"""
    <div class="glass-panel" style="margin-top:28px;">
        <div class="page-kicker">NAVIGATE</div>
        <div class="page-title">{about_title}</div>
        <div class="page-copy" style="max-width:900px;">{about_copy}</div>

        <div style="margin-top:32px;color:#79172b;font-weight:850;font-size:1.05rem;">
            {how_title}
        </div>
        <div style="margin-top:14px;color:#835761;line-height:1.9;font-size:.82rem;">
            {"<br>".join("• " + x for x in how_items)}
        </div>

        <div style="margin-top:30px;color:#79172b;font-weight:850;font-size:1.05rem;">
            {tech_title}
        </div>
        <div style="margin-top:10px;color:#835761;font-size:.82rem;">
            Python · scikit-learn · pandas · Streamlit
        </div>

        <div class="quick-card" style="margin-top:30px;">
            <div class="quick-icon">ⓘ</div>
            <div>
                <div class="quick-title">{limitation_title}</div>
                <div class="quick-copy">{limitation}</div>
            </div>
        </div>
    </div>
    """)


# ============================================================
# RESULTS PAGE
# ============================================================

elif st.session_state.page == "results":

    if st.session_state.analysis_result is None:
        st.session_state.page = "form"
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

    risk_factors = result[
        "risk_factors"
    ]

    actions = result[
        "actions"
    ]

    decision_title = result[
        "decision_title"
    ]

    decision_text = result[
        "decision_text"
    ]


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

        c1, c2, c3 = st.columns(
            3,
            gap="medium"
        )


        # PRIORITY
        with c1:

            badge = get_badge_class(
                priority_level
            )

            render_html(f"""
            <div class="metric-grid-card">

                <div class="metric-top">

                    <div class="metric-heading">

                        <div class="metric-icon">
                            ◎
                        </div>

                        <div class="metric-label">
                            Retention Priority Score
                        </div>

                    </div>

                    <span class="badge badge-{badge}">
                        {html.escape(priority_level)}
                    </span>

                </div>

                <div class="metric-value">

                    {retention_priority:.2f}

                    <span class="metric-unit">
                        / 100
                    </span>

                </div>

                <div class="progress-track">

                    <div
                        class="progress-fill"
                        style="width:{retention_priority}%">
                    </div>

                </div>

                <div class="metric-note">

                    Indicates how strongly this customer
                    should be prioritized for retention.

                </div>

            </div>
            """)


        # CHURN
        with c2:

            badge = get_badge_class(
                risk_level
            )

            render_html(f"""
            <div class="metric-grid-card">

                <div class="metric-top">

                    <div class="metric-heading">

                        <div class="metric-icon">
                            ◇
                        </div>

                        <div class="metric-label">
                            Estimated Churn Risk
                        </div>

                    </div>

                    <span class="badge badge-{badge}">
                        {html.escape(risk_level)}
                    </span>

                </div>

                <div class="metric-value">
                    {churn_risk:.2f}%
                </div>

                <div class="progress-track">

                    <div
                        class="progress-fill"
                        style="width:{churn_risk}%">
                    </div>

                </div>

                <div class="metric-note">

                    Model-estimated likelihood of the
                    customer leaving.

                </div>

            </div>
            """)


        # IMPACT
        with c3:

            render_html(f"""
            <div class="metric-grid-card">

                <div class="metric-top">

                    <div class="metric-heading">

                        <div class="metric-icon">
                            ◉
                        </div>

                        <div class="metric-label">
                            Customer Business Impact
                        </div>

                    </div>

                </div>

                <div class="metric-value">

                    {business_impact:.2f}

                    <span class="metric-unit">
                        / 100
                    </span>

                </div>

                <div class="progress-track">

                    <div
                        class="progress-fill"
                        style="width:{business_impact}%">
                    </div>

                </div>

                <div class="metric-note">

                    Relative customer value based on
                    monthly and total payments.

                </div>

            </div>
            """)


        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        render_html(f"""
        <div class="decision-card">

            <div class="decision-head">

                <div class="decision-icon">
                    ◇
                </div>

                <div class="decision-label">
                    Decision Summary
                </div>

            </div>

            <div class="decision-title">
                {html.escape(decision_title)}
            </div>

            <div class="decision-text">
                {html.escape(decision_text)}
            </div>

        </div>
        """)


        st.write("")


        factor_col, action_col = st.columns(
            [1, 1.35],
            gap="medium"
        )


        # ----------------------------------------------------
        # FACTORS
        # ----------------------------------------------------

        with factor_col:

            if risk_factors:

                factor_html = ""

                for factor in risk_factors:

                    factor_html += f"""
                    <div class="risk-item">

                        <div class="risk-dot"></div>

                        <div>
                            {html.escape(factor)}
                        </div>

                    </div>
                    """

            else:

                factor_html = """
                <div class="risk-item">

                    <div class="risk-dot"></div>

                    <div>
                        No major actionable customer
                        signals identified
                    </div>

                </div>
                """


            render_html(f"""
            <div class="detail-card">

                <div class="detail-header">

                    <div class="detail-icon">
                        ▣
                    </div>

                    <div class="detail-title">
                        Factors to Review
                    </div>

                </div>

                <div class="detail-copy">

                    Customer characteristics that may
                    be useful when reviewing retention risk.

                </div>

                {factor_html}

            </div>
            """)


        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        with action_col:

            action_html = ""

            for i, action in enumerate(
                actions,
                start=1
            ):

                action_html += f"""
                <div class="action-item">

                    <div class="action-number">
                        {i:02d}
                    </div>

                    <div class="action-text">
                        {html.escape(action)}
                    </div>

                </div>
                """


            render_html(f"""
            <div class="detail-card">

                <div class="detail-header">

                    <div class="detail-icon">
                        ◎
                    </div>

                    <div class="detail-title">
                        Suggested Actions
                    </div>

                </div>

                <div class="detail-copy">

                    Suggested actions based on the
                    customer information entered.

                </div>

                {action_html}

            </div>
            """)



        st.write("")

        pdf_bytes = create_pdf_report(result)
        st.download_button(
            label=(
                "↓ تحميل تقرير التحليل PDF"
                if st.session_state.language == "العربية"
                else "↓ Download Analysis Report (PDF)"
            ),
            data=pdf_bytes,
            file_name="NAVIGATE_customer_retention_analysis.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_analysis_pdf"
        )

        render_html("""
        <div class="note">

            NAVIGATE is a decision-support prototype.
            Scores are based on the trained model and project
            scoring assumptions. Suggested actions are
            rule-based recommendations and do not guarantee
            a specific customer outcome.

        </div>
        """)
