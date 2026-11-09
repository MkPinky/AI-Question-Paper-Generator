import os
import re
from io import BytesIO
from html import escape as html_escape
from xml.sax.saxutils import escape as xml_escape

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Preformatted
)

APP_TITLE = "AI-Powered Question Paper Generator"
APP_VERSION = "5.3"
load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
BLOOM_LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")


# ----------------------------- PDF extraction -----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract selectable text from every page of an uploaded PDF."""
    try:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError("This PDF is password-protected.") from exc

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[Page {page_number}]\n{page_text.strip()}")

        extracted = "\n\n".join(pages).strip()
        if not extracted:
            raise ValueError(
                "No selectable text was found. This may be a scanned/image-only PDF. "
                "Please upload a text-based PDF."
            )
        return extracted
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read this PDF: {exc}") from exc


# ----------------------------- Generation -----------------------------
def build_prompt(
    syllabus: str,
    subject: str,
    difficulty: str,
    question_types: list[str],
    num_questions: int,
    total_marks: int,
    duration: str,
) -> str:
    syllabus = syllabus[:24000]
    types_text = ", ".join(question_types)

    return f"""
You are an experienced examination setter. Create a formal question paper using ONLY the
syllabus text supplied below.

NON-NEGOTIABLE SOURCE RULES:
- Treat the supplied syllabus as the only source of examinable subject content.
- Every question must be directly traceable to a topic, concept, term, or learning outcome in it.
- Do not introduce unrelated programming languages, libraries, tools, platforms, or examples.
- Do not infer the subject syllabus from the subject name. The uploaded syllabus is authoritative.
- If the syllabus is too short to support the requested number of distinct questions, say so clearly
  in a short note rather than inventing topics.
- Do not write questions about the process of uploading, reading, or generating this paper.

EXAM SETTINGS
Subject: {subject}
Difficulty: {difficulty}
Question types: {types_text}
Exact number of questions: {num_questions}
Exact total marks: {total_marks}
Duration: {duration}

SYLLABUS (source of truth):
----- BEGIN SYLLABUS -----
{syllabus}
----- END SYLLABUS -----

PAPER REQUIREMENTS:
1. Create exactly {num_questions} numbered questions, numbered 1 to {num_questions}.
2. Allocate marks so the sum is exactly {total_marks}; show marks for every question.
3. State Total Marks: {total_marks} and Duration: {duration}.
4. Follow the requested question types. For MCQs, provide options A, B, C, D.
5. Every MCQ must have exactly ONE unambiguously correct answer. Distractors must be incorrect.
6. Do not make two options equivalent or both correct.
7. Add exactly one Bloom label to each question: [Bloom: Remember], [Bloom: Understand],
   [Bloom: Apply], [Bloom: Analyze], [Bloom: Evaluate], or [Bloom: Create].
8. Use clear, grammatically correct, examination-style language.
9. Do not include answers or hints in the student paper.
10. Create a separate answer key with the correct answer and a concise explanation or marking points
    for every question.
11. The answer key must contain exactly one entry for each question.
12. Check your arithmetic, answer key, marks total, and syllabus relevance before responding.
13. Do not claim the output is guaranteed error-free.

Return exactly these two sections:

STUDENT QUESTION PAPER
[Formal student paper]

TEACHER ANSWER KEY
[Answer key]
""".strip()


def generate_paper(
    syllabus: str,
    subject: str,
    difficulty: str,
    question_types: list[str],
    num_questions: int,
    total_marks: int,
    duration: str,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. Add it to your .env file or Streamlit Cloud secrets."
        )

    prompt = build_prompt(
        syllabus, subject, difficulty, question_types,
        num_questions, total_marks, duration
    )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful academic assessment designer. "
                    "Obey source restrictions and formatting instructions. "
                    "Never invent syllabus content."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_completion_tokens=6000,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Groq returned an empty response. Please try again.")
    return content.strip()


# ----------------------------- Parsing and checks -----------------------------
def _normalise_heading(line: str) -> str:
    """Normalize Markdown-wrapped section headings returned by an LLM."""
    value = line.strip()
    value = re.sub(r"^\s{0,3}#{1,6}\s*", "", value)
    value = re.sub(r"^\s*[*_`~]+|[*_`~]+\s*$", "", value)
    value = re.sub(r"[*_`]", "", value)
    value = re.sub(r"[:\-\s]+$", "", value)
    return re.sub(r"\s+", " ", value).strip().upper()


def extract_sections(text: str) -> tuple[str, str]:
    """
    Split the model response robustly, including headings wrapped in Markdown
    bold/italics or preceded by Markdown heading markers.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    student_start = None
    answer_start = None
    answer_heading_end = None

    for index, line in enumerate(lines):
        heading = _normalise_heading(line)
        if heading in {"STUDENT QUESTION PAPER", "QUESTION PAPER"}:
            if student_start is None:
                student_start = index + 1
        elif heading in {"TEACHER ANSWER KEY", "ANSWER KEY", "TEACHER KEY"}:
            answer_start = index
            answer_heading_end = index + 1
            break

    if answer_start is not None:
        student_lines = lines[student_start:answer_start] if student_start is not None else lines[:answer_start]
        answer_lines = lines[answer_heading_end:]
        student = "\n".join(student_lines).strip()
        answer_key = "\n".join(answer_lines).strip()
        # Remove separator-only lines at section boundaries.
        student = re.sub(r"(?m)^\s*(?:---+|___+|\*\*\*+)\s*$", "", student).strip()
        answer_key = re.sub(r"(?m)^\s*(?:---+|___+|\*\*\*+)\s*$", "", answer_key).strip()
        return student, answer_key

    # If only a student heading exists, do not pretend the answer key was found.
    if student_start is not None:
        return "\n".join(lines[student_start:]).strip(), ""
    return text.strip(), ""


def count_numbered_questions(text: str) -> int:
    return len(re.findall(r"(?m)^\s*(\d{1,2})[.)]\s+", text))


def declared_total_marks(text: str):
    match = re.search(
        r"(?im)^\s*(?:total\s+marks|maximum\s+marks|full\s+marks)\s*[:\-]?\s*(\d+)\b",
        text,
    )
    return int(match.group(1)) if match else None


def validate_question_paper(
    student_paper: str,
    answer_key: str,
    expected_questions: int,
    expected_marks: int,
) -> list[str]:
    issues = []
    if not student_paper.strip():
        return ["The student question paper is empty."]
    if not answer_key.strip():
        issues.append("The teacher answer key is missing.")

    question_count = count_numbered_questions(student_paper)
    if question_count != expected_questions:
        issues.append(f"Expected {expected_questions} questions, but detected {question_count}.")

    question_blocks = re.split(r"(?m)^\s*\d{1,2}[.)]\s+", student_paper)[1:]
    for index, block in enumerate(question_blocks, start=1):
        if not re.search(
            r"\[?\s*Bloom\s*:\s*(Remember|Understand|Apply|Analyze|Analyse|Evaluate|Create)\s*\]?",
            block, re.IGNORECASE
        ):
            issues.append(f"Question {index} may be missing a Bloom's Taxonomy label.")

    marks = declared_total_marks(student_paper)
    if marks is None:
        issues.append("The paper does not clearly state its total marks.")
    elif marks != expected_marks:
        issues.append(f"Expected {expected_marks} total marks; paper declares {marks}.")

    key_count = count_numbered_questions(answer_key)
    if answer_key.strip() and key_count and key_count != expected_questions:
        issues.append(f"The answer key has {key_count} numbered entries; expected {expected_questions}.")

    # Check MCQ option structure when option labels are present.
    student_blocks = re.split(r"(?m)^\s*\d{1,2}[.)]\s+", student_paper)[1:]
    for index, block in enumerate(student_blocks, start=1):
        option_labels = re.findall(r"(?m)^\s*([A-D])[.)]\s+", block, re.IGNORECASE)
        if option_labels and sorted(label.upper() for label in option_labels) != ["A", "B", "C", "D"]:
            issues.append(
                f"Question {index} should have exactly one each of options A, B, C and D."
            )

    # Check explicit per-question mark labels and compare their sum with requested marks.
    mark_values = re.findall(r"\[\s*(\d+)\s*marks?\s*\]", student_paper, re.IGNORECASE)
    if len(mark_values) == expected_questions:
        mark_sum = sum(map(int, mark_values))
        if mark_sum != expected_marks:
            issues.append(f"Individual question marks add up to {mark_sum}, not {expected_marks}.")
    else:
        issues.append("Could not reliably read a mark allocation for every question.")

    # Detect explicit marking-scheme totals that contradict the marks assigned in the paper.
    # This is a text-pattern check; it does not judge the academic quality of the rubric.
    if answer_key.strip():
        declared_question_marks = re.findall(
            r"(?im)^\s*\d{1,2}[.)]\s+.*?\((\d+)\s*marks?\)",
            student_paper
        )
        if not declared_question_marks:
            declared_question_marks = re.findall(
                r"(?im)^\s*\d{1,2}[.)]\s+.*?\[(\d+)\s*marks?\]",
                student_paper
            )
        key_totals = re.findall(
            r"(?i)\btotal\s+(\d+)\s*,?\s*but\s+only\s+(\d+)\s+marks?\s+allocated",
            answer_key
        )
        for key_total, allocated_total in key_totals:
            if key_total != allocated_total:
                issues.append(
                    f"Answer-key marking scheme totals {key_total} marks, "
                    f"but the question is allocated {allocated_total} marks. Review the rubric."
                )

    if not issues:
        issues.append(
            "Structural checks passed. A teacher should still verify subject accuracy and every answer."
        )
    return issues


# ----------------------------- Professional PDF layout -----------------------------
def _clean_line(line: str) -> str:
    """Remove common Markdown wrappers while preserving readable text."""
    line = line.strip()
    # Preserve Markdown code-fence delimiters so the PDF renderer can detect code blocks.
    if line.startswith("```"):
        return line
    line = re.sub(r"^\s*[-*]\s+", "• ", line)
    line = re.sub(r"`([^`]*)`", r"\1", line)
    line = line.replace("**", "").replace("__", "")
    # Preserve literal asterisks used in C pointers and multiplication (e.g. int *ptr; a * b).
    # Strip italic markers only when they wrap the entire line.
    if len(line) >= 2 and line.startswith("*") and line.endswith("*") and not line.startswith("**"):
        line = line[1:-1].strip()
    replacements = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u00d7": "x", "\u2022": "-", "\u00a0": " ",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    for old_char, new_char in replacements.items():
        line = line.replace(old_char, new_char)
    return line.strip()


def split_inline_mcq_options(line: str) -> list[str]:
    """Split A) ... B) ... C) ... D) options returned on a single line."""
    matches = list(re.finditer(r"(?<!\w)([A-D])\s*[).]\s*", line, re.IGNORECASE))
    if len(matches) < 2 or line[:matches[0].start()].strip():
        return [line]
    parts = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        part = line[match.start():end].strip()
        if part:
            parts.append(part)
    return parts if len(parts) >= 2 else [line]



def _format_pdf_inline(text: str) -> str:
    """Escape text and render a small safe subset of Markdown emphasis."""
    safe = xml_escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"\*(Explanation:|Marking points:|Answer:)\*", r"<b>\1</b>", safe, flags=re.IGNORECASE)
    return safe


def create_exam_pdf(
    document_title: str,
    content: str,
    subject: str,
    duration: str = "",
    total_marks: int | None = None,
    institution: str = "QUESTION PAPER",
) -> bytes:
    """Render a clean, print-ready A4 exam document with structured question blocks."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=19 * mm, leftMargin=19 * mm,
        topMargin=16 * mm, bottomMargin=19 * mm,
        title=document_title,
        author="AI-Powered Question Paper Generator",
    )
    styles = getSampleStyleSheet()
    institution_style = ParagraphStyle(
        "Institution", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8.5, leading=11, alignment=TA_CENTER,
        textColor=colors.HexColor("#526477"), spaceAfter=2 * mm,
    )
    title_style = ParagraphStyle(
        "ExamTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=17, leading=21, alignment=TA_CENTER,
        textColor=colors.HexColor("#142D4E"), spaceAfter=2 * mm,
    )
    subject_style = ParagraphStyle(
        "Subject", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=11, leading=15, alignment=TA_CENTER,
        textColor=colors.HexColor("#176B87"), spaceAfter=5 * mm,
    )
    body_style = ParagraphStyle(
        "QuestionBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.5, leading=14, alignment=TA_LEFT, spaceAfter=2 * mm,
    )
    option_style = ParagraphStyle(
        "Option", parent=body_style, leftIndent=8 * mm, firstLineIndent=0,
        spaceAfter=1.2 * mm,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9, leading=12,
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8, leading=11, textColor=colors.HexColor("#526477"),
    )
    subquestion_style = ParagraphStyle(
        "SubQuestion", parent=body_style, leftIndent=7 * mm,
        firstLineIndent=-6 * mm, spaceAfter=2 * mm,
    )
    code_style = ParagraphStyle(
        "CodeBlock", parent=styles["Code"], fontName="Courier",
        fontSize=7.8, leading=10, leftIndent=4 * mm, rightIndent=2 * mm,
        borderColor=colors.HexColor("#D8E1EA"), borderWidth=0.5,
        borderPadding=5, backColor=colors.HexColor("#F4F7FA"),
        spaceBefore=2 * mm, spaceAfter=3 * mm,
    )


    story = [
        Paragraph(xml_escape(institution), institution_style),
        Paragraph(xml_escape(document_title.upper()), title_style),
        Paragraph(xml_escape(subject), subject_style),
    ]
    meta_table = Table(
        [[
            Paragraph(f"<b>Duration:</b> {xml_escape(duration or 'As specified')}", meta_style),
            Paragraph(f"<b>Total Marks:</b> {total_marks if total_marks is not None else 'As specified'}", meta_style),
        ]],
        colWidths=[86 * mm, 86 * mm],
    )
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF3F8")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([meta_table, Spacer(1, 4 * mm)])

    if document_title.lower().startswith("student"):
        identity = Table(
            [[
                Paragraph("<b>Name:</b> ____________________________________", meta_style),
                Paragraph("<b>Roll No.:</b> __________________", meta_style),
            ]],
            colWidths=[105 * mm, 67 * mm],
        )
        identity.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(identity)

    # Build content in logical blocks so a question and its options stay together.
    # Repeated metadata is omitted because the document header already displays it.
    lines = [_clean_line(raw) for raw in content.splitlines()]
    lines = [line for line in lines if line and not re.fullmatch(r"(?:---+|___+|\*\*\*+)", line)]

    filtered_lines = []
    for line in lines:
        normalized = _normalise_heading(line)
        if normalized in {
            "STUDENT QUESTION PAPER", "TEACHER ANSWER KEY",
            "QUESTION PAPER", "ANSWER KEY",
        }:
            continue
        if re.match(
            r"^(?:Subject|Difficulty|Question Types?|Question Formats?|Total Marks|Maximum Marks|Duration)\s*:",
            line, re.IGNORECASE
        ):
            continue
        filtered_lines.append(line)

    question_start = re.compile(r"^\d{1,2}[.)]\s+")

    def make_line_flowable(line):
        safe_line = _format_pdf_inline(line)
        if re.match(r"^[A-D][.)]\s+", line, re.IGNORECASE):
            return Paragraph(safe_line, option_style)
        if re.match(r"^[a-z][.)]\\s+", line, re.IGNORECASE):
            return Paragraph(safe_line, subquestion_style)
        if re.match(r"^(?:\[?\s*Bloom\s*:|Marks?\s*:|\[\s*\d+\s*marks?)", line, re.IGNORECASE):
            return Paragraph(f"<font color='#526477'><i>{safe_line}</i></font>", label_style)
        if re.match(r"^(?:Correct answer|Answer|Marking points)\s*:", line, re.IGNORECASE):
            return Paragraph(f"<b>{safe_line}</b>", body_style)
        if line.lower().startswith(("instructions:", "note:", "section ")):
            return Paragraph(f"<b>{safe_line}</b>", body_style)
        if line.startswith("• "):
            return Paragraph(safe_line, option_style)
        return Paragraph(safe_line, body_style)

    blocks = []
    current_block = []
    for line in filtered_lines:
        if question_start.match(line):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        elif current_block:
            current_block.append(line)
        else:
            blocks.append([line])
    if current_block:
        blocks.append(current_block)

    for block in blocks:
        flowables = []
        is_numbered_entry = bool(question_start.match(block[0]))
        if is_numbered_entry:
            flowables.append(Spacer(1, 2 * mm))
        line_index = 0
        while line_index < len(block):
            line = block[line_index]

            # Render fenced source code as a shaded monospace block.
            if line.strip().startswith("```"):
                code_lines = []
                next_index = line_index + 1
                while next_index < len(block) and not block[next_index].strip().startswith("```"):
                    code_lines.append(block[next_index].expandtabs(4))
                    next_index += 1
                if code_lines:
                    code_text = "\n".join(code_lines).rstrip()
                    flowables.append(Preformatted(code_text, code_style, maxLineLength=92))
                line_index = next_index + 1 if next_index < len(block) else len(block)
                continue

            # Join wrapped continuation lines belonging to a lettered subquestion.
            if re.match(r"^[a-z][.)]\s+", line, re.IGNORECASE):
                sub_lines = [line]
                next_index = line_index + 1
                while next_index < len(block):
                    following = block[next_index]
                    if (
                        re.match(r"^[a-z][.)]\s+", following, re.IGNORECASE)
                        or re.match(r"^(?:\[?\s*Bloom\s*:|Marks?\s*:)", following, re.IGNORECASE)
                        or re.match(r"^[A-D][.)]\s+", following, re.IGNORECASE)
                        or following.strip().startswith("```")
                    ):
                        break
                    sub_lines.append(following)
                    next_index += 1
                merged = " ".join(part.strip() for part in sub_lines)
                flowables.append(Paragraph(_format_pdf_inline(merged), subquestion_style))
                line_index = next_index
                continue

            inline_options = split_inline_mcq_options(line)
            if len(inline_options) > 1:
                flowables.extend(Paragraph(_format_pdf_inline(option), option_style) for option in inline_options)
            elif line_index == 0 and is_numbered_entry:
                flowables.append(Paragraph(f"<b>{_format_pdf_inline(line)}</b>", body_style))
            else:
                flowables.append(make_line_flowable(line))
            line_index += 1

        if is_numbered_entry and flowables:
            story.append(KeepTogether(flowables))
        else:
            story.extend(flowables)

    def page_footer(canvas, document):
        canvas.saveState()
        width, _ = A4
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.setLineWidth(0.5)
        canvas.line(19 * mm, 13 * mm, width - 19 * mm, 13 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(19 * mm, 8 * mm, "Generated draft - Verify before examination use")
        canvas.drawRightString(width - 19 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return buffer.getvalue()


# ----------------------------- Streamlit UI -----------------------------
def inject_styles():
    st.markdown("""
    <style>
    .stApp { background: #f4f7fb; }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px; color: white;
        background: linear-gradient(115deg, #102a43 0%, #176b87 58%, #2c7da0 100%);
        margin-bottom: 1.2rem; box-shadow: 0 10px 28px rgba(16,42,67,.13);
    }
    .hero h1 { color: white; font-size: 2rem; margin: 0 0 .35rem 0; }
    .hero p { color: #e6f1f7; margin: 0; font-size: 1rem; }
    .section-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1rem 1.2rem; margin: .6rem 0 1rem 0;
    }
    .paper-preview {
        background: white; border: 1px solid #dbe3ec; border-radius: 10px;
        padding: 1.6rem 1.8rem; color: #1f2937; line-height: 1.65;
        box-shadow: 0 5px 18px rgba(15,23,42,.05);
        white-space: pre-wrap; font-family: Georgia, 'Times New Roman', serif;
    }
    div[data-testid="stDownloadButton"] button {
        border-radius: 9px; font-weight: 600; min-height: 2.8rem;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")
    inject_styles()

    st.markdown(
        '<div class="hero"><h1>AI Question Paper Generator</h1>'
        '<p>Create structured examination papers and separate teacher answer keys from your syllabus.</p></div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Teacher review is required before an examination. AI-generated questions and answers "
        "may contain mistakes; verify subject accuracy, marks, and answer keys."
    )

    with st.sidebar:
        st.header("⚙️ Examination settings")
        subject = st.text_input("Subject", value="C Programming")
        difficulty = st.selectbox("Difficulty level", ["Easy", "Medium", "Hard", "Mixed"])
        question_types = st.multiselect(
            "Question formats",
            ["MCQs", "Short Answer", "Long Answer"],
            default=["MCQs", "Short Answer", "Long Answer"],
        )
        num_questions = st.slider("Number of questions", 5, 20, 5)
        total_marks = st.selectbox("Total marks", [20, 30, 50, 60, 100], index=0)
        duration = st.selectbox("Examination duration", ["30 minutes", "1 hour", "2 hours", "3 hours"])
        st.caption(f"AI model: {GROQ_MODEL}")

    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.subheader("📚 Syllabus source")
        uploaded_file = st.file_uploader(
            "Upload your syllabus PDF", type=["pdf"],
            help="Use a text-based PDF. Scanned image-only PDFs need OCR before they can be read.",
        )
        typed_syllabus = st.text_area(
            "Optional: paste syllabus text",
            height=110,
            placeholder="Only use this if you want to provide or supplement the PDF syllabus.",
        )
    with right:
        st.subheader("How it works")
        st.markdown(
            '<div class="section-card"><b>1. Upload</b><br>Provide a syllabus PDF.<br><br>'
            '<b>2. Configure</b><br>Select subject, difficulty, format and marks.<br><br>'
            '<b>3. Generate</b><br>Review the student paper and teacher key, then download print-ready PDFs.</div>',
            unsafe_allow_html=True,
        )

    syllabus_text = ""
    if uploaded_file is not None:
        try:
            pdf_text = extract_text_from_pdf(uploaded_file)
            st.success(f"PDF read successfully: {uploaded_file.name}")
            with st.expander("Preview extracted syllabus (verify before generating)", expanded=False):
                st.text(pdf_text[:7000])
                if len(pdf_text) > 7000:
                    st.caption("Preview shortened. The full extracted text (up to the prompt limit) is used.")
            syllabus_text = pdf_text
        except ValueError as exc:
            st.error(str(exc))

    # Pasted text intentionally overrides PDF only when the user supplies non-empty text.
    if typed_syllabus.strip():
        syllabus_text = typed_syllabus.strip()
        if uploaded_file is not None:
            st.warning("Pasted syllabus text is being used instead of the uploaded PDF.")

    if uploaded_file is None and not typed_syllabus.strip():
        st.caption("Upload a syllabus PDF to begin. You do not need to paste the syllabus separately.")

    generate = st.button("✨ Generate Question Paper", type="primary", use_container_width=True)
    if generate:
        if not subject.strip():
            st.error("Please enter a subject.")
        elif not question_types:
            st.error("Select at least one question format.")
        elif not syllabus_text.strip():
            st.error("Upload a readable syllabus PDF or provide syllabus text.")
        else:
            try:
                with st.spinner(f"Generating with Groq ({GROQ_MODEL})…"):
                    raw_output = generate_paper(
                        syllabus_text, subject.strip(), difficulty,
                        question_types, num_questions, total_marks, duration
                    )
                student_paper, answer_key = extract_sections(raw_output)
                issues = validate_question_paper(
                    student_paper, answer_key, num_questions, total_marks
                )
                st.session_state["student_paper"] = student_paper
                st.session_state["answer_key"] = answer_key
                st.session_state["paper_subject"] = subject.strip()
                st.session_state["paper_duration"] = duration
                st.session_state["paper_marks"] = total_marks
                st.session_state["paper_questions"] = num_questions
                st.session_state["validation_issues"] = issues
                st.session_state["raw_output"] = raw_output
                st.session_state["syllabus_filename"] = (
                    uploaded_file.name if uploaded_file is not None and not typed_syllabus.strip()
                    else "Pasted syllabus text"
                )
                st.success("Question paper generated.")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

    if "student_paper" in st.session_state:
        student_paper = st.session_state["student_paper"]
        answer_key = st.session_state["answer_key"]
        saved_subject = st.session_state["paper_subject"]
        saved_duration = st.session_state["paper_duration"]
        saved_marks = st.session_state["paper_marks"]

        st.divider()
        st.header("📄 Generated examination documents")
        st.caption(
            f"Subject: {saved_subject}  ·  {st.session_state['paper_questions']} questions  ·  "
            f"{saved_marks} marks  ·  {saved_duration}"
        )

        paper_tab, key_tab, quality_tab = st.tabs(
            ["Student question paper", "Teacher answer key", "Quality checks"]
        )
        with paper_tab:
            st.markdown("#### Student copy")
            st.markdown(
                f'<div class="paper-preview">{html_escape(student_paper)}</div>',
                unsafe_allow_html=True,
            )
            pdf_data = create_exam_pdf(
                "Student Question Paper", student_paper, saved_subject,
                saved_duration, saved_marks, "STUDENT COPY",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "⬇ Download Student Paper · PDF",
                    data=pdf_data, file_name="student_question_paper.pdf",
                    mime="application/pdf", use_container_width=True,
                )
            with col_b:
                st.download_button(
                    "⬇ Download Student Paper · TXT",
                    data=student_paper, file_name="student_question_paper.txt",
                    mime="text/plain", use_container_width=True,
                )

        with key_tab:
            st.markdown("#### Confidential teacher copy")
            if not answer_key.strip():
                st.error("The answer-key section was not detected. Generate the paper again; do not distribute this output.")
            else:
                st.markdown(
                    f'<div class="paper-preview">{html_escape(answer_key)}</div>',
                    unsafe_allow_html=True,
                )
            key_pdf = create_exam_pdf(
                "Teacher Answer Key", answer_key, saved_subject,
                saved_duration, saved_marks, "CONFIDENTIAL • TEACHER COPY",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "⬇ Download Answer Key · PDF",
                    data=key_pdf, file_name="teacher_answer_key.pdf",
                    mime="application/pdf", use_container_width=True,
                )
            with col_b:
                st.download_button(
                    "⬇ Download Answer Key · TXT",
                    data=answer_key, file_name="teacher_answer_key.txt",
                    mime="text/plain", use_container_width=True,
                )

        with quality_tab:
            st.markdown("#### Automatic structural checks")
            for issue in st.session_state.get("validation_issues", []):
                if "passed" in issue.lower():
                    st.success(issue)
                else:
                    st.warning(issue)
            st.caption(
                "These checks verify document structure and some mark totals. They cannot independently "
                "prove that every subject answer is factually correct or that an MCQ has only one valid answer."
            )
            with st.expander("View complete raw Groq response"):
                st.text(st.session_state.get("raw_output", ""))

    st.divider()
    st.caption(f"Version {APP_VERSION} · AI provider: Groq · Review all generated content before using it in an examination.")


if __name__ == "__main__":
    main()