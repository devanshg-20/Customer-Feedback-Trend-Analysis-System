from flask import Flask, render_template, request, send_file
import pickle
import re
import io
import os
import pandas as pd
from datetime import datetime
import speech_recognition as sr
import pyttsx3

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

STRONG_NEGATIVE_PHRASES = [
    # value / money
    "waste of money", "not worth", "not worth the price",
    "not worth buying", "total waste", "complete waste",
    "waste of time", "rip off", "ripoff", "overpriced",
    # quality
    "very poor", "very bad", "really bad", "extremely bad",
    "poor build quality", "build quality is poor", "very poor build",
    "poor quality", "low quality", "terrible quality", "horrible quality",
    # usability
    "completely useless", "totally useless", "absolutely useless",
    "does not work", "doesnt work", "stopped working", "stopped work",
    "broke after", "dead on arrival", "doa",
    # regret / recommendation
    "do not buy", "dont buy", "don't buy", "avoid this", "avoid at all costs",
    "would not recommend", "wouldn't recommend", "cannot recommend",
    "returning this", "returned this", "asked for refund",
    # dissatisfaction
    "deeply disappointed", "very disappointed", "extremely disappointed",
    "not happy", "not satisfied", "not pleased",
    "biggest mistake", "worst purchase", "worst decision", "worst product",
    # other
    "not good at all", "not great at all", "absolutely terrible",
    "absolutely horrible", "absolutely awful",
]

# ── Strong-negative single words ──────────────────────────────────────────
STRONG_NEGATIVE_WORDS = {
    "useless", "worthless", "garbage", "junk", "trash", "rubbish",
    "hate", "hated", "awful", "terrible", "horrible", "dreadful",
    "worst", "atrocious", "abysmal", "appalling",
    "broken", "defective", "faulty", "fraudulent", "scam",
    "disappointing", "disappointed", "disgusted", "regret",
    "unusable", "malfunctioning", "pathetic", "disastrous",
}

# ── Regular-negative (used in keyword fallback) ───────────────────────────
NEGATIVE_WORDS = STRONG_NEGATIVE_WORDS | {
    "bad", "poor", "slow", "cheap", "flimsy", "fragile", "unreliable",
    "issue", "problem", "failure", "fail", "failed", "fails",
    "uncomfortable", "annoying", "frustrating", "frustration",
    "inadequate", "inferior", "mediocre", "deficient",
    "damage", "damaged", "scratch", "scratched", "broken",
}

# ── Negation words ────────────────────────────────────────────────────────
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "without",
    "cant", "cannot", "wont", "dont", "doesnt",
    "didn't", "doesn't", "won't", "can't", "don't", "isn't",
    "wasn't", "aren't", "weren't", "haven't", "hasn't",
    "couldn't", "wouldn't", "shouldn't",
}

# ── Strong-positive phrases ───────────────────────────────────────────────
STRONG_POSITIVE_PHRASES = [
    "absolutely love", "really love", "totally love",
    "highly recommend", "strongly recommend",
    "exceeded expectations", "beyond expectations", "exceeds expectations",
    "best purchase", "best product", "best buy",
    "very happy", "extremely happy", "very satisfied", "extremely satisfied",
    "works perfectly", "works great", "works flawlessly",
    "absolutely amazing", "absolutely fantastic", "absolutely brilliant",
    "great value", "great quality", "excellent quality", "excellent value",
    "worth every penny", "worth the price", "well worth",
    "very impressed", "really impressed", "extremely impressed",
    "best ever", "love this product", "love this item",
]

# ── Strong-positive single words ─────────────────────────────────────────
STRONG_POSITIVE_WORDS = {
    "excellent", "outstanding", "superb", "brilliant", "wonderful",
    "fantastic", "amazing", "awesome", "perfect", "flawless",
    "love", "loved", "recommend",
}

# ── Regular-positive ─────────────────────────────────────────────────────
POSITIVE_WORDS = STRONG_POSITIVE_WORDS | {
    "good", "great", "nice", "satisfied", "happy", "pleased",
    "comfortable", "reliable", "durable", "solid", "efficient",
    "effective", "helpful", "useful", "convenient", "impressive",
    "delightful", "enjoy", "enjoyed", "quality",
}

# ── Neutral ───────────────────────────────────────────────────────────────
NEUTRAL_WORDS = {
    "okay", "ok", "average", "fine", "decent", "normal",
    "moderate", "acceptable", "ordinary", "fair", "alright",
    "standard", "basic",
}

NEUTRAL_PHRASES = [
    "nothing special", "not bad", "does the job", "works fine",
    "could be better", "could be worse", "meets expectations",
    "as expected", "as described", "neither good nor bad",
    "average quality", "okay for the price", "works okay",
    "functions fine", "serves its purpose", "gets the job done",
]

# ── Product-domain keywords ───────────────────────────────────────────────
PRODUCT_KEYWORDS = {
    # product types
    "product", "item", "device", "gadget", "machine", "tool",
    "phone", "laptop", "tablet", "camera", "charger", "cable",
    "headphone", "earphone", "speaker", "keyboard", "mouse", "battery",
    "watch", "shoes", "shirt", "jacket", "bag", "bottle", "book",
    "appliance", "equipment", "accessory", "hardware", "software",
    # purchase / commerce
    "price", "cost", "value", "buy", "bought", "purchase", "purchased",
    "order", "ordered", "delivery", "shipped", "shipping",
    "return", "returned", "refund", "seller", "brand",
    "warranty", "packaging", "unboxed",
    # quality / experience
    "quality", "build", "material", "design", "size", "color",
    "colour", "durable", "durability", "comfortable", "comfort",
    "performance", "feature", "recommend", "review",
    "works", "working", "use", "used", "using",
}

# ── Extra vocab to merge into TF-IDF ─────────────────────────────────────
EXTRA_VOCAB = list(
    set(list(NEGATIVE_WORDS) + list(POSITIVE_WORDS) + list(NEUTRAL_WORDS) +
        STRONG_NEGATIVE_PHRASES + STRONG_POSITIVE_PHRASES + NEUTRAL_PHRASES +
        list(PRODUCT_KEYWORDS))
)


app = Flask(__name__)

# ── Load model ───────────────────────────────────────────────────────────────
model      = pickle.load(open('sentiment_model4.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer4.pkl', 'rb'))


# ── NLP helpers ──────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Lowercase → keep letters & spaces → collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_phrase(text: str, phrases) -> bool:
    return any(ph in text for ph in phrases)


def _has_word(text: str, word_set: set) -> bool:
    return bool(set(text.split()) & word_set)


def _negated_positive(text: str) -> bool:
    """
    Detects genuinely negative negations like 'not good', 'not worth', 'not happy'.
    Returns True ONLY when a strong positive word is negated — i.e. the reviewer
    is expressing clear dissatisfaction.

    Does NOT return True for:
      "not bad"      — double-negative → neutral/ok
      "not great"    — mild → neutral, not negative
      "not amazing"  — mild → neutral, not negative
      "not perfect"  — mild → neutral, not negative

    Returns True for:
      "not worth"    — clearly negative value judgement
      "not happy"    — clear dissatisfaction
      "not satisfied"— clear dissatisfaction
      "not good"     — clear negative
    """
    # Words that when negated produce a NEUTRAL (not negative) outcome
    MILD_POSITIVE_WORDS = {
        "great", "amazing", "perfect", "excellent", "fantastic",
        "outstanding", "brilliant", "superb", "flawless", "wonderful",
        "bad", "terrible", "awful",  # double-negative = neutral
    }

    # Words that when negated produce a CLEARLY NEGATIVE outcome
    STRONGLY_NEGATION_SENSITIVE = {
        "good", "happy", "satisfied", "pleased", "worth", "recommend",
        "reliable", "durable", "comfortable", "useful", "helpful",
        "effective", "efficient", "working", "works",
    }

    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok in NEGATION_WORDS and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            # Skip: "not bad/terrible/awful" → neutral double-negative
            # Skip: "not great/amazing/perfect/…" → mild, not strongly negative
            if nxt in MILD_POSITIVE_WORDS:
                continue
            # Only flip to Negative for words where negation = clear dissatisfaction
            if nxt in STRONGLY_NEGATION_SENSITIVE:
                return True
    return False


def is_strongly_negative(text: str) -> bool:
    """
    True if the text has any strong-negative signal:
      • a strong-negative phrase, OR
      • a strong-negative word, OR
      • a negated positive/neutral word
    """
    if _has_phrase(text, STRONG_NEGATIVE_PHRASES):
        return True
    if _has_word(text, STRONG_NEGATIVE_WORDS):
        return True
    if _negated_positive(text):
        return True
    return False


def is_strongly_positive(text: str) -> bool:
    """True if the text has clear positive signals."""
    if _has_phrase(text, STRONG_POSITIVE_PHRASES):
        return True
    hits = sum(1 for w in text.split() if w in STRONG_POSITIVE_WORDS)
    if hits >= 2:
        return True
    return False


def is_product_related(text: str) -> bool:
    """True if text contains at least one product-domain keyword."""
    if set(text.split()) & PRODUCT_KEYWORDS:
        return True
    return False


def is_sentiment_bearing(text: str) -> bool:
    """
    Broader gate — True if the text has ANY sentiment signal at all.
    Allows 'Completely useless' to pass even without a product keyword.

    Special case: exclude generic life-sentiment words ("love", "enjoy", "like")
    when there is no product context — 'i love to play cricket' should stay Invalid.
    """
    if is_strongly_negative(text) or _has_word(text, STRONG_NEGATIVE_WORDS):
        return True
    if _has_phrase(text, STRONG_POSITIVE_PHRASES):
        return True
    if _has_word(text, NEGATIVE_WORDS):
        return True

    # For positive/neutral words, only count them as product-sentiment
    # if combined with a product keyword OR if they are product-specific words
    GENERIC_LIFE_WORDS = {"love", "loved", "enjoy", "enjoyed", "like", "liked",
                          "happy", "sad", "fun", "exciting", "beautiful", "nice"}
    product_positive = POSITIVE_WORDS - GENERIC_LIFE_WORDS
    if _has_word(text, product_positive):
        return True

    # Generic positive words only count if there's also a product keyword nearby
    if _has_word(text, GENERIC_LIFE_WORDS) and is_product_related(text):
        return True

    if _has_word(text, NEUTRAL_WORDS) or _has_phrase(text, NEUTRAL_PHRASES):
        return True

    return False


def is_weak_neutral(text: str) -> bool:
    """
    True when the text contains mild/neutral language AND no strong negative signal.

    Handles:
      "not bad, not great either"          → Neutral (soft double-negation)
      "decent quality, not amazing though" → Neutral (decent + mild not-amazing)
      "fine but nothing special"           → Neutral
      "bad product"                        → NOT neutral (hard negative word)
    """
    # Hard negative single words that disqualify (only when NOT preceded by negation)
    HARD_NEGATIVE = STRONG_NEGATIVE_WORDS | {
        "poor", "broken", "defective", "faulty", "failure",
        "fail", "failed", "disappointing", "disappointed", "frustrating",
    }

    # Check for hard negative words, but ignore "bad" when it's part of "not bad"
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok in HARD_NEGATIVE:
            # Allow "bad" if preceded by "not" → "not bad" = neutral
            if tok == "bad" and i > 0 and tokens[i - 1] == "not":
                continue
            return False

    # Strong negative phrases disqualify
    if _has_phrase(text, STRONG_NEGATIVE_PHRASES):
        return False

    # Genuine negative negation (e.g. "not happy", "not worth") disqualifies
    if _negated_positive(text):
        return False

    # Mild negations like "not amazing", "not great", "not perfect" → still neutral
    MILD_NEGATED_POSITIVE = ["not amazing", "not great", "not perfect",
                              "not excellent", "not fantastic", "not outstanding"]

    has_neutral_word   = _has_word(text, NEUTRAL_WORDS)
    has_neutral_phrase = _has_phrase(text, NEUTRAL_PHRASES)
    has_not_bad        = "not bad" in text
    has_mild_neg       = _has_phrase(text, MILD_NEGATED_POSITIVE)

    return has_neutral_word or has_neutral_phrase or has_not_bad or has_mild_neg


def is_confident(text: str, threshold: float = 0.60) -> bool:
    """True if the ML model's top class probability > threshold."""
    vec   = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    return max(probs) > threshold


def keyword_sentiment(text: str):
    """Keyword-only fallback. Returns label string or None."""
    if is_strongly_negative(text) or _has_word(text, NEGATIVE_WORDS):
        return "Negative"
    if is_strongly_positive(text) or _has_word(text, POSITIVE_WORDS):
        return "Positive"
    if is_weak_neutral(text):
        return "Neutral"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_sentiment(review: str) -> str:
    """
    Classifies a review as:
        "Positive Review" | "Negative Review" | "Neutral Review" | "Invalid Review"
    """

    # 1. Preprocess
    clean = preprocess(review)

    # 2. Empty / too short
    if len(clean.split()) < 2:
        return "Invalid"

    # 3. Strong-negative guard — HIGHEST priority
    #    Catches: "Waste of money", "Completely useless",
    #             "Not worth the price", "I hate this product"
    if is_strongly_negative(clean):
        return "Negative"

    # 4. Strong-positive guard
    if is_strongly_positive(clean):
        return "Positive"

    # 5. Off-topic / random sentence filter
    #    Reject only when there is ZERO product signal AND ZERO sentiment signal
    #    "my name is devansh", "i love to play cricket" → Invalid
    if not is_product_related(clean) and not is_sentiment_bearing(clean):
        return "Invalid"

    # 6. Weak / neutral catch (only when no negative word present)
    #    "The device works fine but nothing special" → Neutral
    if is_weak_neutral(clean):
        return "Neutral"

    # 7. ML model
    vec        = vectorizer.transform([clean])
    prediction = str(model.predict(vec)[0])

    # 8. Low-confidence fallback to keywords
    if not is_confident(clean, threshold=0.60):
        kw = keyword_sentiment(clean)
        if kw:
            return kw

    # 9. Normalise ML label
    pred_lower = prediction.lower()
    if "positive" in pred_lower:
        return "Positive"
    if "negative" in pred_lower:
        return "Negative"
    if "neutral"  in pred_lower:
        return "Neutral"

    # Final fallback: keyword or neutral
    return keyword_sentiment(clean) or "Neutral"


def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass   # silently skip TTS errors in headless environments


# ── PDF builder ──────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    'positive': colors.HexColor('#059669'),
    'negative': colors.HexColor('#DC2626'),
    'neutral':  colors.HexColor('#6B7280'),
    'invalid':  colors.HexColor('#D97706'),
}

def build_pdf(df, creator="SentimentScope System"):
    """
    Build a professional PDF report from a DataFrame with columns
    ['review', 'sentiment'] and return it as a BytesIO buffer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A1410'),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#6B5F54'),
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#A89D93'),
        alignment=TA_CENTER,
    )
    cell_style = ParagraphStyle(
        'Cell',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1A1410'),
        alignment=TA_LEFT,
    )

    now = datetime.now().strftime('%d %B %Y, %H:%M')

    story = []

    # ── Header ──
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Review Sentiment Analysis", title_style))
    story.append(Paragraph("Bulk Analysis Report", subtitle_style))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(f"Created by: {creator}  |  Generated: {now}", meta_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#E5DDD6'), spaceAfter=12))

    # ── Summary stats ──
    total   = len(df)
    counts  = df['sentiment'].value_counts()
    pos_n   = int(counts.get('positive', 0))
    neg_n   = int(counts.get('negative', 0))
    neu_n   = int(counts.get('neutral',  0))
    inv_n   = int(counts.get('invalid',  0))

    summary_data = [
        ['Total Reviews', 'Positive', 'Negative', 'Neutral', 'Invalid'],
        [str(total), str(pos_n), str(neg_n), str(neu_n), str(inv_n)],
    ]
    summary_table = Table(summary_data, colWidths=[1.3 * inch] * 5)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), colors.HexColor('#F2EDE7')),
        ('TEXTCOLOR',   (0, 0), (-1, 0), colors.HexColor('#6B5F54')),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, 0), 8),
        ('FONTNAME',    (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 1), (-1, 1), 16),
        ('TEXTCOLOR',   (1, 1), (1, 1), SENTIMENT_COLORS['positive']),
        ('TEXTCOLOR',   (2, 1), (2, 1), SENTIMENT_COLORS['negative']),
        ('TEXTCOLOR',   (3, 1), (3, 1), SENTIMENT_COLORS['neutral']),
        ('TEXTCOLOR',   (4, 1), (4, 1), SENTIMENT_COLORS['invalid']),
        ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#F2EDE7'), colors.white]),
        ('BOX',         (0, 0), (-1, -1), 0.5, colors.HexColor('#D9D1C7')),
        ('INNERGRID',   (0, 0), (-1, -1), 0.5, colors.HexColor('#D9D1C7')),
        ('TOPPADDING',  (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.25 * inch))

    # ── Data table ──
    header_row = ['#', 'Review', 'Sentiment']
    table_data = [header_row]
    for i, row in df.iterrows():
        review_para = Paragraph(str(row['review']), cell_style)
        sentiment   = str(row['sentiment']).lower()
        sent_color  = SENTIMENT_COLORS.get(sentiment, colors.HexColor('#1A1410'))
        sent_style  = ParagraphStyle(
            'SentCell', parent=cell_style,
            textColor=sent_color, fontName='Helvetica-Bold'
        )
        sent_para = Paragraph(sentiment.capitalize(), sent_style)
        table_data.append([str(i + 1), review_para, sent_para])

    col_widths = [0.4 * inch, 4.8 * inch, 1.2 * inch]
    data_table = Table(table_data, colWidths=col_widths, repeatRows=1)

    row_colors = []
    for idx in range(1, len(table_data)):
        bg = colors.white if idx % 2 == 0 else colors.HexColor('#FAFAF8')
        row_colors.append(('BACKGROUND', (0, idx), (-1, idx), bg))

    data_table.setStyle(TableStyle([
        # Header
        ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor('#1A1410')),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 9),
        ('ALIGN',         (0, 0), (-1, 0), 'CENTER'),
        ('TOPPADDING',    (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        # Body
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('ALIGN',         (0, 1), (0, -1),  'CENTER'),
        ('ALIGN',         (2, 1), (2, -1),  'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 7),
        ('LEFTPADDING',   (1, 0), (1, -1),  8),
        # Grid
        ('BOX',           (0, 0), (-1, -1), 0.5, colors.HexColor('#D9D1C7')),
        ('LINEBELOW',     (0, 0), (-1, -1), 0.3, colors.HexColor('#E5DDD6')),
        *row_colors,
    ]))
    story.append(data_table)

    # ── Footer note ──
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor('#E5DDD6'), spaceAfter=8))
    story.append(Paragraph(
        f"Report generated by SentimentScope &nbsp;|&nbsp; {now} &nbsp;|&nbsp; {total} reviews processed",
        meta_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ── Shared CSV validation & processing ───────────────────────────────────────
def validate_and_process_csv(file):
    """
    Returns (df, error_message).
    df is None on error; error_message is None on success.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return None, f"Could not read CSV file: {e}"

    # Validation rule 1: exactly one column
    if df.shape[1] != 1:
        return None, "Invalid file format. The CSV must contain exactly one column named 'review'."

    # Validation rule 2: column name must be exactly 'review'
    if df.columns[0] != 'review':
        return None, "Invalid file format. The CSV must contain exactly one column named 'review'."

    # Drop empty rows gracefully
    df = df.dropna(subset=['review'])
    df = df[df['review'].astype(str).str.strip() != '']
    df = df.reset_index(drop=True)

    if df.empty:
        return None, "The uploaded CSV contains no valid review rows."

    # Process sentiments
    df['sentiment'] = df['review'].apply(predict_sentiment)

    return df, None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    text = ""
    if request.method == 'POST':
        text = request.form['review']
        prediction = predict_sentiment(text)
        speak_text(f"The sentiment of the {text} is {prediction}")
    return render_template('index.html',
                           prediction=prediction,
                           review_text=text,
                           active_tab='text')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('index.html',
                               csv_error="No file uploaded.",
                               active_tab='csv')

    df, error = validate_and_process_csv(file)

    if error:
        return render_template('index.html',
                               csv_error=error,
                               active_tab='csv')

    # Save output CSV (only review + sentiment columns)
    output_df = df[['review', 'sentiment']].copy()
    output_df.to_csv('output.csv', index=False)

    # Store for PDF generation (we'll rebuild from file next request, so save df to a temp csv)
    output_df.to_csv('output_pdf_source.csv', index=False)

    # Counts
    pred_lower = df['sentiment'].str.lower()
    count_pos = int(pred_lower.eq('positive').sum())
    count_neg = int(pred_lower.eq('negative').sum())
    count_neu = int(pred_lower.eq('neutral').sum())
    count_inv = int(pred_lower.eq('invalid').sum())

    return render_template(
        'index.html',
        table=output_df.to_html(classes='result-table', index=False, border=0),
        download=True,
        active_tab='csv',
        count_pos=count_pos,
        count_neg=count_neg,
        count_neu=count_neu,
        count_inv=count_inv,
    )


@app.route('/download/csv')
def download_csv():
    return send_file('output.csv', as_attachment=True, download_name='sentiment_results.csv')


@app.route('/download/pdf')
def download_pdf():
    if not os.path.exists('output_pdf_source.csv'):
        return "No processed data found. Please upload a CSV first.", 404
    df = pd.read_csv('output_pdf_source.csv')
    pdf_buffer = build_pdf(df)
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='sentiment_report.pdf'
    )


@app.route('/download')
def download():
    """Legacy route — kept for backward compatibility."""
    return send_file('output.csv', as_attachment=True, download_name='sentiment_results.csv')


@app.route('/voice')
def voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
    except Exception:
        text = "Could not understand"
    prediction = predict_sentiment(text)
    speak_text(f"The sentiment of the {text} is {prediction}")
    return render_template('index.html',
                           prediction=prediction,
                           review_text=text,
                           active_tab='voice')


if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
