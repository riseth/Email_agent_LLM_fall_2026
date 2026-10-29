import re

from django.shortcuts import render

from .classifier import classify_email
from .llm_classifier import classify_with_llm

# A name is assumed to be one or more capitalized words, e.g. "Foo" or
# "Mary Jane". Deliberately simple: this is pattern matching, not NLP.
# (?<!^) below is not used; instead callers exclude the first word of a
# sentence themselves, since greeting words are also capitalized.
NAME = r"([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)?)"

# --- Receiver's name (from the greeting) -----------------------------------

# Direct form: "Hello Foo,", "Hi Foo,", "Dear Foo,", "Hey Foo!"
GREETING_DIRECT_RE = re.compile(
    r"(?i:\b(?:Hi|Hello|Hey|Dear)\b)[,\s]*" + NAME + r"\s*[,.!]"
)

# Words that can start a greeting sentence but are never themselves a name.
# Used to skip the first word of a sentence when it's just capitalized
# because it's at the start of a sentence (e.g. "How", "Its", "It's").
_SENTENCE_START_STOPWORDS = {
    "hi", "hello", "hey", "dear", "how", "its", "it's", "i", "im", "i'm",
    "just", "so", "well",
}

_WORD_RE = re.compile(r"[A-Za-z']+")


def _extract_trailing_name(sentence):
    """
    Fallback for greetings with no explicit "Hi/Hello" marker next to the
    name, e.g. "How are you doing Foo" or "It's been a long time since I
    heard from you Foo". Looks at the first sentence and returns the last
    capitalized word that isn't just a sentence-starting capital and isn't
    a common pronoun/filler like "I".
    """
    words = _WORD_RE.findall(sentence)
    candidates = []
    for i, word in enumerate(words):
        if i == 0:
            continue  # skip sentence-initial capitalization
        if word.lower() in _SENTENCE_START_STOPWORDS:
            continue
        if re.match(r"^[A-Z][a-z]+$", word):
            candidates.append(word)
    return candidates[-1] if candidates else None


def extract_receiver_name(text):
    direct_match = GREETING_DIRECT_RE.search(text)
    if direct_match:
        return direct_match.group(1)

    # Fall back to the first sentence of the message.
    first_sentence_match = re.match(r"[^.!?]*", text)
    first_sentence = first_sentence_match.group(0) if first_sentence_match else text
    trailing_name = _extract_trailing_name(first_sentence)
    return trailing_name or "(receiver not found)"


# --- Sender's name (from the sign-off / self-introduction) -----------------

# Classic sign-off at the end of the message, e.g. "Regards, Bryan",
# "Bye, Kevin", "Thanks - Bryan". If no name follows the sign-off word
# (e.g. just "Bye"), this simply won't match.
SIGNOFF_RE = re.compile(
    r"(?i:\b(?:Regards|Best\s+regards|Best|Sincerely|Thanks|Thank\s+you|"
    r"Cheers|Yours\s+truly|Yours\s+sincerely|Warm\s+regards|Bye|"
    r"Goodbye|Farewell|Take\s+care)\b)[,\s-]*"
    + NAME + r"\s*[.!]?\s*$"
)

# Self-introduction anywhere in the message, e.g. "This is Bryan enquiring
# about my order.", "I am Bryan", "My name is Bryan".
SELF_INTRO_RE = re.compile(
    r"(?i:\b(?:This\s+is|I\s*am|I'm|My\s+name\s+is)\b)\s+" + NAME + r"\b"
)


def extract_sender_name(text):
    signoff_match = SIGNOFF_RE.search(text)
    if signoff_match:
        return signoff_match.group(1)

    self_intro_match = SELF_INTRO_RE.search(text)
    if self_intro_match:
        return self_intro_match.group(1)

    return "(sender not found)"


# --- Classification ----------------------------------------------------


def classify_message(text):
    """
    Classify with the Gemini LLM. If the call fails (no API key, network
    error, bad response), fall back to the local Naive Bayes model so the
    page still works. Returns (label, reason, method).
    """
    if not text:
        return "(nothing to classify)", "", ""
    try:
        label, reason = classify_with_llm(text)
        return label, reason, "Gemini LLM"
    except Exception:
        return classify_email(text), "", "Local fallback model (LLM unavailable)"


# --- Views -------------------------------------------------------------


def index(request):
    """Show the text box where the user pastes/enters an email message."""
    return render(request, "emailapp/index.html")


def result(request):
    """
    Read the submitted email text and:
      - extract the receiver's name from the greeting
      - extract the sender's name from the sign-off / self-introduction
      - classify the message as Phishing, Spam, or Legitimate

    Name extraction is simple pattern matching. Classification is done by
    a Naive Bayes model trained on labeled examples (see classifier.py).
    """
    raw_message = request.POST.get("email_message", "") if request.method == "POST" else ""
    text = raw_message.strip()

    classification, reason, method = classify_message(text)

    context = {
        "sender_name": extract_sender_name(text),
        "receiver_name": extract_receiver_name(text),
        "classification": classification,
        "reason": reason,
        "method": method,
    }
    return render(request, "emailapp/result.html", context)
