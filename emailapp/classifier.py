"""
A small, self-contained Multinomial Naive Bayes text classifier used to
label a submitted email as "Phishing", "Spam", or "Legitimate".

It is trained, at import time, on the 24 labeled examples supplied for
this project (see TRAINING_DATA below). No external ML library is
required -- everything is plain Python so the project only needs Django
installed to run.
"""
import math
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Training data: (subject, body, label)
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    ("Your account will be suspended in 24 hours",
     "Dear Customer, We detected unusual activity on your account. Click here to verify your identity within 24 hours or your account will be permanently suspended: http://secure-verify-account.com/login. Failure to act will result in loss of access.",
     "phishing"),
    ("Invoice #4471 overdue - action required",
     "Your payment for invoice #4471 is now 30 days overdue. Please remit payment via the attached link to avoid service interruption and late fees: http://billing-payportal.net/pay?id=4471",
     "phishing"),
    ("IT Helpdesk: Password expiring today",
     "Your corporate password expires today. To avoid being locked out, log in immediately at http://it-helpdesk-portal.info/reset and confirm your current credentials.",
     "phishing"),
    ("You have received a secure document from HR",
     "A confidential HR document has been shared with you. Click below to view it using your company email login: http://hr-docs-share.co/view?ref=98213",
     "phishing"),
    ("Unusual sign-in attempt on your account",
     "We noticed a sign-in attempt from a new device in Minsk, Belarus. If this wasn't you, verify your identity now: http://account-security-check.com/verify",
     "phishing"),
    ("Congratulations! You've been selected for a refund",
     "Our records show you are eligible for a refund of $284.50 from your recent purchase. Submit your banking details here to receive your refund within 24 hours: http://refund-claims-center.net",
     "phishing"),
    ("URGENT: Wire transfer needed before end of day",
     "Hi, I'm in a meeting and can't talk. I need you to process a wire transfer of $9,500 to a new vendor today. Reply with your availability and I'll send the account details. Please keep this confidential for now. - Sent from my iPhone",
     "phishing"),
    ("Final notice: shipment held at customs",
     "Your package could not be delivered due to unpaid customs fees. Pay $3.99 immediately to release your shipment: http://parcel-fee-payment.com/track",
     "phishing"),
    ("Verify your Apple ID to continue using iCloud",
     "Your Apple ID has been locked for security reasons. Verify your identity now to restore access to iCloud, Photos, and Find My iPhone: http://appleid-verify-support.com",
     "phishing"),
    ("Action required: Update your payroll direct deposit",
     "Please update your direct deposit information before Friday's payroll run using the secure form below, or your paycheck may be delayed: http://payroll-update-form.biz",
     "phishing"),
    ("50% off everything this weekend only!!!",
     "Don't miss out! Our biggest sale of the season is happening now. Shop hundreds of deals before they're gone. Unsubscribe anytime at the link in the footer.",
     "spam"),
    ("Congratulations, you've won a $500 gift card!",
     "You've been randomly selected to receive a $500 gift card. Claim your prize now by completing a short survey. Limited time offer, click to claim.",
     "spam"),
    ("Lose 20 pounds in 2 weeks with this one trick",
     "Doctors hate this! Discover the secret supplement that celebrities are using to shed weight fast. Order now and get a free bottle.",
     "spam"),
    ("Hot singles in your area want to meet you",
     "Thousands of members are online right now near you. Sign up free and start chatting today. No credit card required.",
     "spam"),
    ("Extend your car's warranty before it's too late",
     "Our records indicate your vehicle's warranty is about to expire. Call now to extend your coverage and avoid costly repairs.",
     "spam"),
    ("Make $5,000 a week working from home",
     "Join thousands who have already quit their 9-to-5 with this simple online business model. No experience needed, start today.",
     "spam"),
    ("Get a free credit score check instantly",
     "Check your credit score for free, no credit card needed. See offers you qualify for in seconds.",
     "spam"),
    ("Refinance your mortgage and save thousands",
     "Interest rates are at historic lows. See if you qualify for a lower monthly payment in under 2 minutes.",
     "spam"),
    ("Team standup notes - July 3",
     "Hi team, here are today's standup notes: Sarah finished the auth refactor and it's in review. Mike is blocked on the staging environment, ticket filed as INFRA-482. I'll follow up with DevOps this afternoon. No meeting tomorrow due to the holiday.",
     "legitimate"),
    ("Reminder: Dentist appointment Thursday at 2pm",
     "Hi, this is a reminder that your appointment with Dr. Alvarez is scheduled for Thursday at 2:00 PM. Please arrive 10 minutes early. Call us at the office number if you need to reschedule.",
     "legitimate"),
    ("Re: Draft budget for Q3",
     "Thanks for putting this together. A couple of notes: line item 4 looks high compared to last quarter, can you double check the vendor quote? Otherwise this looks good to send to finance.",
     "legitimate"),
    ("Your order has shipped",
     "Good news! Your order #10293 has shipped and is on its way. You can track your package using the carrier's tracking number in your order history on our website (no link needed, just log in directly as usual).",
     "legitimate"),
    ("Lunch on Friday?",
     "Hey, a few of us are grabbing lunch on Friday around noon at the taco place near the office. Let me know if you want to join.",
     "legitimate"),
    ("Meeting moved to 3pm",
     "Quick note - I moved our 1:1 from 2pm to 3pm today, same room. Let me know if that doesn't work.",
     "legitimate"),
]

LABELS = ("phishing", "spam", "legitimate")

_TOKEN_RE = re.compile(r"[a-z']+")


def _tokenize(text):
    """Lowercase and split into simple word tokens."""
    return _TOKEN_RE.findall(text.lower())


class NaiveBayesClassifier:
    """A minimal multinomial Naive Bayes classifier with Laplace smoothing."""

    def __init__(self, training_data):
        self.class_doc_counts = Counter()
        self.class_word_counts = defaultdict(Counter)
        self.class_total_words = Counter()
        self.vocabulary = set()

        for subject, body, label in training_data:
            tokens = _tokenize(subject + " " + body)
            self.class_doc_counts[label] += 1
            self.class_word_counts[label].update(tokens)
            self.class_total_words[label] += len(tokens)
            self.vocabulary.update(tokens)

        self.total_docs = sum(self.class_doc_counts.values())
        self.vocab_size = len(self.vocabulary)

    def _log_prob(self, tokens, label):
        log_prior = math.log(self.class_doc_counts[label] / self.total_docs)
        denom = self.class_total_words[label] + self.vocab_size
        log_likelihood = 0.0
        for token in tokens:
            count = self.class_word_counts[label][token]
            log_likelihood += math.log((count + 1) / denom)
        return log_prior + log_likelihood

    def classify(self, text):
        """Return (best_label, {label: score}) for the given text."""
        tokens = _tokenize(text)
        scores = {label: self._log_prob(tokens, label) for label in LABELS}
        best_label = max(scores, key=scores.get)
        return best_label, scores


# Train once at import time; reused for every request.
_classifier = NaiveBayesClassifier(TRAINING_DATA)

DISPLAY_LABELS = {
    "phishing": "Phishing",
    "spam": "Spam",
    "legitimate": "Legitimate",
}


def classify_email(text):
    """
    Classify the given email text as Phishing, Spam, or Legitimate.
    Returns the human-readable label (e.g. "Phishing").
    """
    if not text.strip():
        return "(nothing to classify)"
    label, _scores = _classifier.classify(text)
    return DISPLAY_LABELS[label]
