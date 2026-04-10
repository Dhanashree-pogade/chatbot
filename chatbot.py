"""
AI-Powered Customer Support Chatbot
====================================
Upgraded with:
  - RAG (Retrieval-Augmented Generation) style response pipeline
  - TF-IDF + Cosine Similarity for intent classification (LLM-style retrieval)
  - Model Training & Evaluation (train/test split, accuracy, F1 score)
  - Sentiment Analysis with negation handling
  - Conversation memory / context tracking
  - GUI via Tkinter
"""

import tkinter as tk
from tkinter import scrolledtext, ttk
import re
import time
import random
import json
import os
import math
from collections import Counter, defaultdict

# ─────────────────────────────────────────────
# 1.  TOKENIZER & TF-IDF UTILITIES
# ─────────────────────────────────────────────

def tokenize(text):
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def compute_tf(tokens):
    """Term-frequency dict for a token list."""
    freq = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: count / total for term, count in freq.items()}


def compute_idf(corpus_tokens):
    """Inverse-document-frequency across a list of token lists."""
    N = len(corpus_tokens)
    df = defaultdict(int)
    for doc in corpus_tokens:
        for term in set(doc):
            df[term] += 1
    return {term: math.log((N + 1) / (count + 1)) + 1 for term, count in df.items()}


def tfidf_vector(tokens, idf):
    """Return a TF-IDF weighted vector (dict)."""
    tf = compute_tf(tokens)
    return {term: tf[term] * idf.get(term, 1.0) for term in tokens}


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two TF-IDF dicts."""
    keys = set(vec_a) & set(vec_b)
    if not keys:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in keys)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────
# 2.  TRAINING DATA  (intent → sample queries)
# ─────────────────────────────────────────────

INTENT_CORPUS = {
    "greeting": [
        "hello", "hi there", "hey", "good morning", "good afternoon",
        "greetings", "howdy", "what's up", "hey there", "hi bot"
    ],
    "farewell": [
        "bye", "goodbye", "see you later", "take care", "good night",
        "I'm done", "that's all", "thanks and bye", "exit", "quit"
    ],
    "business_hours": [
        "when are you open", "what are your business hours", "what time do you close",
        "are you open on weekends", "when can I call", "opening hours",
        "what time do you open", "do you work on Saturday"
    ],
    "return_refund": [
        "how do I return an item", "refund policy", "can I get my money back",
        "return policy", "how long for refund", "send item back",
        "return process", "I want to return my purchase"
    ],
    "shipping": [
        "how long does shipping take", "delivery time", "when will my order arrive",
        "track my order", "express shipping", "shipping options",
        "where is my package", "shipping cost", "how do I track delivery"
    ],
    "product_info": [
        "tell me about your products", "what do you sell", "product details",
        "features of laptop", "smartphone specs", "headphones review",
        "product information", "what are the specs", "product catalogue"
    ],
    "pricing": [
        "how much does it cost", "what is the price", "pricing details",
        "cost of laptop", "phone price", "discount available",
        "any offers", "how much for headphones", "price list",
        "how much is a smartphone", "what does the laptop cost",
        "give me a price quote", "is there a discount", "product pricing",
        "how much for the headphones", "tell me the price"
    ],
    "technical_support": [
        "my product is broken", "not working", "technical issue",
        "device is faulty", "error on screen", "need help fixing",
        "troubleshoot my device", "problem with order", "defective product"
    ],
    "contact_info": [
        "how can I contact you", "customer support number", "email address",
        "phone number", "website link", "reach your team",
        "how do I get in touch", "support email"
    ],
    "warranty": [
        "warranty details", "is my product under warranty", "guarantee policy",
        "warranty period", "how long is the warranty", "warranty claim"
    ],
    "human_agent": [
        "I want to speak to a human", "connect me to an agent",
        "talk to representative", "real person please", "human support",
        "escalate my issue", "transfer to agent"
    ],
    "thanks": [
        "thank you", "thanks a lot", "many thanks", "much appreciated",
        "great help", "you're helpful", "awesome thanks", "cheers"
    ]
}


# ─────────────────────────────────────────────
# 3.  INTENT CLASSIFIER  (train + evaluate)
# ─────────────────────────────────────────────

class IntentClassifier:
    """
    Nearest-centroid TF-IDF classifier — analogous to a lightweight
    retrieval step in a RAG pipeline.
    """

    def __init__(self):
        self.idf = {}
        self.centroids = {}        # intent → mean TF-IDF vector
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.f1_scores = {}
        self.is_trained = False

    # ── helpers ──────────────────────────────
    def _build_dataset(self, corpus):
        """Flatten corpus into (tokens, label) pairs."""
        data = []
        for intent, examples in corpus.items():
            for ex in examples:
                data.append((tokenize(ex), intent))
        return data

    def _train_test_split(self, data, test_ratio=0.25, seed=42):
        random.seed(seed)
        shuffled = data[:]
        random.shuffle(shuffled)
        split = int(len(shuffled) * (1 - test_ratio))
        return shuffled[:split], shuffled[split:]

    def _compute_centroid(self, vectors):
        """Average a list of TF-IDF dicts into one centroid vector."""
        centroid = defaultdict(float)
        for vec in vectors:
            for term, val in vec.items():
                centroid[term] += val
        n = len(vectors)
        return {t: v / n for t, v in centroid.items()}

    def _predict(self, tokens):
        vec = tfidf_vector(tokens, self.idf)
        scores = {intent: cosine_similarity(vec, c) for intent, c in self.centroids.items()}
        return max(scores, key=scores.get), max(scores.values())

    def _accuracy(self, dataset):
        correct = sum(1 for tokens, label in dataset if self._predict(tokens)[0] == label)
        return correct / len(dataset) if dataset else 0.0

    def _f1(self, dataset):
        intents = list(self.centroids.keys())
        tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
        for tokens, label in dataset:
            pred, _ = self._predict(tokens)
            if pred == label:
                tp[label] += 1
            else:
                fp[pred] += 1
                fn[label] += 1
        f1 = {}
        for intent in intents:
            precision = tp[intent] / (tp[intent] + fp[intent] + 1e-9)
            recall    = tp[intent] / (tp[intent] + fn[intent] + 1e-9)
            f1[intent] = 2 * precision * recall / (precision + recall + 1e-9)
        return f1

    # ── public API ───────────────────────────
    def train(self, corpus=INTENT_CORPUS):
        dataset = self._build_dataset(corpus)
        train_data, test_data = self._train_test_split(dataset, test_ratio=0.25)

        # Build IDF over training corpus only
        self.idf = compute_idf([tokens for tokens, _ in train_data])

        # Build per-intent centroids
        intent_vecs = defaultdict(list)
        for tokens, label in train_data:
            intent_vecs[label].append(tfidf_vector(tokens, self.idf))
        self.centroids = {intent: self._compute_centroid(vecs)
                          for intent, vecs in intent_vecs.items()}

        # Evaluate
        self.train_accuracy = self._accuracy(train_data)
        self.test_accuracy  = self._accuracy(test_data)
        self.f1_scores      = self._f1(test_data)
        self.is_trained     = True

        return {
            "train_samples": len(train_data),
            "test_samples":  len(test_data),
            "train_accuracy": round(self.train_accuracy * 100, 2),
            "test_accuracy":  round(self.test_accuracy  * 100, 2),
            "f1_scores":      {k: round(v, 3) for k, v in self.f1_scores.items()}
        }

    def classify(self, text):
        if not self.is_trained:
            self.train()
        tokens = tokenize(text)
        if not tokens:
            return "unknown", 0.0
        intent, score = self._predict(tokens)
        return intent, round(score, 4)


# ─────────────────────────────────────────────
# 4.  RAG KNOWLEDGE RETRIEVER
# ─────────────────────────────────────────────

class KnowledgeRetriever:
    """
    Lightweight RAG retriever:
    1. Index documents (passages) with TF-IDF
    2. At query time, retrieve the top-k most similar passages
    3. The chatbot's response is generated from the retrieved context
    """

    def __init__(self):
        self.documents = []      # list of {"id": ..., "text": ..., "meta": ...}
        self.doc_vectors = []
        self.idf = {}

    def index(self, documents):
        """Index a list of document dicts with 'id', 'text', 'meta' keys."""
        self.documents = documents
        corpus_tokens = [tokenize(doc["text"]) for doc in documents]
        self.idf = compute_idf(corpus_tokens)
        self.doc_vectors = [tfidf_vector(tok, self.idf) for tok in corpus_tokens]

    def retrieve(self, query, top_k=2):
        """Return top-k most relevant documents for the query."""
        q_tokens = tokenize(query)
        q_vec = tfidf_vector(q_tokens, self.idf)
        scores = [cosine_similarity(q_vec, dv) for dv in self.doc_vectors]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.documents[i], score) for i, score in ranked[:top_k] if score > 0.0]


# ─────────────────────────────────────────────
# 5.  SENTIMENT ANALYSER
# ─────────────────────────────────────────────

POSITIVE_WORDS = {
    "happy", "satisfied", "great", "good", "excellent", "love", "like",
    "thanks", "thank", "awesome", "amazing", "perfect", "wonderful",
    "fantastic", "helpful", "pleased", "glad", "appreciate", "impressive",
    "enjoy", "nice", "superb", "outstanding", "brilliant"
}

NEGATIVE_WORDS = {
    "unhappy", "dissatisfied", "bad", "poor", "terrible", "hate", "dislike",
    "angry", "disappointed", "horrible", "awful", "useless", "worst",
    "frustrating", "annoying", "slow", "expensive", "broken", "faulty",
    "complaint", "issue", "problem", "waste", "fail", "defective", "refund"
}

NEGATIONS = {"not", "no", "never", "don't", "doesn't", "didn't",
             "wasn't", "aren't", "isn't", "hardly", "barely"}


def analyze_sentiment(text):
    tokens = tokenize(text)
    pos = neg = 0
    negate = False
    for tok in tokens:
        if tok in NEGATIONS:
            negate = True
            continue
        if tok in POSITIVE_WORDS:
            if negate:
                neg += 1
            else:
                pos += 1
        elif tok in NEGATIVE_WORDS:
            if negate:
                pos += 1
            else:
                neg += 1
        negate = False
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


# ─────────────────────────────────────────────
# 6.  KNOWLEDGE BASE  (used as RAG documents)
# ─────────────────────────────────────────────

DEFAULT_KB = {
    "business": {
        "hours": "Monday to Friday 9 AM – 5 PM, Saturday 10 AM – 2 PM",
        "location": "123 Main Street, Business City",
        "contact": {
            "phone": "1-800-555-1234",
            "email": "support@example.com",
            "website": "www.example.com"
        }
    },
    "policies": {
        "return": "Returns accepted within 30 days with a valid receipt",
        "refund": "Refunds processed within 5-7 business days",
        "warranty": "1-year warranty covering manufacturing defects",
        "shipping": "Standard 3-5 days; express 1-2 days"
    },
    "products": {
        "electronics": {
            "smartphone": {"price": "$599-$999",  "features": "Latest processor, 5G, high-resolution camera"},
            "laptop":     {"price": "$799-$1599", "features": "Fast CPU, long battery, lightweight"},
            "headphones": {"price": "$99-$299",   "features": "Noise cancellation, wireless, long battery"}
        }
    }
}

RAG_DOCUMENTS = [
    {"id": "hours",    "text": "business hours open close time Monday Friday Saturday", "meta": "business_hours"},
    {"id": "return",   "text": "return refund policy send back money purchase receipt",  "meta": "return_refund"},
    {"id": "shipping", "text": "shipping delivery tracking order arrive express standard","meta": "shipping"},
    {"id": "product",  "text": "product information specs features smartphone laptop headphones electronics", "meta": "product_info"},
    {"id": "pricing",  "text": "price cost discount offer how much pricing",             "meta": "pricing"},
    {"id": "warranty", "text": "warranty guarantee policy defect manufacturing claim",   "meta": "warranty"},
    {"id": "contact",  "text": "contact phone email website support reach team",         "meta": "contact_info"},
    {"id": "support",  "text": "broken defective not working technical issue troubleshoot error fault repair", "meta": "technical_support"},
    {"id": "agent",    "text": "human agent representative escalate real person speak transfer", "meta": "human_agent"},
]


# ─────────────────────────────────────────────
# 7.  CHATBOT APPLICATION
# ─────────────────────────────────────────────

class CustomerServiceChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Customer Support Chatbot  |  RAG + LLM-style NLP")
        self.root.geometry("750x620")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f4f8")

        # ── Models ──────────────────────────
        self.classifier = IntentClassifier()
        self.retriever  = KnowledgeRetriever()
        self.retriever.index(RAG_DOCUMENTS)
        self.knowledge_base = self._load_knowledge_base()

        # ── State ───────────────────────────
        self.conversation_history = []   # list of {"role": "user"|"bot", "text": ..., "intent": ...}
        self.context = {
            "customer_name": None,
            "current_product": None,
            "mentioned_issues": [],
            "order_number": None,
            "last_intent": None
        }
        self.sentiment_history = []
        self.current_sentiment = "neutral"
        self.model_metrics = {}

        # ── UI ──────────────────────────────
        self._build_ui()

        # ── Train on startup ────────────────
        self.root.after(200, self._train_and_report)

    # ─────────────── UI SETUP ────────────────

    def _build_ui(self):
        # Title bar
        title_frame = tk.Frame(self.root, bg="#1a3c5e", pady=6)
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="🤖  AI Customer Support  |  RAG · Sentiment · Intent Classification",
                 font=("Arial", 11, "bold"), bg="#1a3c5e", fg="white").pack()

        # Main chat area
        chat_frame = tk.Frame(self.root, bg="#f0f4f8")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, font=("Arial", 10),
            bg="white", relief=tk.FLAT, bd=1
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.tag_config("user_msg", foreground="#1a3c5e", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("bot_msg",  foreground="#2e7d32", font=("Arial", 10))
        self.chat_display.tag_config("sys_msg",  foreground="#888888", font=("Arial", 9, "italic"))

        # Metrics bar
        metrics_frame = tk.Frame(self.root, bg="#e8edf2", pady=3)
        metrics_frame.pack(fill=tk.X, padx=12)
        self.metrics_label = tk.Label(
            metrics_frame, text="⏳ Training intent classifier...",
            font=("Arial", 9), bg="#e8edf2", fg="#555555", anchor=tk.W
        )
        self.metrics_label.pack(fill=tk.X, padx=4)

        # Status / sentiment bar
        status_frame = tk.Frame(self.root, bg="#dde3ea", pady=3)
        status_frame.pack(fill=tk.X, padx=12)
        self.intent_label = tk.Label(
            status_frame, text="Intent: —",
            font=("Arial", 9), bg="#dde3ea", fg="#333333", anchor=tk.W, width=35
        )
        self.intent_label.pack(side=tk.LEFT, padx=4)
        self.sentiment_label = tk.Label(
            status_frame, text="Sentiment: Neutral 😐",
            font=("Arial", 9), bg="#dde3ea", fg="blue", anchor=tk.E
        )
        self.sentiment_label.pack(side=tk.RIGHT, padx=4)

        # RAG retrieval label
        self.rag_label = tk.Label(
            self.root, text="RAG: —",
            font=("Arial", 9, "italic"), bg="#f0f4f8", fg="#888888", anchor=tk.W
        )
        self.rag_label.pack(fill=tk.X, padx=16)

        # Input row
        input_frame = tk.Frame(self.root, bg="#f0f4f8", pady=8)
        input_frame.pack(fill=tk.X, padx=12)
        self.user_input = tk.Entry(
            input_frame, font=("Arial", 11), relief=tk.SOLID, bd=1
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.user_input.bind("<Return>", self.process_input)
        send_btn = tk.Button(
            input_frame, text="Send ➤", command=self.process_input,
            bg="#1a3c5e", fg="white", font=("Arial", 10, "bold"),
            relief=tk.FLAT, padx=12, pady=4
        )
        send_btn.pack(side=tk.RIGHT)

    # ─────────── TRAINING + METRICS ──────────

    def _train_and_report(self):
        metrics = self.classifier.train()
        self.model_metrics = metrics
        summary = (
            f"✅ Model trained  |  Train acc: {metrics['train_accuracy']}%  "
            f"|  Test acc: {metrics['test_accuracy']}%  "
            f"|  Samples: {metrics['train_samples']} train / {metrics['test_samples']} test"
        )
        self.metrics_label.config(text=summary)
        self._post_system_message(f"[Model] {summary}")
        self._post_bot_message(
            "Hello! I'm your AI-powered customer support assistant. "
            "Ask me about orders, returns, products, shipping, and more! 😊"
        )

    # ─────────── KNOWLEDGE BASE I/O ──────────

    def _load_knowledge_base(self):
        if os.path.exists("chatbot_knowledge.json"):
            try:
                with open("chatbot_knowledge.json") as f:
                    return json.load(f)
            except Exception:
                pass
        return DEFAULT_KB

    def _save_knowledge_base(self):
        with open("chatbot_knowledge.json", "w") as f:
            json.dump(self.knowledge_base, f, indent=4)

    # ─────────── CHAT DISPLAY ────────────────

    def _post_message(self, sender, text, tag):
        self.chat_display.config(state=tk.NORMAL)
        ts = time.strftime("%H:%M")
        self.chat_display.insert(tk.END, f"{ts}  {sender}: {text}\n\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _post_user_message(self, text):
        self._post_message("You", text, "user_msg")

    def _post_bot_message(self, text):
        self._post_message("Bot", text, "bot_msg")

    def _post_system_message(self, text):
        self._post_message("", text, "sys_msg")

    # ─────────── CONTEXT EXTRACTION ──────────

    def _extract_context(self, text):
        lower = text.lower()
        m = re.search(r"my name is (\w+)", lower)
        if m:
            self.context["customer_name"] = m.group(1).capitalize()

        m = re.search(r"order\s*#?(\d+)", lower)
        if m:
            self.context["order_number"] = m.group(1)

        for product in ["smartphone", "laptop", "headphones"]:
            if product in lower:
                self.context["current_product"] = product

        for issue in ["broken", "defective", "not working", "problem", "issue", "error", "fault"]:
            if issue in lower and issue not in self.context["mentioned_issues"]:
                self.context["mentioned_issues"].append(issue)

    # ─────────── RAG RESPONSE GENERATOR ──────

    def _generate_response(self, user_text, intent, confidence):
        """
        RAG Pipeline:
          1. Retrieve relevant knowledge passages (retriever)
          2. Use intent + retrieved context to generate response
        """
        kb = self.knowledge_base
        name = self.context["customer_name"]
        name_str = f" {name}" if name else ""

        # ── Retrieve top passages ─────────────
        retrieved = self.retriever.retrieve(user_text, top_k=2)
        retrieved_intents = [doc["meta"] for doc, _ in retrieved]
        self.rag_label.config(
            text=f"RAG retrieved: {retrieved_intents}  |  Intent: {intent} (conf: {confidence})"
        )

        lower = user_text.lower()

        # ── Intent-based response ─────────────
        if intent == "greeting":
            return f"Hello{name_str}! How can I assist you today? 😊"

        if intent == "farewell":
            return random.choice([
                "Thank you for reaching out! Have a great day! 👋",
                "Goodbye! Don't hesitate to come back if you need help.",
                "Take care! We're always here if you need us."
            ])

        if intent == "thanks":
            return random.choice([
                "You're welcome! Is there anything else I can help with?",
                "Happy to help! Let me know if you need anything else. 😊",
                "My pleasure! Feel free to ask anytime."
            ])

        if intent == "business_hours":
            return f"Our business hours are: {kb['business']['hours']}."

        if intent == "return_refund":
            return (f"📦 Return Policy: {kb['policies']['return']}.\n"
                    f"💰 Refund: {kb['policies']['refund']}.")

        if intent == "shipping":
            if self.context["order_number"]:
                statuses = ["processing", "shipped", "out for delivery", "delivered"]
                return (f"I've checked order #{self.context['order_number']} — "
                        f"it's currently {random.choice(statuses)}. 🚚\n"
                        f"Shipping info: {kb['policies']['shipping']}.")
            return f"🚚 Shipping: {kb['policies']['shipping']}. Share your order number for tracking."

        if intent == "warranty":
            return f"🛡️ Warranty Policy: {kb['policies']['warranty']}."

        if intent == "contact_info":
            c = kb["business"]["contact"]
            return (f"📞 Phone: {c['phone']}\n"
                    f"📧 Email: {c['email']}\n"
                    f"🌐 Website: {c['website']}")

        if intent == "human_agent":
            return ("I'll connect you with a human representative right away. "
                    "Please hold for a moment... 👤")

        if intent == "technical_support":
            issues = ", ".join(self.context["mentioned_issues"]) or "your issue"
            return (f"I'm sorry to hear about {issues}. 🔧\n"
                    "Let me help troubleshoot:\n"
                    "1. Please restart the device.\n"
                    "2. Check for software updates.\n"
                    "3. If the issue persists, I can connect you to our technical team.")

        if intent == "product_info":
            product = self.context["current_product"]
            if product:
                electronics = kb["products"].get("electronics", {})
                if product in electronics:
                    p = electronics[product]
                    return (f"📱 {product.capitalize()} — Price: {p['price']}\n"
                            f"✨ Features: {p['features']}")
            return ("We carry smartphones, laptops, and headphones. "
                    "Which product would you like details on?")

        if intent == "pricing":
            product = self.context["current_product"]
            if product:
                electronics = kb["products"].get("electronics", {})
                if product in electronics:
                    return f"💲 The {product} is priced at {electronics[product]['price']}."
            return "Could you tell me which product you'd like pricing for?"

        # ── RAG fallback: use retrieved context ──
        if retrieved:
            top_intent = retrieved[0][0]["meta"]
            return (f"Based on your query, here's what I found (RAG-retrieved topic: {top_intent}):\n"
                    "Could you provide more details so I can give you a precise answer?")

        # ── Final fallback ──
        if self.context["last_intent"]:
            return (f"I see you're asking about {self.context['last_intent'].replace('_', ' ')}. "
                    "Could you rephrase or add more details?")

        return random.choice([
            "I'm not quite sure I understand. Could you rephrase that?",
            "Could you provide more details? I want to make sure I help you correctly.",
            "I'd like to help! Could you tell me a bit more about your question?"
        ])

    # ─────────── PROCESS INPUT ───────────────

    def process_input(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        self.user_input.delete(0, tk.END)

        # Display user message
        self._post_user_message(user_text)
        self.conversation_history.append({"role": "user", "text": user_text})

        # Extract context entities
        self._extract_context(user_text)

        # Sentiment analysis
        sentiment = analyze_sentiment(user_text)
        self.sentiment_history.append(sentiment)
        self.current_sentiment = sentiment
        self._update_sentiment_ui(sentiment)

        # Intent classification (TF-IDF retrieval)
        intent, confidence = self.classifier.classify(user_text)
        self.context["last_intent"] = intent
        self.intent_label.config(
            text=f"Intent: {intent.replace('_', ' ').title()}  ({confidence:.2f})"
        )

        # Generate RAG-style response
        response = self._generate_response(user_text, intent, confidence)

        # Save to history
        self.conversation_history.append({"role": "bot", "text": response, "intent": intent})

        # Delayed response display
        self.root.after(700, lambda: self._post_bot_message(response))

        # Negative sentiment escalation
        if (self.current_sentiment == "negative"
                and len(self.sentiment_history) >= 3
                and self.sentiment_history[-3:].count("negative") >= 2):
            escalation = ("⚠️ I notice you seem frustrated. "
                          "Would you like me to escalate this to a human representative?")
            self.root.after(1600, lambda: self._post_bot_message(escalation))

    # ─────────── SENTIMENT UI ────────────────

    def _update_sentiment_ui(self, sentiment):
        config = {
            "positive": ("Sentiment: Positive 😊", "green"),
            "negative": ("Sentiment: Negative 😞", "red"),
            "neutral":  ("Sentiment: Neutral 😐",  "blue")
        }
        text, color = config[sentiment]
        self.sentiment_label.config(text=text, fg=color)


# ─────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = CustomerServiceChatbot(root)
    root.mainloop()


if __name__ == "__main__":
    main()
