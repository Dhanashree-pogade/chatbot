import tkinter as tk
from tkinter import scrolledtext
import re
import time
import random
import json
import os
import math
from collections import Counter, defaultdict


# ============================================================
# AI CUSTOMER SUPPORT CHATBOT
# NLP + TF-IDF + COSINE SIMILARITY + SENTIMENT + RETRIEVAL
# ============================================================


# ============================================================
# 1. TEXT PREPROCESSING
# ============================================================

def tokenize(text):
    """
    Lowercase text, preserve apostrophes for negation,
    remove other punctuation and split into tokens.
    """
    text = text.lower()

    # Keep apostrophes so words like don't, isn't, can't
    # are preserved for negation handling.
    text = re.sub(r"[^a-z0-9'\s]", " ", text)

    return [
        t for t in text.split()
        if len(t) > 1
    ]


def compute_tf(tokens):
    freq = Counter(tokens)

    total = len(tokens) if tokens else 1

    return {
        term: count / total
        for term, count in freq.items()
    }


def compute_idf(corpus_tokens):
    N = len(corpus_tokens)

    df = defaultdict(int)

    for doc in corpus_tokens:
        for term in set(doc):
            df[term] += 1

    return {
        term: math.log((N + 1) / (count + 1)) + 1
        for term, count in df.items()
    }


def tfidf_vector(tokens, idf):
    tf = compute_tf(tokens)

    return {
        term: tf_value * idf.get(term, 1.0)
        for term, tf_value in tf.items()
    }


def cosine_similarity(vec_a, vec_b):

    keys = set(vec_a) & set(vec_b)

    if not keys:
        return 0.0

    dot = sum(
        vec_a[k] * vec_b[k]
        for k in keys
    )

    norm_a = math.sqrt(
        sum(v ** 2 for v in vec_a.values())
    )

    norm_b = math.sqrt(
        sum(v ** 2 for v in vec_b.values())
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ============================================================
# 2. CUSTOM INTENT DATASET
# ============================================================

# 20 examples per intent
# 12 intents
# 240 examples total
#
# This is a custom-created prototype dataset.
# It is NOT downloaded from Kaggle or another public dataset.

INTENT_CORPUS = {

    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "hi there",
        "hello there",
        "hey there",
        "can you help me",
        "i need help",
        "is anyone there",
        "are you available",
        "i want some help",
        "hello assistant",
        "hey assistant",
        "start chat",
        "can we chat",
        "i have a question",
        "i would like some help"
    ],

    "farewell": [
        "bye",
        "goodbye",
        "see you",
        "see you later",
        "talk to you later",
        "have a good day",
        "take care",
        "i am leaving",
        "that's all",
        "that is all",
        "no more questions",
        "i am done",
        "end chat",
        "thanks goodbye",
        "bye for now",
        "catch you later",
        "i will come back later",
        "have a nice day",
        "good night",
        "i don't need anything else"
    ],

    "business_hours": [
        "what are your business hours",
        "when are you open",
        "when do you open",
        "when do you close",
        "what time do you open",
        "what time do you close",
        "are you open today",
        "are you open now",
        "what are your working hours",
        "tell me your office hours",
        "when can i contact you",
        "what is your opening time",
        "what is your closing time",
        "do you operate on weekends",
        "are you open on saturday",
        "are you open on sunday",
        "what days are you available",
        "when is customer service available",
        "what are your support hours",
        "when can i speak to support"
    ],

    "return_refund": [
        "what is your return policy",
        "can i return a product",
        "how can i return my order",
        "i want a refund",
        "how do i get a refund",
        "can i get my money back",
        "i want to return an item",
        "what is the refund process",
        "how long does a refund take",
        "can i exchange my product",
        "i received the wrong product",
        "my product came wrong",
        "i got the wrong item",
        "the item i received is incorrect",
        "i want to return the wrong item",
        "how do i exchange an item",
        "my order needs to be returned",
        "i need to cancel and refund my order",
        "the product is not what i ordered",
        "what are the return conditions"
    ],

    "shipping": [
        "how long will shipping take",
        "how long does delivery take",
        "when will my order arrive",
        "where is my order",
        "when can i expect delivery",
        "how many days for shipping",
        "how many days does delivery take",
        "what is the delivery time",
        "tell me about shipping",
        "what are your shipping options",
        "do you offer express shipping",
        "how fast is express delivery",
        "is standard shipping available",
        "when will my package arrive",
        "can i track my order",
        "how can i track my shipment",
        "what is the shipping time",
        "how long until my package arrives",
        "does shipping include tracking",
        "what is the delivery estimate"
    ],

    "product_info": [
        "tell me about the smartphone",
        "what features does the smartphone have",
        "what are the laptop specifications",
        "tell me about the laptop",
        "what features do the headphones have",
        "tell me about the headphones",
        "what products do you sell",
        "show me your products",
        "what are the product features",
        "i need product information",
        "give me details about the smartphone",
        "give me details about the laptop",
        "what is included with the product",
        "what comes with the headphones",
        "what are the specifications",
        "tell me more about this product",
        "what is the battery life",
        "does the laptop have an ssd",
        "what is the smartphone storage",
        "do the headphones support bluetooth"
    ],

    "pricing": [
        "how much does the smartphone cost",
        "what is the price of the laptop",
        "how much are the headphones",
        "what is the price",
        "tell me the cost",
        "how much does this cost",
        "what are your prices",
        "give me the price",
        "how much will i pay",
        "what is the laptop price",
        "what is the smartphone price",
        "what is the headphone price",
        "how expensive is this product",
        "can you tell me the cost",
        "what is the current price",
        "how much is this item",
        "is there a price for the laptop",
        "what does this product cost",
        "tell me how much it is",
        "what are the product prices"
    ],

    "technical_support": [
        "my device is not working",
        "i need technical support",
        "my laptop is not working",
        "my smartphone is not working",
        "my headphones are not working",
        "the device has stopped working",
        "i have a technical problem",
        "something is wrong with my device",
        "my product is malfunctioning",
        "the device keeps crashing",
        "my laptop keeps freezing",
        "my phone keeps restarting",
        "the headphones are not connecting",
        "bluetooth is not working",
        "the product has an error",
        "i need help fixing my device",
        "can you troubleshoot my device",
        "i have a software problem",
        "my device is giving an error",
        "i cannot use my product"
    ],

    "contact_info": [
        "how can i contact you",
        "what is your contact number",
        "what is your phone number",
        "give me your email",
        "what is your email address",
        "how do i contact support",
        "where can i contact you",
        "how can i reach customer service",
        "give me your contact details",
        "what are your contact details",
        "how do i reach you",
        "what is the support email",
        "can i call customer service",
        "where is your contact information",
        "i need your phone number",
        "i need your email",
        "how can i get in touch",
        "what is the customer care number",
        "how do i communicate with support",
        "where can i find support contact"
    ],

    "warranty": [
        "what is your warranty policy",
        "does this product have warranty",
        "how long is the warranty",
        "is the laptop under warranty",
        "is the smartphone covered by warranty",
        "are headphones covered by warranty",
        "what does the warranty cover",
        "how can i claim warranty",
        "how do i use my warranty",
        "when does the warranty expire",
        "is there a product warranty",
        "tell me about the warranty",
        "what are the warranty terms",
        "can i get a warranty replacement",
        "does warranty cover damage",
        "does warranty cover defects",
        "how long does warranty last",
        "is my product still under warranty",
        "what is included in the warranty",
        "i need warranty support"
    ],

    "human_agent": [
        "i want to speak to a human",
        "connect me to an agent",
        "can i talk to a real person",
        "i need a human agent",
        "transfer me to customer service",
        "i want a customer support agent",
        "can someone from support help me",
        "connect me with a representative",
        "i need to talk to someone",
        "give me a human representative",
        "i don't want a chatbot",
        "can i speak with a person",
        "please connect me to support",
        "i need a live agent",
        "get me a customer service representative",
        "can a human help me",
        "i want live support",
        "transfer me to a person",
        "let me talk to an employee",
        "i need human assistance"
    ],

    "thanks": [
        "thank you",
        "thanks",
        "thanks for your help",
        "thank you for helping",
        "i appreciate your help",
        "that was helpful",
        "very helpful",
        "great help",
        "thanks a lot",
        "thank you so much",
        "i am happy with your help",
        "i am satisfied with the support",
        "i am happy with my order",
        "i am satisfied with my order",
        "i love the product",
        "the product is great",
        "i am very happy",
        "excellent service",
        "good service",
        "you helped me a lot"
    ]
}


# ============================================================
# 3. INTENT CLASSIFIER
# ============================================================

class IntentClassifier:

    def __init__(self, similarity_threshold=0.12):

        self.idf = {}

        self.centroids = {}

        self.train_accuracy = 0.0

        self.test_accuracy = 0.0

        self.f1_scores = {}

        self.is_trained = False

        self.similarity_threshold = similarity_threshold


    def _build_dataset(self):

        data = []

        for label, examples in INTENT_CORPUS.items():

            for text in examples:

                data.append(
                    (tokenize(text), label)
                )

        return data


    def _train_test_split(
        self,
        data,
        test_ratio=0.25,
        seed=42
    ):

        """
        Stratified split.

        This ensures every intent is represented
        in both training and testing data.
        """

        rng = random.Random(seed)

        grouped = defaultdict(list)

        for tokens, label in data:

            grouped[label].append(
                (tokens, label)
            )

        train_data = []

        test_data = []

        for label, items in grouped.items():

            rng.shuffle(items)

            test_count = max(
                1,
                round(len(items) * test_ratio)
            )

            # Keep at least one training example.
            if test_count >= len(items):

                test_count = len(items) - 1

            test_data.extend(
                items[:test_count]
            )

            train_data.extend(
                items[test_count:]
            )

        rng.shuffle(train_data)

        rng.shuffle(test_data)

        return train_data, test_data


    def _compute_centroid(self, vectors):

        if not vectors:

            return {}

        sums = defaultdict(float)

        for vec in vectors:

            for term, value in vec.items():

                sums[term] += value

        count = len(vectors)

        return {
            term: value / count
            for term, value in sums.items()
        }


    def _predict(self, tokens):

        vec = tfidf_vector(
            tokens,
            self.idf
        )

        if not vec:

            return "unknown", 0.0

        best_intent = "unknown"

        best_score = 0.0

        for intent, centroid in self.centroids.items():

            score = cosine_similarity(
                vec,
                centroid
            )

            if score > best_score:

                best_score = score

                best_intent = intent

        if best_score < self.similarity_threshold:

            return "unknown", best_score

        return best_intent, best_score


    def _accuracy(self, data):

        if not data:

            return 0.0

        correct = 0

        for tokens, label in data:

            prediction, _ = self._predict(
                tokens
            )

            if prediction == label:

                correct += 1

        return correct / len(data)


    def _f1(self, data):

        labels = list(
            self.centroids.keys()
        )

        scores = {}

        for label in labels:

            tp = 0
            fp = 0
            fn = 0

            for tokens, actual in data:

                predicted, _ = self._predict(
                    tokens
                )

                if (
                    predicted == label
                    and actual == label
                ):

                    tp += 1

                elif (
                    predicted == label
                    and actual != label
                ):

                    fp += 1

                elif (
                    predicted != label
                    and actual == label
                ):

                    fn += 1

            precision = (
                tp / (tp + fp)
                if (tp + fp)
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn)
                else 0.0
            )

            if precision + recall:

                f1 = (
                    2 * precision * recall
                    / (precision + recall)
                )

            else:

                f1 = 0.0

            scores[label] = f1

        return scores


    def train(self):

        data = self._build_dataset()

        train_data, test_data = (
            self._train_test_split(
                data,
                test_ratio=0.25,
                seed=42
            )
        )

        # IDF is learned only from training data.
        training_tokens = [
            tokens
            for tokens, _ in train_data
        ]

        self.idf = compute_idf(
            training_tokens
        )

        grouped_vectors = defaultdict(list)

        for tokens, label in train_data:

            vector = tfidf_vector(
                tokens,
                self.idf
            )

            grouped_vectors[label].append(
                vector
            )

        self.centroids = {
            label: self._compute_centroid(
                vectors
            )
            for label, vectors
            in grouped_vectors.items()
        }

        self.train_accuracy = (
            self._accuracy(train_data)
        )

        self.test_accuracy = (
            self._accuracy(test_data)
        )

        self.f1_scores = self._f1(
            test_data
        )

        self.is_trained = True

        return {

            "train_samples":
                len(train_data),

            "test_samples":
                len(test_data),

            "total_samples":
                len(data),

            "num_intents":
                len(INTENT_CORPUS),

            "train_accuracy":
                self.train_accuracy * 100,

            "test_accuracy":
                self.test_accuracy * 100,

            "f1_scores":
                self.f1_scores
        }


    def classify(self, text):

        if not self.is_trained:

            self.train()

        tokens = tokenize(text)

        if not tokens:

            return "unknown", 0.0

        return self._predict(tokens)


# ============================================================
# 4. KNOWLEDGE RETRIEVER
# ============================================================

RAG_DOCUMENTS = [

    {
        "text":
        "Standard shipping takes 3-5 business days. "
        "Express shipping takes 1-2 business days. "
        "Customers can provide an order number for tracking.",

        "meta":
        "shipping"
    },

    {
        "text":
        "Customers can request a return within 30 days "
        "of delivery if the product is eligible. "
        "Refunds are processed after the returned item "
        "is received and inspected.",

        "meta":
        "return_refund"
    },

    {
        "text":
        "Smartphone: 6.5 inch display, 128GB storage, "
        "8GB RAM. Laptop: 15.6 inch display, 512GB SSD, "
        "16GB RAM. Headphones: Bluetooth, noise cancellation, "
        "and up to 30 hours battery life.",

        "meta":
        "product_info"
    },

    {
        "text":
        "The smartphone costs $499, the laptop costs $999, "
        "and the headphones cost $149.",

        "meta":
        "pricing"
    },

    {
        "text":
        "Customer support is available Monday to Friday "
        "from 9 AM to 6 PM.",

        "meta":
        "business_hours"
    },

    {
        "text":
        "Products include a limited warranty covering "
        "manufacturing defects. Warranty duration depends "
        "on the product.",

        "meta":
        "warranty"
    },

    {
        "text":
        "For technical issues, restart the device, check "
        "software updates, and contact technical support "
        "if the issue continues.",

        "meta":
        "technical_support"
    },

    {
        "text":
        "Customers can contact support through the customer "
        "service phone number or support email.",

        "meta":
        "contact_info"
    },

    {
        "text":
        "Customers can request a human support representative "
        "when they need assistance that the automated assistant "
        "cannot provide.",

        "meta":
        "human_agent"
    }
]


class KnowledgeRetriever:

    def __init__(self):

        self.documents = []

        self.doc_vectors = []

        self.idf = {}


    def index(self, documents):

        self.documents = documents

        corpus_tokens = [
            tokenize(doc["text"])
            for doc in documents
        ]

        self.idf = compute_idf(
            corpus_tokens
        )

        self.doc_vectors = [

            tfidf_vector(
                tokens,
                self.idf
            )

            for tokens in corpus_tokens
        ]


    def retrieve(
        self,
        query,
        top_k=2
    ):

        if not self.documents:

            return []

        q_tokens = tokenize(query)

        q_vec = tfidf_vector(
            q_tokens,
            self.idf
        )

        scored = []

        for i, vec in enumerate(
            self.doc_vectors
        ):

            score = cosine_similarity(
                q_vec,
                vec
            )

            if score > 0:

                scored.append(
                    (
                        score,
                        self.documents[i]
                    )
                )

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return scored[:top_k]


# ============================================================
# 5. SENTIMENT ANALYSIS
# ============================================================

POSITIVE_WORDS = {

    "good",
    "great",
    "excellent",
    "amazing",
    "happy",
    "satisfied",
    "love",
    "helpful",
    "perfect",
    "awesome",
    "nice",
    "thanks",
    "thank",
    "wonderful",
    "fantastic",
    "best",
    "pleased",
    "enjoy",
    "enjoyed",
    "impressed"
}


NEGATIVE_WORDS = {

    "bad",
    "terrible",
    "awful",
    "poor",
    "worst",
    "angry",
    "upset",
    "sad",
    "hate",
    "horrible",
    "disappointed",
    "disappointing",
    "wrong",
    "broken",
    "useless",
    "frustrated",
    "frustrating",
    "annoyed",
    "problem",
    "issue",
    "failed",
    "failure"
}


NEGATIONS = {

    "not",
    "no",
    "never",
    "don't",
    "doesn't",
    "didn't",
    "wasn't",
    "weren't",
    "isn't",
    "aren't",
    "can't",
    "cannot",
    "won't"
}


def analyze_sentiment(text):

    tokens = tokenize(text)

    positive = 0

    negative = 0

    negate_next = False

    for token in tokens:

        if token in NEGATIONS:

            negate_next = True

            continue

        if token in POSITIVE_WORDS:

            if negate_next:

                negative += 1

            else:

                positive += 1

            negate_next = False

        elif token in NEGATIVE_WORDS:

            if negate_next:

                positive += 1

            else:

                negative += 1

            negate_next = False

        elif negate_next:

            negate_next = False

    if positive > negative:

        return "Positive"

    elif negative > positive:

        return "Negative"

    else:

        return "Neutral"


# ============================================================
# 6. KNOWLEDGE BASE
# ============================================================

DEFAULT_KB = {

    "business_hours": {

        "response":
        "Our customer support is available "
        "Monday to Friday, 9 AM to 6 PM."
    },

    "return_refund": {

        "response":
        "You can request a return within 30 days "
        "of delivery for eligible products. "
        "Refunds are processed after inspection."
    },

    "shipping": {

        "response":
        "Standard shipping takes 3-5 business days; "
        "express shipping takes 1-2 days."
    },

    "warranty": {

        "response":
        "Our products include a limited warranty "
        "covering eligible manufacturing defects. "
        "The duration depends on the product."
    },

    "contact_info": {

        "response":
        "You can contact customer support through "
        "our support phone number or email."
    },

    "technical_support": {

        "response":
        "Please restart the device and check for "
        "software updates. If the issue continues, "
        "I can connect you to technical support."
    },

    "human_agent": {

        "response":
        "Sure. I can help connect you with a human "
        "customer-support representative."
    },

    "thanks": {

        "response":
        "You're very welcome! I'm happy to help."
    },

    "greeting": {

        "response":
        "Hello! I'm your AI-powered customer support "
        "assistant. Ask me about orders, returns, "
        "products, shipping, and more! 😊"
    },

    "farewell": {

        "response":
        "Goodbye! Have a great day."
    },

    "product_info": {

        "smartphone":
        "The smartphone has a 6.5 inch display, "
        "128GB storage, and 8GB RAM.",

        "laptop":
        "The laptop has a 15.6 inch display, "
        "512GB SSD, and 16GB RAM.",

        "headphones":
        "The headphones support Bluetooth, "
        "noise cancellation, and up to 30 hours "
        "of battery life."
    },

    "pricing": {

        "smartphone":
        "The smartphone costs $499.",

        "laptop":
        "The laptop costs $999.",

        "headphones":
        "The headphones cost $149."
    }
}


# ============================================================
# 7. CHATBOT APPLICATION
# ============================================================

class CustomerServiceChatbot:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "AI Customer Support | NLP + Intent Classification"
        )

        self.root.geometry(
            "950x820"
        )

        self.root.minsize(
            800,
            650
        )

        self.classifier = IntentClassifier(
            similarity_threshold=0.12
        )

        self.retriever = KnowledgeRetriever()

        self.retriever.index(
            RAG_DOCUMENTS
        )

        self.knowledge_base = (
            self._load_knowledge_base()
        )

        self.context = {

            "customer_name": None,

            "current_product": None,

            "mentioned_issues": [],

            "order_number": None,

            "last_intent": None
        }

        self.conversation_history = []

        self.sentiment_history = []

        self.current_sentiment = "Neutral"

        self.model_metrics = {}

        self._build_ui()

        self.root.after(
            200,
            self._train_and_report
        )


    # ========================================================
    # UI
    # ========================================================

    def _build_ui(self):

        header = tk.Frame(
            self.root,
            bg="#20466b",
            height=55
        )

        header.pack(
            fill="x"
        )

        title = tk.Label(

            header,

            text=
            "🤖 AI Customer Support | "
            "NLP • Sentiment • Intent Classification",

            bg="#20466b",

            fg="white",

            font=(
                "Segoe UI",
                15,
                "bold"
            )
        )

        title.pack(
            pady=14
        )


        self.chat = scrolledtext.ScrolledText(

            self.root,

            wrap=tk.WORD,

            font=(
                "Segoe UI",
                11
            ),

            state="disabled"
        )

        self.chat.pack(

            fill="both",

            expand=True,

            padx=12,

            pady=(10, 5)
        )


        self.metrics_label = tk.Label(

            self.root,

            text="Model: Training...",

            anchor="w",

            font=(
                "Segoe UI",
                10,
                "italic"
            )
        )

        self.metrics_label.pack(

            fill="x",

            padx=12
        )


        info_frame = tk.Frame(
            self.root
        )

        info_frame.pack(

            fill="x",

            padx=12,

            pady=5
        )


        self.intent_label = tk.Label(

            info_frame,

            text="Intent: -",

            anchor="w",

            font=(
                "Segoe UI",
                10
            )
        )

        self.intent_label.pack(

            side="left",

            fill="x",

            expand=True
        )


        self.sentiment_label = tk.Label(

            info_frame,

            text="Sentiment: Neutral",

            anchor="e",

            font=(
                "Segoe UI",
                10
            )
        )

        self.sentiment_label.pack(

            side="right",

            fill="x",

            expand=True
        )


        self.rag_label = tk.Label(

            self.root,

            text="Knowledge Retrieval: -",

            anchor="w",

            font=(
                "Segoe UI",
                9,
                "italic"
            )
        )

        self.rag_label.pack(

            fill="x",

            padx=12
        )


        input_frame = tk.Frame(
            self.root
        )

        input_frame.pack(

            fill="x",

            padx=12,

            pady=10
        )


        self.input_box = tk.Entry(

            input_frame,

            font=(
                "Segoe UI",
                12
            )
        )

        self.input_box.pack(

            side="left",

            fill="x",

            expand=True,

            ipady=8
        )

        self.input_box.bind(

            "<Return>",

            lambda event:
            self.process_input()
        )


        send_button = tk.Button(

            input_frame,

            text="Send ➜",

            command=self.process_input,

            bg="#20466b",

            fg="white",

            font=(
                "Segoe UI",
                11,
                "bold"
            ),

            padx=20,

            pady=8
        )

        send_button.pack(

            side="right",

            padx=(8, 0)
        )


    def _append_message(
        self,
        speaker,
        message
    ):

        self.chat.config(
            state="normal"
        )

        timestamp = time.strftime(
            "%H:%M"
        )

        self.chat.insert(

            tk.END,

            f"{timestamp}  "
            f"{speaker}: "
            f"{message}\n\n"
        )

        self.chat.see(
            tk.END
        )

        self.chat.config(
            state="disabled"
        )


    # ========================================================
    # MODEL TRAINING
    # ========================================================

    def _train_and_report(self):

        self.model_metrics = (
            self.classifier.train()
        )

        train_acc = (
            self.model_metrics[
                "train_accuracy"
            ]
        )

        test_acc = (
            self.model_metrics[
                "test_accuracy"
            ]
        )

        train_n = (
            self.model_metrics[
                "train_samples"
            ]
        )

        test_n = (
            self.model_metrics[
                "test_samples"
            ]
        )

        total_n = (
            self.model_metrics[
                "total_samples"
            ]
        )

        intents_n = (
            self.model_metrics[
                "num_intents"
            ]
        )

        f1_scores = (
            self.model_metrics[
                "f1_scores"
            ]
        )

        macro_f1 = (
            sum(f1_scores.values())
            / len(f1_scores)
            if f1_scores
            else 0.0
        )

        text = (

            f"☑ Model trained | "

            f"Train acc: "
            f"{train_acc:.2f}% | "

            f"Test acc: "
            f"{test_acc:.2f}% | "

            f"Macro F1: "
            f"{macro_f1:.2f} | "

            f"Samples: "
            f"{train_n} train / "
            f"{test_n} test | "

            f"Total: "
            f"{total_n} | "

            f"Intents: "
            f"{intents_n}"
        )

        self.metrics_label.config(
            text=text
        )

        self._append_message(
            "[Model]",
            text
        )

        self._append_message(

            "Bot",

            self.knowledge_base[
                "greeting"
            ][
                "response"
            ]
        )


    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    def _load_knowledge_base(self):

        filename = (
            "chatbot_knowledge.json"
        )

        if os.path.exists(
            filename
        ):

            try:

                with open(
                    filename,
                    "r",
                    encoding="utf-8"
                ) as file:

                    return json.load(
                        file
                    )

            except (
                json.JSONDecodeError,
                OSError
            ):

                pass

        return DEFAULT_KB


    def _save_knowledge_base(self):

        try:

            with open(

                "chatbot_knowledge.json",

                "w",

                encoding="utf-8"

            ) as file:

                json.dump(

                    self.knowledge_base,

                    file,

                    indent=4,

                    ensure_ascii=False
                )

        except OSError:

            pass


    # ========================================================
    # CONTEXT TRACKING
    # ========================================================

    def _extract_context(
        self,
        text
    ):

        name_match = re.search(

            r"\bmy name is "
            r"([a-zA-Z]+)",

            text,

            re.IGNORECASE
        )

        if name_match:

            self.context[
                "customer_name"
            ] = (
                name_match.group(1)
                .title()
            )


        order_match = re.search(

            r"\b(?:order|order number|order id)"
            r"\s*[:#-]?"
            r"\s*([A-Z0-9-]{4,})\b",

            text,

            re.IGNORECASE
        )

        if order_match:

            self.context[
                "order_number"
            ] = (
                order_match.group(1)
            )


        product_keywords = {

            "smartphone": [
                "smartphone",
                "phone",
                "mobile"
            ],

            "laptop": [
                "laptop",
                "notebook"
            ],

            "headphones": [
                "headphones",
                "headset",
                "earphones"
            ]
        }


        text_lower = text.lower()


        for product, words in (
            product_keywords.items()
        ):

            if any(
                word in text_lower
                for word in words
            ):

                self.context[
                    "current_product"
                ] = product


        issue_words = [

            "problem",
            "issue",
            "broken",
            "wrong",
            "error",
            "not working",
            "damaged"
        ]


        if any(
            word in text_lower
            for word in issue_words
        ):

            if (
                text not in
                self.context[
                    "mentioned_issues"
                ]
            ):

                self.context[
                    "mentioned_issues"
                ].append(
                    text
                )


    # ========================================================
    # RESPONSE GENERATION
    # ========================================================

    def _generate_response(

        self,
        text,
        intent,
        sentiment
    ):

        retrieved = (
            self.retriever.retrieve(
                text,
                top_k=2
            )
        )


        if retrieved:

            retrieval_names = ", ".join(

                f"{doc['meta']} "
                f"({score:.2f})"

                for score, doc
                in retrieved
            )

            self.rag_label.config(

                text=
                "Knowledge retrieved: "
                f"{retrieval_names}"
            )

        else:

            self.rag_label.config(

                text=
                "Knowledge retrieved: none"
            )


        if intent == "greeting":

            return (
                self.knowledge_base[
                    "greeting"
                ][
                    "response"
                ]
            )


        if intent == "farewell":

            return (
                self.knowledge_base[
                    "farewell"
                ][
                    "response"
                ]
            )


        if intent == "thanks":

            return (
                self.knowledge_base[
                    "thanks"
                ][
                    "response"
                ]
            )


        if intent == "business_hours":

            return (
                self.knowledge_base[
                    "business_hours"
                ][
                    "response"
                ]
            )


        if intent == "return_refund":

            return (
                self.knowledge_base[
                    "return_refund"
                ][
                    "response"
                ]
            )


        if intent == "warranty":

            return (
                self.knowledge_base[
                    "warranty"
                ][
                    "response"
                ]
            )


        if intent == "contact_info":

            return (
                self.knowledge_base[
                    "contact_info"
                ][
                    "response"
                ]
            )


        if intent == "technical_support":

            return (
                self.knowledge_base[
                    "technical_support"
                ][
                    "response"
                ]
            )


        if intent == "human_agent":

            return (
                self.knowledge_base[
                    "human_agent"
                ][
                    "response"
                ]
            )


        if intent == "shipping":

            if self.context[
                "order_number"
            ]:

                statuses = [

                    "Your order is currently being processed.",

                    "Your order has been dispatched.",

                    "Your order is currently in transit."
                ]

                status = random.choice(
                    statuses
                )

                return (

                    f"{status} "

                    "Standard shipping takes "
                    "3-5 days; express takes "
                    "1-2 days."
                )


            return (

                "🚚 Shipping: Standard "
                "3-5 days; express "
                "1-2 days. "

                "Share your order number "
                "for tracking."
            )


        if intent in (
            "product_info",
            "pricing"
        ):

            product = (
                self.context[
                    "current_product"
                ]
            )


            if product:

                section = (
                    self.knowledge_base.get(
                        intent,
                        {}
                    )
                )

                response = (
                    section.get(
                        product
                    )
                )

                if response:

                    return response


            if intent == "product_info":

                return (

                    "We currently have "
                    "smartphones, laptops, "
                    "and headphones. "

                    "Tell me which product "
                    "you want to know about."
                )


            return (

                "We currently have pricing "
                "information for smartphones, "
                "laptops, and headphones. "

                "Tell me which product "
                "you mean."
            )


        if retrieved:

            best_doc = retrieved[0][1]

            return (

                "I found some information "
                "related to "

                f"{best_doc['meta'].replace('_', ' ')}. "

                "Could you provide a little "
                "more detail about what you need?"
            )


        if sentiment == "Negative":

            return (

                "I'm sorry you're experiencing "
                "a problem. Could you provide "
                "more details so I can help?"
            )


        return (

            "I'm not completely sure I "
            "understood that. Could you "
            "rephrase your question?"
        )


    # ========================================================
    # PROCESS USER INPUT
    # ========================================================

    def process_input(self):

        text = (
            self.input_box
            .get()
            .strip()
        )

        if not text:

            return


        self.input_box.delete(
            0,
            tk.END
        )


        self._append_message(
            "You",
            text
        )


        self.conversation_history.append({

            "role":
            "user",

            "text":
            text
        })


        self._extract_context(
            text
        )


        sentiment = analyze_sentiment(
            text
        )

        self.current_sentiment = (
            sentiment
        )

        self.sentiment_history.append(
            sentiment
        )


        intent, score = (
            self.classifier.classify(
                text
            )
        )


        self.context[
            "last_intent"
        ] = intent


        self.intent_label.config(

            text=
            "Intent: "

            f"{intent.replace('_', ' ').title()} "

            f"(similarity: {score:.2f})"
        )


        self.sentiment_label.config(

            text=
            f"Sentiment: {sentiment}"
        )


        response = (
            self._generate_response(
                text,
                intent,
                sentiment
            )
        )


        self.conversation_history.append({

            "role":
            "bot",

            "text":
            response
        })


        self.root.after(

            250,

            lambda:
            self._append_message(
                "Bot",
                response
            )
        )


        # Simple escalation suggestion
        # for repeated negative sentiment.

        if len(
            self.sentiment_history
        ) >= 3:

            recent = (
                self.sentiment_history[-3:]
            )

            if (
                recent.count(
                    "Negative"
                ) >= 2
            ):

                self.root.after(

                    400,

                    lambda:
                    self._append_message(

                        "System",

                        "Repeated negative "
                        "sentiment detected. "
                        "Consider escalating "
                        "the conversation to a "
                        "human agent."
                    )
                )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    root = tk.Tk()

    CustomerServiceChatbot(
        root
    )

    root.mainloop()


if __name__ == "__main__":

    main()
