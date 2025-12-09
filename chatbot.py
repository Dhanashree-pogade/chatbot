import tkinter as tk
from tkinter import scrolledtext
import re
import time
import random
import json
import os

class CustomerServiceChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Service Chatbot")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=65, height=20, font=("Arial", 10))
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)
        
        # User input field
        self.user_input = tk.Entry(root, width=55, font=("Arial", 10))
        self.user_input.grid(row=1, column=0, padx=10, pady=10)
        self.user_input.bind("<Return>", self.process_input)
        
        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.process_input, width=10)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)
        
        # Status bar for sentiment display
        self.sentiment_label = tk.Label(root, text="Sentiment: Neutral", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.sentiment_label.grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E, padx=10)
        
        # Knowledge base for dynamic responses
        self.knowledge_base = {
            "business": {
                "hours": "Monday to Friday, 9 AM to 5 PM and Saturday 10 AM to 2 PM",
                "location": "123 Main Street, Business City",
                "contact": {
                    "phone": "1-800-555-1234",
                    "email": "support@example.com",
                    "website": "www.example.com"
                }
            },
            "policies": {
                "return": "Returns accepted within 30 days of purchase with a valid receipt",
                "refund": "Refunds are processed within 5-7 business days",
                "warranty": "Standard 1-year warranty covering manufacturing defects",
                "shipping": "Standard shipping takes 3-5 business days; express shipping is 1-2 days"
            },
            "products": {
                "categories": ["electronics", "clothing", "home goods", "accessories"],
                "popular": ["smartphone", "wireless headphones", "laptop", "smart watch"],
                "electronics": {
                    "smartphone": {"price": "$599-$999", "features": "Latest processor, 5G capability, high-resolution camera"},
                    "laptop": {"price": "$799-$1599", "features": "Fast performance, long battery life, lightweight design"},
                    "headphones": {"price": "$99-$299", "features": "Noise cancellation, wireless, long battery life"}
                }
            }
        }
        
        # Conversation context
        self.context = {
            "last_topic": None,
            "customer_name": None,
            "current_product": None,
            "mentioned_issues": [],
            "order_number": None
        }
        
        # Conversation history for more context-aware responses
        self.conversation_history = []
        
        # Sentiment analysis word lists
        self.positive_words = [
            "happy", "satisfied", "great", "good", "excellent", "love", "like", "thanks", "thank", 
            "awesome", "amazing", "perfect", "wonderful", "fantastic", "helpful", "pleased", "glad",
            "appreciate", "impressive", "enjoy", "nice", "superb", "outstanding", "brilliant"
        ]
        
        self.negative_words = [
            "unhappy", "dissatisfied", "bad", "poor", "terrible", "hate", "dislike", "angry", 
            "disappointed", "horrible", "awful", "useless", "worst", "frustrating", "annoying", 
            "slow", "expensive", "broken", "faulty", "complaint", "issue", "problem", "waste", 
            "fail", "defective", "refund"
        ]
        
        # Customer sentiment tracking
        self.sentiment_history = []
        self.current_sentiment = "neutral"
        
        # Check if we have a saved KB and load it
        self.load_knowledge_base()
        
        # Start with a greeting
        self.update_chat("Bot", "Hello! I'm your customer service assistant. How can I help you today?")
    
    def load_knowledge_base(self):
        """Load knowledge base from file if it exists"""
        if os.path.exists("chatbot_knowledge.json"):
            try:
                with open("chatbot_knowledge.json", "r") as f:
                    self.knowledge_base = json.load(f)
            except:
                pass
    
    def save_knowledge_base(self):
        """Save knowledge base to file"""
        with open("chatbot_knowledge.json", "w") as f:
            json.dump(self.knowledge_base, f, indent=4)
    
    def update_chat(self, sender, message):
        """Update the chat display with a new message."""
        self.chat_display.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M")
        
        if sender == "User":
            self.chat_display.insert(tk.END, f"{timestamp} - You: {message}\n", "user_message")
        else:
            self.chat_display.insert(tk.END, f"{timestamp} - Bot: {message}\n", "bot_message")
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment of the user's message."""
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Check for negation words which could reverse sentiment
        negation_words = ["not", "no", "never", "don't", "doesn't", "didn't", "wasn't", "aren't", "isn't"]
        for negation in negation_words:
            if negation in text_lower:
                # Simple negation handling
                temp = positive_count
                positive_count = negative_count
                negative_count = temp
                break
        
        # Determine sentiment
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def update_sentiment_display(self, sentiment):
        """Update the sentiment display in the UI."""
        self.current_sentiment = sentiment
        
        if sentiment == "positive":
            self.sentiment_label.config(text="Sentiment: Positive 😊", fg="green")
        elif sentiment == "negative":
            self.sentiment_label.config(text="Sentiment: Negative 😞", fg="red")
        else:
            self.sentiment_label.config(text="Sentiment: Neutral 😐", fg="blue")
    
    def extract_context(self, user_input):
        """Extract context from user input"""
        user_input_lower = user_input.lower()
        
        # Extract name if mentioned
        name_match = re.search(r"my name is (\w+)", user_input_lower)
        if name_match:
            self.context["customer_name"] = name_match.group(1).capitalize()
        
        # Extract order number
        order_match = re.search(r"order\s+#?(\d{5,})", user_input_lower)
        if order_match:
            self.context["order_number"] = order_match.group(1)
        
        # Extract product mentions
        for category in self.knowledge_base["products"]:
            if isinstance(self.knowledge_base["products"][category], dict):
                for product in self.knowledge_base["products"][category]:
                    if product in user_input_lower:
                        self.context["current_product"] = product
        
        # Extract issues
        issue_words = ["broken", "defective", "not working", "problem", "issue", "error", "fault"]
        for issue in issue_words:
            if issue in user_input_lower and issue not in self.context["mentioned_issues"]:
                self.context["mentioned_issues"].append(issue)
                
        # Determine topic
        topics = {
            "business hours": ["hours", "open", "close", "time"],
            "returns": ["return", "send back", "money back"],
            "shipping": ["ship", "delivery", "arrive", "tracking"],
            "product info": ["information", "details", "specs", "features"],
            "pricing": ["price", "cost", "how much", "discount"],
            "technical support": ["help", "fix", "repair", "troubleshoot"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in user_input_lower for keyword in keywords):
                self.context["last_topic"] = topic
                break
    
    def generate_dynamic_response(self, user_input):
        """Generate a dynamic response based on the context and user input"""
        user_input_lower = user_input.lower()
        
        # Update context based on user input
        self.extract_context(user_input)
        
        # Check for greetings
        greetings = ["hello", "hi", "hey", "greetings"]
        if any(greeting in user_input_lower for greeting in greetings):
            if self.context["customer_name"]:
                return f"Hello {self.context['customer_name']}! How can I help you today?"
            else:
                return "Hello! How can I help you today?"
        
        # Check for thank you
        if "thank" in user_input_lower or "thanks" in user_input_lower:
            responses = [
                "You're welcome! Is there anything else I can help with?",
                "My pleasure! Do you have any other questions?",
                "Happy to help! Let me know if you need anything else."
            ]
            return random.choice(responses)
        
        # Check for goodbye
        goodbyes = ["bye", "goodbye", "see you", "later"]
        if any(goodbye in user_input_lower for goodbye in goodbyes):
            responses = [
                "Thank you for chatting with us today. Have a great day!",
                "Goodbye! Feel free to return if you have more questions.",
                "Thanks for reaching out. Come back anytime!"
            ]
            return random.choice(responses)
        
        # Handle business hours inquiry
        if "hours" in user_input_lower or "when" in user_input_lower and ("open" in user_input_lower or "close" in user_input_lower):
            return f"Our business hours are {self.knowledge_base['business']['hours']}."
        
        # Handle order status
        if "order" in user_input_lower and ("status" in user_input_lower or "track" in user_input_lower):
            if self.context["order_number"]:
                # Simulate order tracking with random status
                statuses = ["processing", "shipped", "out for delivery", "delivered"]
                return f"I've checked order #{self.context['order_number']}, and it's currently {random.choice(statuses)}."
            else:
                return "Could you provide your order number so I can check its status for you?"
        
        # Handle return policy
        if "return" in user_input_lower or "refund" in user_input_lower:
            return f"Our return policy: {self.knowledge_base['policies']['return']}. {self.knowledge_base['policies']['refund']}."
        
        # Handle product information
        if self.context["current_product"]:
            product = self.context["current_product"]
            # Search for product in knowledge base
            for category in self.knowledge_base["products"]:
                if isinstance(self.knowledge_base["products"][category], dict) and product in self.knowledge_base["products"][category]:
                    prod_info = self.knowledge_base["products"][category][product]
                    return f"The {product} is priced at {prod_info['price']} and features {prod_info['features']}."
        
        # Handle contact information request
        if "contact" in user_input_lower or "phone" in user_input_lower or "email" in user_input_lower:
            contact = self.knowledge_base["business"]["contact"]
            return f"You can reach us by phone at {contact['phone']} or by email at {contact['email']}."
        
        # Handle human representative request
        if "human" in user_input_lower or "representative" in user_input_lower or "agent" in user_input_lower:
            return "I'll connect you with a human representative shortly. Please wait a moment while I transfer your chat."
            
        # Handle warranty information
        if "warranty" in user_input_lower or "guarantee" in user_input_lower:
            return f"Our warranty policy: {self.knowledge_base['policies']['warranty']}."
        
        # Handle shipping inquiry
        if "shipping" in user_input_lower or "delivery" in user_input_lower:
            return f"Our shipping policy: {self.knowledge_base['policies']['shipping']}."
        
        # Learn from user input (very basic learning)
        if "actually" in user_input_lower and self.context["last_topic"]:
            # The user might be correcting information
            topic = self.context["last_topic"]
            self.conversation_history.append(f"User provided new info about {topic}: {user_input}")
            return f"Thank you for that information about {topic}. I'll make a note of it."
        
        # Handle problems or issues
        if self.context["mentioned_issues"] and self.current_sentiment == "negative":
            issue_list = ", ".join(self.context["mentioned_issues"])
            return f"I'm sorry to hear you're experiencing issues with {issue_list}. Let me help troubleshoot or connect you with our technical support team."
        
        # If no other matches, give a contextual default response
        if self.context["last_topic"]:
            return f"I see you're asking about {self.context['last_topic']}. Could you provide more details so I can help you better?"
        
        # Final fallback
        fallbacks = [
            "I'm not sure I understand. Could you please rephrase that?",
            "Could you provide more details about what you need help with?",
            "I'm still learning. Could you explain what you're looking for in a different way?",
            "I'd like to help, but I need a bit more information. What specifically are you interested in?"
        ]
        return random.choice(fallbacks)
    
    def process_input(self, event=None):
        """Process user input and generate a response."""
        user_message = self.user_input.get()
        if user_message.strip() == "":
            return
        
        # Save to conversation history
        self.conversation_history.append(f"User: {user_message}")
        
        # Display user message
        self.update_chat("User", user_message)
        self.user_input.delete(0, tk.END)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(user_message)
        self.sentiment_history.append(sentiment)
        self.update_sentiment_display(sentiment)
        
        # Generate dynamic response
        response = self.generate_dynamic_response(user_message)
        
        # Save to conversation history
        self.conversation_history.append(f"Bot: {response}")
        
        # Simulate typing delay
        self.root.after(1000, lambda: self.update_chat("Bot", response))
        
        # Escalation for persistent negative sentiment
        if sentiment == "negative" and len(self.sentiment_history) >= 3:
            if self.sentiment_history[-3:].count("negative") >= 2:
                escalation_msg = "I notice you seem frustrated. Would you like me to connect you with a human customer service representative?"
                self.root.after(2000, lambda: self.update_chat("Bot", escalation_msg))

def main():
    root = tk.Tk()
    app = CustomerServiceChatbot(root)
    root.mainloop()

if __name__ == "__main__":
    main()