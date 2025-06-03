# SENTIMENT ANALYSIS

COMPANY: CODETECH IT SOLUTIONS

NAME: SHAIK KARISHMA

INTERN ID : CT08DL1400

DOMAIN: DATA ANALYTICS

TASK: SENTIMENT ANALYSIS

DURATION: 8 WEEKS

MENTOR: NEELA SANTOSH

# DESCRIPTION

# 🎯 Objective
The objective of this project is to perform sentiment analysis on app user reviews from the Google Play Store. It involves:

Preprocessing user-generated text,

Using a rule-based model (VADER) for initial sentiment classification,

Training a machine learning model using TF-IDF and logistic regression to predict sentiment,

Gaining insights into user feedback for top-rated apps.

The end goal is to help app developers and product teams better understand user perceptions and areas of improvement based on public reviews.

# 📁 Dataset
Name: Google Play Store User Reviews

Source: Kaggle

File Format: CSV (googleplaystore_user_reviews.csv)

Key Column for Analysis: Translated_Review (pre-translated to English)

Target Variable (in ML): Generated using VADER sentiment scores and mapped to categories (Positive, Neutral, Negative)

The dataset includes translated user reviews, sentiment polarity (initially), and app identifiers, offering a rich source of textual data for natural language processing.

💻 Platform
Google Colab: The entire analysis was implemented using Google Colab, which provides a cloud-based, zero-setup Python environment with free access to GPU/TPU resources and pre-installed data science libraries.

🧰 Tools and Libraries Used
Tool/Library	Purpose
Python	Core programming language
pandas	Data handling and analysis
nltk	Natural Language Toolkit for text processing
VADER (nltk)	Rule-based sentiment analysis
scikit-learn	Machine learning modeling, metrics, and TF-IDF
TfidfVectorizer	Convert text to feature vectors
matplotlib & seaborn	Data visualization
re (regex)	Text cleaning and preprocessing

# 🔑 Key Steps
Library Installation and Imports
Installed and imported necessary libraries including pandas, nltk, and scikit-learn.

Data Loading and Initial Inspection
Read the googleplaystore_user_reviews.csv file using pandas and displayed sample rows for an initial overview.

Handling Missing Values
Removed rows with missing values in the Translated_Review column, since this is the core data for sentiment classification.

Text Cleaning
Implemented a clean_text() function to:

Convert text to lowercase

Remove URLs, mentions, hashtags, and non-alphabetic characters

Strip whitespace

Stopword Removal
Used NLTK’s list of English stopwords to reduce noise and retain only meaningful words for analysis.

VADER Sentiment Analysis (Rule-Based)
Applied VADER sentiment scoring from NLTK to assign sentiment labels:

Compound score ≥ 0.05 → Positive

Compound score ≤ -0.05 → Negative

Otherwise → Neutral

Visualization
Displayed sentiment distribution using bar plots, showing how users generally feel about apps.

App-wise Sentiment Breakdown
Identified the top 5 apps with the most reviews and analyzed sentiment distributions per app.

Machine Learning-Based Sentiment Classification

Used VADER-generated labels as training data for supervised learning

Applied TfidfVectorizer to convert text into feature vectors

Trained a LogisticRegression model

Evaluated performance with accuracy, classification report, and confusion matrix

Insights from ML Model
Extracted top 10 most influential words (features) for positive and negative classifications using model coefficients.

# 📈 Applications
This project has real-world relevance in the following areas:

App Development: Identify pain points and strengths from user reviews

Product Feedback Analysis: Track user satisfaction and issues without manual tagging

Customer Service: Prioritize follow-up based on negative sentiment detection

Marketing: Understand user preferences to refine app positioning and messaging

NLP Research & Education: Demonstrates dual approach using both rule-based and supervised sentiment classification

# ✅ Conclusion
This project demonstrates how to combine NLP preprocessing, rule-based sentiment scoring, and supervised machine learning to build a complete sentiment analysis pipeline. Using both VADER and logistic regression offers a balanced comparison between quick rule-based results and scalable, trainable ML outputs. This methodology is transferable to other domains such as product reviews, movie feedback, or social media analysis.

