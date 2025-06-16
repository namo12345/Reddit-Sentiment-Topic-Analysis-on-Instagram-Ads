import praw
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from nltk.tokenize import sent_tokenize

# Step 2: Initialize Reddit API credentials (replace with your credentials)
reddit = praw.Reddit(client_id='vkJorP5FIqpCHo3yqFqs5Q',
                     client_secret='ANMY35tmcX_borqrjT1TLvFkhXtEaw',
                     user_agent='YOUR_USER_AGENT')

# Step 3: Fetch data from Reddit about "Instagram Ads"
posts = []
for submission in reddit.subreddit("marketing").search("Instagram ads", limit=50):
    posts.append([submission.title, submission.selftext])

# Step 4: Convert collected data into a pandas DataFrame for easier analysis
posts_df = pd.DataFrame(posts, columns=["Title", "Post Content"])

# Step 5: Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Step 6: Function to get VADER sentiment score
def get_vader_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # The compound score summarizes overall sentiment

# Step 7: Apply VADER sentiment analysis to the Reddit posts
posts_df["Sentiment Score"] = posts_df["Post Content"].apply(get_vader_sentiment)

# Step 8: Display the DataFrame with sentiment scores
print("🧠 Sentiment Analysis for Reddit Posts:\n")
print(posts_df[["Title", "Post Content", "Sentiment Score"]])

# Step 9: Visualize Sentiment Scores for Reddit Data
# Plot histogram for Reddit sentiment distribution
plt.figure(figsize=(8, 5))
plt.hist(posts_df["Sentiment Score"], bins=10, color="lightgreen", edgecolor="black")
plt.title("Reddit Sentiment Distribution")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Step 10: Generate Word Cloud for Reddit posts
feedback_text = " ".join(posts_df["Post Content"].values)  # Combine all text

# Create word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(feedback_text)

# Step 11: Plot word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("🌟 Word Cloud - Instagram Ads Sentiment")
plt.show()

# Step 12: Named Entity Recognition (NER) using SpaCy
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Apply NER to Reddit posts
posts_df["Entities"] = posts_df["Post Content"].apply(extract_entities)

# Step 13: Topic Modeling with LDA (Latent Dirichlet Allocation)
corpus = list(posts_df["Post Content"].values)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Print out topics
for index, topic in enumerate(lda.components_):
    print(f"Topic {index + 1}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6 - 1:-1]])
    print("\n")

# Step 14: Summarize Posts using Hugging Face
summarizer = pipeline("summarization")

def summarize_text(text):
    sentences = sent_tokenize(text)
    summary = summarizer(' '.join(sentences[:3]), max_length=100, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Apply text summarization on selected Reddit posts
posts_df["Summary"] = posts_df["Post Content"].apply(summarize_text)

# Step 15: Key Insights and Summary
# Best performing sentiment in Reddit
best_sentiment_post = posts_df[posts_df["Sentiment Score"] == posts_df["Sentiment Score"].max()]["Title"].values[0]

print("\n🎯 Key Insights from Sentiment Analysis:")
print(f"❤️ Best Sentiment in Reddit: {best_sentiment_post}")
