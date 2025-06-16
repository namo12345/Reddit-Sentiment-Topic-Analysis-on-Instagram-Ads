


# Reddit Sentiment & Topic Analysis on Instagram Ads

This project performs **Sentiment Analysis**, **Topic Modeling**, **NER**, **Summarization**, and **Word Cloud Visualization** on Reddit posts related to _Instagram Ads_ using NLP libraries like **VADER**, **SpaCy**, **LDA**, and **Hugging Face Transformers**.

---

## Project Highlights

- **Reddit Scraping**: Uses PRAW to extract real Reddit posts.
- **Sentiment Analysis**: VADER analyzes the tone of each post.
- **Word Cloud**: Visualizes frequent terms in Reddit discussions.
- **Topic Modeling**: LDA uncovers hidden topics from post content.
- **NER (Named Entity Recognition)**: SpaCy detects key entities.
- **Text Summarization**: Hugging Face transformers generate short summaries for each post.
- **Visualizations**: Histogram of sentiment scores and a word cloud.

---

## Project Structure

```bash
.
├── reddit_instagram_ads_analysis.py   # Main script
├── requirements.txt                   # Required packages
└── README.md                          # This file
````

---

##  Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/reddit-instagram-ads-analysis.git
cd reddit-instagram-ads-analysis
```

2. **Create Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Required Packages**

```bash
pip install -r requirements.txt
```

4. **Download SpaCy English Model**

```bash
python -m spacy download en_core_web_sm
```

5. **Run the Script**

```bash
python reddit_instagram_ads_analysis.py
```

---

## Dependencies

* `praw`
* `pandas`
* `matplotlib`
* `vaderSentiment`
* `wordcloud`
* `spacy`
* `scikit-learn`
* `transformers`
* `nltk`

Install via:

```bash
pip install praw pandas matplotlib vaderSentiment wordcloud spacy scikit-learn transformers nltk
```

---

## Output Samples

### Sentiment Distribution

Shows how positive/negative Reddit posts are.

![Sentiment Histogram](path/to/your/histogram.png)

### Word Cloud

A visual of the most frequent terms in Instagram Ads discussions.

![Word Cloud](path/to/your/wordcloud.png)

---

## Example Output: Topic Modeling

```
Topic 1:
['instagram', 'ads', 'reach', 'targeting', 'roi']

Topic 2:
['budget', 'conversion', 'audience', 'clicks', 'performance']

...
```

---

## Notes

* This script is educational and uses Hugging Face's `pipeline()` for summarization, which may require internet access and a model download.
* You should insert your own **Reddit API credentials** in the script (`client_id`, `client_secret`, `user_agent`).

---

## Future Improvements

* Store outputs in CSV or database
* Deploy as a Streamlit dashboard
* Add classification model for auto-tagging posts

---

## License

This project is licensed under the MIT License. Feel free to use, modify, and share!

---

## Contributing

Pull requests and suggestions are welcome! If you find bugs or want to improve the project, feel free to contribute.

---

## Author

* **Venky Reddy**
  [GitHub](https://github.com/venkyreddy10) | [LinkedIn](https://linkedin.com/in/venkyreddy)

```
