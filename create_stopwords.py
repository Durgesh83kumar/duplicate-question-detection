import pickle
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Save to pickle file
with open('stopwords.pkl', 'wb') as f:
    pickle.dump(stop_words, f)

print(f"Created stopwords.pkl with {len(stop_words)} stopwords")
