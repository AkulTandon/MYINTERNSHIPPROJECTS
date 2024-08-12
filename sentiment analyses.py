import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

nltk.download('movie_reviews')

def load_data():
    docs = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
    return docs

def preprocess_data(docs):
    texts = [" ".join(doc) for doc, _ in docs]
    labels = [label for _, label in docs]
    return texts, labels

docs = load_data()
texts, labels = preprocess_data(docs)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

model = make_pipeline(CountVectorizer(), MultinomialNB())


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


print(metrics.classification_report(y_test, y_pred))
def predict_sentiment(text):
    prediction = model.predict([text])
    return prediction[0]


sample_text = "I love this movie! It was fantastic and thrilling."
print(f"Sentiment: {predict_sentiment(sample_text)}")

