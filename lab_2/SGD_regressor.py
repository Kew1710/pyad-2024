import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

def load_data():
    ratings_data = pd.read_csv("Ratings.csv")
    books_data = pd.read_csv("Books.csv")
    return ratings_data, books_data

def preprocess_data(ratings_data, books_data):
    books_data['Year-Of-Publication'] = pd.to_numeric(books_data['Year-Of-Publication'], errors='coerce')
    books_data = books_data[books_data['Year-Of-Publication'] <= 2025]
    books_data.dropna(subset=['Book-Author', 'Publisher', 'Year-Of-Publication'], inplace=True)

    books_data = books_data.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

    ratings_data = ratings_data[ratings_data['Book-Rating'] > 0]

    avg_ratings = ratings_data.groupby('ISBN')['Book-Rating'].mean().reset_index()
    avg_ratings.rename(columns={'Book-Rating': 'Avg-Rating'}, inplace=True)

    merged_data = pd.merge(books_data, avg_ratings, on='ISBN', how='inner')

    return merged_data

def train_model(merged_data):
    X = merged_data[['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']]
    y = merged_data['Avg-Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('title_tfidf', TfidfVectorizer(max_features=500), 'Book-Title'),
            ('author_ohe', OneHotEncoder(handle_unknown='ignore'), ['Book-Author']),  # Оборачиваем в список
            ('publisher_ohe', OneHotEncoder(handle_unknown='ignore'), ['Publisher']),  # Оборачиваем в список
            ('year_scaler', StandardScaler(), ['Year-Of-Publication'])  # Оборачиваем в список
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', SGDRegressor(random_state=42, max_iter=1000, tol=1e-3))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return pipeline, mae

def save_model(model, mae, threshold=1.5):
    if mae < threshold:
        with open("sgd_model.pkl", "wb") as file:
            pickle.dump(model, file)
        print(f"Модель сохранена с MAE: {mae}")
    else:
        print(f"Модель не сохранена, так как MAE: {mae} превышает порог {threshold}")

def main():
    ratings_data, books_data = load_data()
    merged_data = preprocess_data(ratings_data, books_data)
    model, mae = train_model(merged_data)
    save_model(model, mae)

if __name__ == "__main__":
    main()