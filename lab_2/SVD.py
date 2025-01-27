import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle

def load_data():
    ratings_data = pd.read_csv("Ratings.csv")
    books_data = pd.read_csv("Books.csv")
    return ratings_data, books_data

def preprocess_data(ratings_data, books_data):
    books_data = books_data[books_data['ISBN'].notnull()]
    books_data['Year-Of-Publication'] = pd.to_numeric(books_data['Year-Of-Publication'], errors='coerce')
    current_year = 2025
    books_data = books_data[books_data['Year-Of-Publication'] <= current_year]

    books_data['Book-Author'].fillna('Unknown', inplace=True)
    books_data['Publisher'].fillna('Unknown', inplace=True)
    books_data['Year-Of-Publication'].fillna(0, inplace=True)

    books_data = books_data.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

    filtered_ratings = ratings_data[ratings_data['Book-Rating'] > 0]

    book_counts = filtered_ratings['ISBN'].value_counts()
    valid_books = book_counts[book_counts > 1].index
    filtered_ratings = filtered_ratings[filtered_ratings['ISBN'].isin(valid_books)]

    user_counts = filtered_ratings['User-ID'].value_counts()
    valid_users = user_counts[user_counts > 1].index
    filtered_ratings = filtered_ratings[filtered_ratings['User-ID'].isin(valid_users)]

    return filtered_ratings, books_data

def train_svd_model(filtered_ratings):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(filtered_ratings[["User-ID", "ISBN", "Book-Rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(random_state=42)
    model.fit(trainset)

    predictions = model.test(testset)
    mae = accuracy.mae(predictions)

    return model, mae

def save_model(model, mae, threshold=1.3):
    if mae < threshold:
        with open("svd_model.pkl", "wb") as file:
            pickle.dump(model, file)
        print(f"Модель сохранена с MAE: {mae}")
    else:
        print(f"Модель не сохранена, так как MAE: {mae} превышает порог {threshold}")

def main():
    ratings_data, books_data = load_data()
    filtered_ratings, filtered_books = preprocess_data(ratings_data, books_data)
    model, mae = train_svd_model(filtered_ratings)
    save_model(model, mae)

if __name__ == "__main__":
    main()
