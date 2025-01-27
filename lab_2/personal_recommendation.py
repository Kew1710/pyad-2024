import pandas as pd
import pickle
from surprise import Dataset, Reader

def load_models():
    with open("svd_model.pkl", "rb") as svd_file:
        svd_model = pickle.load(svd_file)
    with open("sgd_model.pkl", "rb") as linreg_file:
        linreg_model = pickle.load(linreg_file)
    return svd_model, linreg_model

def find_target_user(ratings_data):
    zero_ratings = ratings_data[ratings_data['Book-Rating'] == 0]
    target_user = zero_ratings['User-ID'].value_counts().idxmax()
    return target_user, zero_ratings[zero_ratings['User-ID'] == target_user]

def recommend_books(svd_model, linreg_model, target_user, zero_ratings, books_data):
    predictions = []
    for _, row in zero_ratings.iterrows():
        predicted_rating = svd_model.predict(row['User-ID'], row['ISBN']).est
        if predicted_rating >= 8:
            predictions.append((row['ISBN'], predicted_rating))

    if not predictions:
        return []

    recommended_books = pd.DataFrame(predictions, columns=['ISBN', 'SVD-Predicted-Rating'])
    books_for_linreg = pd.merge(recommended_books, books_data, on='ISBN', how='inner')

    X = books_for_linreg[['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']]
    X_prepared = linreg_model.named_steps['preprocessor'].transform(X)

    books_for_linreg['LinReg-Predicted-Rating'] = linreg_model.named_steps['regressor'].predict(X_prepared)

    sorted_books = books_for_linreg.sort_values(by='LinReg-Predicted-Rating', ascending=False)

    return sorted_books[['ISBN', 'Book-Title', 'LinReg-Predicted-Rating']].values.tolist()

def main():
    ratings_data = pd.read_csv("Ratings.csv")
    books_data = pd.read_csv("Books.csv")

    svd_model, linreg_model = load_models()

    target_user, user_zero_ratings = find_target_user(ratings_data)

    recommendations = recommend_books(svd_model, linreg_model, target_user, user_zero_ratings, books_data)

    if recommendations:
        print("Рекомендации для пользователя с ID", target_user, ":")
        for rec in recommendations:
            print(f"ISBN: {rec[0]}, Title: {rec[1]}, Predicted Rating: {rec[2]:.2f}")
    else:
        print("Нет подходящих рекомендаций для пользователя с ID", target_user)

if __name__ == "__main__":
    main()


