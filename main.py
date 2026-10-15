from src.recommender import recommend

if __name__ == "__main__":
    print("🎬 Movie Recommendation System 🎬\n")
    genre = input("Enter a genre (e.g. Action, Drama, Comedy): ")
    recommendations = recommend(genre, n=5)

    if len(recommendations):
        print("\nRecommended Movies:\n")
        for _, row in recommendations.iterrows():
            print(f"🎞️ {row['title']}  |  {row['genres']}  |  Dir: {row['director']}")
