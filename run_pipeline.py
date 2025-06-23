import os
from agents import ingest_prices, ask_insights
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env", override=True)

    csv_path = "urls.csv"
    question = "Перечисли все sku компании REALCAPS"

    df = ingest_prices(csv_path)
    print(df.head())
    answer = ask_insights(df, question)
    print(answer)
