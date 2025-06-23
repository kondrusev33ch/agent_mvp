import os
from agents import ingest_prices, ask_insights
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env", override=True)

    csv_path = "urls.csv"
    question = "Что дешевле, аквадетрим капли 30 мл или фортодетрим 30 капсул?"

    df = ingest_prices(csv_path)
    print(df.head())
    answer = ask_insights(df, question)
    print(answer)
