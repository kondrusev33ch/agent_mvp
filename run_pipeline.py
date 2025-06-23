from agents import ingest_prices, ask_insights
import argparse

parser = argparse.ArgumentParser(description="Pharma price pipeline")
parser.add_argument("--csv", required=True, help="CSV file with urls column")
parser.add_argument("--question", required=True, help="Question for insights agent")
args = parser.parse_args()

if __name__ == "__main__":
    df = ingest_prices(args.csv)
    print(df.head())
    answer = ask_insights(df, args.question)
    print(answer)
