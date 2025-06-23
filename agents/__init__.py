import os
import sys
import json
import random
from typing import List, Dict

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, RootModel
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import create_openai_tools_agent, AgentExecutor

__all__ = ["ingest_prices", "ask_insights"]

# Pydantic model for structured output
class Product(BaseModel):
    sku: str = Field(..., description="product name")
    manufacturer: str = Field(..., description="manufacturer name")
    price: float = Field(..., description="price as float")

class Products(RootModel[List[Product]]):
    """Root model representing a list of products."""

# Output parser for the LLM
_products_parser = PydanticOutputParser(pydantic_object=Products)

# Prompt template for ingestion agent
_ingest_prompt = ChatPromptTemplate.from_template(
    """Ты — эксперт-фармацевт и веб-скрейпер.
Проанализируй HTML-контент страницы и верни JSON-массив товаров.
{format_instructions}
<HTML>{html_text}</HTML>
Только массив JSON, без комментариев."""
).partial(format_instructions=_products_parser.get_format_instructions())

# LLM used for both agents
def _get_llm(temp: float) -> ChatOpenAI:
    return ChatOpenAI(model_name="gpt-4o-mini",
                      api_key=os.getenv("OPENAI_API_KEY"),
                      temperature=temp)


def _fetch_html(url: str) -> str:
    """Download and sanitize page content using retries and random UA."""
    proxy = os.getenv("PROXY")
    proxies = {"http": proxy, "https": proxy} if proxy else None

    user_agents = [
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        ),
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.0 Safari/605.1.15"
        ),
        (
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) "
            "Gecko/20100101 Firefox/121.0"
        ),
    ]

    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    resp = session.get(url, timeout=15, proxies=proxies, headers=headers)
    resp.raise_for_status()

    html_text = resp.text

    return BeautifulSoup(html_text, "lxml").get_text(" ", strip=True)


def _parse_product(html_text: str) -> List[Product]:
    """Call LLM to parse product info from HTML text.

    The model may return a single product or a list of products. This
    function normalizes the output to ``List[Product]``.
    """
    llm = _get_llm(0.0)
    # Obtain raw JSON text from the LLM
    raw = (_ingest_prompt | llm).invoke({"html_text": html_text}).content
    raw = raw[raw.find("[") : raw.rfind("]") + 1]

    # First try to parse using the structured output parser
    try:
        products = _products_parser.invoke(raw)
        return products.__root__
    except Exception:
        pass

    # Fallback: parse JSON manually to handle lists
    try:
        data = json.loads(raw)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(f"Failed to load JSON: {exc}") from exc

    if isinstance(data, list):
        return [Product(**item) for item in data]
    if isinstance(data, dict):
        return [Product(**data)]
    raise ValueError("Unsupported JSON format")


def ingest_prices(csv_path: str) -> pd.DataFrame:
    """Read URLs, scrape pages and return dataframe."""
    urls = pd.read_csv(csv_path)["urls"].dropna()
    records: List[Dict] = []
    for url in urls:
        try:
            text = _fetch_html(url)
            products = _parse_product(text)
            for product in products:
                records.append(product.model_dump())
        except Exception as exc:
            print(f"Failed to process {url}: {exc}", file=sys.stderr)
    df = pd.DataFrame(records, columns=["sku", "manufacturer", "price"])
    tmp = "prices.parquet.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, "prices.parquet")
    return df


def ask_insights(df: pd.DataFrame, question: str) -> str:
    """Answer analytical question about dataframe."""
    llm = _get_llm(0.0)
    tool = PythonAstREPLTool(locals={"df": df})
    tools = [tool]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты — фармацевт-аналитик данных. "
                "Тебе доступен объект pandas DataFrame `df` со столбцами sku, manufacturer, price. "
                "Отвечай на вопрос пользователя, применяя код Python внутри текстовых блоков ```python ... ```. "
                "Если код что-то выводит (таблицу или график), покажи результат пользователю. "
                "В конце — краткий вывод.",
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "Вопрос: {question}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
  
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor.invoke({"question": question})["output"]
