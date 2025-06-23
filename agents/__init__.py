import os
import sys
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_experimental.tools import PythonAstREPLTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_openai_tools_agent, AgentExecutor

__all__ = ["ingest_prices", "ask_insights"]

# Pydantic model for structured output
class Product(BaseModel):
    sku: str = Field(..., description="product name")
    manufacturer: str = Field(..., description="manufacturer name")
    price: float = Field(..., description="price as float")

# Output parser for the LLM
_product_parser = PydanticOutputParser(pydantic_object=Product)

# Prompt template for ingestion agent
_ingest_prompt = ChatPromptTemplate.from_template(
    """Ты — эксперт-фармацевт и веб-скрейпер.
Проанализируй HTML-контент страницы и верни данные.
{format_instructions}
<HTML>{html_text}</HTML>
Только JSON, без комментариев."""
).partial(format_instructions=_product_parser.get_format_instructions())

# LLM used for both agents
def _get_llm(temp: float) -> ChatOpenAI:
    return ChatOpenAI(model_name="gpt-4o-mini",
                      api_key=os.getenv("OPENAI_API_KEY"),
                      temperature=temp)


def _fetch_html(url: str) -> str:
    """Download and sanitize page content."""
    proxy = os.getenv("PROXY")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, timeout=15, proxies=proxies, headers=headers)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml").get_text(" ", strip=True)


def _parse_product(html_text: str) -> Product:
    """Call LLM to parse product info from HTML text."""
    llm = _get_llm(0.0)
    chain = _ingest_prompt | llm | _product_parser
    return chain.invoke({"html_text": html_text})


def ingest_prices(csv_path: str) -> pd.DataFrame:
    """Read URLs, scrape pages and return dataframe."""
    urls = pd.read_csv(csv_path)["urls"].dropna()
    records: List[Dict] = []
    for url in urls:
        try:
            text = _fetch_html(url)
            product = _parse_product(text)
            records.append(product.dict())
        except Exception as exc:
            print(f"Failed to process {url}: {exc}", file=sys.stderr)
    df = pd.DataFrame(records, columns=["sku", "manufacturer", "price"])
    tmp = "prices.parquet.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, "prices.parquet")
    return df


def ask_insights(df: pd.DataFrame, question: str) -> str:
    """Answer analytical question about dataframe."""
    llm = _get_llm(0.2)
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
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
    return executor.invoke({"question": question})["output"]
