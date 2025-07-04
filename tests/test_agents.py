import pandas as pd
from unittest import mock
import agents


class DummyResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


def test_ingest_prices(tmp_path):
    csv = tmp_path / "urls.csv"
    csv.write_text("urls\nhttp://example.com\n")

    html = "<html><body>Ibuprofen ACME 100</body></html>"

    class DummySession:
        def __init__(self):
            self.get = mock.Mock(return_value=DummyResponse(html))

        def mount(self, *_args, **_kwargs):
            pass

    with mock.patch("agents.requests.Session", return_value=DummySession()):
        with mock.patch("agents._parse_product") as mock_parse:
            mock_parse.return_value = [agents.Product(sku="Ibu", manufacturer="ACME", price=100.0)]
            df = agents.ingest_prices(str(csv))

    assert list(df.columns) == ["sku", "manufacturer", "price"]
    assert df.iloc[0]["sku"] == "Ibu"


def test_ask_insights():
    df = pd.DataFrame({"sku": ["A"], "manufacturer": ["B"], "price": [1.0]})
    class DummyExec:
        def __init__(self, *_, **__):
            pass

        def invoke(self, *_args, **_kwargs):
            return {"output": "ok"}

    with mock.patch("agents.AgentExecutor", DummyExec):
        with mock.patch("agents.create_openai_tools_agent"):
            with mock.patch("agents._get_llm"):
                result = agents.ask_insights(df, "question")
    assert result == "ok"


def test_fetch_html_uses_user_agent():
    resp = DummyResponse("<html></html>")

    class DummySession:
        def __init__(self):
            self.get = mock.Mock(return_value=resp)

        def mount(self, *_args, **_kwargs):
            pass

    dummy = DummySession()
    with mock.patch("agents.requests.Session", return_value=dummy):
        agents._fetch_html("http://example.com")
        headers = dummy.get.call_args.kwargs.get("headers", {})
        assert "User-Agent" in headers


def test_parse_product_handles_list():
    content = (
        '[{"sku": "A", "manufacturer": "B", "price": 1.0},'
        ' {"sku": "C", "manufacturer": "D", "price": 2.0}]'
    )

    class DummyPrompt:
        def __or__(self, _other):
            class Seq:
                def invoke(self, *_args, **_kwargs):
                    class Resp:
                        def __init__(self, text):
                            self.content = text

                    return Resp(content)

            return Seq()

    with mock.patch("agents._get_llm"), mock.patch("agents._ingest_prompt", DummyPrompt()):
        result = agents._parse_product("<html></html>")

    assert len(result) == 2
    assert result[0].sku == "A"
