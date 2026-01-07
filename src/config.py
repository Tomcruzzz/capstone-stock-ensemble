# config.py

DEFAULT_TARGET = "AAPL"
DEFAULT_MARKET = "QQQ"
DEFAULT_SECTOR_ETF = None

# A small peer basket; filtered by availability in data/ at runtime.
DEFAULT_PEERS = [
    "MSFT",
    "AMZN",
    "GOOGL",
    "GOOG",
    "META",
    "NFLX",
    "NVDA",
    "INTC",
    "ADBE",
    "CSCO",
    "ORCL",
    "CRM",
    "TSLA",
    "AVGO",
    "QCOM",
    "AMD",
]

TRAIN_START = "2018-01-01"
TRAIN_END = "2018-12-31"
VAL_START = "2019-01-01"
VAL_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2020-03-31"

MIN_HISTORY_DAYS = 70

