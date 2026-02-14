# Importing all required packages
import os
import json
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.logging import LoggerFactory
from core.graph import MarketSignals

# Initializing logger
logger_factory = LoggerFactory()
logger = logger_factory.get_logger(__name__)


# Main
if __name__ == "__main__":
    df = pd.read_json("./dataset/lme-al-news-corpus.json")
    df = df.reset_index()
    logger.info(f"Raw articles loaded - {df.shape}")

    text = df["article"].values[len(df)-1]

    signal = MarketSignals(local_inference=False)
    response = signal.get_signals(full_article=text)
    print(response["paraphrase"])

    