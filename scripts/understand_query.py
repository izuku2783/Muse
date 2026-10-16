import argparse
import json

from app.llm.factory import get_llm_provider
from app.query_understanding.service import QueryUnderstandingService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Muse's validated intent and filters from a music query."
    )
    parser.add_argument("query", help="Natural-language music query")
    args = parser.parse_args()

    provider = get_llm_provider()
    try:
        result = QueryUnderstandingService(provider).understand(args.query)
    finally:
        provider.close()

    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
