import argparse
import json

from app.agent.runtime import build_agent_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Run the bounded Muse agent graph.")
    parser.add_argument("query", help="Natural-language music request")
    parser.add_argument(
        "--follow-up",
        action="append",
        default=[],
        help="Additional turn to run in the same short-term conversation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = build_agent_runtime()
    try:
        results = []
        conversation_id = None
        for query in (args.query, *args.follow_up):
            result = runtime.run(query, conversation_id=conversation_id)
            conversation_id = result.conversation.conversation_id
            results.append(result.model_dump(mode="json"))
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
