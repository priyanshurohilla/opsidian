import os, json, pathlib
from slack_sdk import WebClient
from dotenv import load_dotenv
from tools.search import initialize_rag

load_dotenv()
client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

DOCS_DIR = pathlib.Path("./knowledge_base")
DOCS_DIR.mkdir(exist_ok=True)

def fetch_threads_from_channel(channel_id):
    """Fetch recent threads and replies from a Slack channel"""
    threads = []
    history = client.conversations_history(channel=channel_id, limit=200)
    for msg in history["messages"]:
        if "thread_ts" in msg:
            thread_ts = msg["thread_ts"]
            replies = client.conversations_replies(channel=channel_id, ts=thread_ts)["messages"]
            threads.append({
                "thread_ts": thread_ts,
                "channel": channel_id,
                "messages": replies
            })
    return threads

def sync_slack_data():
    channels = client.conversations_list(types="public_channel,private_channel")["channels"]

    for channel in channels:
        channel_id = channel["id"]
        name = channel["name"]
        if name=="new-channel":
            continue
        print(f"Fetching threads from #{name} ({channel_id})")
        threads = fetch_threads_from_channel(channel_id)

        if threads:
            output_file = DOCS_DIR / f"{name}_{channel_id}.json"
            with open(output_file, "w") as f:
                json.dump(threads, f, indent=2)
            print(f"Saved {len(threads)} threads to {output_file}")

    print("Rebuilding vectorstore...")
    initialize_rag()
    print("✅ Sync complete.")

if __name__ == "__main__":
    sync_slack_data()
