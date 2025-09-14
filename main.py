from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langgraph.prebuilt import create_react_agent
import os
from tools.search import get_vectorstore
from langchain.tools.retriever import create_retriever_tool
load_dotenv()

bot_token = os.environ.get("SLACK_BOT_TOKEN")
app_token = os.environ.get("SLACK_APP_TOKEN")

if not bot_token or not app_token:
    raise ValueError(
        "Missing required environment variables: SLACK_BOT_TOKEN and/or SLACK_APP_TOKEN"
    )
app = App(token=bot_token)

vectorstore = get_vectorstore()
retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(),
    name="search",
    #description="Use this tool to look up accurate information from the company's knowledge base. Always use this when answering user questions about policies, office hours, remote work benefits, or any factual company information."
    description=(
        "Retrieve information from the knowledge base, which includes:\n"
        "- Company documents (e.g., policies, work culture, benefits).\n"
        "- Historical Slack threads where past issues and their solutions were discussed.\n\n"
        "Use this tool whenever a user asks about company policies OR needs help with a technical issue "
        "that might have already been solved in Slack discussions. "
        "Always give the steps to solve the issue if available in the retrieved documents. "
        "Always include all relevant clickable links to the discussion threads. My workspace is stoic-ent5530.slack.com. So make the links according to this "
        "if available, so the user can trace back to the original Slack thread."
    
    ),
)

agent = create_react_agent(model="openai:gpt-4o-mini", tools=[retriever_tool])

@app.event("app_mention")
def handle_hello(body, say, client):
    event = body["event"]
    message = event["text"]
    thread_ts = event.get("thread_ts", event["ts"])

    
    response = agent.invoke({"messages": [{"role": "user", "content": message}]})

    text = response["messages"][-1].content

    say(text=text, thread_ts=thread_ts)

if __name__ == "__main__":
    print("starting")
    handler = SocketModeHandler(app, app_token)
    handler.start()

