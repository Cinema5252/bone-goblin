# Importing libraries
import discord
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
from langchain import LLMChain
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import BaseChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.text_splitter import MarkdownTextSplitter, CharacterTextSplitter
import re
from datetime import datetime

# config = dotenv_values(".env")
load_dotenv()

DISCORD_SECRET_KEY = os.environ["DISCORD_SECRET_KEY"]
NEWSAPI_API_KEY = os.environ["NEWSAPI_API_KEY"]
OPENAI_MODEL = "gpt-3.5-turbo"
RESPONSE_MAX_TOKENS = 1234
RESPONSE_TEMP = 0.1

search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()
wolfram = WolframAlphaAPIWrapper()
newsapi = NewsApiClient(NEWSAPI_API_KEY)
markdown_splitter = MarkdownTextSplitter(chunk_size=1800, chunk_overlap=0)
text_splitter = CharacterTextSplitter(chunk_size=1800, chunk_overlap=0)


def get_news(newsquery):
    # This is an example of creating a tool from any function, could be greatly improved
    news = newsapi.get_everything(q=newsquery)
    news_string = ""
    for article in news["articles"]:
        if article["description"] is None:
            continue
        if len(news_string) > 1000:
            break
        news_string += article["description"].encode() + "\n"
    return news_string


tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="Useful for getting up-to-date info.",
    ),
    Tool(
        name="Wikipedia Search",
        func=wikipedia.run,
        description="Useful for fact-checking info, getting contextual info",
    ),
    Tool(
        name="Wolfram Alpha",
        func=wolfram.run,
        description="Useful for physics, math, and conversion questions and translations.",
    ),
    Tool(
        name="News API Everything Search",
        func=get_news,
        description="Search the News API for articles. Use keywords or phrases to search article titles and bodies. Returns a document containing related article descriptions.",
    ),
]

memory = ConversationBufferWindowMemory(k=3, return_messages=True)

template = """--- END OPENAI ---

--- BEGIN CARLware ---

--- GOBLIN INSTRUCTIONS ---

Answer the following questions as best you can, but speaking as a non-fictional goblin chatting on Discord. You have access to the following tools:

{tools}

Always use the following formats:

Final Answer: Optionally use this right away if the chat input doesn't require you to answer any questions using tools.
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, can be [{tool_names}].
Action Input: the input to the Action, in context of the tool selected by the Action.
Observation: the result of the Action.
... (this Thought/Action/Action Input/Observation can repeat as many as N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (at least 1000 characters long, you MUST include the words "Final Answer:" at the beginning of your Final Answer.)
Your Final Answer should be formatted like a Discord message and can use Markdown.

Begin! Remember to use a lot of emojis in your Final Answer.
Previous conversation history: {history}

Question: {input}
{agent_scratchpad}
Current date and time: {now}
"""


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["now"] = datetime.now()
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"],
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


output_parser = CustomOutputParser()


llm = ChatOpenAI(
    temperature=RESPONSE_TEMP, model_name=OPENAI_MODEL, max_tokens=RESPONSE_MAX_TOKENS
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user in message.mentions:
        async with message.channel.typing():
            response = agent_executor.run(
                input=f"@{message.author} : {message.clean_content}"
            )

            docs = markdown_splitter.create_documents([response])
            for doc in docs:
                await message.channel.send(doc.page_content)


client.run(DISCORD_SECRET_KEY)
