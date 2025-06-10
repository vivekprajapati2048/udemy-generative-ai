from crewai import Agent, LLM
from tools import yt_tool
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"

llm = LLM(model="gpt-4o")

# Blog Content Researcher
blog_researcher = Agent(
    role='Blog Researcher from Youtube Videos',
    goal='Get the relevant video content for the topic {topic} from Youtube Channel',
    verbose=True,
    backstory=(
        'Expert in understanding videos in AI, Data Science, Machine Learning and Generative AI and providing suggestions.'
    ),
    memory=True,
    llm=llm,
    tools=[yt_tool],
    allow_delegation=True
)

# Blog Writer Agent
blog_writer = Agent(
    role='Blog Writer from Youtube Videos',
    goal='Narrate compelling tech stories about the video {topic} from Youtube Channel',
    verbose=True,
    backstory=(
        'With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate, bringing new discoveries to light in an accessible manner.'
    ),
    memory=True,
    llm=llm,
    tools=[yt_tool],
    allow_delegation=False
)