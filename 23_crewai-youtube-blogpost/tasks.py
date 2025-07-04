from crewai import Task
from tools import yt_tool
from agents import blog_researcher, blog_writer

# Research Task
research_task = Task(
  description=(
    "Identify the video {topic}."
    "Get detailed information about the video from the channel video."
  ),
  expected_output='A comprehensive 3 paragraphs long report based on the {topic} of video content.',
  tools=[yt_tool],
  agent=blog_researcher,
)

# Writing Task with language model configuration
write_task = Task(
  description=(
    "Get the information from the youtube channel on the topic {topic}."
  ),
  expected_output='Summarize the information from the youtube channel video on the topic {topic} and create the content for the blog.',
  tools=[yt_tool],
  agent=blog_writer,
  async_execution=False,  # If True, both agents will work parallely
  output_file='new-blog-post.md'
)
