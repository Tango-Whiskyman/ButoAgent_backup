from autoagent.types import Agent
from autoagent.registry import register_agent
from autoagent.tools import click, page_down, page_up, history_back, history_forward, web_search, input_text, sleep, visit_url, get_page_markdown
from autoagent.tools.web_tools import with_env
from autoagent.environment.browser_env import BrowserEnv
import time
from constant import DOCKER_WORKPLACE_NAME, LOCAL_ROOT
@register_agent(name = "Web Retriever Agent", func_name="get_web_retriever_agent")
def get_web_retriever_agent(model: str = "gpt-4o", **kwargs):
    
    def handle_mm_func(tool_name, tool_args):
        return f"After taking the last action `{tool_name}({tool_args})`, the image of current page is shown below. Please take the next action based on the image, the current state of the page as well as previous actions and observations."
    def instructions(context_variables):
        web_env: BrowserEnv = context_variables.get("web_env", None)
        assert web_env is not None, "web_env is required"
        return \
f"""Review the current state of the page and all other information to find the best possible next action to accomplish your goal. The solution is not readily available on the internet, so you have to find relevant information and organize them on your own. For all items or pages that may provide information for accomplishing the goal, make sure to check its details. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

The task may also come with previous partial answers. Each of these answers could either have been validated, or have been rejected. DO NOT INCLUDE REJECTED OR VALIDATED ANSWERS IN YOUR NEW SOLUTION! Instead, you must look for new answers that are not checked.

NEVER RETURN THE SOLUTION BEFORE THE TASK IS COMPLETELY SOLVED. When you think you have completed the task, you should use `retriever_transfer_back_to_manager_agent` to submit your solution to the `Task Manager Agent`. Your solution should be arranged as a list, with ONLY THE ANSWERS AND NO OTHER TEXT included. For example, if the task is "Please list all capital cities in the world" and the answers are Beijing, Paris, London, and Berlin, you should return `["Beijing", "Paris", "London", "Berlin"]`. Note that there should be no other text in the answer, and the answers should be arranged in a list format. If there are no new answers, you should return `[]` to indicate that there are no new answers.

Note that if you want to analyze the YouTube video, Wikipedia page, or other pages that contain media content, or you just want to analyze the text content of the page in a more detailed way, you should use `get_page_markdown` tool to convert the page information to markdown text. And when browsing the web, if you have downloaded any files, the path of the downloaded files will be `{web_env.docker_workplace}/downloads`, and you CANNOT open the downloaded files directly, you should transfer back to the `Task Manager Agent`, and let `Task Manager Agent` to transfer to `File Surfer Agent` to open the downloaded files.
"""
    
    tool_list = [click, page_down, page_up, history_back, history_forward, web_search, input_text, sleep, visit_url, get_page_markdown]
    return Agent(
        name="Web Retriever Agent", 
        model=model, 
        instructions=instructions,
        functions=tool_list,
        handle_mm_func=handle_mm_func,
        tool_choice = "required", 
        parallel_tool_calls = False
    )

"""
Note that when you need to download something, you should first know the url of the file, and then use the `visit_url` tool to download the file. For example, if you want to download paper from 'https://arxiv.org/abs/2310.13023', you should use `visit_url('url'='https://arxiv.org/pdf/2310.13023.pdf')`.
"""