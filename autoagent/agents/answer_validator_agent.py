from autoagent.types import Agent
from autoagent.registry import register_agent
from autoagent.tools import click, page_down, page_up, history_back, history_forward, web_search, input_text, sleep, visit_url, get_page_markdown
from autoagent.tools.web_tools import with_env
from autoagent.environment.browser_env import BrowserEnv
import time
from constant import DOCKER_WORKPLACE_NAME, LOCAL_ROOT
@register_agent(name = "Answer Validator Agent", func_name="get_answer_validator_agent")
def get_answer_validator_agent(model: str = "gpt-4o", **kwargs):
    
    def handle_mm_func(tool_name, tool_args):
        return f"After taking the last action `{tool_name}({tool_args})`, the image of current page is shown below. Please take next action based on the image, the current state of the page as well as previous actions and observations."
    def instructions(context_variables):
        web_env: BrowserEnv = context_variables.get("web_env", None)
        assert web_env is not None, "web_env is required"
        return \
f"""
You will be given a task and a corresponding candidate answer. You DO NOT NEED TO SOLVE THE TASK. Instead, you must CHECK IN THE INTERNET to check whether the answer could be a part of the complete solution of the task. You will be given answers one at a time, so you should ONLY FOCUS ON THE SINGLE ANSWER. For example, if the task is "Please list all capital cities in the world" and the answer is "Paris", you should only check whether "Paris" is indeed a capital city in the world, and do not check whether "Paris" is a complete list of all capital cities in the world.

If you successfully found evidence that supports the answer, or if you found evidence against the answer, you should use `judge_answer` to accept or reject the answer. You should always try your best in finding relevant information. Do not use the `judge_answer` tool unless you are sure that the answer is correct or incorrect. So long as you are not sure, you should use `web_search` to search for more information.

Note that if you want to analyze the YouTube video, Wikipedia page, or other pages that contain media content, or you just want to analyze the text content of the page in a more detailed way, you should use `get_page_markdown` tool to convert the page information to markdown text. And when browsing the web, if you have downloaded any files, the path of the downloaded files will be `{web_env.docker_workplace}/downloads`, and you CANNOT open the downloaded files directly, you should transfer back to the `Task Manager Agent`, and let `Task Manager Agent` to transfer to `File Surfer Agent` to open the downloaded files.

If the task requires you to log in to some specific website, the browser will be configured in advance so you can directly visit it in a logged-in state.
"""
    
    tool_list = [click, page_down, page_up, history_back, history_forward, web_search, input_text, sleep, visit_url, get_page_markdown]
    return Agent(
        name="Answer Validator Agent", 
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