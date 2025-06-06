import sys
sys.path.append('/home/wangyumeng/projects/ButoAgent/')
from autoagent.registry import register_tool
from browsergym.core.action.highlevel import HighLevelActionSet
from typing import Literal
from autoagent.environment import DockerEnv, DockerConfig, BrowserEnv, VIEWPORT
from browsergym.utils.obs import flatten_axtree_to_str
from dataclasses import dataclass, field
from typing import Dict
from urllib.parse import quote_plus
from autoagent.types import Result
from functools import partial, update_wrapper
from inspect import signature
import tiktoken
from datetime import datetime
from autoagent.util import function_to_json
from autoagent.tools.md_obs import flatten_md_axtree_to_str
from firecrawl import FirecrawlApp, ScrapeOptions
from browsergym.core.action.functions import goto, page
import base64
from io import BytesIO
import requests
from PIL import Image
from autoagent.core import rag, processed_websites, ENABLE_RAG

current_url = None
current_clickables = []
# app = FirecrawlApp(api_key = "fc-075bac3f08f24bb9bcef3d61d499d7cf", api_url="http://localhost:3002")
url_list = ['about:blank']
url_head = 0

def with_env(env: BrowserEnv):
    """将env注入到工具函数中的装饰器"""
    def decorator(func):
        def wrapped(*args, **kwargs):
            return func(env=env, *args, **kwargs)
        
        # 保留原始函数的所有属性
        update_wrapper(wrapped, func)
        # 修改signature，移除env参数
        wrapped.__signature__ = signature(func).replace(
            parameters=[p for p in signature(func).parameters.values() if p.name != 'env']
        )
        return wrapped
    return decorator

def with_two_envs(env: BrowserEnv, code_env: DockerEnv):
    """将env注入到工具函数中的装饰器"""
    def decorator(func):
        def wrapped(*args, **kwargs):
            return func(env=env, code_env=code_env, *args, **kwargs)
        
        # 保留原始函数的所有属性
        update_wrapper(wrapped, func)
        # 修改signature，移除env参数
        wrapped.__signature__ = signature(func).replace(
            parameters=[p for p in signature(func).parameters.values() if p.name not in ['env', 'code_env']]
        )
        return wrapped
    return decorator
@dataclass
class WebObservation:
    content: str  # text content of the page
    url: str # URL of the page
    screenshot: str  # base64-encoded screenshot, png
    open_pages_urls: list[str] # list of open pages
    active_page_index: int  # index of the active page
    dom_object: dict  # DOM object
    axtree_object: dict  # accessibility tree object
    extra_element_properties: dict
    focused_element_bid: str  # focused element bid
    last_browser_action: str  # last browser env action performed
    last_browser_action_error: str # last browser env action error
    error: bool  # error flag

def to_web_obs(obs) -> WebObservation:
    global current_url
    global url_list
    global url_head
    global current_clickables
    new_url = obs.get('url', '')
    if new_url != url_list[url_head]:
        if url_head + 1 < len(url_list) and url_list[url_head + 1] == new_url:
            url_head += 1
            print("history forward!")
            print(url_list)
            print(url_head)
        elif url_head - 1 >= 0 and url_list[url_head - 1] == new_url:
            url_head -= 1
            print("history backward!")
            print(url_list)
            print(url_head)
        else:
            while len(url_list) > 0 and url_list[-1] != current_url:
                url_list.pop()
            url_list.append(new_url)
            url_head = len(url_list) - 1
            print("new page!")
            print(url_list)
            print(url_head)
        current_url = new_url
    obs_dict = dict(
        content=obs['text_content'],  # text content of the page
        url=obs.get('url', ''),  # URL of the page
        # screenshot=obs.get('screenshot', None),  # base64-encoded screenshot, png
        screenshot=None,  # base64-encoded screenshot, png
        open_pages_urls=obs.get('open_pages_urls', []),  # list of open pages
        active_page_index=obs.get(
            'active_page_index', -1
        ),  # index of the active page
        dom_object=obs.get('dom_object', {}),  # DOM object
        axtree_object=obs.get('axtree_object', {}),  # accessibility tree object
        extra_element_properties=obs.get('extra_element_properties', {}),
        focused_element_bid=obs.get(
            'focused_element_bid', None
        ),  # focused element bid
        last_browser_action=obs.get(
            'last_action', ''
        ),  # last browser env action performed
        last_browser_action_error=obs.get('last_action_error', ''),
        error=True if obs.get('last_action_error', '') else False,  # error flag
    )
    #print("extra element properties:\n\n", obs_dict['extra_element_properties'])
    current_clickables = []
    for id in obs_dict['extra_element_properties'].keys():
        if obs_dict['extra_element_properties'][id]['clickable']:
                current_clickables.append(id)
    return WebObservation(**obs_dict)
def wrap_return_value(web_obs: WebObservation, action_description: str = ""):
    error_prefix = ""
    if web_obs.error:
        error_prefix = get_error_prefix(web_obs.last_browser_action, web_obs.last_browser_action_error)
    cur_url = web_obs.url
    try:
        cur_axtree_txt = flatten_axtree_to_str(
            web_obs.axtree_object,
            extra_properties=web_obs.extra_element_properties,
            with_clickable=True,
            filter_visible_only=True,
        )
    except Exception as e:
        cur_axtree_txt = f'Error encountered when browsing.\nError when trying to process the accessibility tree:{str(e)}'
    ret_value = f"""\
{error_prefix}
{action_description}

# Current Page URL:
{cur_url}

# Current Accessibility Tree:
{cur_axtree_txt}

Here is an example with chain of thought of a valid action when clicking on a button:
"
In order to accomplish my goal I need to click on the button with bid 12
```click("12")```
"
""".strip()
    return ret_value

def wrap_return_value_markdown(web_obs: WebObservation, action_description: str = ""):
    error_prefix = ""
    if web_obs.error:
        error_prefix = get_error_prefix(web_obs.last_browser_action, web_obs.last_browser_action_error)
    cur_url = web_obs.url
    try:
        cur_axtree_txt = flatten_md_axtree_to_str(
            web_obs.axtree_object,
            extra_properties=web_obs.extra_element_properties,
            with_clickable=True,
            filter_visible_only=True,
        )
    except Exception as e:
        cur_axtree_txt = f'Error encountered when browsing.\nError when trying to process the accessibility tree:{str(e)}'
    ret_value = f"""\
{error_prefix}
{action_description}

# The current page is converted to markdown format:
{cur_axtree_txt}

If the content is too long, you can use `page_down()` and `page_up()` to navigate through the text.
If you have not yet got the answer and want to back to the previous page, please use `history_back()` to navigate back.
""".strip()
    return ret_value
def get_error_prefix(last_browser_action: str, last_browser_action_error: str) -> str:
    if 'parent_tags' in last_browser_action_error:
        return None
    return f'IMPORTANT! Last action is incorrect:\n{last_browser_action}\nThink again with the current observation of the page.\nThe error message is:\n{last_browser_action_error}'

# @register_tool("click")
# def click(env: BrowserEnv, bid: str, button: Literal["left", "middle", "right"] = "left", modifiers: list[Literal["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]] = []):
#     """
#     Clicks the mouse on the target with the given element bid.
#     Args:
#         bid: The bid of the element to click.
#         button: The button to click.
#         modifiers: The modifiers to click.
#     """
#     try:
#         # 执行动作
#         # action = action_func(*args, **kwargs)
#         button_str = f''', button="{button}"''' if button else ''
#         modifiers_str = f', modifiers={modifiers}' if modifiers else ''
#         action_str = f"""click('{bid}'{button_str}{modifiers_str})"""
        
#         # 与环境交互
#         obs = env.step(action_str)
#         web_obs = to_web_obs(obs)
        
#     except Exception as e:
#         return f"Error encountered when taking action: {action_str}\nError: {e}"
#     ret_value = wrap_return_value(web_obs)
#     return Result(
#             value=ret_value,
#             image=web_obs.screenshot, 
#         )
@register_tool("click")
def click(context_variables, bid: str, button: Literal["left", "middle", "right"] = "left"):
    """
    Clicks the mouse on the target with the given element bid. Note that you can only click on elements in the current page.
    Args:
        bid: The bid of the element to click.
        button: The button to click.
    """
    #print("current_clickables", current_clickables)
    env: BrowserEnv = context_variables.get("web_env", None)
    assert env is not None, "web_env is not set"
    if bid not in current_clickables:
        return f"Error: The element with bid {bid} is not in current page. Please check the current page and try again."
    try:
        # 执行动作
        # action = action_func(*args, **kwargs)
        button_str = f''', button="{button}"''' if button else ''
        action_str = f"""_click_id('{bid}'{button_str})"""
        
        # 与环境交互
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
        
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("page_down")
def page_down(context_variables):
    """
    Scrolls the entire browser viewport one page DOWN towards the end.
    """
    env: BrowserEnv = context_variables.get("web_env", None)
    assert env is not None, "web_env is not set"
    try:    
        action_str = f'scroll(0, {VIEWPORT["height"]-50})'
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    if web_obs.url.startswith("data:text/html;base64,"): 
        ret_value = wrap_return_value_markdown(web_obs)
    else:
        ret_value = wrap_return_value(web_obs)

    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("page_up")
def page_up(context_variables):
    """
    Scrolls the entire browser viewport one page UP towards the beginning.
    """
    env: BrowserEnv = context_variables.get("web_env", None)
    assert env is not None, "web_env is not set"
    try:    
        action_str = f'scroll(0, -{VIEWPORT["height"]-50})'
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    if web_obs.url.startswith("data:text/html;base64,"): 
        ret_value = wrap_return_value_markdown(web_obs)
    else:
        ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("history_back")
def history_back(context_variables):
    """
    Navigates back one page in the browser's history. This is equivalent to clicking the browser back button.
    """
    env: BrowserEnv = context_variables.get("web_env", None)
    assert env is not None, "web_env is not set"
    try:
        #action_str = 'go_back()'
        if url_head - 1 < 0:
            return f"Error: No backward history available."
        url = url_list[url_head - 1]
        action_str = f"_visit_page('{url}')"
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("history_forward")
def history_forward(context_variables):
    """
    Navigates forward one page in the browser's history. This is equivalent to clicking the browser forward button.
    """
    env: BrowserEnv = context_variables.get("web_env", None)
    assert env is not None, "web_env is not set"
    try:
        #action_str = 'go_forward()'
        if url_head + 1 >= len(url_list):
            return f"Error: No forward history available."
        url = url_list[url_head + 1]
        action_str = f"_visit_page('{url}')"
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
@register_tool("input_text")
def input_text(context_variables, bid: str, text: str):
    """
    Types the given text value into the specified field.
    Args:
        bid: The bid of the element to type into.
        text: The text to type into the input field.
    """
    env: BrowserEnv = context_variables.get("web_env", None)
    assert env is not None, "web_env is not set"
    try:
        action_str = f"fill('{bid}', '{text}')"
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )

@register_tool("visit_url")
def visit_url(context_variables, url: str): 
    """
    Navigate directly to a provided URL using the browser's address bar. Prefer this tool over other navigation techniques in cases where the user provides a fully-qualified URL (e.g., choose it over clicking links, or inputing queries into search boxes).
    Args:
        url: The URL to navigate to.
    """
    env: BrowserEnv = context_variables.get("web_env", None)
    assert env is not None, "web_env is not set"
    try:
        if url.startswith(("https://", "http://", "file://", "about:")):
            action_str = f"_visit_page('{url}')"
            obs = env.step(action_str)
            web_obs = to_web_obs(obs)
        elif " " in url:
            query = quote_plus(url)
            action_str = f"_visit_page('https://www.google.com.sg/search?q={query}&hl=en&gl=US')"
            obs = env.step(action_str)
            web_obs = to_web_obs(obs)
        else:
            action_str = f"_visit_page('https://{url}')"
            obs = env.step(action_str)
            web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )

@register_tool("web_search")
def web_search(context_variables, query: str):
    """
    Performs a web search on 'https://www.bing.com/search' with the given query.
    Args:
        query: The query to search for.
    """
    env: BrowserEnv = context_variables.get("web_env", None)
    assert env is not None, "web_env is not set"
    try:
        app = FirecrawlApp(api_url="http://localhost:3002")
        search_result = app.search(query, limit=15)
    except Exception as e:
        return f"Error encountered when performing web search. \nError: {e}"
    ret_value = f"Here are the top 15 search results for your query '{query}':\n"
    for result in search_result.data:
        title = result['title']
        url = result['url']
        #content = result['markdown'][:1000] + "..."
        #ret_value += f"- **Title**: {title}\n  **URL**: {url}\n  **Content**: {content}\n\n"
        description = result['description']
        ret_value += f"- **Title**: {title}\n  **URL**: {url}\n  **Description**: {description}\n\n"
    ret_value += f"If there are multiple websites you want to visit, make sure to visit the most authoritative website first. In order to get details of the pages, please use `visit_url()` first to open the page you are interested in."
    return Result(
            value=ret_value
            #image=web_obs.screenshot, 
        )

@register_tool("sleep")
def sleep(context_variables):
    """
    Wait a short period of time. Call this function if the page has not yet fully loaded, or if it is determined that a small delay would increase the task's chances of success.
    """
    env: BrowserEnv = context_variables.get("web_env", None)
    assert env is not None, "web_env is not set"

    try: 
        action_str = f"noop(3000)"
        obs = env.step(action_str)
        web_obs = to_web_obs(obs)
    except Exception as e:
        return f"Error encountered when taking action: {action_str}\nError: {e}"
    ret_value = wrap_return_value(web_obs)
    return Result(
            value=ret_value,
            image=web_obs.screenshot, 
        )
def truncate_by_tokens(env: DockerEnv, text, max_tokens = 4096, model="gpt-4o-2024-08-06"):
    from autoagent.tools.terminal_tools import create_file, create_directory
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    create_directory(f"{env.docker_workplace}/web_page_markdown_output", context_variables={"code_env": env})
    output_path = f"{env.docker_workplace}/web_page_markdown_output/full_output_{timestamp}.md"
    create_msg = create_file(output_path, content = text, context_variables={"code_env": env})
    # 截断tokens并解码回字符串
    truncated_tokens_bos = tokens[:max_tokens//2]
    truncated_tokens_eos = tokens[-(max_tokens - len(truncated_tokens_bos)):]
    if create_msg.startswith("Error"):
        return encoding.decode(truncated_tokens_bos) + "\n...\n" + encoding.decode(truncated_tokens_eos) + "\n\nThe full convert markdown output is too long, so I want to save them into the file: {0}\n\nBut I got an error: {1}".format(output_path, create_msg)
    
    return encoding.decode(truncated_tokens_bos) + "\n...\n" + encoding.decode(truncated_tokens_eos) + "\n\nThe full convert markdown output is too long, so it is saved in the file: {0}\n\nYou may use the `File Surfer Agent` to view the full output.".format(output_path)

@register_tool("get_page_markdown")
def get_page_markdown(context_variables):
    """
    Get the markdown content of the current page. 
    Use this tool if you need to watch the Youtube video, Wikipedia page, or other pages that contain media content. 
    Note that this tool can only be used after you have visited a valid page.
    """
    env: BrowserEnv = context_variables.get("web_env", None)
    assert env is not None, "web_env is not set"
    try:
        app = FirecrawlApp()
        scrape_result = app.scrape_url(current_url, formats=['markdown'])
    except Exception as e:
        return f"Error encountered when getting page markdown\nError: {e}"
    
    global rag
    global processed_websites
    # if ENABLE_RAG:
    #     if current_url not in processed_websites:
    #         processed_websites.add(current_url)
    #         rag.insert(scrape_result.markdown)
    # Fetch the image from the URL
    #try:
        #response = requests.get(scrape_result.screenshot)
        #response.raise_for_status()
        #image_data = response.content
        # Convert the image to base64
        #image_base64 = base64.b64encode(image_data).decode('utf-8')
    #except Exception as e:
    #    return f"Error encountered when fetching the image: {e}"

    return Result(
            value=scrape_result.markdown,
    #        image=image_base64, 
        )

if __name__ == "__main__":
    env = BrowserEnv(browsergym_eval_env = None, local_root="/home/wangyumeng/projects/ButoAgent/", workplace_name="Auto-Agent")
    # code_env = DockerEnv(DockerConfig(container_name = "gaia_lite_eval", 
    # workplace_name = "workplace_gaia_eval", 
    # communication_port = 12345, 
    # conda_path = "/root/miniconda3"))
    # code_env.init_container()
    # import json
    # web_search_with_env = with_env(env)(web_search)
    # print(json.dumps(function_to_json(web_search_with_env), indent=4))
    # visit_url(env, "https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=LLMRec&oq=")
    # res = page_down(env)
    # print(res.value)
    # res = visit_url(env, 'https://arxiv.org/pdf/2310.13023')
    # print(res.value)


    context_variables = {"web_env": env}
    res = web_search(context_variables, 'Night Heron')
    # res = visit_url(env, 'https://www.reddit.com/r/ChatGPT/comments/1h5ey4m/chatgpt_helped_me_not_blow_up_on_my_boss/')
    print("******search_result\n", res.value)  

    res = visit_url(context_variables, 'https://www.bilibili.com/')
    #print("******visit_url\n", res.value)
    res = get_page_markdown(context_variables)
    print("******get_page_markdown\n", res)
    # res = page_down(context_variables)
    # print("******page_down", res.value)
    # res = history_back(context_variables)
    # print("******history_back", res.value)
    
    # print(function_to_json(page_down))