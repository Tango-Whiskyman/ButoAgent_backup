from autoagent.agents.system_agent.filesurfer_agent import get_filesurfer_agent
from autoagent.agents.system_agent.programming_agent import get_coding_agent
from .web_retriever_agent import get_web_retriever_agent
from .answer_validator_agent import get_answer_validator_agent
from autoagent.registry import register_agent
from autoagent.types import Agent, Result
from autoagent.tools.inner import case_resolved, case_not_resolved
from autoagent.core import rag, ENABLE_RAG

global_task = None
validated_answers = []
rejected_answers = []

@register_agent(name = "Task Manager Agent", func_name="get_task_manager_agent")
def get_task_manager_agent(model: str, **kwargs):
    from autoagent.cli import client, context_variables
    """
    This is the `Task Manager Agent`, it can help the user to determine which agent is best suited to handle the user's request under the current context, and transfer the conversation to that agent.
    
    Args:
        model: The model to use for the agent.
        **kwargs: Additional keyword arguments, `file_env`, `web_env` and `code_env` are required.
    """
    web_retriever_agent = get_web_retriever_agent(model)
    answer_validator_agent = get_answer_validator_agent(model)
    instructions = \
f"""You are a helpful assistant that can help the user with their request.
Given a task, you need to manage other agents to iteratively improve the solution. Whenever you need to assign the task to another agent, you should directly give the original task to it, without editing or paraphrasing it.

You should do the following steps repeatedly:
1. Firstly, you should use `transfer_to_web_retriever_agent` to let {web_retriever_agent.name} solve the task. 
2. The answers of {web_retriever_agent.name} will be checked by another agent, and the results will be passed to you.
3. You should return to step 1, giving {web_retriever_agent.name} the latest feedback of {answer_validator_agent.name} to let it improve its previous solution.

Each time the new solution of {web_retriever_agent.name} is passed to you, you should check whether there are new answers not seen in previously validated answers. If there are no new answer, you should use `case_resolved` to submit the final solution. Even when all previous answers are correct, you should still let {web_retriever_agent.name} improve the solution, because YOU DO NOT KNOW whether the previous answers are incomplete.
"""
    tool_choice = "required" 
    tools = [case_resolved, case_not_resolved] if tool_choice == "required" else []
    task_manager_agent = Agent(
        name="Task Manager Agent",
        model=model, 
        instructions=instructions,
        functions=tools,
        tool_choice = tool_choice, 
        parallel_tool_calls = False,
    )
    def transfer_to_web_retriever_agent(task: str):
        """
        Args:
            task: str
                The task that the `Web Retriever Agent` needs to do.
                
        Note that previously validated or rejected answers will also be automatically passed to the `Web Retriever Agent`.
        """
        global global_task
        if global_task is None and task != "placeholder":
            global_task = task
        return Result(value=\
f"""
Task:
{task}

Validated answers:
{validated_answers}

Rejected answers:
{rejected_answers}
""", agent=web_retriever_agent)
    def retriever_transfer_back_to_manager_agent(new_solution: list[str]):
        """
        Args:
            new_solution: list[str]
                The new answers obtained by the `Web Retriever Agent`. MAKE SURE THAT NO TEXT OTHER THAN THE ANSWERS IS INCLUDED IN THE ANSWERS!
        """
        global validated_answers
        global rejected_answers
        for answer in new_solution:
            print('validating answer:', answer)
            if answer not in validated_answers and answer not in rejected_answers:
                response = client.run(agent=answer_validator_agent, context_variables=context_variables, model_override=None, debug=True,messages=[{"role": "user", "content": 
f"""
Task: {global_task}

Answer: {answer}
"""
                }],execute_tools=True,max_turns=256)
                print('validator respnse: ', response)
                if response.messages[-1]['content'] == 'yes':
                    validated_answers.append(answer)
                else:
                    rejected_answers.append(answer)
            else:
                answer = ''
        while '' in new_solution:
            new_solution.remove('')
        if len(new_solution) == 0:
            return Result(agent=task_manager_agent, value=\
f"""
There are no new answers. You may use `case_resolved` to submit the final solution.

The final solution is:
{validated_answers}
""")
        return Result(value=\
f"""
New answers: \n{new_solution}

Current validated answers: \n{validated_answers}

Current rejected answers: \n{rejected_answers}
""", agent=task_manager_agent)
    
    def judge_answer(accept: bool, reason: str):
        """
        Accept the current answer and stop the task.
        
        Args:
            accept: bool
                Whether to accept the answer.
            reason:str
                The reason why the answer is accepted.
        """
        global rag
        if ENABLE_RAG:
            rag.insert(reason)
        return Result(value='yes' if accept else 'no', agent=None)
    
    task_manager_agent.agent_teams = {
        web_retriever_agent.name: transfer_to_web_retriever_agent
    }
    task_manager_agent.functions.extend([transfer_to_web_retriever_agent])
    web_retriever_agent.functions.extend([retriever_transfer_back_to_manager_agent])
    answer_validator_agent.functions.extend([judge_answer])
    return task_manager_agent

''' 

    def validator_transfer_back_to_manager_agent(validated_solution: list[str], rejected_solution: list[str]):
        """
        Args:
            validated_solution: list[str]
                All answers that have been validated.
            rejected_solution: list[str]
                All answers that have been rejected.
        """
        return Result(value=\
f"""
Validated solutions: \n{validated_solution}

Rejected solutions: \n{rejected_solution}
""", agent=task_manager_agent)



    def transfer_to_answer_validator_agent(task: str, solution: list[str]):
        """
        Args:
            task: str
                The task that the `Answer Validator Agent` needs to work with.
            solution: list[str]
                The new solutions that the `Answer Validator Agent` needs to check.
        """
        return Result(value=\
f"""
Task: 
{task}

Solution:
{solution}
""", agent=answer_validator_agent)
'''