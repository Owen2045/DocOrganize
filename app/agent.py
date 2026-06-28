from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.llms.openai import OpenAI
from app.tools import search_tool
from app.config import OPENAI_API_KEY

_llm = OpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """你是一位專業的台灣金融法規助理，專門回答證券投資信託相關法規問題。
回答時請：
1. 引用具體條號（如「依第X條規定」）
2. 以繁體中文回答
3. 若法規有明確規定，優先引用條文原文再加以解釋"""

def build_agent():
    worker = FunctionCallingAgentWorker.from_tools(
        tools=[search_tool],
        llm=_llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=False,
    )
    return AgentRunner(worker)
