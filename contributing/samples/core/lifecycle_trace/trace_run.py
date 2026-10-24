"""Traces one Runner.run_async() invocation end to end, without an API key.

The model is a scripted BaseLlm (call 1 -> function call, call 2 -> text);
everything else (Runner, node runtime, LlmAgent, flow, tools, session service,
plugins) is real ADK code.

Run:  python contributing/samples/core/lifecycle_trace/trace_run.py
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import AsyncGenerator

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.genai import types

STEP = 0


def log(who: str, msg: str) -> None:
  global STEP
  STEP += 1
  print(f"[{STEP:02d}] {who:<22} {msg}")


class ScriptedLlm(BaseLlm):
  """Answers turn 1 with a tool call and turn 2 with text."""

  model: str = "scripted"

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    roles = [
        f"{c.role}:{'/'.join(_part_kind(p) for p in c.parts)}"
        for c in llm_request.contents
    ]
    log("model", f"generate_content_async contents={roles}")
    log("model", f"  system_instruction={llm_request.config.system_instruction!r}")
    log("model", f"  tools={list(llm_request.tools_dict)}")
    last = llm_request.contents[-1].parts[0]
    if last.function_response is None:
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_function_call(
                      name="get_weather", args={"city": "New York"}
                  )
              ],
          )
      )
    else:
      report = last.function_response.response["report"]
      yield LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part(text=f"Answer: {report}")]
          )
      )


def _part_kind(p: types.Part) -> str:
  if p.function_call:
    return f"fc({p.function_call.name})"
  if p.function_response:
    return f"fr({p.function_response.name})"
  return "text"


class TracePlugin(BasePlugin):
  """Logs every runner-level lifecycle hook."""

  def __init__(self) -> None:
    super().__init__(name="trace")

  async def on_user_message_callback(self, *, invocation_context, user_message):
    log("plugin", f"on_user_message inv={invocation_context.invocation_id}")

  async def before_run_callback(self, *, invocation_context):
    log("plugin", "before_run")

  async def before_agent_callback(self, *, agent, callback_context):
    log("plugin", f"before_agent agent={agent.name}")

  async def before_model_callback(self, *, callback_context, llm_request):
    log("plugin", f"before_model ({len(llm_request.contents)} contents)")

  async def after_model_callback(self, *, callback_context, llm_response):
    log("plugin", "after_model")

  async def before_tool_callback(self, *, tool, tool_args, tool_context):
    log("plugin", f"before_tool {tool.name}({tool_args}) fc_id={tool_context.function_call_id}")

  async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
    log("plugin", f"after_tool {tool.name} -> {result}")

  async def on_event_callback(self, *, invocation_context, event):
    log("plugin", f"on_event author={event.author} id={event.id[:8]}")

  async def after_agent_callback(self, *, agent, callback_context):
    log("plugin", f"after_agent agent={agent.name}")

  async def after_run_callback(self, *, invocation_context):
    log("plugin", "after_run")


def get_weather(city: str) -> dict:
  """Retrieves the current weather report for a specified city."""
  log("tool", f"get_weather(city={city!r}) executing")
  # Uncomment to see who calls the tool:
  # traceback.print_stack(limit=12)
  return {"status": "success", "report": f"{city}: sunny, 25C"}


root_agent = Agent(
    name="weather_agent",
    model=ScriptedLlm(),
    instruction="Answer weather questions.",
    tools=[get_weather],
    output_key="last_answer",
)


def dump_event(e) -> str:
  d = e.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
  for k in ("id", "invocation_id"):
    if k in d:
      d[k] = d[k][:8]
  return json.dumps(d, indent=1)


async def main() -> None:
  runner = InMemoryRunner(agent=root_agent, plugins=[TracePlugin()])
  session = await runner.session_service.create_session(
      app_name=runner.app_name, user_id="u1", state={"user_tier": "free"}
  )
  log("main", f"session created id={session.id[:8]} events={len(session.events)}")

  async for event in runner.run_async(
      user_id="u1",
      session_id=session.id,
      new_message=types.UserContent("What's the weather in New York?"),
  ):
    log("main", f"<- yielded author={event.author} final={event.is_final_response()}")
    print(dump_event(event))

  session = await runner.session_service.get_session(
      app_name=runner.app_name, user_id="u1", session_id=session.id
  )
  print("\n=== persisted session.events ===")
  for i, e in enumerate(session.events):
    kinds = [_part_kind(p) for p in (e.content.parts if e.content else [])]
    print(f"{i}: author={e.author:<14} parts={kinds} "
          f"state_delta={dict(e.actions.state_delta)}")
  print("=== session.state ===", session.state)


if __name__ == "__main__":
  asyncio.run(main())
