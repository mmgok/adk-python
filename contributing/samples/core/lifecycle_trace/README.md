# Lifecycle trace

Runs one `Runner.run_async()` invocation of a one-tool `LlmAgent` and prints
every lifecycle hook, the `LlmRequest` the model receives, each yielded
`Event`, and the persisted session afterwards.

The model is a scripted `BaseLlm` (call 1 returns a `get_weather` function
call, call 2 returns text), so no API key is needed. Everything else is real
ADK code.

```bash
pip install -e .
python contributing/samples/core/lifecycle_trace/trace_run.py
```

Call path for a root `LlmAgent` (ADK 2.x):

```
Runner.run_async                         runners.py
 └ _node_runner_utils.run_node_async     workflow/_node_runner_utils.py
    ├ append user event, before_run
    ├ asyncio.create_task(_drive_root_node)   ── producer
    │   └ Context._run_node_internal → NodeRunner.run → BaseNode.run
    │      └ LlmAgent._run_impl → run_llm_agent_as_node (chat mode)
    │         └ BaseAgent.run_async → _run_with_lifecycle → LlmAgent._run_async_impl
    │            └ SingleFlow.run_async: loop of _run_one_step_async (one LLM call each)
    └ Runner._consume_event_queue             ── consumer: on_event, append_event, yield
```
