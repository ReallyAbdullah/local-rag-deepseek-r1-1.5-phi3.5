from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    LLMCallStartedEvent,
    LLMStreamChunkEvent,
    TaskStartedEvent,
    TaskCompletedEvent
)
import logging

logger = logging.getLogger(__name__)

class CustomCrewEventListener(BaseEventListener):
    def __init__(self, progress_update_fn):
        super().__init__()
        self.progress_update_fn = progress_update_fn
        logger.info("CustomCrewEventListener initialized.")

    def setup_listeners(self, crewai_event_bus):
        logger.info("Setting up listeners for CustomCrewEventListener.")

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            self.progress_update_fn("🚀 **Crew Execution Started**")
            logger.info(f"Crew '{event.crew_name}' has started execution!")

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            self.progress_update_fn("🏁 **Crew Execution Finished**")
            # self.progress_update_fn(f"Final output: {event.output}") # Can be verbose
            logger.info(f"Crew '{event.crew_name}' has completed execution! Output: {event.output}")

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(source, event: AgentExecutionStartedEvent):
            task_description = "No specific task description"
            # crewai's AgentExecutionStartedEvent doesn't directly provide task description easily.
            # We might need to infer it or it might be part of the agent's internal state/logging if verbose.
            # For now, we use the agent's role.
            # If event.task is available and has a description, use it. (Event structure might vary)
            # Based on docs, event.agent.role should be available.
            # Let's assume event.task.description might not be directly on this event,
            # so we rely on TaskStartedEvent for task details.
            self.progress_update_fn(f"▶️ **Agent:** {event.agent.role} - Starting execution...")
            logger.info(f"Agent '{event.agent.role}' (ID: {event.agent.id}) started execution.")


        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
            self.progress_update_fn(f"✅ **Agent:** {event.agent.role} - Finished execution.")
            # Output can be very verbose, so avoid sending it directly to UI for now.
            # self.progress_update_fn(f"Agent {event.agent.role} output: {event.output}")
            logger.info(f"Agent '{event.agent.role}' (ID: {event.agent.id}) completed task. Output: {event.output}")

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_started(source, event: TaskStartedEvent):
            self.progress_update_fn(f"📝 **Task Started:** {event.task.description}")
            logger.info(f"Task started: {event.task.description}")

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source, event: TaskCompletedEvent):
            self.progress_update_fn(f"✔️ **Task Completed:** {event.task.description}")
            logger.info(f"Task completed: {event.task.description}")
            # self.progress_update_fn(f"Task output for '{event.task.description}': {event.output}")


        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_call_started(source, event: LLMCallStartedEvent):
            # LLMCallStartedEvent has 'agent' and 'task' attributes
            agent_role = event.agent.role if event.agent else "Unknown Agent"
            self.progress_update_fn(f"🧠 ({agent_role}): Thinking...")
            logger.info(f"LLM call started by agent '{agent_role}' for task '{event.task.description if event.task else 'N/A'}'.")


        @crewai_event_bus.on(LLMStreamChunkEvent)
        def on_llm_stream_chunk(source, event: LLMStreamChunkEvent):
            # LLMStreamChunkEvent has 'agent', 'task', and 'chunk' attributes
            # To avoid too many "Agent X is writing..." messages, just pass the chunk
            self.progress_update_fn(event.chunk)
            logger.debug(f"LLM stream chunk from agent '{event.agent.role if event.agent else ''}': {event.chunk}")

        logger.info("CustomCrewEventListener setup complete.")

# To ensure the listener is registered, an instance needs to be created
# where the crew is defined or run. This file only defines the class.
# The instantiation will happen in rag/agents.py.
