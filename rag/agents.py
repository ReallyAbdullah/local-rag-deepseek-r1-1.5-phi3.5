from crewai import Agent, Task, Crew, LLM
from langchain_core.messages import AIMessage
from typing import List, Dict, Any, Callable
import logging
from .models import LocalModels
from config import MODEL_CONFIG
# from langchain_ollama import OllamaLLM # Not directly used here, LocalModels encapsulates it
from langchain_core.callbacks import CallbackManager # Added missing import
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from .listeners import CustomCrewEventListener # Added import for the custom listener

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Custom exception for agent-related errors"""

    pass


class ProgressCallback:
    def __init__(self, callback_fn: Callable[[str], None]):
        self.callback_fn = callback_fn

    def on_update(self, message: str):
        if self.callback_fn:
            self.callback_fn(message)


class RAGAgents:
    def __init__(self):
        self.models = LocalModels()
        self.progress_callback = None # This will be set by RAGChain via set_progress_callback

        # Instantiate the custom event listener
        # This instance will automatically register its handlers with the crewai_event_bus
        # because CustomCrewEventListener's __init__ calls super().__init__() and
        # setup_listeners is correctly implemented.
        self.custom_event_listener = CustomCrewEventListener(progress_update_fn=self._emit_progress)

        # Initialize Ollama LLMs with proper configuration
        # Kept StreamingStdOutCallbackHandler for console logging.
        self.complex_llm = LLM( # crewAI's LLM wrapper
            model="ollama/" + MODEL_CONFIG["llm"]["complex"], # Assuming this is how crewAI expects model name for Ollama
            base_url="http://localhost:11434", # Assuming this is how crewAI's LLM wrapper takes base_url
            config={"temperature": 0.7, "top_p": 0.9, "context_window": 4096}, # Config for the model
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        self.simple_llm = LLM( # crewAI's LLM wrapper
            model="ollama/" + MODEL_CONFIG["llm"]["simple"],
            base_url="http://localhost:11434",
            config={"temperature": 0.5, "top_p": 0.9, "context_window": 4096},
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        
        logger.info("RAGAgents initialized with CustomCrewEventListener.")

    def set_progress_callback(self, callback: ProgressCallback):
        """Set callback for progress updates"""
        self.progress_callback = callback
        # The _emit_progress method checks if self.progress_callback is set,
        # so it's safe even if this is called after listener instantiation.

    def _emit_progress(self, message: str):
        """Emit progress update through callback and log it"""
        if self.progress_callback:
            self.progress_callback.on_update(message)
        logger.info(f"[PROGRESS] {message}") # Enhanced logging for progress

    def create_planner_agent(self) -> Agent:
        self._emit_progress("Creating Planner Agent...") # Example of emitting progress for agent creation
        return Agent(
            role="Task Planner",
            goal="Break down complex queries into actionable steps and create execution plans",
            backstory=("""You are an expert at analyzing complex queries and breaking them down 
            into smaller, manageable tasks. You understand document analysis, information retrieval,
            and how to create effective execution plans. You have access to the document context 
            and can use it to create targeted research directives."""),
            allow_delegation=True,
            llm=self.complex_llm, # Use the crewAI LLM wrapper
            tools=[],
            memory=True,
            verbose=True,
        )

    def create_researcher_agent(self) -> Agent:
        self._emit_progress("Creating Researcher Agent...")
        return Agent(
            role="Document Researcher",
            goal="Analyze documents and extract relevant information based on the query context",
            backstory=("""You are a skilled researcher with expertise in document analysis,
            information extraction, and connecting relevant pieces of information. You have 
            access to the full document context and can extract, analyze, and synthesize 
            information effectively. Always ground your analysis in the provided documents."""),
            allow_delegation=True,
            llm=self.complex_llm, # Use the crewAI LLM wrapper
            tools=[],
            memory=True,
            verbose=True,
        )

    def create_writer_agent(self) -> Agent:
        self._emit_progress("Creating Writer Agent...")
        return Agent(
            role="Content Writer",
            goal="Create well-structured, informative responses based on research findings",
            backstory=("""You are an expert writer who creates clear, concise, and informative
            responses. You have access to both the research findings and original documents.
            Always ground your writing in the source material and include specific citations.
            Never make claims without evidence from the provided context."""),
            allow_delegation=False,
            llm=self.simple_llm, # Use the crewAI LLM wrapper
            tools=[],
            memory=True,
            verbose=True,
        )

    def create_crew(self) -> Crew:
        """Creates a crew of agents for handling complex queries"""
        self._emit_progress("🔧 Assembling agent crew...")
        planner = self.create_planner_agent()
        researcher = self.create_researcher_agent()
        writer = self.create_writer_agent()

        crew = Crew(
            agents=[planner, researcher, writer],
            tasks=[], # Tasks will be added in process_query
            verbose=True,
            # process_concurrency=1, # Already sequential by default if not specified
        )
        return crew

    def process_query(
        self, query: str, context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process a complex query using the agent crew"""
        try:
            # self._emit_progress("🚀 Initializing agent crew...") # Covered by listener: CrewKickoffStartedEvent
            crew = self.create_crew()

            # Format context with metadata and content
            formatted_context = []
            for i, doc in enumerate(context):
                doc_content = doc["content"]
                doc_metadata = doc.get("metadata", {})
                formatted_doc = f"""Document {i+1}:
                Source: {doc_metadata.get('source', 'Unknown')}
                Page: {doc_metadata.get('page', 'N/A')}
                Content: {doc_content}
                ---"""
                formatted_context.append(formatted_doc)

            context_str = "\n\n".join(formatted_context)

            # Define tasks with proper context handling and dependencies
            planning_task = Task(
                description=f"""[PLANNING PHASE] Analyze this query and create a research plan:
                Query: {query}
                
                Available Documents:
                {context_str}
                
                Create a detailed plan that:
                1. Identifies key information needs from the query
                2. Maps these needs to specific sections in the available documents
                3. Outlines specific research directives for analyzing the documents
                
                Your output will guide the researcher in extracting relevant information.""",
                agent=crew.agents[0],
                expected_output="A detailed research plan with specific document references",
                # Removed context from task definition as it's not standard for crewAI Task
                # and can be passed via agent tools or main description if necessary.
            )

            research_task = Task(
                description=f"""[RESEARCH PHASE] Execute the research plan on these documents:
                
                Documents:
                {context_str}
                
                Research Plan:
                {{planning_task.output}}
                
                Your task:
                1. Follow the research plan exactly
                2. Extract relevant quotes and evidence from the documents
                3. Analyze and synthesize the information
                4. Maintain clear document references for citations""",
                agent=crew.agents[1], # Researcher
                expected_output="Research findings with citations",
                dependencies=[planning_task],
            )

            writing_task = Task(
                description=f"""[FINAL PHASE] Create a comprehensive response:
                
                Original Query: {query}
                
                Research Findings:
                {{research_task.output}}
                
                Available Documents:
                {context_str}
                
                Requirements:
                1. Directly address the original query
                2. Use only information from the provided documents
                3. Include specific quotes and citations
                4. Structure the response clearly
                5. Maintain academic rigor and accuracy""",
                agent=crew.agents[2], # Writer
                expected_output="Evidence-based response with citations",
                dependencies=[research_task],
            )
            
            crew.tasks = [planning_task, research_task, writing_task]

            # self._emit_progress("🎯 Starting task execution...") # Covered by listener
            result = crew.kickoff()
            # self._emit_progress("✨ Agent processing completed") # Covered by listener

            return {
                "answer": result,
                "model_used": "complex (crewAI)", # Updated model_used
                "agent_info": {
                    "crew_size": len(crew.agents),
                    "tasks_completed": len(crew.tasks), # This is number of defined tasks, not necessarily all executed if error.
                                                       # CrewAI event 'CrewKickoffCompletedEvent' might have better stats.
                },
            }

        except Exception as e:
            error_msg = f"Failed to process query with agents: {str(e)}"
            self._emit_progress(f"❌ Error in agent processing: {error_msg}")
            logger.error(error_msg, exc_info=True)
            raise AgentError(error_msg)
