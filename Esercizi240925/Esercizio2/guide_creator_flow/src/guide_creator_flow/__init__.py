"""Guide Creator Flow package.

This package contains the components for the "Travel Guide Assistant" demo built
on top of CrewAI and Opik integrations. The package exposes a small Flow
implementation, a Streamlit application and several "crew" implementations
used to evaluate query relevance, perform retrieval (RAG) and generate the
final output.

The intent of this module is purely informational: importing the package will
not perform side effects beyond making the modules available. Use the
submodules to access the public API:

- ``main``: Flow orchestration and the PoemFlow entrypoint
- ``streamlit_app``: Streamlit user interface
- ``crews``: Crew implementations (poem, rag, output, research)
- ``utilis``: helper utilities for conversion, vector store and checks

Examples
--------
>>> from guide_creator_flow.main import PoemFlow
>>> flow = PoemFlow()
>>> flow.state.user_query = "Weekend a Londra"
>>> flow.kickoff()

"""

