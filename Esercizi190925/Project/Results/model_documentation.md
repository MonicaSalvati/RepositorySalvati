# Model Documentation Template
 
 
### Model type
The model type is described as "advanced large language models" used for intelligent report creation, integrated with Retrieval-Augmented Generation (RAG) capabilities.
 
 
### Model Description
The model used in the system is GPT-4, as specified in the environment configuration.
 
 
### Status
Our RAG was not able to find the information.
 
 
### Relevant Links
Here are the relevant links and resources based on the provided context:
 
1. **GitHub Repository**: [AI-Academy-Final-Project](https://github.com/alessio-buda/AI-Academy-Final-Project) [source:Documents\rag_markdown.txt].
 
2. **Qdrant Database Setup**:
   - **Qdrant Cloud**: [Qdrant Cloud](https://cloud.qdrant.io/) [source:Documents\rag_markdown.txt].
   - **Local Qdrant Server**: [Qdrant Releases](https://github.com/qdrant/qdrant/releases) [source:Documents\rag_markdown.txt].
 
Other requested links (e.g., paper/documentation, initiative demo, conference talk, API link) are not present in the provided context.
 
 
### Developers
Our RAG was not able to find the information.
 
 
### Owner
The author of the project is not explicitly mentioned in the provided context. However, the project repository is associated with "alessio-buda" on GitHub [source:Documents\rag_markdown.txt].
 
 
## Version Details and Artifacts
The version details and artifacts mentioned in the context are:
 
- **Model Configuration**: `MODEL=gpt-4`
- **Azure OpenAI API Version**: `2024-02-15-preview`
- **Azure OpenAI Embedding Deployment**: `text-embedding-ada-002`
- **Qdrant Vector Database Configuration**: Local (`http://localhost:6333`) or Cloud (`https://your-cluster.qdrant.io`) with optional API key for cloud usage.      
 
Artifacts include:
- **Flexible Output**: Multiple output formats with detailed artifacts and process summaries [source:Documents\rag_markdown.txt].
- **MLflow Integration**: Built-in experiment tracking and evaluation capabilities [source:Documents\rag_markdown.txt].
 
For further details, testing can be performed using the command `crewai run`
 
 
## Intended and Known Usage
The intended usage of the system includes:
 
1. **Modular Workflow**: Sequential processing by specialized crews for input sanitization, analysis, and report writing.      
2. **AI-Powered Report Creation**: Utilizing advanced large language models for intelligent content generation.
3. **RAG Integration**: Employing Retrieval-Augmented Generation with Qdrant vector database to enhance knowledge capabilities.
4. **Security Measures**: Comprehensive input validation, secure infrastructure, and regular security updates.
5. **Experiment Tracking**: Built-in MLflow integration for tracking and evaluating experiments.
6. **Flexible Outputs**: Supporting multiple output formats with detailed artifacts and summaries.
 
These features are designed to ensure robust, secure, and intelligent processing for diverse use cases.
 
 
## Model Architecture
The model architecture involves several key components:
 
1. **Data Collection and Preprocessing**:
   - Input sanitization is performed by a specialized crew to ensure data consistency and security. This includes comprehensive input validation and security checks before processing [source:Documents\rag_markdown.txt].
 
2. **Data Splitting**:
   - The system uses a Pydantic-based state model (`ReportState`) to manage data flow across different processing stages. This model includes fields for user input, sanitized input, security checks, analysis results, and the final report [source:Documents\rag_markdown.txt].
 
3. **Sequential Processing**:
   - The architecture follows a linear communication pattern where each crew processes the output of the previous crew. This ensures a structured and modular workflow [source:Documents\rag_markdown.txt].
 
4. **RAG Integration**:
   - Retrieval-Augmented Generation (RAG) is implemented using a Qdrant vector database to enhance knowledge capabilities during analysis and report generation [source:Documents\rag_markdown.txt].
 
5. **MLflow Integration**:
   - The system includes MLflow for experiment tracking and evaluation, supporting iterative improvements and performance monitoring [source:Documents\rag_markdown.txt].
 
6. **Output Generation**:
   - The final output is flexible, offering multiple formats with detailed artifacts and process summaries [source:Documents\rag_markdown.txt].
 
This modular and security-first design ensures robust data handling and intelligent report creation.
 
 
## Model Training Process
Our RAG was not able to find the information.
 
 
## Model Training and Validation (metrics)
The performance metrics used include:
 
1. **Security Accuracy** with a score of 91.3% for the "Sanitize Crew" component.
2. **LLM Relevance** with a score of 8.5/10 for the "Analysis Crew" component.
3. **Context Precision** for the "Writer Crew (RAG)" component, though the score is not yet determined (TBD) [source:Documents\rag_markdown.txt].
 
 
## Model Interpretability and Explainability
Our RAG was not able to find the information.
 
 
## Documentation Metadata
Our RAG was not able to find the information.