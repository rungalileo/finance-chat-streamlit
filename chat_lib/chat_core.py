import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from galileo import GalileoLogger

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger_debug = logging.getLogger(__name__)

class ChatConfig:
    """Configuration class for chat application"""
    def __init__(self, 
                 openai_api_key: str,
                 pinecone_api_key: str,
                 pinecone_index_name: str,
                 galileo_api_key: str,
                 galileo_project: str,
                 galileo_log_stream: str):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.galileo_api_key = galileo_api_key
        self.galileo_project = galileo_project
        self.galileo_log_stream = galileo_log_stream

# Define RAG response type
class RagResponse:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents

class ChatCore:
    """Core chat functionality shared between Streamlit and Flask apps"""
    
    def __init__(self, config: ChatConfig):
        self.config = config
        
        # Initialize OpenAI client
        logger_debug.info("Initializing OpenAI client")
        self.openai_client = OpenAI(
            api_key=config.openai_api_key
        )
        logger_debug.debug(f"OpenAI API Key loaded: {'*' * 8}{config.openai_api_key[-4:] if config.openai_api_key else 'Not found'}")

        # Initialize Pinecone
        logger_debug.info("Initializing Pinecone client")
        self.pc = Pinecone(
            api_key=config.pinecone_api_key,
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
        logger_debug.debug(f"Pinecone API Key loaded: {'*' * 8}{config.pinecone_api_key[-4:] if config.pinecone_api_key else 'Not found'}")
        
        # Initialize Galileo logger
        os.environ["GALILEO_API_KEY"] = config.galileo_api_key
        os.environ["GALILEO_PROJECT_NAME"] = config.galileo_project
        os.environ["GALILEO_LOG_STREAM_NAME"] = config.galileo_log_stream
        
        self.galileo_logger = GalileoLogger(
            project=config.galileo_project,
            log_stream=config.galileo_log_stream
        )
        
        # Define tools
        self.tools = {
            "getTickerSymbol": {
                "description": "Get the ticker symbol for a company",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company": {
                            "type": "string",
                            "description": "The name of the company"
                        }
                    },
                    "required": ["company"]
                }
            },
            "purchaseStocks": {
                "description": "Purchase a specified number of shares of a stock at a given price.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol to purchase"
                        },
                        "quantity": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "The number of shares to purchase"
                        },
                        "price": {
                            "type": "number",
                            "minimum": 0.01,
                            "description": "The price per share at which to purchase"
                        }
                    },
                    "required": ["ticker", "quantity", "price"]
                }
            },
            "getStockPrice": {
                "description": "Get the current stock price and other market data for a given ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol to look up"
                        }
                    },
                    "required": ["ticker"]
                }
            }
        }
        
        # Format tools for OpenAI API
        self.openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
            for name, tool in self.tools.items()
        ]

    async def get_rag_response(self, query: str, namespace: str, top_k: int) -> Optional[RagResponse]:
        """Get RAG response using Pinecone vector store."""
        try:
            logger_debug.info(f"Making RAG request - Query: {query}, Namespace: {namespace}, Top K: {top_k}")
            
            # Get embeddings for the query
            logger_debug.info("Getting embeddings for query")
            embedding_response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = embedding_response.data[0].embedding
            logger_debug.debug(f"Generated embedding of length: {len(query_embedding)}")
            
            # Initialize Pinecone index
            index_name = self.config.pinecone_index_name
            logger_debug.debug(f"Using Pinecone index: {index_name}")
            if not index_name:
                logger_debug.error("PINECONE_INDEX_NAME environment variable is not set")
                return None
                
            index = self.pc.Index(index_name)
            
            # Query Pinecone
            logger_debug.info("Querying Pinecone index")
            try:
                query_response = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    namespace=namespace if namespace and namespace != "" else None,
                    include_metadata=True
                )
                # Log query response in a serializable format
                logger_debug.debug("Pinecone query response:")
                logger_debug.debug(f"Number of matches: {len(query_response.matches)}")
                if query_response.matches:
                    for i, match in enumerate(query_response.matches):
                        logger_debug.debug(f"Match {i + 1}:")
                        logger_debug.debug(f"  ID: {match.id}")
                        logger_debug.debug(f"  Score: {match.score}")
                        logger_debug.debug(f"  Metadata: {json.dumps(match.metadata, indent=2)}")
            except Exception as e:
                logger_debug.error(f"Error querying Pinecone: {str(e)}", exc_info=True)
                return None
            
            # Process results
            if not query_response.matches:
                logger_debug.warning("No matches found in Pinecone")
                return None
                
            logger_debug.info(f"Found {len(query_response.matches)} matches in Pinecone")
            logger_debug.debug(f"First match score: {query_response.matches[0].score}")
            
            # Log the full metadata structure of the first match
            if query_response.matches:
                first_match = query_response.matches[0]
                logger_debug.info("Metadata structure of first match:")
                logger_debug.debug(f"Full metadata: {json.dumps(first_match.metadata, indent=2)}")
                logger_debug.debug(f"Available metadata keys: {list(first_match.metadata.keys())}")
                logger_debug.debug(f"Match ID: {first_match.id}")
                logger_debug.debug(f"Match score: {first_match.score}")
            
            # Format documents
            documents = [
                {
                    "content": match.metadata.get("text", ""),
                    "metadata": {
                        "score": match.score,
                        **match.metadata  # Include all metadata fields
                    }
                }
                for match in query_response.matches
            ]
            
            logger_debug.info(f"Formatted {len(documents)} documents for response")
            if documents:
                logger_debug.debug(f"First document content preview: {documents[0]['content'][:200]}")
            
            return RagResponse(documents=documents)
            
        except Exception as e:
            logger_debug.error(f"Error in RAG request: {str(e)}", exc_info=True)
            return None
    
    def format_message(self, role: str, content: str = None, tool_calls=None, tool_call_id=None) -> dict:
        """Format a message for the chat.
        
        Args:
            role: The role of the message (system, user, assistant, tool)
            content: The content of the message
            tool_calls: Tool calls for assistant messages
            tool_call_id: Tool call ID for tool messages
            
        Returns:
            A properly formatted message dictionary
        """
        message = {"role": role}
        
        if content is not None:
            message["content"] = content
            
        if role == "assistant" and tool_calls is not None:
            message["tool_calls"] = [{
                "id": tool_call.get("id", f"toolcall-{i}"),
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": tool_call.get("function", {}).get("name", ""),
                    "arguments": tool_call.get("function", {}).get("arguments", "{}")
                }
            } for i, tool_call in enumerate(tool_calls)]
            
        if role == "tool" and tool_call_id is not None:
            message["tool_call_id"] = tool_call_id
            
        return message
            
    async def process_chat_request(self, 
                          messages: List[Dict[str, Any]], 
                          system_prompt: str, 
                          use_rag: bool = True, 
                          namespace: str = "sp500-qa-demo", 
                          top_k: int = 10,
                          model: str = "gpt-4") -> Dict[str, Any]:
        """
        Process a chat request with optional RAG support
        
        Args:
            messages: List of message objects (role, content)
            system_prompt: The system prompt to use
            use_rag: Whether to use RAG
            namespace: Pinecone namespace for RAG
            top_k: Number of top documents to retrieve in RAG
            model: The OpenAI model to use
            
        Returns:
            Dict containing response message and any tool calls
        """
        start_time = time.time()
        
        # Start a new trace
        trace = self.galileo_logger.start_trace(
            input=messages[-1]["content"] if messages else "",
            name="Chat Workflow",
            tags=["chat"],
        )
        
        try:
            messages_to_use = messages.copy()
            prompt = messages[-1]["content"] if messages else ""
            
            # Handle RAG if enabled
            if use_rag and prompt:
                logger_debug.info("RAG enabled, fetching relevant documents")
                rag_response = await self.get_rag_response(prompt, namespace, top_k)
                
                if rag_response and rag_response.documents:
                    logger_debug.info(f"RAG returned {len(rag_response.documents)} documents")
                    
                    # Log RAG retrieval to Galileo
                    self.galileo_logger.add_retriever_span(
                        input=prompt,
                        output=[doc['content'] for doc in rag_response.documents],
                        name="RAG Retriever",
                        duration_ns=int((time.time() - start_time) * 1000000),
                        metadata={
                            "document_count": str(len(rag_response.documents)),
                            "namespace": namespace
                        }
                    )
                    
                    # Add context to system message
                    context = "\n\n".join(doc['content'] for doc in rag_response.documents)
                    logger_debug.debug(f"Adding RAG context to messages: {context[:200]}...")
                    
                    messages_to_use = [
                        {
                            "role": "system",
                            "content": f"{system_prompt}\n\nHere is the relevant context that you should use to answer the user's questions:\n\n{context}\n\nMake sure to use this context when answering questions."
                        },
                        *messages_to_use
                    ]
                else:
                    logger_debug.warning("No RAG documents found for query")
                    messages_to_use = [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        *messages_to_use
                    ]
            elif system_prompt:
                logger_debug.info("Adding system prompt without RAG")
                messages_to_use = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    *messages_to_use
                ]
            
            # Get response from OpenAI
            logger_debug.info("Calling OpenAI API")
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages_to_use,
                tools=self.openai_tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            logger_debug.info("Received response from OpenAI")
            
            # Calculate token counts safely
            input_tokens = len(prompt.split()) if prompt else 0
            output_tokens = len(response_message.content.split()) if response_message.content else 0
            total_tokens = input_tokens + output_tokens

            # Log the API call
            logger_debug.info("Logging API call to Galileo")
            self.galileo_logger.add_llm_span(
                input=[self.format_message(msg["role"], msg["content"]) for msg in messages_to_use],
                output={
                    "role": response_message.role,
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": call.type,
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments
                            }
                        } for call in (response_message.tool_calls or [])
                    ] if response_message.tool_calls else None
                },
                model=model,
                name="OpenAI API Call",
                tools=[{"name": name, "parameters": list(tool["parameters"]["properties"].keys())} 
                      for name, tool in self.tools.items()],
                duration_ns=int((time.time() - start_time) * 1000000),
                metadata={"temperature": "0.7", "model": model},
                tags=["api-call"],
                num_input_tokens=input_tokens,
                num_output_tokens=output_tokens,
                total_tokens=total_tokens
            )
            
            # Conclude the trace
            logger_debug.info("Concluding Galileo trace")
            self.galileo_logger.conclude(
                output=response_message.content,
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=200
            )
            self.galileo_logger.flush()
            
            # Return the response message
            return {
                "role": response_message.role,
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments
                        }
                    } for call in (response_message.tool_calls or [])
                ] if response_message.tool_calls else None
            }
            
        except Exception as e:
            logger_debug.error(f"Error occurred: {str(e)}", exc_info=True)
            
            # Log error and conclude trace
            logger_debug.info("Logging error to Galileo")
            self.galileo_logger.conclude(
                output=f"Error: {str(e)}",
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=500
            )
            self.galileo_logger.flush()
            
            # Return error
            return {
                "role": "assistant",
                "content": f"Error: {str(e)}",
                "tool_calls": None
            } 