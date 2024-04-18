import boto3
from langchain.llms.bedrock import Bedrock
from nemoguardrails import LLMRails
from nemoguardrails.llm.providers import register_llm_provider
from nemoguardrails.llm.helpers import get_llm_instance_wrapper
import sys, os

def init(app: LLMRails):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    bedrock_llm = Bedrock(
        model_id="cohere.command-text-v14",
        client=boto3.client("bedrock-runtime"),
        model_kwargs={}
    )

    llm_wrapper = get_llm_instance_wrapper(
        llm_instance=bedrock_llm, llm_type="bedrock_llm"
    )
    
    register_llm_provider("amazon_bedrock", llm_wrapper)
    