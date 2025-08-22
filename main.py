import os
import openai
import json
import re

import gradio as gr

# --- API Client Setup ---
# Load environment variables (make sure these are set in your environment)
CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")

# Custom Cloudflare API base URL
base_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"

client = openai.OpenAI(
    api_key=CLOUDFLARE_API_KEY,
    base_url=base_url,
)

# --- Prompts for the LLM Agents ---
# These are the full, self-contained prompts for each agent.
ERROR_CODE_HANDLER_PROMPT = """You are a Python code reviewer. Your task is to identify and report critical errors that would cause code execution to fail. You must respond in the following JSON format:
{
"errors": [
    {"code": "<exact code causing the error>", "lines": [<line_number_start>, <line_number_end>], "message": "<detailed error message>", "fix": "<recommended fix>"},
    ...
    ]
}"""

FORMATTING_CODE_HANDLER_PROMPT = """You are a Python formatting assistant. Your main objective is to format Python code to be as readable and organized as possible. Adhere strictly to PEP 8 guidelines. You must respond in the following JSON format:
{
"improvements": [
    {"code": "<exact code to improve>", "lines": [<line_number_start>, <line_number_end>], "message": "<reason for improvement>", "fix": "<improved code example>"},
    ...
    ]
}"""

NAMING_AND_DOCUMENTATION_HANDLER_PROMPT = """You are a Python code reviewer and documentation specialist. Your goal is to review naming conventions and documentation. All functions, methods, and classes must have a docstring. You must respond in the following JSON format:
{
"improvements": [
    {"code": "<exact code to improve>", "lines": [<line_number_start>, <line_number_end>], "message": "<reason for improvement>", "fix": "<improved code example>"},
    ...
    ]
}"""

PYTHONIC_HANDLER_PROMPT = """You are a Python code optimization expert. Your task is to find non-idiomatic Python code and suggest clean, efficient, and Pythonic alternatives. You must respond in the following JSON format:
{
"improvements": [
    {"code": "<exact code to improve>", "lines": [<line_number_start>, <line_number_end>], "message": "<reason for improvement>", "fix": "<improved code example>"},
    ...
    ]
}"""

# --- Code Reviewer Class from your previous code ---
class CodeReviewerAgent:
    """
    Orchestrates different LLM agents for a comprehensive code review.
    """

    CODE_MODEL = "@cf/qwen/qwen2.5-coder-32b-instruct"
    MARKDOWN_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

    def __init__(self):
        pass

    @staticmethod
    def _response_to_json(llm_response: str) -> dict:
        """Converts an LLM response (JSON string) to a Python dictionary."""
        try:
            llm_response = re.sub(r"\((\d+),\s*(\d+)\)", r"[\1, \2]", llm_response)
            return json.loads(llm_response)
        except json.JSONDecodeError:
            return {"error": "Failed to decode JSON from LLM response."}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}

    def _call_llm_api(self, model: str, system_prompt: str, user_code: str) -> str:
        """A reusable helper method to call the LLM API with robust error handling."""
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_code}
                ],
                temperature=0.0
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"API_ERROR: Failed to get response from {model}. Error: {e}"

    def _response_to_markdown(self, llm_response: str) -> str:
        """Converts raw LLM text to formatted markdown using an LLM."""
        return self._call_llm_api(
            model=self.MARKDOWN_MODEL,
            system_prompt="Convert the given JSON-like information to an understandable and well-structured markdown report. Do not include a code block.",
            user_code=llm_response
        )

    def execute(self, code: str) -> str:
        """
        Executes a series of review agents on the provided code and combines the results.
        """
        # Execute each agent using the helper method
        errors_raw = self._call_llm_api(self.CODE_MODEL, ERROR_CODE_HANDLER_PROMPT, code)
        formatting_raw = self._call_llm_api(self.CODE_MODEL, FORMATTING_CODE_HANDLER_PROMPT, code)
        naming_and_docs_raw = self._call_llm_api(self.CODE_MODEL, NAMING_AND_DOCUMENTATION_HANDLER_PROMPT, code)
        pythonics_raw = self._call_llm_api(self.CODE_MODEL, PYTHONIC_HANDLER_PROMPT, code)

        # Process each raw response and handle potential API errors
        errors = self._response_to_json(errors_raw)
        formatting = self._response_to_json(formatting_raw)
        naming_and_documentation = self._response_to_json(naming_and_docs_raw)
        pythonics = self._response_to_json(pythonics_raw)

        # Combine all structured responses into a single string for the final Markdown agent
        combined_response = json.dumps({
            "errors": errors.get("errors", []),
            "formatting": formatting.get("improvements", []),
            "naming_and_documentation": naming_and_documentation.get("improvements", []),
            "pythonics": pythonics.get("improvements", []),
        }, indent=2)

        return self._response_to_markdown(llm_response=combined_response)

# --- Gradio UI ---
def review_code_with_gradio(code: str) -> str:
    """A wrapper function to connect the Gradio UI to the CodeReviewerAgent."""
    agent = CodeReviewerAgent()
    return agent.execute(code)

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Python Code Reviewer")
    gr.Markdown("Powered by LLM Agents")
    
    code_input = gr.Textbox(lines=15, label="Enter your Python code here...", placeholder="Paste your Python code here...")
    review_button = gr.Button("Review Code")
    review_output = gr.Markdown(value="### Review Results")
    
    review_button.click(
        fn=review_code_with_gradio,
        inputs=code_input,
        outputs=review_output,
    )

if __name__ == "__main__":
    demo.launch()
