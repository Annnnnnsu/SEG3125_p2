import gradio as gr
from groq import Groq  # Ensure the groq library is installed

# Set your Groq API key (provided from https://console.groq.com)
API_KEY = "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"

# Create a Groq client instance using the provided API key
client = Groq(api_key=API_KEY)

def groq_cooking_assistant(user_input, temperature, max_tokens, top_p, top_k):
    """
    This function constructs a prompt that includes a system instruction and the user's query,
    then calls the Groq API to generate a cooking assistant response.

    Parameters:
      user_input: The cooking-related query from the user.
      temperature: Controls the randomness of the output.
      max_tokens: Maximum number of tokens to generate.
      top_p: Nucleus sampling parameter.
      top_k: Top-k sampling parameter.

    Returns:
      The generated response from the Groq API.
    """
    # System instruction to define the role and behavior of the assistant
    system_instruction = (
        "You are a helpful and knowledgeable cooking assistant. "
        "Provide clear and concise cooking instructions, recipe suggestions, and ingredient tips."
    )

    # Build the complete prompt by combining the system instruction and the user's input
    prompt = f"{system_instruction}\nUser: {user_input}\nAssistant:"

    # Call the Groq API to generate a response.
    # Adjust the client.generate() method parameters based on Groq's API documentation.
    response = client.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    # Assume the response is a dictionary with a "text" field containing the generated text.
    generated_text = response.get("text", "").strip()

    # Remove the prompt portion from the generated text if present.
    if generated_text.startswith(prompt):
        answer = generated_text[len(prompt):].strip()
    else:
        answer = generated_text

    return answer

# Define the Gradio interface with adjustable input controls.
iface = gr.Interface(
    fn=groq_cooking_assistant,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter your cooking query here...", label="Your Query"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=50, maximum=300, value=150, step=10, label="Max Tokens"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Top-p"),
        gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Top-k")
    ],
    outputs=gr.Textbox(label="Assistant Response"),
    title="AI Cooking Assistant Chatbot using Groq",
    description="Ask for cooking recipes, instructions, or ingredient tips. Powered by the Groq API."
)

if __name__ == "__main__":
    iface.launch()
