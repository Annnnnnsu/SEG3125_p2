import gradio as gr
import os
from groq import Groq

# Initialize Groq client with your provided API key
client = Groq(api_key="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")

# Global variable to store conversation history
conversation_history = []


def set_interview_context(interview_env, interview_position):
    """
    Reset the conversation history and set a system instruction that defines the interview context.

    Parameters:
      interview_env: Interview environment (e.g., Virtual, In-person)
      interview_position: Interview position (e.g., Software Engineer, Data Scientist)

    Returns:
      A status message indicating that the context has been set.
    """
    global conversation_history
    conversation_history = []  # Reset conversation history
    system_message = (
        f"You are an expert interview coach. You help candidates prepare for interviews. "
        f"The interview environment is: {interview_env}. "
        f"The interview position is: {interview_position}. "
        "Provide detailed, practical, and structured advice for interview preparation and answer interview questions accordingly."
    )
    conversation_history.append({"role": "system", "content": system_message})
    return "Interview context has been set."


def chat_with_bot_stream(user_input):
    """
    Append the user's message to the conversation history, call the Groq API to generate a response,
    and return the updated conversation for display.
    """
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model="llama3-70b-8192",  # Use your desired model
        messages=conversation_history,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response_content = ""
    for chunk in completion:
        response_content += chunk.choices[0].delta.content or ""

    conversation_history.append({"role": "assistant", "content": response_content})

    # Format conversation history for display: only show user and assistant messages.
    formatted_conversation = [
        (msg["content"] if msg["role"] == "user" else None,
         msg["content"] if msg["role"] == "assistant" else None)
        for msg in conversation_history if msg["role"] in ["user", "assistant"]
    ]

    return formatted_conversation


# Custom CSS to set the background color to light green.
BACKGROUND_STYLE = """
<style>
body {
    background-color: gray;
}
</style>
"""

# HTML styling for the title with updated font size and light green color
TITLE = """
<style>
h1 { text-align: center; font-size: 28px; margin-bottom: 20px; color: #90EE90; }
</style>
<h1>💼 Interview Coach Chatbot</h1>
"""

with gr.Blocks(theme=gr.themes.Glass(primary_hue="green", secondary_hue="gray", neutral_hue="gray")) as demo:
    # Inject the custom background style
    gr.HTML(BACKGROUND_STYLE)

    with gr.Tabs():
        with gr.TabItem("💬 Chat"):
            gr.HTML(TITLE)
            gr.Markdown("### Set Interview Context")
            with gr.Row():
                interview_env = gr.Textbox(label="Interview Environment", placeholder="e.g., Virtual, In-person, etc.")
                interview_position = gr.Textbox(label="Interview Position",
                                                placeholder="e.g., Software Engineer, Data Scientist, etc.")
                set_context_btn = gr.Button("Set Interview Context")
            context_status = gr.Textbox(label="Context Status", interactive=False)

            # Set the interview context when the button is clicked.
            set_context_btn.click(set_interview_context, inputs=[interview_env, interview_position],
                                  outputs=context_status)

            gr.Markdown("### Interview Chat")
            chatbot = gr.Chatbot(label="Interview Coach Chatbot")
            with gr.Row():
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask your interview question here...",
                    lines=1
                )
                send_button = gr.Button("✋ Ask Question")

            send_button.click(
                fn=chat_with_bot_stream,
                inputs=user_input,
                outputs=chatbot,
                queue=True
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=user_input
            )

            # Display example questions at the bottom of the Chat tab.
            gr.Markdown("#### Example Questions")
            gr.Markdown(
                """
                - **Software Engineer Interview**: "What are some common technical questions in a software engineering interview?"
                - **Self-Introduction**: "How should I introduce myself to make a strong impression in an interview?"
                - **Behavioral Question**: "Can you describe a time when you worked effectively in a team?"
                - **Technical Challenge**: "How do I prepare for a whiteboard coding interview?"
                - **Interview Strategy**: "How should I answer difficult questions during an interview?"
                """
            )

        # Additional tabs can be added if needed.

demo.launch()

