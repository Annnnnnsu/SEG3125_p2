import gradio as gr
import os
from groq import Groq

# Global preferences dictionary
user_preferences = {}

# Initialize Groq client with your provided API key
client = Groq(api_key="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")

# Global variable to store conversation history
conversation_history = []

# Track consecutive unknown responses
consecutive_unknown_count = 0


def set_interview_context(interview_env, interview_position):
    global conversation_history
    conversation_history = []
    prefs_text = ""
    if user_preferences:
        prefs_text = " You should also consider the user's stored preferences: "
        for k, v in user_preferences.items():
            prefs_text += f"[{k} -> {v}] "

    system_message = (
            f"You are an expert interview coach. You help candidates prepare for interviews. "
            f"The interview environment is: {interview_env}. "
            f"The interview position is: {interview_position}. "
            "If asked unrelated questions, politely redirect them back to interview preparation topics and explain how focusing on interview skills is beneficial. "
            "Provide detailed, practical, and structured advice for interview preparation and answer interview questions accordingly."
            + prefs_text
    )
    conversation_history.append({"role": "system", "content": system_message})
    return "Interview context has been set." + (" Preferences found." if prefs_text else "")


def add_preference(env_dropdown, env_custom, pos_dropdown, pos_custom):
    global user_preferences
    env = env_custom if env_dropdown == "Other" and env_custom.strip() else env_dropdown
    pos = pos_custom if pos_dropdown == "Other" and pos_custom.strip() else pos_dropdown
    user_preferences["engagement_env"] = env
    user_preferences["engagement_pos"] = pos
    return f"Engagement preference set: env={env}, style={pos}"


def chat_with_bot_stream(user_input):
    global conversation_history, consecutive_unknown_count
    conversation_history.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
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

    # Simple heuristic check for unknown/unrecognized
    is_unknown_response = False
    if ("do not understand" in response_content.lower()
            or "unrecognized" in response_content.lower()
            or len(response_content.strip()) < 50):
        is_unknown_response = True

    if is_unknown_response:
        consecutive_unknown_count += 1
    else:
        consecutive_unknown_count = 0

    # If it's second time in a row, add helpful tip
    if consecutive_unknown_count >= 2:
        response_content += (
            "\n\nIt seems we've run into difficulty understanding your question twice. "
            "Try rephrasing or asking about interview-related topics. For example:\n"
            "- \"How can I build confidence for an online interview?\"\n"
            "- \"What if I forget an important detail during the interview?\"\n"
            "- \"How do I talk about my weaknesses without sounding negative?\""
        )
        consecutive_unknown_count = 0

    conversation_history.append({"role": "assistant", "content": response_content})

    formatted_conversation = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation_history if msg["role"] in ["user", "assistant"]
    ]
    return formatted_conversation


CUSTOM_CSS = """
body {
    background-color: #2B2B2B;
}
h1, h2, h3, p, label, .gr-textbox label, .gr-chatbot label {
    color: #ffffff;
    font-weight: bold;
}
.gr-button {
    display: inline-block;
    margin: 10px auto;
    border-radius: 10px;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
    background-color: #004080;
    color: white;
    padding: 10px 20px;
}
.gr-button:hover {
    background-color: #002F5F;
}
.gr-row-buttons {
    display: flex;
    justify-content: center;
    gap: 20px;
}

.minimal-card {
    border: 1px solid #444;
    border-radius: 5px;
    padding: 5px; /* make them smaller */
    margin: 5px 0;
    background-color: #3A3A3A;
}
#textbox-col .gr-textbox {
    height: 120px !important;
    width: 100% !important;
}
#audio-col .gr-box {
    height: 120px !important;
    width: 100% !important;
}
"""

TITLE = """
<h1 style='text-align: center; font-size: 28px; color: #ffffff; background-color: #004080; padding: 10px;'>💼 AI Interview Coach Chatbot</h1>
"""

with gr.Blocks(css=CUSTOM_CSS,
               theme=gr.themes.Glass(primary_hue="blue", secondary_hue="gray", neutral_hue="gray")) as demo:
    gr.HTML(TITLE)

    with gr.Tabs():
        with gr.TabItem("💬 Chat"):
            with gr.Row():
                # Left Column: Set Interview Context (scale=0.5 to make smaller)
                with gr.Column(scale=0.5, elem_classes="minimal-card"):
                    gr.Markdown("#### Set Interview Context")

                    # Put environment & custom_env on the same row
                    with gr.Row():
                        interview_env = gr.Dropdown(
                            choices=["Virtual", "In-person", "Phone", "Other"],
                            label="Environment",
                            value="Virtual"
                        )
                        custom_env = gr.Textbox(label="Or enter custom env", lines=1)

                    # Put position & custom_position on the same row
                    with gr.Row():
                        interview_position = gr.Dropdown(
                            choices=["Software Engineer", "Data Scientist", "Project Manager", "Other"],
                            label="Position",
                            value="Software Engineer"
                        )
                        custom_position = gr.Textbox(label="Or enter custom position", lines=1)

                    auto_set_btn = gr.Button("Auto Set Context")
                    context_status = gr.Textbox(label="Context Status", interactive=False)


                    def auto_set_context(env_dropdown, env_custom, pos_dropdown, pos_custom):
                        env = env_custom if env_dropdown == "Other" and env_custom.strip() else env_dropdown
                        pos = pos_custom if pos_dropdown == "Other" and pos_custom.strip() else pos_dropdown
                        return set_interview_context(env, pos)


                    auto_set_btn.click(
                        auto_set_context,
                        inputs=[interview_env, custom_env, interview_position, custom_position],
                        outputs=context_status
                    )

                # Right Column: Engagement and Personalization (also scale=0.5)
                with gr.Column(scale=0.5, elem_classes="minimal-card"):
                    gr.Markdown("#### Engagement and Personalization")
                    # Let user select environment & position to store as preference
                    with gr.Row():
                        engage_env = gr.Dropdown(
                            choices=["Gamification", "Friendly Tone", "Formal Tone", "Other"],
                            label="Engagement Env",
                            value="Gamification"
                        )
                        custom_eng_env = gr.Textbox(label="Or custom env", lines=1)
                    with gr.Row():
                        engage_pos = gr.Dropdown(
                            choices=["Short Answers", "Detailed Answers", "Other"],
                            label="Answer Style",
                            value="Short Answers"
                        )
                        custom_eng_pos = gr.Textbox(label="Or custom style", lines=1)

                    pref_btn = gr.Button("Set Preference")
                    pref_status = gr.Textbox(label="Pref Status", interactive=False)

                    pref_btn.click(
                        add_preference,
                        inputs=[engage_env, custom_eng_env, engage_pos, custom_eng_pos],
                        outputs=pref_status
                    )

            # Chat UI
            gr.Markdown("### Interview Chat")
            chatbot = gr.Chatbot(label="Interview Coach Chatbot", type="messages")
            with gr.Row(elem_id="gr-row-buttons"):
                with gr.Column(elem_id="textbox-col", scale=1):
                    user_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask your interview question here...",
                        lines=2,
                        interactive=True,
                        show_label=True
                    )
                with gr.Column(elem_id="audio-col", scale=1):
                    voice_input = gr.Audio(label="Record Your Question (Voice)")

            # Submit on Enter
            user_input.submit(
                fn=chat_with_bot_stream,
                inputs=user_input,
                outputs=chatbot,
                queue=True
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=user_input
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

            gr.Markdown(
                "#### What Can I Ask the Chatbot?\nYou can ask me about various interview-related topics – from handling anxiety and confidence, to common technical or behavioral questions. Here are some suggestions:")
            with gr.Row():
                example_questions = [
                    "What are common interview questions for people who feel nervous?",
                    "How can I answer if I get anxious during the interview?",
                    "How to calm myself before starting a virtual interview?",
                    "What should I do if I forget something during the interview?",
                    "How do I show confidence even when I'm nervous?"
                ]

                for q in example_questions:
                    gr.Button(q).click(lambda q=q: q, outputs=user_input)

demo.launch()



















