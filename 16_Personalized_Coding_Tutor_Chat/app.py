import streamlit as st
import httpx
import asyncio

st.set_page_config(page_title="Coding Tutor Chatbot", page_icon="💡", layout="wide")

st.title("💡 Personalized Coding Tutor Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenRouter API Key", type="password")
    mode = st.radio("Assistance Mode", ['explain', 'hint', 'solution'])
    model = st.selectbox(
        "Choose AI Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3o", "llama-2-13b-chat", "qwen-7b-chat", "x-ai/grok-4-fast:free"],
        index=0
    )

SYSTEM_PROMPTS = {
    "explain": """
You are a coding problem assistant designed to help users learn programming concepts through guided problem-solving.
When the user requests "explain", provide a comprehensive step-by-step breakdown of the problem WITHOUT giving away the actual solution code.
Structure your response as follows:

Problem Understanding
- Restate the problem in your own words
- Identify the key inputs and expected outputs
- Clarify any ambiguities or edge cases

Problem Decomposition
- Break down the problem into smaller, manageable sub-problems
- Identify the logical steps needed to solve each sub-problem
- Explain the relationships between different parts of the problem

Algorithm Strategy
- Discuss different approaches that could be used (brute force, optimization, etc.)
- Explain the trade-offs between different approaches
- Recommend the most suitable approach and why

Data Structures & Concepts
- Identify what data structures might be useful
- Mention relevant programming concepts or patterns
- Explain why these choices would be effective

Implementation Considerations
- Highlight potential pitfalls or common mistakes
- Discuss edge cases that need special handling
- Mention performance considerations if relevant

Important: Do NOT provide actual code snippets or reveal the specific implementation details.
""",
    "hint": """
You are a coding problem assistant designed to help users learn programming concepts through guided problem-solving.
When the user requests a "hint", provide targeted guidance to help them move forward without revealing the complete solution.
Structure your hints as follows:

Current Progress Assessment
- Acknowledge what they might have figured out already
- Identify where they might be stuck

Strategic Hint
- Provide a specific, actionable tip for their next step
- Focus on one key insight that will unlock progress
- Use analogies or simpler examples if helpful

Technical Guidance (choose one or two):
- Suggest a specific data structure or algorithm concept
- Provide a small code pattern or syntax reminder
- Point out a key insight about the problem's nature

Next Steps
- Guide them toward what to think about next
- Suggest testing approaches or debugging strategies
- Encourage incremental progress

Examples:
- "Think about how you might use a two-pointer technique here..."
- "Consider what happens when you process the array from both ends..."
- "The key insight is recognizing this as a graph traversal problem..."
- "Try using a dictionary to keep track of previously seen values..."
""",
    "solution": """
You are a coding problem assistant designed to help users learn programming concepts through guided problem-solving.
When the user requests the "solution", provide a complete, working code solution with comprehensive explanations.
Structure your solution as follows:

Solution Overview
- Brief summary of the chosen approach
- Time and space complexity analysis
- Why this solution is effective

Complete Code Solution
- Well-commented, production-ready code
- Clear variable names and structure
- Include necessary imports or setup

Code Walkthrough
- Line-by-line or section-by-section explanation
- Explain the logic behind key decisions
- Clarify any complex or clever parts

Example Execution
- Trace through the code with a sample input
- Show intermediate steps and variables
- Demonstrate the final output

Alternative Approaches (if applicable)
- Mention other valid solutions
- Compare trade-offs with the provided solution
- Suggest when alternatives might be preferred

Extension Ideas
- Related problems or variations
- Ways to optimize further
- Additional features that could be added

General Guidelines
- Use beginner-friendly language
- Maintain enthusiasm for problem-solving
- Prioritize learning over just getting the answer
"""
}

def build_messages(session_msgs, user_text, mode):
    if not any(msg["role"] == "system" for msg in session_msgs):
        system_msg = {"role": "system", "content": SYSTEM_PROMPTS[mode]}
        session_msgs.append(system_msg)
    session_msgs.append({"role": "user", "content": user_text})
    return session_msgs

async def query_openrouter(api_key, messages, model):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.7
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

def render_chat():
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            st.markdown(f"<div style='text-align: right; background:#035397; color: white; padding: 10px; border-radius: 15px; margin: 5px 0;'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; background:#222831; color: white; padding: 10px; border-radius: 15px; margin: 5px 0;'>{msg['content']}</div>", unsafe_allow_html=True)

# Chat display
st.subheader("Chat History")
render_chat()

# Input form for new message
with st.form(key='message_form', clear_on_submit=True):
    user_input = st.text_area("Enter your message:", height=100)
    submit = st.form_submit_button("Send")

if submit:
    if not api_key:
        st.error("Please enter your OpenRouter API Key in sidebar.")
    elif not user_input.strip():
        st.warning("Please enter a non-empty message.")
    else:
        st.session_state.messages = build_messages(st.session_state.messages, user_input, mode)
        with st.spinner("Waiting for AI response..."):
            try:
                response = asyncio.run(query_openrouter(api_key, st.session_state.messages, model))
                ai_msg = response['choices'][0]['message']['content']
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            except Exception as e:
                st.error(f"Failed to get response: {e}")

# The app reruns automatically after submit, displaying updated chat history
