# Personalized Coding Tutor Chatbot

An interactive AI-powered chatbot to help users learn programming concepts through guided problem-solving. Supports step-by-step explanations, strategic hints, and complete solutions based on user-selected assistance modes. Built with Streamlit and integrates with the OpenRouter API for intelligent conversational tutoring.

---

## Features

- **Multi-turn Chat Interface:** Natural chat experience with user and assistant message bubbles.
- **Mode Selection:** Choose from `explain`, `hint`, or `solution` modes to tailor assistance.
- **Model Selection:** Select from multiple OpenRouter AI models to customize response style and performance.
- **Conversation Context:** Maintains full dialogue history for contextual and personalized responses.
- **Dark Themed Chat Bubbles:** Easy-on-the-eyes dark mode for user and assistant messages.
- **Secure API Key Input:** Users provide their own OpenRouter API key without storing credentials on the server.
- **Asynchronous API Calls:** Fast and responsive UI backed by async OpenRouter API integration.

---

## Installation

1. Clone the repository or copy the `app.py` file.
2. Ensure you have Python 3.8+ installed.
3. Create and activate a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
```

4. Install required packages:

```
pip install -r requirements.txt
```

---

## Usage

1. Run the Streamlit app:

```
streamlit run app.py
```

2. Open the provided local URL in your browser (usually `http://localhost:8501`).

3. In the sidebar:
   - Enter your OpenRouter API key (get one at [OpenRouter](https://openrouter.ai/)).
   - Select the assistance mode (`explain`, `hint`, or `solution`).
   - Choose your preferred OpenRouter model.

4. In the main chat window:
   - Type your coding problem, question, or request.
   - Click **Send** to get step-by-step help, hints, or full solutions.
   - Continue the conversation naturally; the chatbot will remember context.

---

## Technical Overview

- **Frontend:** Streamlit app with chat bubbles styled in dark theme.
- **Backend Integration:** Calls OpenRouter AI API asynchronously to generate tutoring responses.
- **Session Management:** Uses `st.session_state` to store and render chat history.
- **Prompt Templates:** Mode-specific detailed instructions guide AI responses for teaching-focused interaction.

---

## Example Interaction

**User:** "Can you explain how to solve the Two Sum problem?"  
**Tutor:** Provides a step-by-step breakdown without giving code.

**User:** "I'm stuck implementing the hash map approach. Any hints?"  
**Tutor:** Gives focused strategic hints to help progress.

**User:** "Show me the complete solution."  
**Tutor:** Sends well-commented, production-ready code with explanations.

---

## Customization & Extension

- Add user authentication and persistent chat storage.
- Enhance UI with avatars, timestamps, and markdown rendering.
- Support multiple programming languages.
- Integrate code execution and testing environments.
- Add analytics to track learning progress.