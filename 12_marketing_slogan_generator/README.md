# AI Marketing Slogan Generator

Generate strategic, creative, and high-impact marketing slogans using open-source large language models (LLMs) via the OpenRouter API and Streamlit.

---

## Features

- Input rich product, brand, and market details for highly tailored slogan generation
- Uses a comprehensive expert-level system prompt for top-tier creative outputs
- Supports multiple open-source LLMs (e.g., Mistral, CodeLlama, GPT-3.5 Turbo, LLaMA)
- Option to enter your own OpenRouter API key securely
- Interactive Streamlit web interface with form inputs for ease of use
- Outputs slogans with detailed rationale and usage guidance

---

## Demo

_Run locally via Streamlit._

---

## Getting Started

### Prerequisites

- Python 3.8+
- OpenRouter API key (sign up at [OpenRouter](https://openrouter.ai))

### Installation

1. Clone this repository:

```
git clone https://github.com/abhiyaligar/21_days_21_projects.git
cd 12_marketing_slogan_generator
```

2. Install dependencies:

```
pip install -r requirements.txt
```


3. Save your expert system prompt text in `prompt.txt` in the project root.


### Running the App
```
streamlit run streamlit_slogan_app.py
```

This will open the app in your default browser.

---

## Usage

1. Enter detailed information about your product, brand, audience, and positioning.
2. enter your own API key 
3. Select the desired LLM model.
4. Click **Generate Slogans**.
5. Review the resulting slogans and strategic suggestions.

---

## Code Overview

- `streamlit_slogan_app.py`: Main Streamlit app code
- `prompt.txt`: The detailed expert prompting system message used to guide the LLM
- `.env`: Optional file to securely store your API key

---

## Customization

- Edit `prompt.txt` to refine slogan creation strategy.
- Adjust the list of models in the app for preferred LLMs.
- Extend the app with user authentication or other workflow enhancements.

---

## Troubleshooting

- If slogans do not generate, check API key validity and usage limits.
- Ensure `prompt.txt` is in UTF-8 encoding and correctly formatted.
- Review Streamlit console logs for detailed error information.

---

## Acknowledgements

- OpenRouter AI for open source LLM API access
- Streamlit for rapid app development toolkit
- Inspired by expert marketing strategy principles for creative slogan crafting

---

## Contact

For questions or contributions, please open an issue or pull request on GitHub.

---


