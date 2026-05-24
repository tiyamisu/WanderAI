# 🌍 WanderAI: Enterprise Travel Logistics

WanderAI is a high-end, AI-powered travel logistics engine and personal concierge. Built with Python and Streamlit, it leverages **Groq's ultra-fast LLM inference** (llama3-70b-8192) to provide autonomous itinerary planning, real-time climate tracking, and smart packing management.

## ✨ Core Modules

* **🌍 Global Explorer Dashboard:** A real-time data visualizer that generates destination intelligence, local cultural etiquette, and dynamic financial/climate charts using Plotly.
* **🤖 AI Travel Chatbot:** A 24/7 conversational travel concierge capable of drafting custom itineraries, recommending local cuisines, and fetching weather data using agentic function calling.
* **🧳 Smart Gear Matrix:** An interactive, live-calculating packing algorithm that tracks baggage weight limits and allows users to export their manifest directly to PDF.

## 🛠️ Technology Stack
* **Frontend/Backend:** [Streamlit](https://streamlit.io/)
* **AI Engine:** [Groq API](https://console.groq.com/) (`groq`) — llama3-70b-8192
* **Data Visualization:** Pandas & Plotly
* **File Generation:** FPDF

## 🔑 Required Environment Variables

Create a `.env` file in the project root with:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192
```

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tiyamisu/WanderAI.git
   cd WanderAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your `.env` file** with your Groq API key (see above).

4. **Run the app:**
   ```bash
   streamlit run travel_agent.py
   ```

The app will open at `http://localhost:8501` in your browser.