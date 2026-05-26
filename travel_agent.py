import os
import json
import time
import random
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from groq import Groq
from fpdf import FPDF

# Load API Key
load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")
MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- PAGE CONFIGURATION & PREMIUM CSS ---
st.set_page_config(page_title="WanderAI", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    /* Enlarged, Centered, High-End Gradient Title */
    .brand-title { 
        font-size: 130px; 
        font-weight: 900; 
        text-align: center;
        background: -webkit-linear-gradient(45deg, #2563EB, #4F46E5, #06B6D4); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: -20px;
        padding-bottom: 0px;
        line-height: 1.1;
    }
    .brand-subtitle {
        text-align: center;
        font-size: 22px;
        color: #64748B;
        font-weight: 300;
        letter-spacing: 4px;
        margin-bottom: 50px;
    }
    /* Glassmorphism & Hover Effects for Metrics */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4F46E5, #06B6D4); }
    </style>
""", unsafe_allow_html=True)

# --- FUN FACTS GENERATOR ---
facts = [
    "💡 Did you know? Japan has over 5 million vending machines!",
    "💡 Pro Tip: Rolling your clothes instead of folding them saves 20% more space in your suitcase.",
    "💡 Did you know? The shortest commercial flight in the world is in Scotland and lasts just 57 seconds.",
    "💡 Travel Hack: Download Google Maps for offline use before you leave your hotel Wi-Fi.",
    "💡 Did you know? There are no clocks in Las Vegas casinos to keep you playing longer."
]

# --- DYNAMIC DATA ENGINE ---
def get_destination_intel(city: str):
    city_lower = city.lower()
    if "tokyo" in city_lower or "japan" in city_lower:
        return {"cur": "JPY", "rate": 155.20, "sym": "¥", "base": 180, "temp": [15, 16, 18, 20, 19, 22, 21], "tz": "GMT+9", "culture": "Tipping is considered rude. Bowing is the standard greeting."}
    elif "paris" in city_lower or "europe" in city_lower:
        return {"cur": "EUR", "rate": 0.92, "sym": "€", "base": 220, "temp": [12, 14, 13, 16, 18, 17, 15], "tz": "GMT+1", "culture": "Always say 'Bonjour' when entering a shop. Bread is placed directly on the table."}
    elif "london" in city_lower:
        return {"cur": "GBP", "rate": 0.79, "sym": "£", "base": 250, "temp": [10, 11, 9, 12, 14, 13, 11], "tz": "GMT+0", "culture": "Stand on the right on escalators. Queuing (lining up) is taken very seriously."}
    elif "dhaka" in city_lower:
        return {"cur": "BDT", "rate": 109.50, "sym": "৳", "base": 70, "temp": [28, 30, 29, 32, 31, 33, 30], "tz": "GMT+6", "culture": "Eat with your right hand only. Dress modestly, especially in rural or religious areas."}
    else:
        seed = len(city)
        return {"cur": "USD", "rate": 1.0, "sym": "$", "base": 100 + (seed * 10), "temp": [20, 22, 21, 24, 23, 25, 24], "tz": f"GMT+{seed%12}", "culture": "Research local customs before arrival. Always carry a bit of physical cash."}

def get_weather_forecast(city: str) -> str:
    intel = get_destination_intel(city)
    return f"Weather for {city}: Highs around {max(intel['temp'])}°C, lows around {min(intel['temp'])}°C."

# --- PDF GENERATOR ---
def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="WanderAI Smart Gear Manifest", ln=1, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="-"*70, ln=1, align='C')
    
    for _, row in df.iterrows():
        status = "[ X ]" if row["Secured"] else "[   ]"
        text = f"{status}  {row['Qty']}x  {row['Gear']}  ({row['Category']})  |  Weight: {row['Weight']}kg"
        text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(200, 10, txt=text, ln=1)
        
    return pdf.output(dest='S').encode('latin-1')

# ---------------------------------------------------------------------------
# GROQ CHAT SESSION HELPER
# Wraps the Groq client so the rest of the app can call .send_message()
# exactly like the old Gemini chat session, returning an object with .text
# ---------------------------------------------------------------------------

# Tool schema for Groq (OpenAI-compatible format)
WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather_forecast",
        "description": "Get the current weather forecast for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to get the weather for."
                }
            },
            "required": ["city"]
        }
    }
}

SYSTEM_PROMPT = (
    "You are WanderAI, an elite, high-end travel concierge. "
    "Be extremely polite, fun, and use emojis. Use beautiful formatting. "
    "Check the weather using the get_weather_forecast tool if a city is mentioned."
)

class _Response:
    """Thin wrapper so response.text works identically to Gemini's SDK."""
    def __init__(self, text: str):
        self.text = text

class GroqChatSession:
    """
    Stateful, multi-turn chat session backed by Groq with tool-calling support.
    Mirrors the Gemini chat session interface used in this app.
    """

    def __init__(self, client: Groq, model: str):
        self.client = client
        self.model  = model
        # history is a list of OpenAI-format message dicts
        self.history: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def send_message(self, user_text: str) -> _Response:
        """Send a user message and return a _Response with .text set."""
        self.history.append({"role": "user", "content": user_text})

        # First call — allow tool use
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            tools=[WEATHER_TOOL_SCHEMA],
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # Handle tool calls if the model requested one
        if msg.tool_calls:
            # Append the assistant's tool-call message to history
            self.history.append(msg)

            # Execute every requested tool call
            for tc in msg.tool_calls:
                if tc.function.name == "get_weather_forecast":
                    args = json.loads(tc.function.arguments)
                    result = get_weather_forecast(args.get("city", ""))
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            # Second call — get the final natural-language answer
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
            )
            msg = response.choices[0].message

        # Append final assistant reply to history
        final_text = msg.content or ""
        self.history.append({"role": "assistant", "content": final_text})
        return _Response(final_text)


# --- UNIVERSAL HEADER ---
#st.markdown('<p class="brand-title">WanderAI</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">YOUR INTELLIGENT TRAVEL ECOSYSTEM</p>', unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2060/2060284.png", width=60)
    # Made WanderAI text much larger and bolder
    st.markdown("<h1 style='font-family: \"Comic Sans MS\", cursive, sans-serif; color: #06B6D4; margin-top: -15px; font-size: 42px; font-weight: 900;'>WanderAI ✈️</h1>", unsafe_allow_html=True)
    
    # Changed option to Chatbot
    page = st.radio("Select Interface:", [
        "🌍 Global Explorer", 
        "🤖 AI Travel Chatbot", 
        "🧳 Smart Gear Matrix"
    ])
    st.divider()
    st.caption("✨ *'Adventure is out there!' – Ellie* 🎈")

# --- PAGE 1: GLOBAL EXPLORER ---
if page == "🌍 Global Explorer":
    st.markdown("### Welcome to the Future of Travel Planning 🚀")
    st.write("WanderAI is your autonomous travel logistics engine. Before you take off, here is how to navigate your ecosystem:")
    
    guide1, guide2, guide3 = st.columns(3)
    with guide1:
        st.info("**1. Global Explorer (Below)**\nEnter your flight details to generate a comprehensive destination briefing and capital allocation charts.")
    with guide2:
        st.success("**2. AI Chatbot (Sidebar)**\nChat with our elite AI-powered Concierge for custom itineraries and hidden gem recommendations.")
    with guide3:
        st.warning("**3. Smart Matrix (Sidebar)**\nUse our intelligent algorithm to track luggage payloads in real-time and export your flight manifest.")
    
    st.divider()
    st.info(random.choice(facts), icon="✨")
    
    with st.form("analytics_form"):
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1: source_city = st.text_input("Source Location:", placeholder="e.g., New York")
        with col2: target_city = st.text_input("Destination City:", placeholder="e.g., Tokyo")
        with col3: travelers = st.number_input("Adventurers:", min_value=1, max_value=20, value=2)
        with col4: days = st.number_input("Days:", min_value=1, max_value=30, value=7)
        submit = st.form_submit_button("Launch Destination Intel", use_container_width=True)
        
    if submit and target_city and source_city:
        animation_container = st.empty()
        flight_html = f"""
        <div style="width: 100%; height: 120px; position: relative; border-bottom: 3px dashed #4F46E5; margin: 40px 0; font-family: sans-serif;">
            <div style="position: absolute; bottom: -30px; left: 0; font-weight: 800; font-size: 20px; color: #1E3A8A;">DEPARTURE: {source_city.upper()}</div>
            <div style="position: absolute; bottom: -30px; right: 0; font-weight: 800; font-size: 20px; color: #06B6D4;">ARRIVAL: {target_city.upper()}</div>
            <div style="font-size: 50px; position: absolute; bottom: -25px; animation: fly 3s cubic-bezier(0.4, 0, 0.2, 1) forwards;">✈️</div>
            <style>
                @keyframes fly {{
                    0% {{ left: 0px; transform: translateY(0px) rotate(15deg); opacity: 0; }}
                    10% {{ opacity: 1; transform: translateY(-40px) rotate(15deg); }}
                    80% {{ transform: translateY(-40px) rotate(10deg); }}
                    100% {{ left: calc(100% - 60px); transform: translateY(0px) rotate(0deg); opacity: 1; }}
                }}
            </style>
        </div>
        """
        animation_container.markdown(flight_html, unsafe_allow_html=True)
        time.sleep(3.2)
        animation_container.empty()
            
        st.divider()
        st.subheader(f"📍 Intelligence Briefing: {target_city.upper()}")
        
        intel = get_destination_intel(target_city)
        total_usd = intel["base"] * travelers * days
        total_inr = total_usd * 83.0
        total_local = total_usd * intel["rate"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Safety & Vibe Check", "Excellent", "Tourists Welcome")
        m2.metric(f"Local Currency", f"{intel['cur']}", f"1 USD = {intel['sym']}{intel['rate']:.2f}")
        m3.metric("Local Timezone", intel["tz"], "Plan for Jetlag", delta_color="off")
        m4.metric("Est. Total Investment", f"${total_usd:,.0f} USD", f"₹{total_inr:,.0f} INR | {intel['sym']}{total_local:,.0f} {intel['cur']}")
        
        st.write("")
        t1, t2, t3 = st.tabs(["💳 Financial Distribution", "🌤️ Micro-Climate Trend", "🏛️ Cultural Etiquette"])
        
        with t1:
            df_budget = pd.DataFrame({
                "Expense Type": ["Aero-Transit", "Luxury Stays", "Culinary", "Transport", "Experiences"],
                "Cost_USD": [total_usd*0.35, total_usd*0.30, total_usd*0.20, total_usd*0.05, total_usd*0.10],
                "Cost_INR": [total_inr*0.35, total_inr*0.30, total_inr*0.20, total_inr*0.05, total_inr*0.10]
            })
            fig_pie = px.pie(
                df_budget, 
                values='Cost_USD', 
                names='Expense Type', 
                hole=0.5, 
                color_discrete_sequence=px.colors.sequential.Teal_r,
                custom_data=['Cost_INR']
            )
            fig_pie.update_traces(hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.0f} USD<br>Amount: ₹%{customdata[0]:,.0f} INR')
            fig_pie.update_layout(margin=dict(t=20, b=20, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with t2:
            df_temp = pd.DataFrame({"Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "Temp (°C)": intel["temp"]})
            fig_line = px.line(df_temp, x='Day', y='Temp (°C)', markers=True, line_shape='spline')
            fig_line.update_traces(line_color='#06B6D4', line_width=4, marker_size=10)
            fig_line.update_layout(margin=dict(t=20, b=20, l=0, r=0), yaxis_title="Celsius (°C)")
            st.plotly_chart(fig_line, use_container_width=True)
            
        with t3:
            st.write("")
            st.info(f"**Essential Cultural Rules in {target_city.capitalize()}:**")
            st.write(f"💡 {intel['culture']}")
            st.write("💡 Emergency Number: 112 / 911 (Varies by local region)")
            st.write(f"💡 Default Currency: {intel['cur']} ({intel['sym']})")

# --- PAGE 2: AI CONCIERGE ---
elif page == "🤖 AI Travel Chatbot":
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Your Personal 24/7 Concierge")
        st.caption("I can build itineraries, find coffee shops, and give you live weather updates.")
        
        if "chat_session" not in st.session_state:
            client = Groq(api_key=API_KEY)
            st.session_state.chat_session = GroqChatSession(client=client, model=MODEL)
            st.session_state.messages = [{"role": "model", "content": "Welcome to WanderAI! I am your dedicated travel concierge. Where are we jet-setting to next? ✈️"}]

        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                    <div style='background-color: #06B6D4; color: white; padding: 10px 15px; border-radius: 20px 20px 0px 20px; max-width: 75%; font-family: sans-serif; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); line-height: 1.5;'>
                        {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                if len(message['content']) < 200 and '\n' not in message['content']:
                    st.markdown(f"""
                    <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                        <div style='background-color: #F1F5F9; color: #1E293B; padding: 10px 15px; border-radius: 20px 20px 20px 0px; max-width: 75%; font-family: sans-serif; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); border: 1px solid #E2E8F0; line-height: 1.5;'>
                            {message['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    with st.chat_message("model"):
                        st.markdown(message["content"])

    submitted_prompt = None

    with col2:
        st.markdown("#### Prompt Library")
        st.write("Not sure what to ask? Click an idea below:")
        if st.button("🗺️ 3-Day Paris Itinerary", use_container_width=True):
            submitted_prompt = "Draft a highly detailed 3-day itinerary for Paris."
        if st.button("🍽️ Tokyo Street Food", use_container_width=True):
            submitted_prompt = "What are the best street food spots in Tokyo?"
        if st.button("🌤️ London Weather", use_container_width=True):
            submitted_prompt = "Check the weather for London right now."
        if st.button("🎒 Pack for Dhaka", use_container_width=True):
            submitted_prompt = "What should I pack for a 5-day trip to Dhaka?"

    chat_input = st.chat_input("Ask your concierge anything...")
    if chat_input:
        submitted_prompt = chat_input

    if submitted_prompt:
        with col1: 
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                <div style='background-color: #06B6D4; color: white; padding: 10px 15px; border-radius: 20px 20px 0px 20px; max-width: 75%; font-family: sans-serif; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); line-height: 1.5;'>
                    {submitted_prompt}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "user", "content": submitted_prompt})
            
            # --- THE SAFETY NET FOR API ERRORS ---
            try:
                with st.spinner("Curating the perfect response..."):
                    response = st.session_state.chat_session.send_message(submitted_prompt)
                    
                if len(response.text) < 200 and '\n' not in response.text:
                     st.markdown(f"""
                        <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                            <div style='background-color: #F1F5F9; color: #1E293B; padding: 10px 15px; border-radius: 20px 20px 20px 0px; max-width: 75%; font-family: sans-serif; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); border: 1px solid #E2E8F0; line-height: 1.5;'>
                                {response.text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with st.chat_message("model"): 
                        st.markdown(response.text)
                        
                st.session_state.messages.append({"role": "model", "content": response.text})
                
            except Exception as e:
                st.error(f"🚦 **API Error:** {str(e)}\n\nPlease wait a moment and try again.")

# --- PAGE 3: SMART GEAR MATRIX ---
elif page == "🧳 Smart Gear Matrix":
    st.markdown("### The Ultimate Packing Algorithm")
    st.caption("Never forget a charger again. Check items off to calculate real-time baggage weight and export your manifest.")
    
    if 'packing_data' not in st.session_state:
        st.session_state.packing_data = pd.DataFrame([
            {"Secured": False, "Qty": 1, "Gear": "Passports & Visas", "Category": "Mission Critical", "Weight": 0.1},
            {"Secured": False, "Qty": 2, "Gear": "Universal Adapters", "Category": "Tech Arsenal", "Weight": 0.3},
            {"Secured": False, "Qty": 7, "Gear": "Climate-Ready Shirts", "Category": "Apparel & Threads", "Weight": 1.5},
            {"Secured": False, "Qty": 1, "Gear": "Heavy Winter Coat", "Category": "Apparel & Threads", "Weight": 2.2},
            {"Secured": False, "Qty": 1, "Gear": "Noise-Cancelling Over-Ears", "Category": "Tech Arsenal", "Weight": 0.4},
            {"Secured": False, "Qty": 1, "Gear": "TSA Liquids Bag", "Category": "Wellness & Grooming", "Weight": 0.5},
        ])
    
    col1, col2 = st.columns([3, 1])
    with col1:
        edited_df = st.data_editor(
            st.session_state.packing_data,
            column_config={
                "Secured": st.column_config.CheckboxColumn("Packed?", help="Mark when in the bag!"),
                "Qty": st.column_config.NumberColumn("Quantity", min_value=1, max_value=20),
                "Category": st.column_config.SelectboxColumn("Category", options=["Mission Critical", "Tech Arsenal", "Apparel & Threads", "Wellness & Grooming", "Survival & Misc"]),
                "Weight": st.column_config.NumberColumn("Weight (kg)", format="%.1f")
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic"
        )
        st.session_state.packing_data = edited_df

    with col2:
        st.markdown("#### Live Baggage Load")
        
        packed_items = edited_df[edited_df["Secured"] == True]
        total_packed_weight = (packed_items["Qty"] * packed_items["Weight"]).sum()
        max_allowance = 23.0 
        
        st.metric("Current Weight", f"{total_packed_weight:.1f} kg", delta=f"{max_allowance - total_packed_weight:.1f} kg remaining", delta_color="normal")
        
        progress_val = min(total_packed_weight / max_allowance, 1.0)
        st.progress(progress_val, text="Airline Limit Capacity")
        
        st.divider()
        st.success(f"You have secured {len(packed_items)} out of {len(edited_df)} items.")
        
        pdf_bytes = generate_pdf(edited_df)
        st.download_button(
            label="📥 Download Manifest (PDF)",
            data=pdf_bytes,
            file_name='WanderAI_Packing_Manifest.pdf',
            mime='application/pdf',
            use_container_width=True
        )