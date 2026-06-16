import pandas as pd
import joblib
import gradio as gr

df = pd.read_csv("commute_data.csv")
model = joblib.load("delay_predictor_model.pkl")

ROUTE_NAMES = {
    "R1": "Whitefield → MG Road",
    "R2": "Electronic City → Majestic",
    "R3": "Indiranagar → Koramangala",
    "R4": "HSR Layout → Silk Board",
    "R5": "Yelahanka → Hebbal",
    "R6": "Jayanagar → Banashankari",
    "R7": "BTM Layout → Marathahalli",
    "R8": "Rajajinagar → Malleswaram",
}

def safety_score(lighting, incidents, crowd, hour):
    incident_score = 10 - min(incidents, 10)
    crowd_score = max(0, 10 - abs(crowd - 4.5) * 1.4)
    time_score = 10 if 6 <= hour <= 19 else (6 if hour <= 22 else 2)
    return round(0.4*lighting + 0.3*incident_score + 0.2*crowd_score + 0.1*time_score, 2)

def crowd_label(crowd):
    if crowd <= 3:
        return "Low 🟢"
    elif crowd <= 6:
        return "Moderate 🟡"
    else:
        return "High 🔴"

def delay_recommendation(delay, hour):
    if delay > 15:
        earlier = hour - 1 if hour > 0 else 0
        return f"⚠️ High delay expected. Try leaving by {earlier}:00 to avoid peak congestion."
    elif delay > 5:
        return f"ℹ️ Moderate delay expected. Plan an extra {int(delay)} minutes into your schedule."
    else:
        return "✅ Route looks clear — minimal delay expected. Good time to travel!"

def safety_recommendation(score):
    if score >= 7.5:
        return "🛡️ This route is safe for travel at this time."
    elif score >= 5:
        return "🛡️ Moderate safety — stay aware of your surroundings."
    else:
        return "🛡️ Low safety score — consider traveling with a companion or choosing an earlier time."

def get_commute_plan(route_id, day, hour, weather):
    is_weekend = 1 if day in ["Saturday", "Sunday"] else 0

    columns = model.feature_names_in_
    input_row = pd.DataFrame([[0]*len(columns)], columns=columns)

    input_row["hour"] = hour
    input_row["is_weekend"] = is_weekend

    day_col = f"day_of_week_{day}"
    if day_col in input_row.columns:
        input_row[day_col] = 1

    weather_col = f"weather_{weather}"
    if weather_col in input_row.columns:
        input_row[weather_col] = 1

    route_col = f"route_id_{route_id}"
    if route_col in input_row.columns:
        input_row[route_col] = 1

    predicted_delay = round(model.predict(input_row)[0], 1)

    subset = df[df["route_id"] == route_id]
    lighting = subset["lighting_score"].mean()
    crowd = subset["crowd_level"].mean()
    incidents = round(subset["incident_count_30d"].mean())

    safety = safety_score(lighting, incidents, crowd, hour)
    crowd_lbl = crowd_label(crowd)

    delay_rec = delay_recommendation(predicted_delay, hour)
    safety_rec = safety_recommendation(safety)

    output = f"""
🚍 COMMUTE AI — ROUTE PLAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 Route:         {ROUTE_NAMES.get(route_id, route_id)}
📅 Day:           {day} at {hour}:00
🌤️ Weather:       {weather}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ Predicted Delay:    {predicted_delay} min
👥 Crowd Level:        {crowd_lbl}
🛡️ Safety Score:       {safety} / 10
━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 RECOMMENDATIONS
{delay_rec}
{safety_rec}
    """
    return output

demo = gr.Interface(
    fn=get_commute_plan,
    inputs=[
        gr.Dropdown(list(ROUTE_NAMES.keys()), label="Select Route"),
        gr.Dropdown(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], label="Day of Week"),
        gr.Slider(0, 23, step=1, value=8, label="Departure Hour (24h format)"),
        gr.Dropdown(["Clear","Rain","Heavy Rain","Fog"], label="Weather Condition")
    ],
    outputs=gr.Textbox(label="Your Commute Plan", lines=18),
    title="🚍 Commute AI — Smarter, Safer Urban Mobility",
    description="Enter your route details below to get an AI-powered delay prediction, crowd forecast, safety score, and personalized travel recommendations.",
    flagging_mode="never"
)

demo.launch()