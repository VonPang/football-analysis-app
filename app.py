import streamlit as st
import pandas as pd
from math import exp, factorial

# Poisson Probability Function
poisson_prob = lambda k, lmbda: (lmbda**k * exp(-lmbda)) / factorial(k)

# Function to calculate Expected Value (EV)
def calculate_ev(probability, odds):
    return round((probability * odds - 1) * 100, 2) if probability else "N/A"

# Function to calculate Kelly Criterion Bet Size
def calculate_kelly(probability, odds, bankroll):
    edge = (probability * odds - 1)
    if edge > 0:
        kelly_fraction = edge / (odds - 1)
        return round(bankroll * kelly_fraction, 2)
    return 0.0

# Function to calculate probabilities using Poisson distribution
def calculate_poisson_prob(total_avg, threshold):
    prob_over = 1 - sum(poisson_prob(i, total_avg) for i in range(int(threshold) + 1))
    prob_under = 1 - prob_over
    return round(prob_over, 4), round(prob_under, 4)

# Initialize session state
if "combo_data" not in st.session_state:
    st.session_state["combo_data"] = []

if "bet_tracker" not in st.session_state:
    st.session_state["bet_tracker"] = []

if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = {}

# App Layout
st.title("Football Analysis Tools with All Features")

# Sidebar menu
tool = st.sidebar.radio("Choose a tool:", [
    "Shots Market",
    "Player Shots Market",
    "Corners Market",
    "Team Specific Analysis",
    "Combo Bets",
    "Bet Tracker"
])

# Sidebar for bankroll input
bankroll = st.sidebar.number_input("Enter your current bankroll:", min_value=0.0, value=1000.0)

# Shots Market Analysis
if tool == "Shots Market":
    st.header("Advanced Shots Market Analysis")
    home_avg = st.number_input("Home Team Average Shots per match:", min_value=0.0, value=5.5, step=0.1)
    away_avg = st.number_input("Away Team Average Shots per match:", min_value=0.0, value=4.5, step=0.1)
    home_allowed = st.number_input("Opponent Shots Allowed by Home Team:", min_value=0.0, value=12.0, step=0.1)
    away_allowed = st.number_input("Opponent Shots Allowed by Away Team:", min_value=0.0, value=11.5, step=0.1)
    home_xg = st.number_input("Home Team xG:", min_value=0.0, value=1.5, step=0.1)
    away_xg = st.number_input("Away Team xG:", min_value=0.0, value=1.2, step=0.1)
    threshold = st.number_input("Threshold for Total Shots (e.g., 9.5):", min_value=0.0, value=9.5, step=0.1)
    odds_over = st.number_input("Odds for Over Shots Market:", min_value=1.01, value=1.8, step=0.01)
    odds_under = st.number_input("Odds for Under Shots Market:", min_value=1.01, value=2.0, step=0.01)

    if st.button("Analyze Shots Market"):
        home_total = home_avg + away_allowed + home_xg
        away_total = away_avg + home_allowed + away_xg
        total_avg_shots = home_total + away_total

        prob_over, prob_under = calculate_poisson_prob(total_avg_shots, threshold)
        ev_over = calculate_ev(prob_over, odds_over)
        ev_under = calculate_ev(prob_under, odds_under)
        kelly_over = calculate_kelly(prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(prob_under, odds_under, bankroll)

        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Threshold": [threshold, threshold],
            "Probability (%)": [f"{prob_over * 100:.2f}%", f"{prob_under * 100:.2f}%"],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

# Player Shots Market
elif tool == "Player Shots Market":
    st.header("Player Shots Market Analysis")
    player = st.text_input("Enter Player Name", placeholder="e.g., Haaland")
    avg_shots = st.number_input(f"Average Shots per Match for {player}:", min_value=0.0, value=2.5, step=0.1)
    threshold = st.number_input(f"Threshold for Shots (e.g., 2.5):", min_value=0.0, value=2.5, step=0.1)
    odds_over = st.number_input(f"Odds for Over {threshold} Shots:", min_value=1.01, value=1.9, step=0.01)

    if st.button(f"Analyze Shots Market for {player}"):
        prob_over = 1 - sum(poisson_prob(i, avg_shots) for i in range(int(threshold) + 1))
        ev_over = calculate_ev(prob_over, odds_over)
        kelly_over = calculate_kelly(prob_over, odds_over, bankroll)

        results = pd.DataFrame({
            "Bet Option": ["Over"],
            "Threshold": [threshold],
            "Probability (%)": [f"{prob_over * 100:.2f}%"],
            "Odds": [odds_over],
            "EV (%)": [ev_over],
            "Recommended Bet": [kelly_over]
        })
        st.table(results)

# Corners Market
elif tool == "Corners Market":
    st.header("Corners Market Analysis")
    home_avg = st.number_input("Home Team Average Corners per match:", min_value=0.0, value=5.0, step=0.1)
    away_avg = st.number_input("Away Team Average Corners per match:", min_value=0.0, value=4.5, step=0.1)
    threshold = st.number_input("Threshold for Total Corners (e.g., 9.5):", min_value=0.0, value=9.5, step=0.1)
    odds_over = st.number_input("Odds for Over Corners Market:", min_value=1.01, value=1.8, step=0.01)
    odds_under = st.number_input("Odds for Under Corners Market:", min_value=1.01, value=2.0, step=0.01)

    if st.button("Analyze Corners Market"):
        prob_over, prob_under = calculate_poisson_prob(home_avg + away_avg, threshold)
        ev_over = calculate_ev(prob_over, odds_over)
        ev_under = calculate_ev(prob_under, odds_under)
        kelly_over = calculate_kelly(prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(prob_under, odds_under, bankroll)

        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Threshold": [threshold, threshold],
            "Probability (%)": [f"{prob_over * 100:.2f}%", f"{prob_under * 100:.2f}%"],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

# Team Specific Analysis
elif tool == "Team Specific Analysis":
    st.header("Team Specific Analysis")
    team = st.text_input("Enter Team Name", placeholder="e.g., Chelsea, Arsenal")
    half = st.selectbox("Select Half", ["1st Half", "2nd Half"])
    avg_shots = st.number_input(f"Average Shots for {team} in {half}:", min_value=0.0, value=6.5, step=0.1)
    threshold = st.number_input(f"Threshold for Total Shots (e.g., 5.5):", min_value=0.0, value=5.5, step=0.1)
    odds_over = st.number_input(f"Odds for Over {threshold} Shots:", min_value=1.01, value=1.8, step=0.01)

    if st.button(f"Analyze Team Market for {team}"):
        prob_over = 1 - sum(poisson_prob(i, avg_shots) for i in range(int(threshold) + 1))
        ev_over = calculate_ev(prob_over, odds_over)
        kelly_over = calculate_kelly(prob_over, odds_over, bankroll)

        results = pd.DataFrame({
            "Bet Option": ["Over"],
            "Threshold": [threshold],
            "Probability (%)": [f"{prob_over * 100:.2f}%"],
            "Odds": [odds_over],
            "EV (%)": [ev_over],
            "Recommended Bet": [kelly_over]
        })
        st.table(results)

# Combo Bets
elif tool == "Combo Bets":
    st.header("Combo Bets Analysis")
    num_selections = st.number_input("Number of selections in the combo bet:", min_value=2, max_value=10, value=3, step=1)
    selections = []
    for i in range(int(num_selections)):
        st.subheader(f"Selection {i + 1}")
        probability = st.number_input(f"Probability for Selection {i + 1} (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        odds = st.number_input(f"Odds for Selection {i + 1}", min_value=1.01, value=2.0, step=0.01)
        selections.append({"probability": probability / 100, "odds": odds})

    bookmaker_combo_odds = st.number_input("Bookmaker's offered combo odds:", min_value=1.01, value=5.0, step=0.01)

    if st.button("Analyze Combo Bet"):
        total_probability = 1
        total_odds = 1
        for sel in selections:
            total_probability *= sel["probability"]
            total_odds *= sel["odds"]

        ev = calculate_ev(total_probability, bookmaker_combo_odds)
        kelly_fraction = max(0, (bookmaker_combo_odds * total_probability - 1) / (bookmaker_combo_odds - 1))
        recommended_bet = round(bankroll * kelly_fraction, 2)

        results = pd.DataFrame({
            "Fair Odds": [round(1 / total_probability, 2)],
            "Bookmaker Odds": [bookmaker_combo_odds],
            "EV (%)": [ev],
            "Recommended Bet": [recommended_bet]
        })
        st.table(results)

# Bet Tracker
elif tool == "Bet Tracker":
    st.header("Bet Tracker")
    if st.session_state["bet_tracker"]:
        tracker_df = pd.DataFrame(st.session_state["bet_tracker"])
        st.table(tracker_df)
    else:
        st.write("No bets tracked yet.")

















































