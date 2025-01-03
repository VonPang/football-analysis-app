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
def calculate_poisson_prob(avg, second_team_avg, threshold):
    total_avg = avg + second_team_avg
    prob_over = 1 - sum(poisson_prob(i, total_avg) for i in range(int(threshold) + 1))
    prob_under = 1 - prob_over
    return round(prob_over, 4), round(prob_under, 4)

# Function to calculate vig-adjusted probabilities
def calculate_vig_adjusted_probabilities(odds_over, odds_under):
    implied_prob_over = 1 / odds_over
    implied_prob_under = 1 / odds_under
    total_implied_prob = implied_prob_over + implied_prob_under
    adjusted_prob_over = implied_prob_over / total_implied_prob
    adjusted_prob_under = implied_prob_under / total_implied_prob
    return round(adjusted_prob_over, 4), round(adjusted_prob_under, 4)

# Initialize Streamlit app
st.title("Enhanced Football Analysis App with All Tools")

# Sidebar for bankroll input
bankroll = st.sidebar.number_input("Enter your current bankroll:", min_value=0.0, value=1000.0)

# Tool selection
tool = st.sidebar.radio("Choose a tool:", [
    "Shots Market",
    "Player Shots Market",
    "Corners Market",
    "Cards Market",
    "Team Specific Analysis",
    "Combo Bets"
])

# Shots Market Tool
if tool == "Shots Market":
    st.header("Shots Market Analysis with Tool and Vig Probabilities")

    # Inputs
    home_shots = st.number_input("Home Team Average Shots:", min_value=0.0, value=14.0)
    away_shots = st.number_input("Away Team Average Shots:", min_value=0.0, value=10.0)
    home_allowed_shots = st.number_input("Opponent Shots Allowed by Home Team:", min_value=0.0, value=8.0)
    away_allowed_shots = st.number_input("Opponent Shots Allowed by Away Team:", min_value=0.0, value=12.0)
    
    # Inputs for head-to-head matches
    st.subheader("Recent Head-to-Head Matches")
    num_h2h_matches = st.number_input("Number of recent head-to-head matches to include:", min_value=1, max_value=10, value=4)
    h2h_shots = []
    for i in range(int(num_h2h_matches)):
        h2h_shots.append(st.number_input(f"Total shots in head-to-head match {i + 1}:", min_value=0.0, value=20.0, key=f"h2h_{i}"))
    avg_h2h_shots = round(sum(h2h_shots) / len(h2h_shots), 2) if h2h_shots else 0.0

    threshold = st.number_input("Threshold for Total Shots (e.g., 22.5):", min_value=0.0, value=22.5)
    odds_over = st.number_input("Odds for Over Shots:", min_value=1.01, value=1.8)
    odds_under = st.number_input("Odds for Under Shots:", min_value=1.01, value=2.0)

    # Realism Adjustment
    realism_factor = st.slider("Realism Adjustment Factor:", 0.8, 1.2, 1.0)

    if st.button("Analyze Shots Market"):
        # Combine data to calculate average
        total_avg_shots = (
            (home_shots + away_shots) / 2 +
            (home_allowed_shots + away_allowed_shots) / 2 * 0.5 +
            avg_h2h_shots * 0.2
        ) * realism_factor

        # Calculate tool probabilities
        tool_prob_over, tool_prob_under = calculate_poisson_prob(total_avg_shots, 0, threshold)

        # Calculate vig-adjusted probabilities
        adjusted_prob_over, adjusted_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)

        # Use vig-adjusted probabilities for EV and Kelly calculations
        ev_over = calculate_ev(adjusted_prob_over, odds_over)
        ev_under = calculate_ev(adjusted_prob_under, odds_under)
        kelly_over = calculate_kelly(adjusted_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(adjusted_prob_under, odds_under, bankroll)

        # Display results
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Adjusted Avg Shots": [total_avg_shots, total_avg_shots],
            "Threshold": [threshold, threshold],
            "Tool Probability (%)": [f"{tool_prob_over * 100:.2f}%", f"{tool_prob_under * 100:.2f}%"],
            "Vig-Adjusted Probability (%)": [f"{adjusted_prob_over * 100:.2f}%", f"{adjusted_prob_under * 100:.2f}%"],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

# Player Shots Market Tool
elif tool == "Player Shots Market":
    st.header("Player Shots Market Analysis")
    player_name = st.text_input("Enter Player Name:", "Isak")
    avg_shots = st.number_input(f"Average Shots per Match for {player_name}:", min_value=0.0, value=3.0)
    threshold = st.number_input("Threshold for Shots (e.g., 2.5):", min_value=0.0, value=2.5)
    odds_over = st.number_input("Odds for Over Shots:", min_value=1.01, value=1.8)
    odds_under = st.number_input("Odds for Under Shots:", min_value=1.01, value=2.0)

    if st.button(f"Analyze Shots Market for {player_name}"):
        adjusted_prob_over, adjusted_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        ev_over = calculate_ev(adjusted_prob_over, odds_over)
        ev_under = calculate_ev(adjusted_prob_under, odds_under)
        kelly_over = calculate_kelly(adjusted_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(adjusted_prob_under, odds_under, bankroll)
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Threshold": [threshold, threshold],
            "Vig-Adjusted Probability (%)": [f"{adjusted_prob_over * 100:.2f}%", f"{adjusted_prob_under * 100:.2f}%"],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

# Corners Market Tool
elif tool == "Corners Market":
    st.header("Corners Market Analysis")
    home_corners = st.number_input("Home Team Average Corners:", min_value=0.0, value=6.0)
    away_corners = st.number_input("Away Team Average Corners:", min_value=0.0, value=4.0)
    threshold = st.number_input("Threshold for Total Corners:", min_value=0.0, value=9.5)
    odds_over = st.number_input("Odds for Over Corners:", min_value=1.01, value=1.8)
    odds_under = st.number_input("Odds for Under Corners:", min_value=1.01, value=2.0)

    if st.button("Analyze Corners Market"):
        adjusted_prob_over, adjusted_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        ev_over = calculate_ev(adjusted_prob_over, odds_over)
        ev_under = calculate_ev(adjusted_prob_under, odds_under)
        kelly_over = calculate_kelly(adjusted_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(adjusted_prob_under, odds_under, bankroll)
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Threshold": [threshold, threshold],
            "Vig-Adjusted Probability (%)": [f"{adjusted_prob_over * 100:.2f}%", f"{adjusted_prob_under * 100:.2f}%"],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

# Cards Market Tool
elif tool == "Cards Market":
    st.header("Cards Market Analysis")
    home_cards = st.number_input("Home Team Average Cards:", min_value=0.0, value=2.0)
    away_cards = st.number_input("Away Team Average Cards:", min_value=0.0, value=1.5)
    threshold = st.number_input("Threshold for Total Cards:", min_value=0.0, value=3.5)
    odds_over = st.number_input("Odds for Over Cards:", min_value=1.01, value=1.8)
    odds_under = st.number_input("Odds for Under Cards:", min_value=1.01, value=2.0)

    if st.button("Analyze Cards Market"):
        adjusted_prob_over, adjusted_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        ev_over = calculate_ev(adjusted_prob_over, odds_over)
        ev_under = calculate_ev(adjusted_prob_under, odds_under)
        kelly_over = calculate_kelly(adjusted_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(adjusted_prob_under, odds_under, bankroll)
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Threshold": [threshold, threshold],
            "Vig-Adjusted Probability (%)": [f"{adjusted_prob_over * 100:.2f}%", f"{adjusted_prob_under * 100:.2f}%"],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

# Team Specific Analysis Tool
elif tool == "Team Specific Analysis":
    st.header("Team Specific Analysis")
    team = st.text_input("Enter Team Name:", placeholder="e.g., Chelsea")
    avg_shots = st.number_input(f"Average Shots for {team}:", min_value=0.0, value=12.0)
    opp_shots_allowed = st.number_input(f"Opponent's Shots Allowed per Match:", min_value=0.0, value=10.0)
    threshold = st.number_input("Threshold for Shots:", min_value=0.0, value=9.5)
    odds_over = st.number_input("Odds for Over:", min_value=1.01, value=1.8)
    odds_under = st.number_input("Odds for Under:", min_value=1.01, value=2.0)

    if st.button(f"Analyze {team}'s Shot Market"):
        adjusted_prob_over, adjusted_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        ev_over = calculate_ev(adjusted_prob_over, odds_over)
        ev_under = calculate_ev(adjusted_prob_under, odds_under)
        kelly_over = calculate_kelly(adjusted_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(adjusted_prob_under, odds_under, bankroll)
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Threshold": [threshold, threshold],
            "Vig-Adjusted Probability (%)": [f"{adjusted_prob_over * 100:.2f}%", f"{adjusted_prob_under * 100:.2f}%"],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)
