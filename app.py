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

# Adjusted Fair Odds Calculation
def calculate_fair_odds(probability, margin=0.05):
    return round((1 / (probability * (1 - margin))), 2) if probability > 0 else "N/A"

# Adjusted Min Playable Odds Calculation
def calculate_min_playable_odds(probability, margin=0.05):
    adjusted_prob = probability * (1 - margin)  # Reduce probability to account for margin
    return round((1 / adjusted_prob), 2) if adjusted_prob > 0 else "N/A"

# Function to calculate probabilities using Poisson distribution
def calculate_poisson_prob(avg_shots, threshold):
    prob_over = 1 - sum(poisson_prob(i, avg_shots) for i in range(int(threshold) + 1))
    prob_under = 1 - prob_over
    return round(prob_over, 4), round(prob_under, 4)

# Initialize session state for saving analysis data
if "bet_tracker" not in st.session_state:
    st.session_state["bet_tracker"] = []  # Stores all placed bets

if "combo_data" not in st.session_state:
    st.session_state["combo_data"] = []  # Stores Combo Bets analysis data

if "player_shots_data" not in st.session_state:
    st.session_state["player_shots_data"] = []  # Stores Player Shots Market analysis data

# App Layout
st.title("Football Analysis Tools with Player Shots Market and More")

tool = st.sidebar.radio("Choose a tool:", [
    "Corners Market",
    "Shots Market",
    "Shots on Target Market",
    "Cards Market",
    "Team Specific Analysis",
    "Player Shots Market",
    "Combo Bets",
    "Bet Tracker",
    "Risk Management for Multiple Bets"
])


# Sidebar for bankroll input
bankroll = st.sidebar.number_input("Enter your current bankroll:", min_value=0.0, value=1000.0)

# Shared function for market tools
def input_and_analyze_market(title):
    st.header(f"{title} Analysis")
    home_avg = st.number_input(f"Home Team Average {title} per match:", min_value=0.0, value=5.5, step=0.1, key=f"{title}_home")
    away_avg = st.number_input(f"Away Team Average {title} per match:", min_value=0.0, value=4.5, step=0.1, key=f"{title}_away")
    threshold = st.number_input(f"Threshold for Total {title} (e.g., 9.5):", min_value=0.0, value=9.5)
    odds_over = st.number_input(f"Odds for Over {title}:", min_value=1.01, value=1.8)
    odds_under = st.number_input(f"Odds for Under {title}:", min_value=1.01, value=2.0)

    # Analyze results
    if st.button(f"Analyze {title}", key=f"analyze_{title}"):
        prob_over, prob_under = calculate_poisson_prob(home_avg + away_avg, threshold)
        ev_over = calculate_ev(prob_over, odds_over)
        ev_under = calculate_ev(prob_under, odds_under)
        kelly_over = calculate_kelly(prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(prob_under, odds_under, bankroll)

        # Calculate fair odds and minimum playable odds with margin
        margin = 0.05  # Adjust this margin as needed
        fair_odds_over = calculate_fair_odds(prob_over, margin)
        fair_odds_under = calculate_fair_odds(prob_under, margin)
        min_playable_odds_over = calculate_min_playable_odds(prob_over, margin)
        min_playable_odds_under = calculate_min_playable_odds(prob_under, margin)

        # Display results
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Threshold": [threshold, threshold],
            "Probability": [f"{prob_over * 100:.2f}%", f"{prob_under * 100:.2f}%"],
            "Odds": [odds_over, odds_under],
            "Fair Odds": [fair_odds_over, fair_odds_under],
            "Min Playable Odds": [min_playable_odds_over, min_playable_odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under],
        })
        st.table(results)
        # Tool: Team Specific Analysis
if tool == "Team Specific Analysis":
    st.header("Team Specific Analysis")

    # Input team name
    team = st.text_input("Enter Team Name", placeholder="e.g., Inter, Juventus, Napoli")
    half = st.selectbox("Select Half", ["1st Half", "2nd Half"])

    # Input average shots and odds
    if team:
        avg_shots = st.number_input(f"Average Shots for {team} in {half}:", min_value=0.0, value=6.5)

        threshold = st.number_input(f"Threshold for Shots:", min_value=0.0, value=6.5)
        odds_over = st.number_input(f"Odds for Over {threshold} Shots:", min_value=1.01, value=1.8)
        odds_under = st.number_input(f"Odds for Under {threshold} Shots:", min_value=1.01, value=2.0)

        # Analyze results
        if st.button(f"Analyze Team Market for {team} in {half}"):
            # Calculate probabilities using Poisson distribution
            prob_over, prob_under = calculate_poisson_prob(avg_shots, threshold)

            # Calculate EV and Kelly Criterion
            ev_over = calculate_ev(prob_over, odds_over)
            ev_under = calculate_ev(prob_under, odds_under)
            kelly_over = calculate_kelly(prob_over, odds_over, bankroll)
            kelly_under = calculate_kelly(prob_under, odds_under, bankroll)

            # Calculate fair odds and minimum playable odds
            margin = 0.05
            fair_odds_over = calculate_fair_odds(prob_over, margin)
            fair_odds_under = calculate_fair_odds(prob_under, margin)
            min_playable_odds_over = calculate_min_playable_odds(prob_over, margin)
            min_playable_odds_under = calculate_min_playable_odds(prob_under, margin)

            # Display results
            results = pd.DataFrame({
                "Bet Option": ["Over", "Under"],
                "Threshold": [threshold, threshold],
                "Probability": [f"{prob_over * 100:.2f}%", f"{prob_under * 100:.2f}%"],
                "Odds": [odds_over, odds_under],
                "Fair Odds": [fair_odds_over, fair_odds_under],
                "Min Playable Odds": [min_playable_odds_over, min_playable_odds_under],
                "EV (%)": [ev_over, ev_under],
                "Recommended Bet": [kelly_over, kelly_under],
            })
            st.table(results)

# Tool: Team Specific Analysis
if tool == "Team Specific Analysis":
    st.header("Team Specific Analysis with Optional Opponent Strength Adjustment")

    # Input team name
    team = st.text_input("Enter Team Name", placeholder="e.g., Ipswich, Chelsea")

    # Optional: Use adjusted stats based on opponent strength
    use_adjusted_stats = st.checkbox("Use Adjusted Stats Based on Opponent Strength")

    if use_adjusted_stats:
        # If user opts to adjust stats, allow selection of opponent strength
        opponent_strength = st.selectbox(
            "Select Opponent Strength Category:",
            ["Top Teams (e.g., Chelsea)", "Mid Teams", "Low Teams"]
        )

        # Adjusted stats based on selected opponent strength
        if opponent_strength == "Top Teams (e.g., Chelsea)":
            avg_shots = st.number_input(f"Average Shots for {team} vs Top Teams:", min_value=0.0, value=5.0)
        elif opponent_strength == "Mid Teams":
            avg_shots = st.number_input(f"Average Shots for {team} vs Mid Teams:", min_value=0.0, value=8.0)
        else:
            avg_shots = st.number_input(f"Average Shots for {team} vs Low Teams:", min_value=0.0, value=10.0)
    else:
        # Default stats (season average) if adjusted stats are not used
        avg_shots = st.number_input(f"Season Average Shots for {team}:", min_value=0.0, value=7.0)

    # Input threshold and odds
    threshold = st.number_input(f"Threshold for Shots for {team}:", min_value=0.0, value=6.5)
    odds_over = st.number_input(f"Odds for Over {threshold} Shots:", min_value=1.01, value=1.8)

    # Analyze results
    if st.button(f"Analyze Team Market for {team}"):
        prob_over = calculate_poisson_prob(avg_shots, 0, threshold)[0]
        ev_over = calculate_ev(prob_over, odds_over)
        kelly_over = calculate_kelly(prob_over, odds_over, bankroll)

        # Display results
        results = pd.DataFrame({
            "Bet Option": ["Over"],
            "Adjusted/Default Avg Shots": [avg_shots],
            "Threshold": [threshold],
            "Probability": [f"{prob_over * 100:.2f}%"],
            "Odds": [odds_over],
            "EV (%)": [ev_over],
            "Recommended Bet": [kelly_over]
        })
        st.table(results)


        # Display results
        results = pd.DataFrame({
            "Player": [player],
            "Threshold": [threshold],
            "Probability Over": [f"{prob_over * 100:.2f}%"],
            "Odds Over": [odds_over],
            "Fair Odds": [fair_odds_over],
            "Min Playable Odds": [min_playable_odds_over],
            "EV (%)": [ev_over],
            "Recommended Bet": [kelly_over]
        })
        st.table(results)


# Tool: Combo Bets
elif tool == "Combo Bets":
    st.header("Combo Bets Analysis")

    # Number of selections
    num_selections = st.number_input("Number of selections in the combo bet:", min_value=2, max_value=10, value=3, step=1)

    # Input probabilities and odds for each selection
    selections = []
    for i in range(int(num_selections)):
        st.subheader(f"Selection {i + 1}")
        probability = st.number_input(f"Probability for Selection {i + 1} (%)", min_value=0.0, max_value=100.0, value=50.0, key=f"prob_{i}")
        odds = st.number_input(f"Odds for Selection {i + 1}", min_value=1.01, value=2.0, key=f"odds_{i}")
        selections.append({"probability": probability / 100, "odds": odds})

    # Bookmaker combo odds
    bookmaker_combo_odds = st.number_input("Bookmaker's offered combo odds:", min_value=1.01, value=5.0)

    # Analyze Combo Bet
    if st.button("Analyze Combo Bet"):
        total_odds = 1
        total_prob = 1
        for sel in selections:
            total_odds *= sel["odds"]
            total_prob *= sel["probability"]

        ev = (bookmaker_combo_odds * total_prob - 1) * 100
        kelly_fraction = max(0, (bookmaker_combo_odds * total_prob - 1) / (bookmaker_combo_odds - 1))
        recommended_bet = round(bankroll * kelly_fraction, 2)

        st.session_state["combo_data"].append({
            "Combo Odds": bookmaker_combo_odds,
            "Fair Odds": total_odds,
            "Total Probability": total_prob,
            "EV (%)": ev,
            "Recommended Bet": recommended_bet
        })

        # Display Combo Bet results
        st.success("Combo Bet analyzed successfully!")
        st.table(pd.DataFrame(st.session_state["combo_data"]))

# Other Tools: Corners Market, Shots Market, Shots on Target Market, Cards Market
elif tool in ["Corners Market", "Shots Market", "Shots on Target Market", "Cards Market"]:
    input_and_analyze_market(tool)

# Tool: Bet Tracker
elif tool == "Bet Tracker":
    st.header("Bet Tracker")

    # Display saved bets
    if st.session_state["player_shots_data"]:
        tracker_df = pd.DataFrame(st.session_state["player_shots_data"])
        st.table(tracker_df)

    else:
        st.write("No bets added yet.")

    # Option to clear tracker
    if st.button("Clear Bet Tracker"):
        st.session_state["player_shots_data"] = []
        st.write("Bet Tracker cleared!")

        # Tool: Risk Management for Multiple Bets
if tool == "Risk Management for Multiple Bets":
    st.header("Risk Management for Multiple Bets")

    # Input total bankroll and allowed risk percentage
    total_bankroll = st.number_input("Enter your total bankroll:", min_value=0.0, value=10000.0, step=100.0)
    risk_percentage = st.number_input("Enter the percentage of bankroll to risk on this match:", min_value=0.0, max_value=100.0, value=5.0)

    # Input number of bets
    num_bets = st.number_input("Number of bets to analyze:", min_value=1, max_value=10, value=5, step=1)

    # Input details for each bet
    bets = []
    for i in range(int(num_bets)):
        st.subheader(f"Bet {i + 1}")
        probability = st.number_input(f"Probability for Bet {i + 1} (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"prob_{i}")
        odds = st.number_input(f"Odds for Bet {i + 1}", min_value=1.01, value=2.0, step=0.01, key=f"odds_{i}")
        ev = calculate_ev(probability / 100, odds)
        bets.append({"probability": probability / 100, "odds": odds, "EV (%)": ev})

    # Calculate total risk
    total_risk = total_bankroll * (risk_percentage / 100)

    # Normalize bets based on probability
    total_probability = sum(bet["probability"] for bet in bets)
    if total_probability > 0:
        normalized_bets = [
            round((bet["probability"] / total_probability) * total_risk, 2)
            for bet in bets
        ]
    else:
        normalized_bets = [0 for _ in bets]

    # Display results
    results = pd.DataFrame({
        "Probability (%)": [bet["probability"] * 100 for bet in bets],
        "Odds": [bet["odds"] for bet in bets],
        "EV (%)": [bet["EV (%)"] for bet in bets],
        "Recommended Bet (Probability Weighted)": normalized_bets
    })
    st.table(results)

    # Display total risk
    st.write(f"Total Risk: {total_risk:.2f} kr")

















































