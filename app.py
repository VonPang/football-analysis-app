import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp, factorial

# Function to calculate weighted probability with dynamic adjustment
def calculate_weighted_probability_dynamic(monte_carlo_prob, vig_prob, weight_factor=0.7):
    # Weight more towards Monte Carlo simulation
    return round((monte_carlo_prob * weight_factor + vig_prob * (1 - weight_factor)), 4)


# Poisson Probability Function
poisson_prob = lambda k, lmbda: (lmbda**k * exp(-lmbda)) / factorial(k)

# Monte Carlo Simulation Function
def monte_carlo_simulation(avg, simulations=10000):
    return np.random.poisson(avg, simulations)

# Function to calculate EV
def calculate_ev(probability, odds):
    return round((probability * odds - 1) * 100, 2) if probability else "N/A"

# Function to calculate Kelly Criterion Bet Size
def calculate_kelly(probability, odds, bankroll):
    edge = (probability * odds - 1)
    if edge > 0:
        kelly_fraction = edge / (odds - 1)
        return round(bankroll * kelly_fraction, 2)
    return 0.0

# Function to calculate vig-adjusted probabilities
def calculate_vig_adjusted_probabilities(odds_over, odds_under):
    implied_prob_over = 1 / odds_over
    implied_prob_under = 1 / odds_under
    total_implied_prob = implied_prob_over + implied_prob_under
    adjusted_prob_over = implied_prob_over / total_implied_prob
    adjusted_prob_under = implied_prob_under / total_implied_prob
    return round(adjusted_prob_over, 4), round(adjusted_prob_under, 4)

# Function to calculate weighted probability
def calculate_weighted_probability(monte_carlo_prob, vig_prob, weight):
    return round((monte_carlo_prob * weight + vig_prob * (1 - weight)), 4)

tool = st.sidebar.radio("Choose a tool:", [
    "Shots Market",
    "Player Shots Market",
    "Corners Market",
    "Tackling Market",
    "Combo Bets",
    "Player Fouls Market",
    "Player Tackles Market",
    "Throw-Ins Market",
    "Team-Specific Throw-Ins",
    "Team-Specific Shots"  # Nytt verktyg
])








if tool == "Shots Market":
    st.header("Shots Market Analysis with Monte Carlo Simulation")

    # Inputs
    bankroll = st.number_input("Enter your bankroll:", min_value=0.0, value=100.0)  # Konsistent bankroll
    home_shots = st.number_input("Home Team Average Shots:", min_value=0.0, value=15.0)
    away_shots = st.number_input("Away Team Average Shots:", min_value=0.0, value=10.0)
    home_allowed_shots = st.number_input("Opponent Shots Allowed by Home Team:", min_value=0.0, value=8.0)
    away_allowed_shots = st.number_input("Opponent Shots Allowed by Away Team:", min_value=0.0, value=12.0)
    home_possession = st.slider("Home Team Possession (%):", 0, 100, 55)  # Possession input
    away_possession = 100 - home_possession

    # Recent Form Inputs
    st.subheader("Recent Form (Last 5 Matches)")
    home_recent_wins = st.number_input("Home Team Wins (Last 5 Matches):", min_value=0, max_value=5, value=3)
    away_recent_wins = st.number_input("Away Team Wins (Last 5 Matches):", min_value=0, max_value=5, value=2)

    # Adjust form factor based on recent results
    home_recent_form_factor = 1 + (home_recent_wins - 2.5) * 0.1  # Neutral form is 2.5 wins
    away_recent_form_factor = 1 + (away_recent_wins - 2.5) * 0.1

    # Threshold and Odds Inputs
    threshold = st.number_input("Threshold for Total Shots (e.g., 22.5):", min_value=0.0, value=22.5)
    odds_over = st.number_input("Odds for Over Shots:", min_value=1.01, value=2.0)
    odds_under = st.number_input("Odds for Under Shots:", min_value=1.01, value=1.8)

    # Advanced Settings
    realism_factor = st.slider("Realism Adjustment Factor:", 0.8, 1.2, 1.0)  # Extra realism for simulation
    simulations = st.number_input("Number of Monte Carlo Simulations:", min_value=1000, value=10000, step=1000)
    game_state = st.selectbox("Game State (Expected Outcome):", ["Neutral", "Home Team Leading", "Away Team Leading", "Tight Game"])  # Game State

    if st.button("Analyze Shots Market"):
        # Adjusted average shots based on inputs
        possession_factor = (home_possession / 50)  # Normalize possession impact, where 50% is neutral
        adjusted_avg_home_shots = home_shots * home_recent_form_factor
        adjusted_avg_away_shots = away_shots * away_recent_form_factor
        adjusted_avg_shots = ((adjusted_avg_home_shots + adjusted_avg_away_shots) / 2 +
                              (home_allowed_shots + away_allowed_shots) / 2) * realism_factor * possession_factor

        # Adjust further based on game state
        if game_state == "Home Team Leading":
            adjusted_avg_shots *= 0.9  # Home team defends more, fewer shots overall
        elif game_state == "Away Team Leading":
            adjusted_avg_shots *= 1.1  # Away team defends more, home attacks more
        elif game_state == "Tight Game":
            adjusted_avg_shots *= 1.05  # More aggressive play from both teams

        # Monte Carlo Simulation (using the updated simulation logic from corners tool)
        simulated_shots = monte_carlo_simulation(adjusted_avg_shots, simulations)
        monte_carlo_prob_over = np.sum(simulated_shots > threshold) / len(simulated_shots)
        monte_carlo_prob_under = 1 - monte_carlo_prob_over

        # Vig-adjusted probabilities
        vig_prob_over, vig_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        weighted_prob_over = calculate_weighted_probability(monte_carlo_prob_over, vig_prob_over, 0.5)
        weighted_prob_under = 1 - weighted_prob_over

        # Expected Value (EV)
        ev_over = calculate_ev(weighted_prob_over, odds_over)
        ev_under = calculate_ev(weighted_prob_under, odds_under)

        # Kelly Criterion
        kelly_over = calculate_kelly(weighted_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(weighted_prob_under, odds_under, bankroll)

        # Fair Odds
        fair_odds_over = round(1 / weighted_prob_over, 2) if weighted_prob_over > 0 else "N/A"
        fair_odds_under = round(1 / weighted_prob_under, 2) if weighted_prob_under > 0 else "N/A"

        # Results Table
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Monte Carlo Probability (%)": [f"{monte_carlo_prob_over * 100:.2f}%", f"{monte_carlo_prob_under * 100:.2f}%"],
            "Weighted Probability (%)": [f"{weighted_prob_over * 100:.2f}%", f"{weighted_prob_under * 100:.2f}%"],
            "Fair Odds": [fair_odds_over, fair_odds_under],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

        # Plot Histogram
        st.subheader("Simulated Shots Distribution")
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_shots, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Simulated Shots')
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold} shots")
        plt.axvline(np.mean(simulated_shots), color='green', linestyle='-', label=f"Simulated Mean: {np.mean(simulated_shots):.2f} shots")
        plt.title("Simulated Shots Distribution")
        plt.xlabel("Total Shots")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)




if tool == "Player Shots Market":
    st.header("Player Shots Market Analysis with Monte Carlo Simulation")

    # Inputs
    player_name = st.text_input("Enter Player Name:", value="Player Name")
    avg_shots = st.number_input(f"Average Shots per Match for {player_name}:", min_value=0.0, value=3.0)
    threshold = st.number_input(f"Threshold for Shots for {player_name} (e.g., 2.5):", min_value=0.0, value=2.5)
    odds_over = st.number_input("Odds for Over Shots:", min_value=1.01, value=2.0)
    odds_under = st.number_input("Odds for Under Shots:", min_value=1.01, value=1.8)
    opponent_defense = st.selectbox("Opponent Defensive Strength:", ["Weak", "Average", "Strong"])
    player_position = st.selectbox("Player Position:", ["Forward", "Midfielder", "Defender"])
    expected_minutes = st.number_input("Expected Minutes Played:", min_value=0, max_value=90, value=90)

    # Bankroll input
    bankroll = st.number_input("Enter your bankroll:", min_value=0.0, value=100.0)

    # Adjust factors based on opponent defense, position, and minutes
    defense_factor = {"Weak": 1.2, "Average": 1.0, "Strong": 0.8}[opponent_defense]
    position_factor = {"Forward": 1.2, "Midfielder": 1.0, "Defender": 0.8}[player_position]
    minutes_factor = expected_minutes / 90  # Scale based on minutes played

    if st.button(f"Analyze Shots Market for {player_name}"):
        # Adjusted average shots
        adjusted_avg_shots = avg_shots * defense_factor * position_factor * minutes_factor

        # Monte Carlo Simulation (using updated simulation logic from corners tool)
        simulated_shots = monte_carlo_simulation(adjusted_avg_shots, simulations=10000)
        monte_carlo_prob_over = np.sum(simulated_shots > threshold) / len(simulated_shots)
        monte_carlo_prob_under = 1 - monte_carlo_prob_over

        # Vig-adjusted probabilities
        vig_prob_over, vig_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        weighted_prob_over = calculate_weighted_probability(monte_carlo_prob_over, vig_prob_over, 0.5)
        weighted_prob_under = 1 - weighted_prob_over

        # Expected Value (EV)
        ev_over = calculate_ev(weighted_prob_over, odds_over)
        ev_under = calculate_ev(weighted_prob_under, odds_under)

        # Kelly Criterion
        kelly_over = calculate_kelly(weighted_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(weighted_prob_under, odds_under, bankroll)

        # Fair Odds
        fair_odds_over = round(1 / weighted_prob_over, 2) if weighted_prob_over > 0 else "N/A"
        fair_odds_under = round(1 / weighted_prob_under, 2) if weighted_prob_under > 0 else "N/A"

        # Results Table
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Monte Carlo Probability (%)": [f"{monte_carlo_prob_over * 100:.2f}%", f"{monte_carlo_prob_under * 100:.2f}%"],
            "Weighted Probability (%)": [f"{weighted_prob_over * 100:.2f}%", f"{weighted_prob_under * 100:.2f}%"],
            "Fair Odds": [fair_odds_over, fair_odds_under],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

        # Plot Histogram
        st.subheader(f"Simulated Shots Distribution for {player_name}")
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_shots, bins=30, alpha=0.7, color='blue', edgecolor='black', label=f"{player_name} Simulated Shots")
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold} shots")
        plt.axvline(np.mean(simulated_shots), color='green', linestyle='-', label=f"Simulated Mean: {np.mean(simulated_shots):.2f} shots")
        plt.title(f"{player_name} Simulated Shots Distribution")
        plt.xlabel("Shots")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate weighted probability with dynamic adjustment
def calculate_weighted_probability_dynamic(monte_carlo_prob, pinnacle_prob, weight_factor=0.7):
    """Weighted probability combining Monte Carlo and Pinnacle probabilities."""
    return round((monte_carlo_prob * weight_factor + pinnacle_prob * (1 - weight_factor)), 4)

def monte_carlo_simulation(avg, simulations=10000):
    # Generate Poisson distributed data as floats
    simulated = np.random.poisson(avg, simulations).astype(float)  # Convert to float for addition
    simulated += np.random.normal(0, 0.1, size=simulations)  # Add slight randomness
    simulated = np.clip(simulated, 0, None)  # Ensure no negative values
    return simulated.astype(int)  # Convert back to integers


# Function to calculate EV
def calculate_ev(probability, odds):
    """Expected value calculation."""
    return round((probability * odds - 1) * 100, 2) if probability else "N/A"

# Function to calculate Kelly Criterion Bet Size
def calculate_kelly(probability, odds, bankroll):
    """Calculate recommended bet size using the Kelly criterion."""
    edge = (probability * odds - 1)
    if edge > 0:
        kelly_fraction = edge / (odds - 1)
        return round(bankroll * kelly_fraction, 2)
    return 0.0

# Function to calculate vig-adjusted probabilities
def calculate_vig_adjusted_probabilities(odds_over, odds_under):
    """Calculate probabilities adjusted for bookmaker's vig."""
    implied_prob_over = 1 / odds_over
    implied_prob_under = 1 / odds_under
    total_implied_prob = implied_prob_over + implied_prob_under
    adjusted_prob_over = implied_prob_over / total_implied_prob
    adjusted_prob_under = implied_prob_under / total_implied_prob
    return round(adjusted_prob_over, 4), round(adjusted_prob_under, 4)

if tool == "Corners Market":
    st.header("Corners Market Analysis with Monte Carlo Simulation")

    # Inputs
    bankroll = st.number_input("Enter your bankroll:", min_value=0.0, value=100.0)
    home_avg_corners = st.number_input("Home Team Average Corners Per Match:", min_value=0.0, value=6.0)
    away_avg_corners = st.number_input("Away Team Average Corners Per Match:", min_value=0.0, value=4.0)
    historical_corners = st.number_input("Average Corners in Past Matches Between Teams:", min_value=0.0, value=10.0)
    home_possession = st.slider("Home Team Possession (%):", 0, 100, 55)
    away_possession = 100 - home_possession
    weather_condition = st.selectbox("Weather Condition:", ["Clear", "Rainy", "Windy"])
    team_tactics_home = st.selectbox("Home Team Tactics:", ["Defensive", "Balanced", "Offensive"])
    team_tactics_away = st.selectbox("Away Team Tactics:", ["Defensive", "Balanced", "Offensive"])
    home_fouls_drawn = st.number_input("Home Team Fouls Drawn per Match:", min_value=0.0, value=10.0)
    away_fouls_drawn = st.number_input("Away Team Fouls Drawn per Match:", min_value=0.0, value=8.0)
    home_aggression = st.slider("Home Team Aggression Level:", 0.5, 1.5, 1.0)
    away_aggression = st.slider("Away Team Aggression Level:", 0.5, 1.5, 1.0)
    threshold = st.number_input("Threshold for Total Corners (e.g., 9.5):", min_value=0.0, value=9.5)
    odds_over = st.number_input("Odds for Over Corners:", min_value=1.01, value=2.0)
    odds_under = st.number_input("Odds for Under Corners:", min_value=1.01, value=1.8)
    
    # Pinnacle Odds Input
    pinnacle_odds_over = st.number_input("Pinnacle Odds for Over Corners:", min_value=1.01, value=1.95)
    pinnacle_odds_under = st.number_input("Pinnacle Odds for Under Corners:", min_value=1.01, value=1.85)

    # Number of Monte Carlo Simulations
    simulations = st.number_input("Number of Monte Carlo Simulations:", min_value=1000, value=10000, step=1000)

    # Adjustment factors
    tactics_factor_home = {"Defensive": 0.9, "Balanced": 1.0, "Offensive": 1.1}[team_tactics_home]
    tactics_factor_away = {"Defensive": 0.9, "Balanced": 1.0, "Offensive": 1.1}[team_tactics_away]
    weather_factor = {"Clear": 1.0, "Rainy": 1.1, "Windy": 1.2}[weather_condition]

    if st.button("Analyze Corners Market"):
        # Adjusted averages
        adjusted_home_corners = home_avg_corners * (home_possession / 100) * tactics_factor_home * weather_factor
        adjusted_away_corners = away_avg_corners * (away_possession / 100) * tactics_factor_away * weather_factor
        fouls_factor = (home_fouls_drawn * home_aggression + away_fouls_drawn * away_aggression) / 20
        total_avg_corners = (adjusted_home_corners + adjusted_away_corners + historical_corners) / 3 + fouls_factor

        # Monte Carlo Simulation
        simulated_corners = monte_carlo_simulation(total_avg_corners, simulations)
        monte_carlo_prob_over = np.sum(simulated_corners > threshold) / len(simulated_corners)
        monte_carlo_prob_under = 1 - monte_carlo_prob_over

        # Vig-adjusted probabilities
        vig_prob_over, vig_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        weighted_prob_over = calculate_weighted_probability(monte_carlo_prob_over, vig_prob_over, 0.5)
        weighted_prob_under = 1 - weighted_prob_over

        # Pinnacle-adjusted probabilities
        pinnacle_prob_over = 1 / pinnacle_odds_over
        pinnacle_prob_under = 1 / pinnacle_odds_under
        combined_prob_over = (weighted_prob_over + pinnacle_prob_over) / 2
        combined_prob_under = 1 - combined_prob_over

        # Expected Value (EV)
        ev_over = calculate_ev(combined_prob_over, odds_over)
        ev_under = calculate_ev(combined_prob_under, odds_under)

        # Kelly Criterion
        kelly_over = calculate_kelly(combined_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(combined_prob_under, odds_under, bankroll)

        # Fair Odds
        fair_odds_over = round(1 / combined_prob_over, 2) if combined_prob_over > 0 else "N/A"
        fair_odds_under = round(1 / combined_prob_under, 2) if combined_prob_under > 0 else "N/A"

        # Results Table
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Monte Carlo Probability (%)": [f"{monte_carlo_prob_over * 100:.2f}%", f"{monte_carlo_prob_under * 100:.2f}%"],
            "Weighted Probability (%)": [f"{weighted_prob_over * 100:.2f}%", f"{weighted_prob_under * 100:.2f}%"],
            "Pinnacle Probability (%)": [f"{pinnacle_prob_over * 100:.2f}%", f"{pinnacle_prob_under * 100:.2f}%"],
            "Combined Probability (%)": [f"{combined_prob_over * 100:.2f}%", f"{combined_prob_under * 100:.2f}%"],
            "Fair Odds": [fair_odds_over, fair_odds_under],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

        # Plot Histogram
        st.subheader("Simulated Corners Distribution")
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_corners, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Simulated Corners')
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold} corners")
        plt.axvline(np.mean(simulated_corners), color='green', linestyle='-', label=f"Simulated Mean: {np.mean(simulated_corners):.2f} corners")
        plt.title("Simulated Corners Distribution")
        plt.xlabel("Total Corners")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)






if tool == "Tackling Market":
    st.header("Tackling Market Analysis with Monte Carlo Simulation")

    # Inputs
    bankroll = st.number_input("Enter your bankroll:", min_value=0.0, value=100.0)
    home_tackles = st.number_input("Home Team Average Tackles:", min_value=0.0, value=17.0)
    away_tackles = st.number_input("Away Team Average Tackles:", min_value=0.0, value=19.0)
    threshold = st.number_input("Threshold for Total Tackles (e.g., 35.5):", min_value=0.0, value=35.5)
    odds_over = st.number_input("Odds for Over Tackles:", min_value=1.01, value=1.33)
    odds_under = st.number_input("Odds for Under Tackles:", min_value=1.01, value=3.25)

    # Additional Parameters
    home_possession = st.slider("Home Team Possession (%):", 0, 100, 55)
    away_possession = 100 - home_possession  # Ensures possession sums to 100%
    home_aggression = st.slider("Home Team Aggression Level:", 0.5, 1.5, 1.0)  # Slider for Home Team
    away_aggression = st.slider("Away Team Aggression Level:", 0.5, 1.5, 1.0)  # Slider for Away Team
    match_importance = st.selectbox("Match Importance:", ["Low", "Medium", "High"])
    weather_condition = st.selectbox("Weather Condition:", ["Clear", "Rainy", "Snowy", "Windy"])

    # Adjustment factors
    importance_factor = {"Low": 0.9, "Medium": 1.0, "High": 1.1}[match_importance]
    weather_factor = {"Clear": 1.0, "Rainy": 1.1, "Snowy": 1.2, "Windy": 1.15}[weather_condition]

    if st.button("Analyze Tackling Market"):
        # Adjusted averages based on possession, aggression, importance, and weather
        adjusted_home_tackles = home_tackles * (1 - home_possession / 100) * home_aggression * importance_factor * weather_factor
        adjusted_away_tackles = away_tackles * (1 - away_possession / 100) * away_aggression * importance_factor * weather_factor
        adjusted_avg_tackles = adjusted_home_tackles + adjusted_away_tackles

        # Monte Carlo Simulation (using updated simulation logic from corners tool)
        simulated_tackles = monte_carlo_simulation(adjusted_avg_tackles, simulations=10000)
        monte_carlo_prob_over = np.sum(simulated_tackles > threshold) / len(simulated_tackles)
        monte_carlo_prob_under = 1 - monte_carlo_prob_over

        # Vig-adjusted probabilities
        vig_prob_over, vig_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        weighted_prob_over = calculate_weighted_probability(monte_carlo_prob_over, vig_prob_over, 0.5)
        weighted_prob_under = 1 - weighted_prob_over

        # Expected Value (EV)
        ev_over = calculate_ev(weighted_prob_over, odds_over)
        ev_under = calculate_ev(weighted_prob_under, odds_under)

        # Kelly Criterion
        kelly_over = calculate_kelly(weighted_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(weighted_prob_under, odds_under, bankroll)

        # Fair Odds
        fair_odds_over = round(1 / weighted_prob_over, 2) if weighted_prob_over > 0 else "N/A"
        fair_odds_under = round(1 / weighted_prob_under, 2) if weighted_prob_under > 0 else "N/A"

        # Results Table
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Monte Carlo Probability (%)": [f"{monte_carlo_prob_over * 100:.2f}%", f"{monte_carlo_prob_under * 100:.2f}%"],
            "Weighted Probability (%)": [f"{weighted_prob_over * 100:.2f}%", f"{weighted_prob_under * 100:.2f}%"],
            "Fair Odds": [fair_odds_over, fair_odds_under],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

        # Plot Histogram
        st.subheader("Simulated Tackles Distribution")
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_tackles, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Simulated Tackles')
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold} tackles")
        plt.axvline(np.mean(simulated_tackles), color='green', linestyle='-', label=f"Simulated Mean: {np.mean(simulated_tackles):.2f} tackles")
        plt.title("Simulated Tackles Distribution")
        plt.xlabel("Total Tackles")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)





if tool == "Combo Bets":
    st.header("Combo Bets Analysis")

    # Inputs
    st.subheader("Enter Bet Details for Each Leg of the Combo")
    num_legs = st.number_input("Number of Legs in Combo Bet:", min_value=2, max_value=10, value=2)

    # Collecting details for each leg
    legs = []
    for i in range(1, num_legs + 1):
        with st.expander(f"Leg {i}"):
            probability = st.number_input(f"Implied Probability for Leg {i} (%):", min_value=0.01, max_value=100.0, value=50.0) / 100
            odds = st.number_input(f"Odds for Leg {i}:", min_value=1.01, value=2.0)
            legs.append({"probability": probability, "odds": odds})

    # Input for bookmaker's combo odds
    bookmaker_odds = st.number_input("Bookmaker Combo Odds:", min_value=1.01, value=5.0)

    # Simulation settings (inherited from corners)
    simulations = st.number_input("Number of Monte Carlo Simulations:", min_value=1000, value=10000, step=1000)

    if st.button("Analyze Combo Bet"):
        # Monte Carlo Simulation for combo probabilities
        combined_probabilities = []
        for _ in range(simulations):
            leg_simulations = [np.random.uniform(0, 1) < leg["probability"] for leg in legs]
            combined_probabilities.append(all(leg_simulations))  # All legs must succeed for the combo

        monte_carlo_prob = np.mean(combined_probabilities)

        # Weighted Probability (using bookmaker and Monte Carlo probabilities)
        vig_prob = 1 / bookmaker_odds
        weighted_prob = calculate_weighted_probability_dynamic(monte_carlo_prob, vig_prob, 0.7)

        # Calculated EV
        ev = calculate_ev(weighted_prob, bookmaker_odds)

        # Kelly Criterion
        bankroll = st.number_input("Enter your bankroll:", min_value=0.0, value=100.0)
        kelly_bet_size = calculate_kelly(weighted_prob, bookmaker_odds, bankroll)

        # Fair Odds
        fair_odds = round(1 / weighted_prob, 2) if weighted_prob > 0 else "N/A"

        # Display Results
        st.subheader("Combo Bet Results")
        st.write(f"**Monte Carlo Probability:** {monte_carlo_prob * 100:.2f}%")
        st.write(f"**Weighted Probability:** {weighted_prob * 100:.2f}%")
        st.write(f"**Fair Odds:** {fair_odds}")
        st.write(f"**Bookmaker Odds:** {bookmaker_odds}")
        st.write(f"**Expected Value (EV):** {ev}%")
        st.write(f"**Recommended Bet Size (Kelly Criterion):** {kelly_bet_size}")

        # Optional: Display detailed breakdown for each leg
        breakdown = pd.DataFrame(legs)
        breakdown["Implied Probability (%)"] = breakdown["probability"] * 100
        breakdown["Odds"] = breakdown["odds"]
        st.subheader("Detailed Breakdown for Each Leg")
        st.table(breakdown[["Implied Probability (%)", "Odds"]])

        # Histogram for Simulated Combo Outcomes
        st.subheader("Simulated Combo Bet Outcomes")
        plt.figure(figsize=(8, 5))
        plt.hist([int(success) for success in combined_probabilities], bins=2, alpha=0.7, color='blue', edgecolor='black')
        plt.xticks([0, 1], ['Loss', 'Win'])
        plt.title("Distribution of Simulated Combo Bet Outcomes")
        plt.xlabel("Outcome")
        plt.ylabel("Frequency")
        st.pyplot(plt)


if tool == "Player Fouls Market":
    st.header("Player Fouls Market Analysis with Monte Carlo Simulation")

    # Inputs
    player_name = st.text_input("Enter Player Name:", value="Player Name")
    avg_fouls = st.number_input(f"Average Fouls Per Match for {player_name}:", min_value=0.0, value=1.5)
    position = st.selectbox("Player Position:", ["Defender", "Midfielder", "Forward"])
    disciplinary_record = st.selectbox("Disciplinary Record:", ["Aggressive", "Neutral", "Disciplined"])
    match_importance = st.selectbox("Match Importance:", ["Low", "Medium", "High"])
    opponent_playstyle = st.selectbox("Opponent Playstyle:", ["Defensive", "Balanced", "Offensive"])
    expected_minutes = st.number_input("Expected Minutes Played:", min_value=0, max_value=90, value=90)
    threshold = st.number_input("Threshold for Fouls (e.g., 1.5):", min_value=0.0, value=1.5)
    odds_over = st.number_input("Odds for Over Fouls:", min_value=1.01, value=2.0)
    odds_under = st.number_input("Odds for Under Fouls:", min_value=1.01, value=1.8)

    # Pinnacle Odds Input
    pinnacle_odds_over = st.number_input("Pinnacle Odds for Over Fouls:", min_value=1.01, value=1.95)
    pinnacle_odds_under = st.number_input("Pinnacle Odds for Under Fouls:", min_value=1.01, value=1.85)

    # Simulation settings (from Corners Market)
    simulations = st.number_input("Number of Monte Carlo Simulations:", min_value=1000, value=10000, step=1000)

    # Adjustment factors
    position_factor = {"Defender": 1.2, "Midfielder": 1.0, "Forward": 0.8}[position]
    disciplinary_factor = {"Aggressive": 1.2, "Neutral": 1.0, "Disciplined": 0.8}[disciplinary_record]
    importance_factor = {"Low": 0.9, "Medium": 1.0, "High": 1.1}[match_importance]
    playstyle_factor = {"Defensive": 1.2, "Balanced": 1.0, "Offensive": 0.8}[opponent_playstyle]
    minutes_factor = expected_minutes / 90  # Scale based on expected minutes played

    if st.button(f"Analyze Fouls Market for {player_name}"):
        # Adjusted average fouls based on factors
        adjusted_avg_fouls = avg_fouls * position_factor * disciplinary_factor * importance_factor * playstyle_factor * minutes_factor

        # Monte Carlo Simulation
        simulated_fouls = monte_carlo_simulation(adjusted_avg_fouls, simulations)
        monte_carlo_prob_over = np.sum(simulated_fouls > threshold) / len(simulated_fouls)
        monte_carlo_prob_under = 1 - monte_carlo_prob_over

        # Vig-adjusted probabilities
        vig_prob_over, vig_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        weighted_prob_over = calculate_weighted_probability_dynamic(monte_carlo_prob_over, vig_prob_over, 0.7)
        weighted_prob_under = 1 - weighted_prob_over

        # Pinnacle-adjusted probabilities
        pinnacle_prob_over = 1 / pinnacle_odds_over
        pinnacle_prob_under = 1 / pinnacle_odds_under
        combined_prob_over = (weighted_prob_over + pinnacle_prob_over) / 2
        combined_prob_under = 1 - combined_prob_over

        # Expected Value (EV)
        ev_over = calculate_ev(combined_prob_over, odds_over)
        ev_under = calculate_ev(combined_prob_under, odds_under)

        # Kelly Criterion
        bankroll = st.number_input("Enter your bankroll:", min_value=0.0, value=100.0)
        kelly_over = calculate_kelly(combined_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(combined_prob_under, odds_under, bankroll)

        # Fair Odds
        fair_odds_over = round(1 / combined_prob_over, 2) if combined_prob_over > 0 else "N/A"
        fair_odds_under = round(1 / combined_prob_under, 2) if combined_prob_under > 0 else "N/A"

        # Results Table
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Monte Carlo Probability (%)": [f"{monte_carlo_prob_over * 100:.2f}%", f"{monte_carlo_prob_under * 100:.2f}%"],
            "Weighted Probability (%)": [f"{weighted_prob_over * 100:.2f}%", f"{weighted_prob_under * 100:.2f}%"],
            "Pinnacle Probability (%)": [f"{pinnacle_prob_over * 100:.2f}%", f"{pinnacle_prob_under * 100:.2f}%"],
            "Combined Probability (%)": [f"{combined_prob_over * 100:.2f}%", f"{combined_prob_under * 100:.2f}%"],
            "Fair Odds": [fair_odds_over, fair_odds_under],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

        # Plot Histogram
        st.subheader(f"Simulated Fouls Distribution for {player_name}")
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_fouls, bins=30, alpha=0.7, color='blue', edgecolor='black', label=f"{player_name} Simulated Fouls")
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold} fouls")
        plt.axvline(np.mean(simulated_fouls), color='green', linestyle='-', label=f"Simulated Mean: {np.mean(simulated_fouls):.2f} fouls")
        plt.title(f"{player_name} Simulated Fouls Distribution")
        plt.xlabel("Fouls")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)

      

if tool == "Player Tackles Market":
    st.header("Player Tackles Market Analysis with Monte Carlo Simulation")

    # Inputs
    player_name = st.text_input("Enter Player Name:", value="Player Name")
    avg_tackles = st.number_input(f"Average Tackles Per Match for {player_name}:", min_value=0.0, value=2.5)
    position = st.selectbox("Player Position:", ["Defender", "Midfielder", "Forward"])
    match_importance = st.selectbox("Match Importance:", ["Low", "Medium", "High"])
    opponent_playstyle = st.selectbox("Opponent Playstyle:", ["Offensive", "Balanced", "Defensive"])
    expected_minutes = st.number_input("Expected Minutes Played:", min_value=0, max_value=90, value=90)
    threshold = st.number_input("Threshold for Tackles (e.g., 2.5):", min_value=0.0, value=2.5)
    odds_over = st.number_input("Odds for Over Tackles:", min_value=1.01, value=2.0)
    odds_under = st.number_input("Odds for Under Tackles:", min_value=1.01, value=1.8)

    # Pinnacle Odds Input
    pinnacle_odds_over = st.number_input("Pinnacle Odds for Over Tackles:", min_value=1.01, value=1.95)
    pinnacle_odds_under = st.number_input("Pinnacle Odds for Under Tackles:", min_value=1.01, value=1.85)

    # Simulation settings (from Corners Market)
    simulations = st.number_input("Number of Monte Carlo Simulations:", min_value=1000, value=10000, step=1000)

    # Adjustment factors
    position_factor = {"Defender": 1.2, "Midfielder": 1.0, "Forward": 0.8}[position]
    importance_factor = {"Low": 0.9, "Medium": 1.0, "High": 1.1}[match_importance]
    playstyle_factor = {"Offensive": 1.2, "Balanced": 1.0, "Defensive": 0.8}[opponent_playstyle]
    minutes_factor = expected_minutes / 90  # Scale based on expected minutes played

    if st.button(f"Analyze Tackles Market for {player_name}"):
        # Adjusted average tackles based on factors
        adjusted_avg_tackles = avg_tackles * position_factor * importance_factor * playstyle_factor * minutes_factor

        # Monte Carlo Simulation
        simulated_tackles = monte_carlo_simulation(adjusted_avg_tackles, simulations)
        monte_carlo_prob_over = np.sum(simulated_tackles > threshold) / len(simulated_tackles)
        monte_carlo_prob_under = 1 - monte_carlo_prob_over

        # Vig-adjusted probabilities
        vig_prob_over, vig_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        weighted_prob_over = calculate_weighted_probability_dynamic(monte_carlo_prob_over, vig_prob_over, 0.7)
        weighted_prob_under = 1 - weighted_prob_over

        # Pinnacle-adjusted probabilities
        pinnacle_prob_over = 1 / pinnacle_odds_over
        pinnacle_prob_under = 1 / pinnacle_odds_under
        combined_prob_over = (weighted_prob_over + pinnacle_prob_over) / 2
        combined_prob_under = 1 - combined_prob_over

        # Expected Value (EV)
        ev_over = calculate_ev(combined_prob_over, odds_over)
        ev_under = calculate_ev(combined_prob_under, odds_under)

        # Kelly Criterion
        bankroll = st.number_input("Enter your bankroll:", min_value=0.0, value=100.0)
        kelly_over = calculate_kelly(combined_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(combined_prob_under, odds_under, bankroll)

        # Fair Odds
        fair_odds_over = round(1 / combined_prob_over, 2) if combined_prob_over > 0 else "N/A"
        fair_odds_under = round(1 / combined_prob_under, 2) if combined_prob_under > 0 else "N/A"

        # Results Table
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Monte Carlo Probability (%)": [f"{monte_carlo_prob_over * 100:.2f}%", f"{monte_carlo_prob_under * 100:.2f}%"],
            "Weighted Probability (%)": [f"{weighted_prob_over * 100:.2f}%", f"{weighted_prob_under * 100:.2f}%"],
            "Pinnacle Probability (%)": [f"{pinnacle_prob_over * 100:.2f}%", f"{pinnacle_prob_under * 100:.2f}%"],
            "Combined Probability (%)": [f"{combined_prob_over * 100:.2f}%", f"{combined_prob_under * 100:.2f}%"],
            "Fair Odds": [fair_odds_over, fair_odds_under],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

        # Plot Histogram
        st.subheader(f"Simulated Tackles Distribution for {player_name}")
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_tackles, bins=30, alpha=0.7, color='blue', edgecolor='black', label=f"{player_name} Simulated Tackles")
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold} tackles")
        plt.axvline(np.mean(simulated_tackles), color='green', linestyle='-', label=f"Simulated Mean: {np.mean(simulated_tackles):.2f} tackles")
        plt.title(f"{player_name} Simulated Tackles Distribution")
        plt.xlabel("Tackles")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)


if tool == "Throw-Ins Market":
    st.header("Throw-Ins Market Analysis with Monte Carlo Simulation")

    # Inputs
    home_throw_ins = st.number_input("Home Team Average Throw-Ins Per Match:", min_value=0.0, value=12.0)
    away_throw_ins = st.number_input("Away Team Average Throw-Ins Per Match:", min_value=0.0, value=10.0)
    home_possession = st.slider("Home Team Possession (%):", 0, 100, 55)
    away_possession = 100 - home_possession  # Automatic calculation
    match_importance = st.selectbox("Match Importance:", ["Low", "Medium", "High"])
    weather_conditions = st.selectbox("Weather Conditions:", ["Good", "Rainy", "Windy"])
    threshold = st.number_input("Threshold for Total Throw-Ins (e.g., 25.5):", min_value=0.0, value=25.5)
    odds_over = st.number_input("Odds for Over Throw-Ins:", min_value=1.01, value=2.0)
    odds_under = st.number_input("Odds for Under Throw-Ins:", min_value=1.01, value=1.8)

    # Pinnacle Odds Input
    pinnacle_odds_over = st.number_input("Pinnacle Odds for Over Throw-Ins:", min_value=1.01, value=1.95)
    pinnacle_odds_under = st.number_input("Pinnacle Odds for Under Throw-Ins:", min_value=1.01, value=1.85)

    # Simulation settings (from Corners Market)
    simulations = st.number_input("Number of Monte Carlo Simulations:", min_value=1000, value=10000, step=1000)

    # Adjustment factors
    importance_factor = {"Low": 0.9, "Medium": 1.0, "High": 1.1}[match_importance]
    weather_factor = {"Good": 1.0, "Rainy": 1.1, "Windy": 1.2}[weather_conditions]

    if st.button("Analyze Throw-Ins Market"):
        # Adjusted average throw-ins based on factors
        adjusted_home_throw_ins = home_throw_ins * (1 - home_possession / 100) * importance_factor * weather_factor
        adjusted_away_throw_ins = away_throw_ins * (1 - away_possession / 100) * importance_factor * weather_factor
        adjusted_avg_throw_ins = adjusted_home_throw_ins + adjusted_away_throw_ins

        # Monte Carlo Simulation
        simulated_throw_ins = monte_carlo_simulation(adjusted_avg_throw_ins, simulations)
        monte_carlo_prob_over = np.sum(simulated_throw_ins > threshold) / len(simulated_throw_ins)
        monte_carlo_prob_under = 1 - monte_carlo_prob_over

        # Vig-adjusted probabilities
        vig_prob_over, vig_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        weighted_prob_over = calculate_weighted_probability_dynamic(monte_carlo_prob_over, vig_prob_over, 0.7)
        weighted_prob_under = 1 - weighted_prob_over

        # Pinnacle-adjusted probabilities
        pinnacle_prob_over = 1 / pinnacle_odds_over
        pinnacle_prob_under = 1 / pinnacle_odds_under
        combined_prob_over = (weighted_prob_over + pinnacle_prob_over) / 2
        combined_prob_under = 1 - combined_prob_over

        # Expected Value (EV)
        ev_over = calculate_ev(combined_prob_over, odds_over)
        ev_under = calculate_ev(combined_prob_under, odds_under)

        # Kelly Criterion
        bankroll = st.number_input("Enter your bankroll:", min_value=0.0, value=100.0)
        kelly_over = calculate_kelly(combined_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(combined_prob_under, odds_under, bankroll)

        # Fair Odds
        fair_odds_over = round(1 / combined_prob_over, 2) if combined_prob_over > 0 else "N/A"
        fair_odds_under = round(1 / combined_prob_under, 2) if combined_prob_under > 0 else "N/A"

        # Results Table
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Monte Carlo Probability (%)": [f"{monte_carlo_prob_over * 100:.2f}%", f"{monte_carlo_prob_under * 100:.2f}%"],
            "Weighted Probability (%)": [f"{weighted_prob_over * 100:.2f}%", f"{weighted_prob_under * 100:.2f}%"],
            "Pinnacle Probability (%)": [f"{pinnacle_prob_over * 100:.2f}%", f"{pinnacle_prob_under * 100:.2f}%"],
            "Combined Probability (%)": [f"{combined_prob_over * 100:.2f}%", f"{combined_prob_under * 100:.2f}%"],
            "Fair Odds": [fair_odds_over, fair_odds_under],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

        # Plot Histogram
        st.subheader("Simulated Throw-Ins Distribution")
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_throw_ins, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Simulated Throw-Ins')
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold} throw-ins")
        plt.axvline(np.mean(simulated_throw_ins), color='green', linestyle='-', label=f"Simulated Mean: {np.mean(simulated_throw_ins):.2f} throw-ins")
        plt.title("Simulated Throw-Ins Distribution")
        plt.xlabel("Total Throw-Ins")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)


if tool == "Team-Specific Throw-Ins":
    st.header("Team-Specific Throw-Ins Analysis with Monte Carlo Simulation")

    # Inputs
    team_name = st.text_input("Enter Team Name:", value="Team Name")
    avg_throw_ins = st.number_input(f"Average Throw-Ins Per Match for {team_name}:", min_value=0.0, value=15.0)
    opposition_possession = st.slider("Opposition Possession (%):", 0, 100, 55)
    tactics = st.selectbox("Team Tactics:", ["Defensive", "Balanced", "Offensive"])
    match_importance = st.selectbox("Match Importance:", ["Low", "Medium", "High"])
    weather_conditions = st.selectbox("Weather Conditions:", ["Good", "Rainy", "Windy"])
    threshold = st.number_input(f"Threshold for Throw-Ins for {team_name} (e.g., 14.5):", min_value=0.0, value=14.5)
    odds_over = st.number_input("Odds for Over Throw-Ins:", min_value=1.01, value=2.0)
    odds_under = st.number_input("Odds for Under Throw-Ins:", min_value=1.01, value=1.8)

    # Pinnacle Odds Input
    pinnacle_odds_over = st.number_input("Pinnacle Odds for Over Throw-Ins:", min_value=1.01, value=1.95)
    pinnacle_odds_under = st.number_input("Pinnacle Odds for Under Throw-Ins:", min_value=1.01, value=1.85)

    # Simulation settings (from Corners Market)
    simulations = st.number_input("Number of Monte Carlo Simulations:", min_value=1000, value=10000, step=1000)

    # Adjustment factors
    tactics_factor = {"Defensive": 0.9, "Balanced": 1.0, "Offensive": 1.1}[tactics]
    importance_factor = {"Low": 0.9, "Medium": 1.0, "High": 1.1}[match_importance]
    weather_factor = {"Good": 1.0, "Rainy": 1.1, "Windy": 1.2}[weather_conditions]

    if st.button(f"Analyze Throw-Ins Market for {team_name}"):
        # Adjusted average throw-ins based on factors
        adjusted_avg_throw_ins = avg_throw_ins * (opposition_possession / 100) * tactics_factor * importance_factor * weather_factor

        # Monte Carlo Simulation
        simulated_throw_ins = monte_carlo_simulation(adjusted_avg_throw_ins, simulations)
        monte_carlo_prob_over = np.sum(simulated_throw_ins > threshold) / len(simulated_throw_ins)
        monte_carlo_prob_under = 1 - monte_carlo_prob_over

        # Vig-adjusted probabilities
        vig_prob_over, vig_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        weighted_prob_over = calculate_weighted_probability_dynamic(monte_carlo_prob_over, vig_prob_over, 0.7)
        weighted_prob_under = 1 - weighted_prob_over

        # Pinnacle-adjusted probabilities
        pinnacle_prob_over = 1 / pinnacle_odds_over
        pinnacle_prob_under = 1 / pinnacle_odds_under
        combined_prob_over = (weighted_prob_over + pinnacle_prob_over) / 2
        combined_prob_under = 1 - combined_prob_over

        # Expected Value (EV)
        ev_over = calculate_ev(combined_prob_over, odds_over)
        ev_under = calculate_ev(combined_prob_under, odds_under)

        # Kelly Criterion
        bankroll = st.number_input("Enter your bankroll:", min_value=0.0, value=100.0)
        kelly_over = calculate_kelly(combined_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(combined_prob_under, odds_under, bankroll)

        # Fair Odds
        fair_odds_over = round(1 / combined_prob_over, 2) if combined_prob_over > 0 else "N/A"
        fair_odds_under = round(1 / combined_prob_under, 2) if combined_prob_under > 0 else "N/A"

        # Results Table
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Monte Carlo Probability (%)": [f"{monte_carlo_prob_over * 100:.2f}%", f"{monte_carlo_prob_under * 100:.2f}%"],
            "Weighted Probability (%)": [f"{weighted_prob_over * 100:.2f}%", f"{weighted_prob_under * 100:.2f}%"],
            "Pinnacle Probability (%)": [f"{pinnacle_prob_over * 100:.2f}%", f"{pinnacle_prob_under * 100:.2f}%"],
            "Combined Probability (%)": [f"{combined_prob_over * 100:.2f}%", f"{combined_prob_under * 100:.2f}%"],
            "Fair Odds": [fair_odds_over, fair_odds_under],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

        # Plot Histogram
        st.subheader(f"Simulated Throw-Ins Distribution for {team_name}")
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_throw_ins, bins=30, alpha=0.7, color='blue', edgecolor='black', label=f"{team_name} Simulated Throw-Ins")
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold} throw-ins")
        plt.axvline(np.mean(simulated_throw_ins), color='green', linestyle='-', label=f"Simulated Mean: {np.mean(simulated_throw_ins):.2f} throw-ins")
        plt.title(f"{team_name} Simulated Throw-Ins Distribution")
        plt.xlabel("Throw-Ins")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)

if tool == "Team-Specific Shots":
    st.header("Team-Specific Shots Analysis with Monte Carlo Simulation")

    # Inputs
    team_name = st.text_input("Enter Team Name:", value="Team Name")
    avg_shots = st.number_input(f"Average Shots Per Match for {team_name}:", min_value=0.0, value=15.0)
    opposition_defense = st.selectbox("Opposition Defensive Strength:", ["Weak", "Average", "Strong"])
    tactics = st.selectbox("Team Tactics:", ["Defensive", "Balanced", "Offensive"])
    possession = st.slider(f"{team_name} Possession (%):", 0, 100, 55)
    match_part = st.selectbox("Analyze for:", ["Full Match", "First Half", "Second Half"])

    threshold = st.number_input(f"Threshold for Shots for {team_name} (e.g., 14.5):", min_value=0.0, value=14.5)
    odds_over = st.number_input("Odds for Over Shots:", min_value=1.01, value=2.0)
    odds_under = st.number_input("Odds for Under Shots:", min_value=1.01, value=1.8)

    # Simulation settings
    simulations = st.number_input("Number of Monte Carlo Simulations:", min_value=1000, value=10000, step=1000)

    # Adjustment factors
    tactics_factor = {"Defensive": 0.9, "Balanced": 1.0, "Offensive": 1.1}[tactics]
    defense_factor = {"Weak": 1.2, "Average": 1.0, "Strong": 0.8}[opposition_defense]

    if st.button(f"Analyze Shots Market for {team_name}"):
        # Adjusted average shots based on factors
        adjusted_avg_shots = avg_shots * (possession / 100) * tactics_factor * defense_factor

        # Adjust for match part
        if match_part == "First Half":
            adjusted_avg_shots *= 0.45  # Assume 45% of total shots occur in the first half
        elif match_part == "Second Half":
            adjusted_avg_shots *= 0.55  # Assume 55% of total shots occur in the second half

        # Monte Carlo Simulation
        simulated_shots = monte_carlo_simulation(adjusted_avg_shots, simulations)
        monte_carlo_prob_over = np.sum(simulated_shots > threshold) / len(simulated_shots)
        monte_carlo_prob_under = 1 - monte_carlo_prob_over

        # Vig-adjusted probabilities
        vig_prob_over, vig_prob_under = calculate_vig_adjusted_probabilities(odds_over, odds_under)
        weighted_prob_over = calculate_weighted_probability_dynamic(monte_carlo_prob_over, vig_prob_over, 0.7)
        weighted_prob_under = 1 - weighted_prob_over

        # Expected Value (EV)
        ev_over = calculate_ev(weighted_prob_over, odds_over)
        ev_under = calculate_ev(weighted_prob_under, odds_under)

        # Kelly Criterion
        bankroll = st.number_input("Enter your bankroll:", min_value=0.0, value=100.0)
        kelly_over = calculate_kelly(weighted_prob_over, odds_over, bankroll)
        kelly_under = calculate_kelly(weighted_prob_under, odds_under, bankroll)

        # Fair Odds
        fair_odds_over = round(1 / weighted_prob_over, 2) if weighted_prob_over > 0 else "N/A"
        fair_odds_under = round(1 / weighted_prob_under, 2) if weighted_prob_under > 0 else "N/A"

        # Results Table
        results = pd.DataFrame({
            "Bet Option": ["Over", "Under"],
            "Monte Carlo Probability (%)": [f"{monte_carlo_prob_over * 100:.2f}%", f"{monte_carlo_prob_under * 100:.2f}%"],
            "Weighted Probability (%)": [f"{weighted_prob_over * 100:.2f}%", f"{weighted_prob_under * 100:.2f}%"],
            "Fair Odds": [fair_odds_over, fair_odds_under],
            "Odds": [odds_over, odds_under],
            "EV (%)": [ev_over, ev_under],
            "Recommended Bet": [kelly_over, kelly_under]
        })
        st.table(results)

        # Plot Histogram
        st.subheader(f"Simulated Shots Distribution for {team_name} ({match_part})")
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_shots, bins=30, alpha=0.7, color='blue', edgecolor='black', label=f"{team_name} Simulated Shots")
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold} shots")
        plt.axvline(np.mean(simulated_shots), color='green', linestyle='-', label=f"Simulated Mean: {np.mean(simulated_shots):.2f} shots")
        plt.title(f"{team_name} Simulated Shots Distribution ({match_part})")
        plt.xlabel("Shots")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)
