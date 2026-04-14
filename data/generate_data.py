"""
Generate realistic synthetic IPL dataset for training the DL model.
Comprehensive real IPL 2025 Team Squads.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

teams = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans"
]

venues = [
    "Wankhede Stadium", "MA Chidambaram Stadium", "M. Chinnaswamy Stadium",
    "Eden Gardens", "Arun Jaitley Stadium", "Rajiv Gandhi Intl. Cricket Stadium",
    "Sawai Mansingh Stadium", "PCA Stadium Mohali", "Ekana Cricket Stadium", 
    "Narendra Modi Stadium", "HPCA Stadium Dharamsala", "Barsapara Stadium Guwahati"
]

# Comprehensive ~2024/2025 IPL Squads (Major Batsmen & Bowlers per franchise)
squads = {
    "Mumbai Indians": {
        "batsmen": ["Rohit Sharma", "Suryakumar Yadav", "Ishan Kishan", "Hardik Pandya", "Tilak Varma", "Tim David", "Nehal Wadhera", "Dewald Brevis", "Romario Shepherd"],
        "bowlers": ["Jasprit Bumrah", "Gerald Coetzee", "Akash Madhwal", "Piyush Chawla", "Nuwan Thushara", "Dilshan Madushanka", "Shreyas Gopal", "Luke Wood"]
    },
    "Chennai Super Kings": {
        "batsmen": ["MS Dhoni", "Ruturaj Gaikwad", "Shivam Dube", "Devon Conway", "Ajinkya Rahane", "Daryl Mitchell", "Sameer Rizvi", "Moeen Ali"],
        "bowlers": ["Ravindra Jadeja", "Matheesha Pathirana", "Deepak Chahar", "Mustafizur Rahman", "Tushar Deshpande", "Maheesh Theekshana", "Shardul Thakur", "Mukesh Choudhary"]
    },
    "Royal Challengers Bangalore": {
        "batsmen": ["Virat Kohli", "Faf du Plessis", "Glenn Maxwell", "Rajat Patidar", "Cameron Green", "Dinesh Karthik", "Mahipal Lomror", "Will Jacks", "Anuj Rawat"],
        "bowlers": ["Mohammed Siraj", "Lockie Ferguson", "Reece Topley", "Yash Dayal", "Karn Sharma", "Alzarri Joseph", "Vyshak Vijaykumar", "Akash Deep"]
    },
    "Kolkata Knight Riders": {
        "batsmen": ["Shreyas Iyer", "Rinku Singh", "Andre Russell", "Venkatesh Iyer", "Phil Salt", "Nitish Rana", "Angkrish Raghuvanshi", "Manish Pandey"],
        "bowlers": ["Sunil Narine", "Mitchell Starc", "Varun Chakaravarthy", "Harshit Rana", "Vaibhav Arora", "Chetan Sakariya", "Mujeeb Ur Rahman"]
    },
    "Delhi Capitals": {
        "batsmen": ["Rishabh Pant", "David Warner", "Prithvi Shaw", "Mitchell Marsh", "Tristan Stubbs", "Jake Fraser-McGurk", "Abishek Porel", "Harry Brook", "Yash Dhull"],
        "bowlers": ["Kuldeep Yadav", "Anrich Nortje", "Axar Patel", "Khaleel Ahmed", "Mukesh Kumar", "Ishant Sharma", "Jhye Richardson", "Lalit Yadav"]
    },
    "Sunrisers Hyderabad": {
        "batsmen": ["Heinrich Klaasen", "Travis Head", "Abhishek Sharma", "Aiden Markram", "Nitish Kumar Reddy", "Rahul Tripathi", "Abdul Samad", "Glenn Phillips", "Mayank Agarwal"],
        "bowlers": ["Bhuvneshwar Kumar", "Pat Cummins", "T Natarajan", "Mayank Markande", "Jaydev Unadkat", "Marco Jansen", "Washington Sundar", "Umran Malik"]
    },
    "Rajasthan Royals": {
        "batsmen": ["Sanju Samson", "Jos Buttler", "Yashasvi Jaiswal", "Shimron Hetmyer", "Riyan Parag", "Dhruv Jurel", "Rovman Powell", "Shubham Dubey"],
        "bowlers": ["Yuzvendra Chahal", "Trent Boult", "R Ashwin", "Sandeep Sharma", "Avesh Khan", "Nandre Burger", "Kuldeep Sen", "Navdeep Saini"]
    },
    "Punjab Kings": {
        "batsmen": ["Shikhar Dhawan", "Jonny Bairstow", "Liam Livingstone", "Sam Curran", "Prabhsimran Singh", "Jitesh Sharma", "Shashank Singh", "Ashutosh Sharma", "Rilee Rossouw"],
        "bowlers": ["Kagiso Rabada", "Arshdeep Singh", "Harshal Patel", "Rahul Chahar", "Nathan Ellis", "Kagiso Rabada", "Vidwath Kaverappa", "Harpreet Brar"]
    },
    "Lucknow Super Giants": {
        "batsmen": ["KL Rahul", "Quinton de Kock", "Nicholas Pooran", "Marcus Stoinis", "Ayush Badoni", "Deepak Hooda", "Kyle Mayers", "Devdutt Padikkal", "Krunal Pandya"],
        "bowlers": ["Ravi Bishnoi", "Naveen-ul-Haq", "Mohsin Khan", "Mayank Yadav", "Yash Thakur", "Amit Mishra", "Shamar Joseph", "Matt Henry"]
    },
    "Gujarat Titans": {
        "batsmen": ["Shubman Gill", "Sai Sudharsan", "David Miller", "Rahul Tewatia", "Wriddhiman Saha", "Kane Williamson", "Shahrukh Khan", "Vijay Shankar", "Abhinav Manohar"],
        "bowlers": ["Rashid Khan", "Mohammed Shami", "Mohit Sharma", "Spencer Johnson", "Noor Ahmad", "Umesh Yadav", "Darshan Nalkande", "R Sai Kishore"]
    }
}

NUM_RECORDS = 25000  # Increased for massive dataset width

def simulate_record():
    batting_team = np.random.choice(teams)
    bowling_team = np.random.choice([t for t in teams if t != batting_team])
    venue = np.random.choice(venues)
    
    batsman = np.random.choice(squads[batting_team]["batsmen"])
    bowler = np.random.choice(squads[bowling_team]["bowlers"])

    over = np.random.randint(1, 21)
    ball = np.random.randint(1, 7)

    base_rate = np.random.uniform(7.0, 10.5)
    current_score = int((over - 1) * base_rate + (ball * (base_rate/6)) + np.random.normal(0, 10))
    current_score = max(0, current_score)

    wickets = min(int(np.random.poisson(over * 0.35)), 10)

    runs_last_5 = int(np.random.normal(base_rate * 5, 12))
    runs_last_5 = max(0, min(runs_last_5, 75))

    wickets_last_5 = min(int(np.random.poisson(1.0)), 10 - wickets) if wickets < 10 else 0

    # Dynamic Player Impact for highly rated T20 stats in 2024/2025
    impact = 0
    stars_bat = ["Virat Kohli", "Jos Buttler", "Suryakumar Yadav", "Heinrich Klaasen", "Travis Head", "Phil Salt", "Nicholas Pooran", "Riyan Parag"]
    stars_bowl = ["Jasprit Bumrah", "Rashid Khan", "Sunil Narine", "Matheesha Pathirana", "Trent Boult"]
    
    if batsman in stars_bat: impact += np.random.randint(8, 25)
    if bowler in stars_bowl: impact -= np.random.randint(5, 18)

    projected_rr = current_score / max(over, 1)
    remaining_overs = 20 - (over - 1 + ball/6)
    
    final_score = int(current_score + (projected_rr * remaining_overs) + np.random.normal(impact, 15))
    final_score = max(current_score + 1, min(final_score, 290)) # Allowing extreme 2024 scores

    return {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "batsman": batsman,
        "bowler": bowler,
        "over": over,
        "ball": ball,
        "current_score": current_score,
        "wickets": wickets,
        "runs_last_5_overs": runs_last_5,
        "wickets_last_5_overs": wickets_last_5,
        "final_score": final_score
    }

records = [simulate_record() for _ in range(NUM_RECORDS)]
df = pd.DataFrame(records)

os.makedirs("data", exist_ok=True)
df.to_csv("data/ipl_data.csv", index=False)
print(f"Dataset generated: {len(df)} records covering comprehensive ~2025 IPL squads.")
