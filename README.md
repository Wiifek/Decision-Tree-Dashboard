# Nike Strategic Decision Tree Solver

This project implements an advanced analytical dashboard to evaluate Nike's global strategic options. Utilizing Backward Induction and Expected Utility Theory, the tool assists in determining whether Nike should maintain its current trajectory or pivot its strategy in response to geopolitical tensions, market demand, and competitive pressures.

## Usage
Run the dashboard using the following command:
```bash
streamlit run dashboard.py
```

## Strategic Context

The model evaluates high-stakes decisions including:

- **Market Expansion:** Evaluating a reinforced positioning in the Chinese market versus diversification.

- **Brand Management:** Analyzing local rebranding efforts vs. strategic partnerships with local brands.

- **Financial Tactics:** Comparing aggressive vs. conservative margin adjustments in response to raw material price fluctuations.

- **Operational Shifts:** Assessing marketing investment levels and supply chain stock reductions.

## Features
- **Multi-Scenario Analysis:** 26 unique pure strategies (δ1 to δ26) mapped across Nike's decision nodes (D0 to D6).
- **Risk-Adjusted Modeling:** 
  - **EU Linear:** Balanced expectation for stable market conditions.
  - **EU Concave:** A risk-averse approach, ideal for protecting Nike's market share during economic volatility.
  - **EU Convex:** A risk-seeking approach for aggressive growth in emerging sectors.

- **Robustness Scoring:** Identifies which strategic path remains optimal across the highest number of criteria.


## Analytical Components

1. **Decision Tree Tab:** Explore the structural hierarchy and the "Optimal DP Value" calculated via backward induction.

2. **Strategies Tab:** View a detailed matrix of choices for each strategy and their corresponding evaluation table.

3. **Dashboard Tab:** High-level visual summary of performance, including most robust strategy and utility distributions.
