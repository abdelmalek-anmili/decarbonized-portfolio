import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

# --- 1. Data Loading and Cleaning ---
def load_and_clean_data(filepath):
    """
    Parses the specific layout of the provided Excel file to extract
    returns, deviations, and the correlation matrix.
    """
    # Load the specific Excel sheet
    try:
        # We read the whole file first to handle the split tables
        df_raw = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None, None, None

    # -- Extract Summary Statistics (Mean & Std Dev) --
    # Based on the CSV structure, data starts around row 3/4
    # We look for the row containing 'WEBS' to identify the start
    stats_start_row = df_raw[df_raw.iloc[:, 0] == 'WEBS'].index[0]
    
    # Extract the 8 assets (rows below 'WEBS')
    # Columns: Asset Name (0), Mean Return (1), Std Dev (2)
    stats_df = df_raw.iloc[stats_start_row+1 : stats_start_row+9, 0:3]
    stats_df.columns = ['Asset', 'Mean_Return', 'Std_Dev']
    stats_df = stats_df.set_index('Asset')
    
    # Convert to numeric
    stats_df['Mean_Return'] = pd.to_numeric(stats_df['Mean_Return'])
    stats_df['Std_Dev'] = pd.to_numeric(stats_df['Std_Dev'])

    # -- Extract Correlation Matrix --
    # Find where the correlation matrix starts
    corr_start_row = df_raw[df_raw.iloc[:, 1] == 'Correlation Matrix'].index[0]
    
    # The matrix headers are usually 1 row below the title
    corr_matrix = df_raw.iloc[corr_start_row+2 : corr_start_row+10, 1:9]
    
    # Set index and columns to asset names to ensure alignment
    assets = stats_df.index.tolist()
    corr_matrix.columns = assets
    corr_matrix.index = assets
    corr_matrix = corr_matrix.apply(pd.to_numeric)

    return stats_df, corr_matrix, assets

# --- 2. Financial Math Functions ---
def get_portfolio_metrics(weights, mean_returns, cov_matrix):
    """
    Calculates portfolio return and volatility (standard deviation).
    """
    weights = np.array(weights)
    ret = np.sum(weights * mean_returns)
    # Variance = w.T * Cov * w
    var = np.dot(weights.T, np.dot(cov_matrix, weights))
    vol = np.sqrt(var)
    return ret, vol

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Returns the negative Sharpe Ratio (for minimization).
    """
    p_ret, p_vol = get_portfolio_metrics(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol

def minimize_volatility(weights, mean_returns, cov_matrix):
    """
    Auxiliary function to minimize volatility.
    """
    return get_portfolio_metrics(weights, mean_returns, cov_matrix)[1]

# --- 3. Main Execution ---
if __name__ == "__main__":
    # Settings
    FILE_PATH = 'Min_variance_frontier_input.xlsx' # Ensure this matches your file name
    RISK_FREE_RATE = 2.5  # As per source [7]
    
    # Load Data
    stats, corr_matrix, assets = load_and_clean_data(FILE_PATH)
    
    if stats is not None:
        # Calculate Covariance Matrix: Cov_ij = rho_ij * sigma_i * sigma_j
        # We use outer product of std_devs * correlation matrix
        std_devs = stats['Std_Dev'].values
        means = stats['Mean_Return'].values
        cov_matrix = np.outer(std_devs, std_devs) * corr_matrix.values
        
        num_assets = len(assets)

        # ==========================================
        # Part 1 & 2: Minimum Variance Frontiers
        # ==========================================
        
        # We generate frontiers by finding the min vol for a range of target returns
        target_returns = np.linspace(means.min(), means.max(), 50)
        
        # Storage for frontiers
        constrained_vols = []
        unconstrained_vols = []

        # Initial guess (equal weights)
        init_guess = num_assets * [1. / num_assets,]

        for target in target_returns:
            # Constraints common to both: Portfolio Return = Target
            constraints_base = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: get_portfolio_metrics(x, means, cov_matrix)[0] - target} # Return = target
            )

            # --- A. Constrained (0 <= w <= 1) [Source: 4] ---
            bounds_constrained = tuple((0, 1) for _ in range(num_assets))
            
            result_c = sco.minimize(minimize_volatility, init_guess, 
                                   args=(means, cov_matrix), method='SLSQP', 
                                   bounds=bounds_constrained, constraints=constraints_base)
            constrained_vols.append(result_c.fun)

            # --- B. Unconstrained (Short selling allowed) [Source: 5] ---
            # Bounds are effectively infinite
            bounds_unconstrained = tuple((-np.inf, np.inf) for _ in range(num_assets))
            
            result_u = sco.minimize(minimize_volatility, init_guess, 
                                   args=(means, cov_matrix), method='SLSQP', 
                                   bounds=bounds_unconstrained, constraints=constraints_base)
            unconstrained_vols.append(result_u.fun)

        # ==========================================
        # Part 3: Optimal Risky Portfolio (Tangency)
        # ==========================================
        # [Source: 6, 7]
        
        # Objective: Maximize Sharpe (Minimize Negative Sharpe)
        # Constraint: Sum of weights = 1 (Short selling allowed for optimal in this context usually, 
        # but typically Tangency assumes valid weights. We will assume unconstrained for generality 
        # unless specified, but standard Markowitz often allows shorts. Let's use unconstrained).
        
        args = (means, cov_matrix, RISK_FREE_RATE)
        constraints_sharpe = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds_sharpe = tuple((-np.inf, np.inf) for _ in range(num_assets)) # Unconstrained

        result_sharpe = sco.minimize(neg_sharpe_ratio, init_guess, args=args,
                                     method='SLSQP', bounds=bounds_sharpe,
                                     constraints=constraints_sharpe)
        
        optimal_weights = result_sharpe.x
        opt_ret, opt_vol = get_portfolio_metrics(optimal_weights, means, cov_matrix)
        
        print("\nOptimal Risky Portfolio (Weights):")
        for asset, weight in zip(assets, optimal_weights):
            print(f"{asset}: {weight:.4f}")
        print(f"Optimal Return: {opt_ret:.2f}%")
        print(f"Optimal Volatility: {opt_vol:.2f}%")
        print(f"Max Sharpe Ratio: {(opt_ret - RISK_FREE_RATE)/opt_vol:.4f}")

        # ==========================================
        # Plotting
        # ==========================================
        plt.figure(figsize=(10, 6))

        # 1. Plot Unconstrained Frontier (Dashed)
        plt.plot(unconstrained_vols, target_returns, 'r--', label='Unconstrained Frontier')

        # 2. Plot Constrained Frontier (Solid)
        plt.plot(constrained_vols, target_returns, 'b-', linewidth=2, label='Constrained Frontier (0-1)')

        # 3. Plot Individual Assets
        plt.scatter(stats['Std_Dev'], stats['Mean_Return'], marker='o', s=50, color='black', label='Individual Assets')
        for i, txt in enumerate(assets):
            plt.annotate(txt, (stats['Std_Dev'].iloc[i]+0.5, stats['Mean_Return'].iloc[i]))

        # 4. Plot Optimal Risky Portfolio
        plt.scatter(opt_vol, opt_ret, marker='*', s=300, color='gold', edgecolors='black', label='Optimal Risky Portfolio')

        # 5. Plot Capital Allocation Line (CAL)
        # Line from Risk Free Rate passing through Optimal Portfolio
        cal_x = np.linspace(0, 50, 100)
        cal_y = RISK_FREE_RATE + ((opt_ret - RISK_FREE_RATE) / opt_vol) * cal_x
        plt.plot(cal_x, cal_y, 'g:', label='Capital Allocation Line (CAL)')

        plt.title('Efficient Frontier & Optimal Risky Portfolio')
        plt.xlabel('Volatility (Standard Deviation %)')
        plt.ylabel('Expected Return (%)')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 50)
        plt.ylim(0, 30)
        
        # Save or Show
        plt.savefig('frontier_plot.png')
        print("\nPlot saved as 'frontier_plot.png'")
        plt.show()