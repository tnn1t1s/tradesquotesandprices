import numpy as np
from scipy import stats

def generate_mm_returns_with_informed(n_steps=10000, base_vol=0.01, phi=0.1, J=0.02):
    """
    Generate market maker returns including informed trading effects
    
    Parameters:
    -----------
    n_steps : int
        Number of steps to simulate
    base_vol : float
        Base volatility (random walk component)
    phi : float
        Probability of informed trading (0 to 1)
    J : float
        Size of price jump when informed trading occurs
        
    Returns:
    --------
    numpy.array
        Array of returns with length n_steps
    """
    # Base volatility with some persistence
    vol = base_vol * np.ones(n_steps)
    for i in range(1, n_steps):
        vol[i] = 0.95 * vol[i-1] + 0.05 * base_vol * (1 + 0.5 * np.random.randn())
    
    # Generate base returns (random walk component)
    returns = np.random.randn(n_steps) * vol
    
    # Add informed trading jumps
    informed_trades = np.random.random(n_steps) < phi
    returns[informed_trades] -= J  # Informed traders cause adverse moves of size J
    
    return returns

def calculate_return_distribution_stats(returns):
    """
    Calculate key distribution statistics for a series of returns
    
    Parameters:
    -----------
    returns : numpy.array
        Array of return values
        
    Returns:
    --------
    dict
        Dictionary containing mean, std, skewness, kurtosis
    """
    return {
        'mean': np.mean(returns),
        'std': np.std(returns),
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns)
    }
