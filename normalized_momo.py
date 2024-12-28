import pandas as pd
import numpy as np

def normalize_and_rank_performances(file_path='daily_data/performance_rankings/solana_trending_performance.parquet'):
    # Read the parquet file
    df = pd.read_parquet(file_path)

    # Get the latest data for each pool
    latest_df = df.sort_values('timestamp').groupby('pool_address').last().reset_index()

    # Function to normalize a series to 0-1 range
    def normalize_series(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return series * 0  # Return zeros if all values are the same
        return (series - min_val) / (max_val - min_val)

    # Normalize each performance metric
    latest_df['norm_1d'] = normalize_series(latest_df['performance_1d'])
    latest_df['norm_3d'] = normalize_series(latest_df['performance_3d'])
    latest_df['norm_7d'] = normalize_series(latest_df['performance_7d'])

    # Calculate average normalized performance
    latest_df['avg_norm_performance'] = (
        latest_df['norm_1d'] +
        latest_df['norm_3d'] +
        latest_df['norm_7d']
    ) / 3

    # Sort by average normalized performance
    ranked_df = latest_df.sort_values('avg_norm_performance', ascending=False)

    # Create readable output
    results = ranked_df[[
        'pool_address',
        'avg_norm_performance',
        'performance_1d',
        'performance_3d',
        'performance_7d',
        'volume_1d',
        'volume_3d',
        'volume_7d'
    ]].head(20)

    # Print results
    print("\nTop 20 Pools by Average Normalized Performance:")
    print("==============================================")
    for idx, row in results.iterrows():
        print(f"\nRank {idx + 1}:")
        print(f"Pool Address: {row['pool_address']}")
        print(f"Avg Normalized Performance: {row['avg_norm_performance']:.4f}")
        print(f"1d Performance: {row['performance_1d']:.2f}%")
        print(f"3d Performance: {row['performance_3d']:.2f}%")
        print(f"7d Performance: {row['performance_7d']:.2f}%")
        print(f"1d Volume: {row['volume_1d']:.2f}")
        print(f"3d Volume: {row['volume_3d']:.2f}")
        print(f"7d Volume: {row['volume_7d']:.2f}")

    return ranked_df

if __name__ == "__main__":
    ranked_df = normalize_and_rank_performances()
