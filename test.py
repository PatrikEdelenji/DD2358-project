import pstats

# Load the .stats file
stats = pstats.Stats("profile.stats")

# Print the top 30 slowest functions
stats.strip_dirs().sort_stats("cumulative").print_stats(500)
