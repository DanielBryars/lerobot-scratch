"""
Import this module to register SO100 STS3250 variants with lerobot.

Usage in Python:
    import sts3250_plugin  # Registers the classes

Or add to your scripts before using lerobot:
    import sts3250_plugin
"""

# Import to trigger @register_subclass decorators
from SO100FollowerSTS3250 import SO100FollowerSTS3250, SO100FollowerSTS3250Config
from SO100LeaderSTS3250 import SO100LeaderSTS3250, SO100LeaderSTS3250Config

# Make classes available at module level
__all__ = [
    "SO100FollowerSTS3250",
    "SO100FollowerSTS3250Config",
    "SO100LeaderSTS3250",
    "SO100LeaderSTS3250Config",
]

print("Registered: so100_follower_sts3250, so100_leader_sts3250")
