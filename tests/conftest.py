"""Shared test config: speed up PBKDF2 so the auth-heavy suites stay fast."""
import os

# Must be set before `bhairav.backend.users` is imported anywhere.
os.environ.setdefault("BHAIRAV_PBKDF2_ITERATIONS", "2000")
