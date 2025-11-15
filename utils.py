"""
Some utility functions for the project
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_agent_save_path(agent_name):
    return os.path.join(ROOT_DIR, 'agents', agent_name)