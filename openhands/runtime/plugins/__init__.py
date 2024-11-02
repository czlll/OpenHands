# Requirements
from openhands.runtime.plugins.agent_skills import (
    AgentSkillsPlugin,
    AgentSkillsRequirement,
)
from openhands.runtime.plugins.jupyter import JupyterPlugin, JupyterRequirement
# from openhands.runtime.plugins.location_tools import (
#     LocationToolsPlugin,
#     LocationToolsRequirement,
# )
from openhands.runtime.plugins.requirement import Plugin, PluginRequirement

__all__ = [
    'Plugin',
    'PluginRequirement',
    'AgentSkillsRequirement',
    'AgentSkillsPlugin',
    'JupyterRequirement',
    'JupyterPlugin',
    'LocationToolsPlugin',
    'LocationToolsRequirement',
]

ALL_PLUGINS = {
    'jupyter': JupyterPlugin,
    'agent_skills': AgentSkillsPlugin
}
