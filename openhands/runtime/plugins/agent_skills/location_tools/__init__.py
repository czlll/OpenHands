from dataclasses import dataclass

from plugins.agent_skills.location_tools import locationtools
from plugins.requirement import Plugin, PluginRequirement


@dataclass
class LocationToolsRequirement(PluginRequirement):
    name: str = 'location_tools'
    documentation: str = locationtools.DOCUMENTATION


class LocationToolsPlugin(Plugin):
    name: str = 'location_tools'
