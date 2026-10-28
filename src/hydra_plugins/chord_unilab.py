"""Discover CHORD_UniLab owner configurations through Hydra's plugin API."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class ChordSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append("chord-unilab", "pkg://chord_unilab.conf.flashsac")
