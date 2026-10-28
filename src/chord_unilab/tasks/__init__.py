"""UniLab task registrations for the CHORD reproduction.

Populated as task terms land; the registry bootstrap reads
``__unilab_registry_modules__`` (see ``unilab.base.registry``).
"""

__unilab_registry_modules__: tuple[str, ...] = ("chord_unilab.tasks.sharpa_chord",)
