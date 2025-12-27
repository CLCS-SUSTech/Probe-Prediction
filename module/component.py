
class ComponentRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name=None):
        def decorator(cls):
            key = name or cls.__name__.lower()
            if key in self._registry:
                raise ValueError(f"Component '{key}' already registered.")
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, name):
        try:
            return self._registry[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown component: {name}")

    def list(self):
        return list(self._registry.keys())

registry = ComponentRegistry()
register_component = registry.register
def _load_component_cls():
    import importlib
    import pkgutil
    import module.components
    package = module.components
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{module_name}")

_load_component_cls()

def get_component_cls(name):
    return registry.get(name)