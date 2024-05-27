from typing import Callable, Dict, Optional, Type, TypeVar, List

from utils.import_utils import import_modules_from_folder

RegistryItem = TypeVar("RegistryItem", bound=Callable)


# Inspired by https://github.com/apple/corenet/blob/main/corenet/utils/registry.py
class Registry:
    def __init__(
            self,
            registry_name: str,
            base_class: Optional[Type] = None,
            lazy_load_dirs: Optional[List[str]] = None
    ) -> None:
        """
        Args:
            registry_name: Name der Registry, verwendet für Debugging und Fehlermeldungen.
            base_class: Wenn angegeben, wird sichergestellt, dass alle Elemente in der Registry
                        vom Typ `base_class` sind.
        """
        self.registry_name = registry_name
        self.base_class = base_class
        self.registry: Dict[str, RegistryItem] = {}
        self.flags: Dict[str, Dict[str, bool]] = {}
        self.default_class: Optional[RegistryItem] = None
        self._modules_loaded = False
        self._lazy_load_dirs = lazy_load_dirs or []

    def _load_all(self) -> None:
        if not self._modules_loaded:
            self._modules_loaded = True
            for dir_name in sorted(self._lazy_load_dirs):
                import_modules_from_folder(dir_name)

    def register(
            self,
            name: str,
            is_default: bool = False,
            flags: Dict[str, bool] = None
    ) -> Callable[[RegistryItem], RegistryItem]:
        """Registriert eine Klasse oder Funktion unter dem gegebenen Namen."""
        def register_with_name(item: RegistryItem) -> RegistryItem:
            if name in self.registry:
                raise ValueError(
                    f"Kann kein doppeltes Element in der {self.registry_name} Registry registrieren: {name}")
            if self.base_class and isinstance(item, type):
                if not issubclass(item, self.base_class):
                    raise ValueError(
                        f"{self.registry_name} Klasse ({name}: {item.__name__}) muss {self.base_class} erweitern")
            self.registry[name] = item
            if is_default:
                if self.default_class is not None:
                    raise ValueError(f"A default class was already set for {self.registry_name}.")
                self.default_class = item
            if flags:
                self.flags[name] = flags
            return item
        return register_with_name

    def get(self, name: str) -> RegistryItem:
        self._load_all()
        """Gibt das registrierte Element unter dem gegebenen Namen zurück."""
        if name not in self.registry:
            raise KeyError(f"{name} ist nicht in der {self.registry_name} Registry registriert")
        return self.registry[name]

    def get_default(self) -> RegistryItem:
        self._load_all()
        if not self.default_class:
            raise ValueError(f"No default class set for the {self.registry_name} registry")
        return self.default_class

    def __contains__(self, name: str) -> bool:
        self._load_all()
        return name in self.registry

    def __iter__(self):
        self._load_all()
        return iter(self.registry)

    def items(self):
        self._load_all()

        return self.registry.items()

    def keys(self) -> List[str]:
        self._load_all()
        return list(self.registry.keys())

    def values(self) -> List[RegistryItem]:
        self._load_all()
        return list(self.registry.values())

    def get_all_with_flag(self, flag: str) -> List[RegistryItem]:
        self._load_all()
        keys_with_flag = [k for k, v in self.flags.items() if v[flag] is True]
        return [self.registry[key] for key in keys_with_flag]

    def get_single(self, name: str) -> RegistryItem:
        self._load_all()
        if name == "default":
            return self.get_default()
        else:
            return self.get(name)

    def get_list(self, group: str | List[str]) -> List[RegistryItem]:
        self._load_all()
        if group == "all":
            return self.values()
        elif "all-" in group:  # e.g. all-required
            flag = group.split("-")[1]
            return self.get_all_with_flag(flag)
        elif group == "default" or isinstance(group, str):
            return [self.get_single(group)]
        elif isinstance(group, List):
            return [self.get_single(name) for name in group]
        else:
            raise ValueError("Function get_multiple is not defined for this type.")
