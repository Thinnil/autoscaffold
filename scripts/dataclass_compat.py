"""
scripts/dataclass_compat.py
Zero-Dependency Dataclass Compatibility Layer for AutoScaffold.

Provides seamless backward compatibility for Python 3.6 (where the native
dataclasses module from PEP 557 is absent) and transparently re-exports the
native dataclasses module on Python >= 3.7.

Strictly adheres to standard library only (zero third-party dependencies).
"""

import collections
import copy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

_ATOMIC_TYPES = (int, float, str, bool, bytes, type(None), range, slice, complex)


# ==============================================================================
# 1. Fallback Dataclass Implementation (Pure Python 3.6+ Standard Library)
# ==============================================================================


class _MissingType:
    """Sentinel representation for missing default values and factories."""

    def __repr__(self) -> str:
        return "<MISSING>"


_FALLBACK_MISSING = _MissingType()


class _FallbackFrozenInstanceError(AttributeError):
    """Raised when attempting to modify a frozen dataclass instance."""

    pass


class _FallbackField:
    """
    Representation of a dataclass field descriptor.
    Stores metadata, defaults, and generation flags matching
    stdlib dataclasses.Field.
    Optimized with __slots__ to minimize memory footprint.
    """

    __slots__ = (
        "name",
        "type",
        "default",
        "default_factory",
        "init",
        "repr",
        "hash",
        "compare",
        "metadata",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        type: Any = None,
        default: Any = _FALLBACK_MISSING,
        default_factory: Any = _FALLBACK_MISSING,
        init: bool = True,
        repr: bool = True,
        hash: Optional[bool] = None,
        compare: bool = True,
        metadata: Optional[Dict[Any, Any]] = None,
    ):
        self.name = name
        self.type = type
        self.default = default
        self.default_factory = default_factory
        self.init = init
        self.repr = repr
        self.hash = hash
        self.compare = compare
        self.metadata = metadata if metadata is not None else {}

    def __copy__(self) -> "_FallbackField":
        return _FallbackField(
            name=self.name,
            type=self.type,
            default=self.default,
            default_factory=self.default_factory,
            init=self.init,
            repr=self.repr,
            hash=self.hash,
            compare=self.compare,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        return (
            f"Field(name={self.name!r},type={self.type!r},"
            f"default={self.default!r},"
            f"default_factory={self.default_factory!r},"
            f"init={self.init!r},repr={self.repr!r},hash={self.hash!r},"
            f"compare={self.compare!r},metadata={self.metadata!r})"
        )


def _fallback_field(
    *,
    default: Any = _FALLBACK_MISSING,
    default_factory: Any = _FALLBACK_MISSING,
    init: bool = True,
    repr: bool = True,
    hash: Optional[bool] = None,
    compare: bool = True,
    metadata: Optional[Dict[Any, Any]] = None,
) -> _FallbackField:
    """Creates a Field descriptor for default values or default_factories."""
    if (
        default is not _FALLBACK_MISSING
        and default_factory is not _FALLBACK_MISSING
    ):
        raise ValueError("cannot specify both default and default_factory")
    return _FallbackField(
        name=None,
        type=None,
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )


def _fallback_is_dataclass(obj_or_cls: Any) -> bool:
    """Returns True if the object or class is a dataclass."""
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    return hasattr(cls, "__dataclass_fields__")


def _fallback_asdict_inner(
    obj: Any, dict_factory: Callable[[List[Tuple[Any, Any]]], Any]
) -> Any:
    obj_type = type(obj)
    # 1. Fast-path: Primitives and immutable atomic types require zero copying
    if obj_type in _ATOMIC_TYPES:
        return obj

    # 2. Dataclass instances
    if _fallback_is_dataclass(obj):
        fields = getattr(obj_type, "__dataclass_fields__", {})
        if dict_factory is dict:
            return {
                name: _fallback_asdict_inner(getattr(obj, name), dict_factory)
                for name in fields
            }
        result = [
            (name, _fallback_asdict_inner(getattr(obj, name), dict_factory))
            for name in fields
        ]
        return dict_factory(result)

    # 3. Namedtuples
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return obj_type(
            *[_fallback_asdict_inner(v, dict_factory) for v in obj]
        )

    # 4. Standard sequences: list and tuple
    if obj_type is list:
        return [_fallback_asdict_inner(v, dict_factory) for v in obj]
    if obj_type is tuple:
        return tuple(_fallback_asdict_inner(v, dict_factory) for v in obj)
    if isinstance(obj, (list, tuple)):
        return obj_type(_fallback_asdict_inner(v, dict_factory) for v in obj)

    # 5. Dictionaries
    if obj_type is dict:
        return {
            _fallback_asdict_inner(k, dict_factory): _fallback_asdict_inner(
                v, dict_factory
            )
            for k, v in obj.items()
        }
    if isinstance(obj, dict):
        return obj_type(
            (
                _fallback_asdict_inner(k, dict_factory),
                _fallback_asdict_inner(v, dict_factory),
            )
            for k, v in obj.items()
        )

    # 6. Arrays / custom types with fast .copy() method (e.g. numpy ndarray)
    if hasattr(obj, "copy") and callable(getattr(obj, "copy")):
        try:
            return obj.copy()
        except Exception:
            return copy.deepcopy(obj)

    # 7. Fallback deepcopy for general arbitrary mutable types
    return copy.deepcopy(obj)


def _fallback_asdict(
    obj: Any, *, dict_factory: Callable[[List[Tuple[Any, Any]]], Any] = dict
) -> Any:
    """Converts a dataclass instance recursively to a dictionary."""
    if not _fallback_is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _fallback_asdict_inner(obj, dict_factory)


def _process_fallback_class(
    cls: Type[Any],
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> Type[Any]:
    """Inspects class annotations and dynamically injects dataclass fields."""
    # 1. Collect fields across MRO hierarchy (base classes first)
    fields_dict: Dict[str, _FallbackField] = collections.OrderedDict()

    for base in reversed(cls.__mro__[:-1]):
        base_annotations = getattr(base, "__annotations__", {})
        for name, type_anno in base_annotations.items():
            if name in base.__dict__:
                val = base.__dict__[name]
                if isinstance(val, _FallbackField):
                    f = copy.copy(val)
                    f.name = name
                    f.type = type_anno
                else:
                    f = _FallbackField(name=name, type=type_anno, default=val)
            elif (
                hasattr(base, "__dataclass_fields__")
                and name in base.__dataclass_fields__
            ):
                f = copy.copy(base.__dataclass_fields__[name])
            else:
                f = _FallbackField(
                    name=name, type=type_anno, default=_FALLBACK_MISSING
                )
            fields_dict[name] = f

    # Current class overrides
    cls_annotations = getattr(cls, "__annotations__", {})
    for name, type_anno in cls_annotations.items():
        if name in cls.__dict__:
            val = cls.__dict__[name]
            if isinstance(val, _FallbackField):
                f = copy.copy(val)
                f.name = name
                f.type = type_anno
            else:
                f = _FallbackField(name=name, type=type_anno, default=val)
        else:
            f = _FallbackField(
                name=name, type=type_anno, default=_FALLBACK_MISSING
            )
        fields_dict[name] = f

    # 2. Validate field order: non-default fields cannot follow default fields
    if init:
        seen_default = False
        for f in fields_dict.values():
            if not f.init:
                continue
            has_default = (f.default is not _FALLBACK_MISSING) or (
                f.default_factory is not _FALLBACK_MISSING
            )
            if has_default:
                seen_default = True
            elif seen_default:
                raise TypeError(
                    f"non-default argument {f.name!r} follows default argument"
                )

    # 3. Clean up Field objects from class dict for instance attribute access
    for f in fields_dict.values():
        if f.name in cls.__dict__ and isinstance(
            cls.__dict__[f.name], _FallbackField
        ):
            if f.default is not _FALLBACK_MISSING:
                setattr(cls, f.name, f.default)
            else:
                delattr(cls, f.name)

    # 4. Generate dynamic __init__
    if init:
        init_fields = [f for f in fields_dict.values() if f.init]
        init_field_names = [f.name for f in init_fields]
        init_field_names_set = set(init_field_names)
        non_init_fields = [f for f in fields_dict.values() if not f.init]
        num_init_fields = len(init_fields)
        has_post_init = hasattr(cls, "__post_init__")

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            # Fast path: Complete positional arguments with no kwargs
            if not kwargs and len(args) == num_init_fields:
                if frozen:
                    for f, arg_val in zip(init_fields, args):
                        object.__setattr__(self, f.name, arg_val)
                    for f in non_init_fields:
                        if f.default_factory is not _FALLBACK_MISSING:
                            object.__setattr__(self, f.name, f.default_factory())
                        elif f.default is not _FALLBACK_MISSING and not hasattr(self, f.name):
                            object.__setattr__(self, f.name, f.default)
                    object.__setattr__(self, "_initialized", True)
                else:
                    for f, arg_val in zip(init_fields, args):
                        setattr(self, f.name, arg_val)
                    for f in non_init_fields:
                        if f.default_factory is not _FALLBACK_MISSING:
                            setattr(self, f.name, f.default_factory())
                        elif f.default is not _FALLBACK_MISSING and not hasattr(self, f.name):
                            setattr(self, f.name, f.default)
                if has_post_init:
                    self.__post_init__()
                return

            if len(args) > num_init_fields:
                max_args = num_init_fields + 1
                raise TypeError(
                    f"__init__() takes at most {max_args} positional "
                    f"arguments ({len(args) + 1} given)"
                )

            assigned = set()

            # Positional arguments
            for f, arg_val in zip(init_fields, args):
                if frozen:
                    object.__setattr__(self, f.name, arg_val)
                else:
                    setattr(self, f.name, arg_val)
                assigned.add(f.name)

            # Keyword arguments
            for k, v in kwargs.items():
                if k not in init_field_names_set:
                    raise TypeError(
                        f"__init__() got an unexpected keyword argument {k!r}"
                    )
                if k in assigned:
                    raise TypeError(
                        f"__init__() got multiple values for argument {k!r}"
                    )
                if frozen:
                    object.__setattr__(self, k, v)
                else:
                    setattr(self, k, v)
                assigned.add(k)

            # Defaults & default factories for unassigned init fields
            for f in init_fields:
                if f.name not in assigned:
                    if f.default_factory is not _FALLBACK_MISSING:
                        val = f.default_factory()
                    elif f.default is not _FALLBACK_MISSING:
                        val = f.default
                    else:
                        raise TypeError(
                            "__init__() missing required positional argument: "
                            f"{f.name!r}"
                        )

                    if frozen:
                        object.__setattr__(self, f.name, val)
                    else:
                        setattr(self, f.name, val)

            # Fields with init=False
            for f in non_init_fields:
                if f.default_factory is not _FALLBACK_MISSING:
                    val = f.default_factory()
                    if frozen:
                        object.__setattr__(self, f.name, val)
                    else:
                        setattr(self, f.name, val)
                elif f.default is not _FALLBACK_MISSING and not hasattr(
                    self, f.name
                ):
                    if frozen:
                        object.__setattr__(self, f.name, f.default)
                    else:
                        setattr(self, f.name, f.default)

            if frozen:
                object.__setattr__(self, "_initialized", True)

            # Call __post_init__ if defined on the class
            if has_post_init:
                self.__post_init__()

        cls.__init__ = __init__

    # 5. Generate dynamic __repr__
    if repr:
        repr_fields = [f for f in fields_dict.values() if f.repr]

        def __repr__(self: Any) -> str:
            fields_str = ", ".join(
                f"{f.name}={getattr(self, f.name)!r}" for f in repr_fields
            )
            return f"{cls.__name__}({fields_str})"

        cls.__repr__ = __repr__

    # 6. Generate dynamic __eq__
    if eq:
        compare_fields = [f for f in fields_dict.values() if f.compare]

        def __eq__(self: Any, other: Any) -> bool:
            if other is self:
                return True
            if other.__class__ is not self.__class__:
                return NotImplemented
            self_vals = tuple(getattr(self, f.name) for f in compare_fields)
            other_vals = tuple(getattr(other, f.name) for f in compare_fields)
            return self_vals == other_vals

        cls.__eq__ = __eq__

    # 7. Frozen instance protection
    if frozen:

        def __setattr__(self: Any, name: str, value: Any) -> None:
            if getattr(self, "_initialized", False):
                raise _FallbackFrozenInstanceError(
                    f"cannot assign to field {name!r}"
                )
            object.__setattr__(self, name, value)

        def __delattr__(self: Any, name: str) -> None:
            if getattr(self, "_initialized", False):
                raise _FallbackFrozenInstanceError(
                    f"cannot delete field {name!r}"
                )
            object.__delattr__(self, name)

        cls.__setattr__ = __setattr__
        cls.__delattr__ = __delattr__

    # 8. Attach metadata descriptors and markers
    cls.__dataclass_fields__ = fields_dict
    cls._is_dataclass = True

    return cls


def _FallbackDataclass(
    _cls: Optional[Type[Any]] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> Any:
    """Decorator for creating fallback dataclasses when stdlib is absent."""

    def wrap(cls: Type[Any]) -> Type[Any]:
        return _process_fallback_class(
            cls,
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        )

    if _cls is None:
        return wrap
    return wrap(_cls)


# ==============================================================================
# 2. Transparent Dynamic Re-Export / Stdlib Detection
# ==============================================================================

try:
    import dataclasses

    dataclass = dataclasses.dataclass
    field = dataclasses.field
    asdict = dataclasses.asdict
    is_dataclass = dataclasses.is_dataclass
    FrozenInstanceError = dataclasses.FrozenInstanceError
    Field = dataclasses.Field
    MISSING = dataclasses.MISSING
    _HAS_NATIVE_DATACLASSES = True
except (ImportError, AttributeError):
    dataclass = _FallbackDataclass
    field = _fallback_field
    asdict = _fallback_asdict
    is_dataclass = _fallback_is_dataclass
    FrozenInstanceError = _FallbackFrozenInstanceError
    Field = _FallbackField
    MISSING = _FALLBACK_MISSING
    _HAS_NATIVE_DATACLASSES = False

__all__ = [
    "dataclass",
    "field",
    "asdict",
    "is_dataclass",
    "FrozenInstanceError",
    "Field",
    "MISSING",
    "_FallbackDataclass",
    "_FallbackField",
    "_fallback_field",
    "_fallback_asdict",
    "_fallback_is_dataclass",
    "_FallbackFrozenInstanceError",
    "_FALLBACK_MISSING",
    "_HAS_NATIVE_DATACLASSES",
]
