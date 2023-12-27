import inspect
from inspect import signature

import numpy as np

TYPE_ALIAS = "type_"


class UniversalFactory:
    def __init__(self, *args):
        self._class_names = {}
        for arg in args:
            self._class_names[arg.__name__] = arg

    def make(self, function=None, parameters=None, **kwargs):
        result = self._make_impl(function, parameters, **kwargs)
        return result

    def _make_impl(self, function, parameters, **kwargs):
        if parameters is None:
            raise ValueError("parameters can not be None")
        if function is None:
            if isinstance(parameters, dict) and TYPE_ALIAS in parameters.keys():
                function = self._class_names[parameters[TYPE_ALIAS]]
                return self._make_impl(function, parameters, **kwargs)
            return parameters
        if function is int:
            return int(parameters)
        if function is float:
            return float(parameters)
        if function is str:
            return str(parameters)
        if function == np.array or function == np.ndarray:
            return np.array(parameters)
        if self._is_list(function):
            return self._load_from_list(function, parameters, **kwargs)
        if self._is_dict(function):
            return self._load_from_dict(function, parameters, **kwargs)
        if self._has_from_dict_method(function):
            return function.from_dict(parameters)
        if callable(function):
            return self._load_from_function(function, parameters, **kwargs)
        raise ValueError(f"Can not create {function} from {parameters}")

    # noinspection PyProtectedMember
    def _load_from_function(self, function, parameters, **kwargs):
        function_arguments = signature(function).parameters
        function_parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError(f"parameters {parameters} are not dict")
        for key, value in function_arguments.items():
            annotation = value.annotation
            if annotation is inspect.Signature.empty:
                annotation = None
            if key in parameters.keys():
                function_parameters[key] = self._make_impl(annotation, parameters[key], **kwargs)
                kwargs[key] = function_parameters[key]
            elif key == "parameters" or key == "config":
                function_parameters[key] = self._make_impl(annotation, parameters, **kwargs)
            elif key in kwargs.keys():
                function_parameters[key] = kwargs[key]
            elif value.default is not inspect.Signature.empty:
                pass
            else:
                raise KeyError(f"{key} not found for function {function} with parameters {parameters}")
        return function(**function_parameters)

    @staticmethod
    def _is_list(function):
        return getattr(function, "__origin__", None) is list

    def _load_from_list(self, function, parameters, **kwargs):
        if not isinstance(parameters, list):
            raise ValueError(f"parameters {parameters} are not list")
        result = []
        child_function = function.__args__[0]
        for parameter in parameters:
            result.append(self._make_impl(child_function, parameter, **kwargs))
        return result

    @staticmethod
    def _is_dict(function):
        return getattr(function, "__origin__", None) is dict

    def _load_from_dict(self, function, parameters, **kwargs):
        if not isinstance(parameters, dict):
            raise ValueError(f"parameters {parameters} are not dict")
        result = {}
        key_function = function.__args__[0]
        value_function = function.__args__[1]
        for key, value in parameters.items():
            key_result = self._make_impl(key_function, key, **kwargs)
            value_result = self._make_impl(value_function, value, **kwargs)
            result[key_result] = value_result
        return result

    @staticmethod
    def _has_from_dict_method(function):
        return hasattr(function, "from_dict")
