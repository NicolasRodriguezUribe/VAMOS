from vamos.foundation.registry import Registry
import pytest

def test_registry_basic():
    reg = Registry[int]("TestReg")
    reg.register("foo", 1)
    assert "foo" in reg
    assert reg.get("foo") == 1
    assert reg["foo"] == 1
    assert reg.list() == ["foo"]

def test_registry_decorator():
    reg = Registry[type]("Classes")
    
    @reg.register("my_class")
    class MyClass:
        pass
        
    assert "my_class" in reg
    assert reg.get("my_class") is MyClass

def test_registry_duplicate_error():
    reg = Registry[int]()
    reg.register("a", 1)
    with pytest.raises(ValueError, match="already exists"):
        reg.register("a", 2)

def test_registry_override():
    reg = Registry[int]()
    reg.register("a", 1)
    reg.register("a", 2, override=True)
    assert reg.get("a") == 2

def test_registry_get_default():
    reg = Registry[int]()
    assert reg.get("missing", 99) == 99
    with pytest.raises(KeyError):
        reg.get("missing")
