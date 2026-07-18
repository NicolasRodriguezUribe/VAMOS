import vamos
import vamos.resources as resources


def test_python310_runtime_imports():
    assert vamos.__file__ is not None
    assert resources.__file__ is not None
    assert resources.Traversable is not None
