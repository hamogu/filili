from ..utils import get_flat_element

def test_flat_element():
    inp = [[{'a': '333', 'b':2}],[[[{'c':'qwe'},{'d':2, 'e':3}]]],{'a':1}]
    out = list(get_flat_element(inp, 'a'))
    assert out == ['333', None, None, 1]

    out = list(get_flat_element(inp, 'a', include_missing=False))
    assert out == ['333', 1]
