from ..data import PROJECT_ID, get_output_prefix


def test_get_output_prefix():
    assert get_output_prefix(f"sometext`_{PROJECT_ID}_123`") == "123"
