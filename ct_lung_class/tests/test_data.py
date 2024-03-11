from ..data import get_output_prefix, PROJECT_ID

def test_get_output_prefix():
    assert get_output_prefix(f"sometext`_{PROJECT_ID}_123`") == "123"