from slm_lab.env.registration import get_env_path


def test_get_env_path():
    assert 'node_modules/slm-env-3dball/build/3dball' in get_env_path(
        '3dball')
