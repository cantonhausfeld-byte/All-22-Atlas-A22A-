from a22a.tools.doctor import run_doctor

def test_doctor_runs():
    assert run_doctor(ci=True) is True
